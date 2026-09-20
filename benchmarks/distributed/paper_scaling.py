"""Single-GPU and communication-then-compute forward timings on trusted hosts.

Run one rank per host with matching arguments and source. The two-rank critical
path is the maximum of paired rank durations, not their mean. TCP uses host
staging; NCCL uses device buffers. Neither implies GPUDirect RDMA or GPU overlap.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import importlib.metadata
import os
from pathlib import Path
import platform
import socket
import statistics
import subprocess
import time

import numpy as np
import tiga as tg
from tiga.distributed import DistributedRuntime, TCPTransport, NCCLTransport, nccl_unique_id


class MeanNeighbors(tg.MessagePassing):
    reducer = tg.sum()

    def edge(self, src, dst, edge):
        return src.x

    def node(self, dst, aggregate):
        return aggregate * (1.0 / 16)


class MeshNeighborSum(tg.MessagePassing):
    reducer = tg.sum()

    def edge(self, src, dst, edge):
        return src.x


def native(array, device):
    array = np.ascontiguousarray(array)
    dtype = {np.dtype('float32'): tg.float32, np.dtype('int64'): tg.int64}[array.dtype]
    value = tg.empty(array.shape, dtype=dtype, device=device)
    value._buffer.write(array.tobytes())
    return value


def case(nodes, features, boundary_every=4):
    """25% boundary rows, 16 edges/row; IDs and feature values are deterministic."""
    ids = np.arange(nodes, dtype=np.int64)
    half = nodes // 2
    columns = (ids[:, None] % half + np.arange(1, 17)) % half + (ids[:, None] // half) * half
    if boundary_every:
        columns[ids % boundary_every == 0] = (columns[ids % boundary_every == 0] + half) % nodes
    else:
        # Local ring: only rows near ownership boundaries require a halo.
        columns = (ids[:, None] + np.arange(1, 17)) % nodes
    values = ((ids[:, None] % 97) + np.arange(features)[None, :] % 7).astype(np.float32)
    expected = np.zeros_like(values)
    for e in range(16):
        expected += values[columns[:, e]]
    return np.arange(nodes + 1, dtype=np.int64) * 16, columns.ravel(), values, expected / 16


def mesh_case(side, features):
    """Nonperiodic Q1-hexahedral mesh adjacency (26 neighbors in the interior).

    Nodes sharing an element connect in both directions, excluding diagonal
    self entries. This measures graph aggregation, not a full FEM solver.
    Lexicographic x-major numbering makes the equal-row split a spatial plane.
    """
    if side < 4 or side % 2:
        raise ValueError('mesh side must be even and >=4')
    nodes=side**3
    ids=np.arange(nodes,dtype=np.int64)
    x=ids//(side*side); y=(ids//side)%side; z=ids%side
    columns=np.full((nodes,26),-1,dtype=np.int64)
    values=((ids[:,None]%97)+np.arange(features)[None,:]%7).astype(np.float32)
    expected=np.zeros_like(values)
    slot=0
    for dx in (-1,0,1):
        for dy in (-1,0,1):
            for dz in (-1,0,1):
                if (dx,dy,dz)==(0,0,0): continue
                valid=(x+dx>=0)&(x+dx<side)&(y+dy>=0)&(y+dy<side)&(z+dz>=0)&(z+dz<side)
                source=ids[valid]+dx*side*side+dy*side+dz
                columns[valid,slot]=source
                expected[valid]+=values[source]
                slot+=1
    valid=columns>=0
    rows=np.concatenate(([0],valid.sum(axis=1).cumsum())).astype(np.int64)
    return rows,columns[valid],values,expected


class MeasuredTransport:
    def __init__(self, base, control=None):
        self.base = base
        self.rank, self.world_size = base.rank, base.world_size
        self.events = []
        self.control = base if control is None else control

    def send(self, peer, data):
        return self.base.send(peer, data)

    def receive(self, peer):
        return self.base.receive(peer)

    def send_bytes(self, peer, data):
        start = time.perf_counter_ns()
        self.base.send_bytes(peer, data)
        self.events.append(('send', len(data), start, time.perf_counter_ns()))

    def receive_bytes(self, peer, size):
        start = time.perf_counter_ns()
        value = self.base.receive_bytes(peer, size)
        self.events.append(('receive', size, start, time.perf_counter_ns()))
        return value

    def barrier(self, label):
        if self.rank == 0:
            self.control.send(1, label)
            assert self.control.receive(1) == label
        else:
            assert self.control.receive(0) == label
            self.control.send(0, label)


class MeasuredNCCL(MeasuredTransport):
    def exchange_device(self, sends, receives, *, stream):
        start = time.perf_counter_ns()
        result = self.base.exchange_device(sends, receives, stream=stream)
        end = time.perf_counter_ns()
        self.events.extend(('send', region.byte_count, start, end) for _, region in sends)
        self.events.extend(('receive', region.byte_count, start, end) for _, region in receives)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rank', type=int, choices=(0, 1), required=True)
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', type=int, default=29641)
    parser.add_argument('--sizes', default='8192,32768,131072')
    parser.add_argument('--topology', choices=('ring','mesh'), default='ring')
    parser.add_argument('--mesh-sides', default='32,64,96')
    parser.add_argument('--features', type=int, default=16)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--transport', choices=('tcp', 'nccl'), default='tcp')
    parser.add_argument('--boundary-every', type=int, default=4,
                        help='0: local ring; otherwise every kth row uses remote sources')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sizes = [int(n) for n in args.sizes.split(',')]
    sides = [int(n) for n in args.mesh_sides.split(',')]
    if args.topology == 'mesh':
        if any(s < 4 or s % 2 for s in sides):
            parser.error('mesh sides must be even and >=4')
        sizes = [s**3 for s in sides]
    if any(n < 32 or n % 2 for n in sizes) or min(args.features, args.repeats) < 1 or args.boundary_every < 0:
        parser.error('even sizes >=32 and positive features/repeats required')
    device = 'cuda:0'
    connect = TCPTransport.host if args.rank == 0 else TCPTransport.join
    control = connect(args.rank, 2, host=args.host, port=args.port, timeout=180)
    base = control
    if args.transport == 'nccl':
        if args.rank == 0:
            identifier = nccl_unique_id()
            control.send(1, identifier)
        else:
            identifier = control.receive(0)
        base = NCCLTransport(args.rank, 2, identifier, device=device)
    transport = (MeasuredNCCL if args.transport == 'nccl' else MeasuredTransport)(base, control)
    root = Path(__file__).resolve().parents[2]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted((root/'python/tiga').rglob('*.py'))}
    result = dict(schema='tiga.two-host-scaling.v1', rank=args.rank,
                  hostname=socket.gethostname(), platform=platform.platform(),
                  gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.used,memory.total',
                                               '--format=csv,noheader'], text=True).strip(),
                  transport='nccl-socket' if args.transport == 'nccl' else 'tcp-host-staged', features=args.features, degree=26 if args.topology=='mesh' else 16,
                  topology=args.topology, mesh_sides=sides if args.topology=='mesh' else None,
                  boundary_every=args.boundary_every if args.topology=='ring' else None,
                  boundary_fraction=(1/args.boundary_every if args.boundary_every else None) if args.topology=='ring' else None,
                  warmup=args.warmup, repeats=args.repeats,
                  nccl_version=base.version if args.transport == 'nccl' else None,
                  nccl_library_sha256=hashlib.sha256(Path(base.library_path).read_bytes()).hexdigest() if args.transport == 'nccl' else None,
                  command=__import__('sys').argv, timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  versions={name: importlib.metadata.version(name) for name in ('numpy','triton')},
                  python=platform.python_version(),
                  compiler_hashes={name: hashlib.sha256(Path(os.environ[name]).read_bytes()).hexdigest()
                                   for name in ('TIGA_OPT','TIGA_TRANSLATE') if os.environ.get(name)},
                  source_hashes=hashes, benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='host wall time through synchronized output; topology/input creation, barriers and oracle excluded; changed inputs; forward only', rows=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with DistributedRuntime(transport) as runtime:
            for shape_index,nodes in enumerate(sizes):
                side=sides[shape_index] if args.topology=='mesh' else None
                rowptr, columns, values, expected = (
                    mesh_case(side,args.features) if side is not None
                    else case(nodes, args.features, args.boundary_every))
                full = tg.Graph.from_csr(native(rowptr, device), native(columns, device), num_src=nodes)
                sharded = full.halo(tg.DeviceMesh('cuda', 2), depth=1)
                begin, end = args.rank * (nodes//2), (args.rank+1) * (nodes//2)
                program = MeshNeighborSum() if side is not None else MeanNeighbors()
                local_columns=columns[rowptr[begin]:rowptr[end]]
                ghosts=np.unique(local_columns[(local_columns<begin)|(local_columns>=end)])
                boundary_rows=(side*side if side is not None else None)
                for mode in ('single', 'serialized'):
                    samples, traces, traffic = [], [], []
                    # Exercise the public communication-then-compute default.
                    runtime._force_serialized = None
                    for step in range(args.warmup + args.repeats):
                        scale = float(1 + step % 3)
                        x = native((values if mode == 'single' else values[begin:end]) * scale, device)
                        transport.barrier((nodes, mode, step))
                        transport.events.clear()
                        start = time.perf_counter_ns()
                        y = program(graph=full if mode == 'single' else sharded, src={'x': x}, dst={})
                        y.realize()
                        if y.ready_event is not None:
                            y.ready_event.wait()
                        finished = time.perf_counter_ns()
                        elapsed = (finished - start) / 1e6
                        np.testing.assert_array_equal(y.to_numpy(), (expected if mode == 'single' else expected[begin:end]) * scale)
                        if step >= args.warmup:
                            samples.append(elapsed)
                            traces.append(None if mode == 'single' else runtime.last_execution_trace)
                            events = transport.events[:]
                            traffic.append(dict(send_bytes=sum(e[1] for e in events if e[0]=='send'),
                                                receive_bytes=sum(e[1] for e in events if e[0]=='receive'),
                                                forward_started_ns=start, forward_finished_ns=finished,
                                                exchange_started_ns=min(e[2] for e in events) if events else None,
                                                socket_span_ms=((max(e[3] for e in events)-min(e[2] for e in events))/1e6 if events else 0) if args.transport == 'tcp' else None,
                                                enqueue_span_ms=(max(e[3] for e in events)-min(e[2] for e in events))/1e6 if events else 0))
                        del y, x
                    result['rows'].append(dict(nodes=nodes, mode=mode, correct=True,
                                               edges=int(len(columns)),mesh_side=side,
                                               boundary_rows=boundary_rows,
                                               boundary_fraction=boundary_rows/(end-begin) if side is not None else None,
                                               ghost_nodes=int(len(ghosts)),
                                               median_ms=statistics.median(samples), samples_ms=samples,
                                               traces=traces, traffic=traffic))
                    args.output.write_text(json.dumps(result, indent=2)+'\n')
                    print(args.rank, nodes, mode, round(statistics.median(samples), 3), flush=True)
                del full, sharded
    finally:
        base.close()
        if base is not control:
            control.close()


if __name__ == '__main__':
    main()
