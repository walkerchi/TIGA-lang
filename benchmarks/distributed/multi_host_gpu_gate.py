"""One rank per host: native CUDA forward/VJP with TCP or NCCL halo.

Run rank 0 with --host set to its reachable private IP, and rank 1 with the
same host/port. Local CUDA ordinal is independent of global rank. This is a
correctness gate, not a bandwidth benchmark or a physical GPU-direct claim.
Only connect trusted peers: the existing TCP transport carries pickle data.
"""
from __future__ import annotations

import argparse
import json
import hashlib
import os
from pathlib import Path
import socket

import tiga as tg
from tiga.distributed import (
    DistributedRuntime, TCPTransport, NCCLTransport, nccl_unique_id, owned_range,
)


class NeighborSum(tg.MessagePassing):
    reducer = tg.sum()

    def edge(self, src, dst, edge):
        return src.x


def run_rank(rank, host, port, *, device="cuda:0", timeout=120, transport_kind="tcp"):
    if transport_kind not in {"tcp", "nccl"}:
        raise ValueError("transport_kind must be tcp or nccl")
    os.environ["TIGA_TENSOR_BACKEND"] = "native"
    capability = tg.runtime.cuda_compute_capability(device)
    entities = 8
    graph = tg.Graph.from_csr(
        tg.tensor([3 * row for row in range(entities + 1)], dtype=tg.int64, device=device),
        tg.tensor([source for row in range(entities)
                   for source in ((row - 1) % entities, row, (row + 1) % entities)],
                  dtype=tg.int64, device=device),
        num_src=entities, validate="full",
    ).halo(tg.DeviceMesh("cuda", 2), depth=1)
    begin, end = owned_range(entities, 2, rank)
    x = tg.tensor([float(i) for i in range(begin, end)], device=device, requires_grad=True)
    connect = TCPTransport.host if rank == 0 else TCPTransport.join
    control = connect(rank, 2, host=host, port=port, timeout=timeout)
    transport = control
    try:
        if transport_kind == "nccl":
            if rank == 0:
                identifier = nccl_unique_id()
                control.send(1, identifier)
            else:
                identifier = control.receive(0)
            transport = NCCLTransport(rank, 2, identifier, device=device)
        schedule = ("interior||device-halo->boundary" if transport_kind == "nccl"
                    else "interior||host-staged-halo->boundary")
        records = []
        with DistributedRuntime(transport) as runtime:
            # Exercise both paths. Include repeated automatic launches to catch
            # cached topology/halo state accidentally reusing the first output.
            for serialized, scale in ((True, 1.), (False, 1.), (False, 2.)):
                runtime._force_serialized = None if serialized else False
                output = NeighborSum()(graph=graph, src={"x": x * scale}, dst={})
                values = output.tolist()
                gradient = tg.autograd.grad(output.sum(), x)
                gradients = gradient.tolist()
                expected = ([8., 3., 6., 9.], [12., 15., 18., 13.])[rank]
                trace = runtime.last_execution_trace
                correct = (
                    values == [scale * v for v in expected]
                    and gradients == [3. * scale] * 4
                    and output.execution["backend"] == "cuda-ttir-triton"
                    and gradient.execution["backend"] == "cuda-ttir-triton"
                    and (serialized or trace["schedule"] == schedule)
                )
                records.append(dict(serialized=serialized, scale=scale, output=values,
                                    gradient=gradients, trace=trace, correct=correct,
                                    forward_backend=output.execution["backend"],
                                    backward_backend=gradient.execution["backend"]))
        metadata = {}
        if transport_kind == "nccl":
            metadata = dict(nccl_version=transport.version,
                            nccl_library_sha256=hashlib.sha256(
                                Path(transport.library_path).read_bytes()).hexdigest(),
                            socket_interface=os.environ.get("NCCL_SOCKET_IFNAME"),
                            network_scope="NCCL device-buffer API; physical network is deployment-specific")
        root = Path(__file__).resolve().parents[2]
        source_hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted((root / "python/tiga").rglob("*.py"))}
        return dict(schema="tiga.multi-host-gpu.v1", rank=rank, world_size=2,
                    source_hashes=source_hashes,
                    benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    hostname=socket.gethostname(), device=device, capability=capability,
                    transport="nccl" if transport_kind == "nccl" else "tcp-host-staged",
                    transport_metadata=metadata, records=records,
                    correct=all(record["correct"] for record in records))
    finally:
        try:
            transport.close()
        finally:
            if transport is not control:
                control.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=29570)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--timeout", type=float, default=120.)
    parser.add_argument("--transport", choices=("tcp", "nccl"), default="tcp",
                        help="NCCL uses TCP only to bootstrap its communicator; use an external timeout")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_rank(args.rank, args.host, args.port, device=args.device,
                      timeout=args.timeout, transport_kind=args.transport)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)
    if not result["correct"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
