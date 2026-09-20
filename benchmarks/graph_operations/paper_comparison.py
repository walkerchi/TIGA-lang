"""Fresh-process, matched forward latency and allocated-memory comparisons.

Dynamic cases alternate two genuinely different coordinate snapshots and rebuild
the relation on every call. No dense distance matrix is used by the PyG peer.
Memory is PyTorch peak allocated bytes (including live inputs/output/workspace),
not reserved bytes, a model, or whole-device usage. Run providers separately.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import sys
import subprocess
import time

import torch
import tiga as tg
from torch_geometric.nn import MessagePassing, knn_graph, radius_graph
from benchmarks.graph_operations.torch_graph_baseline import PAIR_BUDGET, knn_indices, radius_csr


class Weighted(tg.MessagePassing):
    reducer = tg.sum()

    def edge(self, src, dst, edge):
        return src.x * edge.w


class Distance(tg.MessagePassing):
    reducer = tg.sum()

    def edge(self, src, dst, edge):
        return src.x * edge.distance


class PyGWeighted(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add', node_dim=0)

    def forward(self, edge_index, x, w):
        return self.propagate(edge_index, x=x, w=w, size=(x.shape[0], x.shape[0]))

    def message(self, x_j, w):
        return x_j * w.reshape((-1,) + (1,) * (x_j.ndim-1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workload', choices=['csr', 'radius', 'knn'], required=True)
    parser.add_argument('--provider', choices=['tiga', 'pyg', 'sparse', 'materialized'], required=True)
    parser.add_argument('--nodes', type=int, required=True)
    parser.add_argument('--features', type=int, default=32)
    parser.add_argument('--degree', type=int, default=32)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.nodes, args.features, args.degree, args.repeats) < 1 or args.degree >= args.nodes:
        parser.error('positive dimensions and degree<nodes required')
    if args.workload != 'csr' and args.features != 1:
        parser.error('this generated dynamic envelope uses scalar node features')
    if (args.workload == 'csr' and args.provider == 'materialized') or (args.workload != 'csr' and args.provider == 'sparse'):
        parser.error('unsupported comparison pair')
    torch.set_num_threads(1)
    rng = torch.Generator(device='cuda').manual_seed(20260919)
    n, k, f = args.nodes, args.degree, args.features
    x = torch.rand((n, f) if args.workload == 'csr' else (n,), generator=rng, device='cuda')
    pyg = PyGWeighted()
    kernel = Distance() if args.workload == 'radius' else Weighted()
    expected = []
    edge_counts = []
    max_degrees = []
    if args.workload == 'csr':
        rows = torch.arange(n+1, device='cuda', dtype=torch.int32) * k
        cols = torch.randint(n, (n*k,), device='cuda', generator=rng, dtype=torch.int32)
        weights = torch.rand(n*k, device='cuda', generator=rng)
        destinations = torch.arange(n, device='cuda').repeat_interleave(k)
        indices = torch.stack((cols.long(), destinations))
        expected = [pyg(indices, x, weights).cpu()]
        edge_counts = [n*k]
        if args.provider == 'tiga':
            graph = tg.Graph.from_csr(rows, cols, num_src=n)
            call = lambda _: kernel(graph=graph, src={'x': x}, dst={}, edge={'w': weights})
        elif args.provider == 'sparse':
            graph = torch.sparse_csr_tensor(rows, cols, weights, size=(n, n))
            call = lambda _: torch.sparse.mm(graph, x)
        else:
            call = lambda _: pyg(indices, x, weights)
        del destinations
        if args.provider != 'pyg':
            del indices
    else:
        snapshots = [torch.rand(n, 3, device='cuda', generator=rng) for _ in range(2)]
        if args.workload == 'radius':
            # Binary-grid squared distances are exact in FP32. Put the cutoff
            # between integer squared-distance levels, not on an FP boundary.
            snapshots = [(p*1024).floor()/1024 for p in snapshots]
        positions = snapshots[0].clone()
        cutoff = (k / (n * 4 * math.pi/3)) ** (1/3)
        if args.workload == 'radius':
            cutoff = math.sqrt(math.floor((cutoff*1024)**2)+.5)/1024
        weights = torch.ones(n*k, device='cuda') if args.workload == 'knn' else None

        def build_peer(p):
            if args.workload == 'knn':
                return knn_graph(p, k=k, loop=False, flow='source_to_target')
            # The full edge-set check below proves this cap does not truncate
            # either measured snapshot. Avoid an artificial N*N output buffer.
            return radius_graph(p, r=cutoff, loop=False, max_num_neighbors=4*k,
                                flow='source_to_target')

        for p in snapshots:
            indices = build_peer(p)
            if args.workload == 'radius':
                graph = radius_csr(p, cutoff)
                rows, cols = graph.crow_indices(), graph.col_indices()
            else:
                # Direct squared distances, bounded row chunks even at 128K.
                cols = knn_indices(p, k).flatten()
                rows = torch.arange(n+1, device='cuda')*k
                graph = None
            # Compare edge sets, not only an aggregation that could hide errors.
            ids = cols.long() + torch.arange(n, device='cuda').repeat_interleave((rows[1:]-rows[:-1]).long()) * n
            peer_ids = indices[0] + indices[1]*n
            torch.testing.assert_close(ids.sort().values, peer_ids.sort().values, rtol=0, atol=0)
            edge_counts.append(indices.shape[1])
            max_degrees.append(int((rows[1:]-rows[:-1]).max()))
            w = (p[indices[0]]-p[indices[1]]).norm(dim=1) if args.workload == 'radius' else torch.ones(indices.shape[1], device='cuda')
            expected.append(pyg(indices, x, w).cpu())
            del indices, graph, rows, cols, ids, peer_ids, w

        def call(step):
            positions.copy_(snapshots[step % 2])  # version changes: no stale builder cache
            if args.provider == 'pyg':
                indices = build_peer(positions)
                w = ((positions[indices[0]]-positions[indices[1]]).norm(dim=1)
                     if args.workload == 'radius' else torch.ones(indices.shape[1], device='cuda'))
                return pyg(indices, x, w)
            if args.provider == 'materialized' and args.workload == 'knn':
                indices = knn_indices(positions, k)
                return x[indices].sum(dim=1)
            if args.provider == 'materialized':
                csr = radius_csr(positions, cutoff)
                return torch.sparse.mm(csr, x[:, None])[:, 0]
            graph = (tg.Graph.radius(positions, cutoff) if args.workload == 'radius'
                     else tg.Graph.knn(positions, k))
            if args.provider == 'tiga':
                return kernel(graph=graph, src={'x': x}, dst={},
                              **({'edge': {'w': weights}} if weights is not None else {}))
            raise AssertionError('unhandled provider')

    first = time.perf_counter()
    actual = call(0)
    torch.cuda.synchronize()
    first_ms = (time.perf_counter()-first)*1000
    torch.testing.assert_close(actual.cpu(), expected[0], rtol=3e-4, atol=3e-4)
    del actual
    for step in range(4):
        actual = call(step)
        torch.testing.assert_close(actual.cpu(), expected[step % len(expected)], rtol=3e-4, atol=3e-4)
        del actual
    gc.collect()
    torch.cuda.empty_cache()
    samples, peaks, errors = [], [], []
    for step in range(args.repeats):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        actual = call(step)
        torch.cuda.synchronize()
        samples.append((time.perf_counter()-started)*1000)
        peaks.append(torch.cuda.max_memory_allocated())
        observed = actual.cpu()
        reference = expected[step % len(expected)]
        torch.testing.assert_close(observed, reference, rtol=3e-4, atol=3e-4)
        errors.append(float((observed-reference).abs().max()))
        del actual, observed
    root = Path(__file__).resolve().parents[2]
    report = dict(schema='tiga.three-questions.comparison.v2',
                  **(vars(args) | {'output': str(args.output)}), device=torch.cuda.get_device_name(),
                  samples_ms=samples, median_ms=statistics.median(samples),
                  peak_allocated_bytes=max(peaks), peak_samples_bytes=peaks,
                  max_abs_error=max(errors), correct=True, edges=edge_counts,
                  max_degrees=max_degrees, first_call_ms=first_ms, warmup=4,
                  scope='warm synchronized wall forward; dynamic coordinate copy + rebuild + consume; no backward',
                  memory_scope='Torch allocated, live inputs + output + intermediates; excludes allocator reserve, CUDA context and other processes',
                  torch_distance_pair_budget=PAIR_BUDGET,
                  torch_baseline_sha256=hashlib.sha256((root/'benchmarks/graph_operations/torch_graph_baseline.py').read_bytes()).hexdigest(),
                  versions={name: importlib.metadata.version(name) for name in ('torch','torch-geometric','pyg-lib','torch-cluster')},
                  command=sys.argv, seed=20260919,
                  coordinates=('binary grid 1024^3; cutoff between squared-distance levels' if args.workload == 'radius' else 'uniform FP32'),
                  timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  gpu_state=subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.used,memory.total', '--format=csv,noheader'], text=True).strip(),
                  compiler_sha256={key: hashlib.sha256(Path(os.environ[key]).read_bytes()).hexdigest() for key in ('TIGA_OPT','TIGA_TRANSLATE')},
                  source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root/'python/tiga').rglob('*.py'))},
                  benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  diagnostics=str(kernel.explain()) if args.provider == 'tiga' else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(args.workload, args.provider, n, report['median_ms'], report['peak_allocated_bytes'], flush=True)


if __name__ == '__main__':
    main()
