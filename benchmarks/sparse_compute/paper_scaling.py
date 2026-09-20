"""Current ordinary-Torch forward latency versus graph size, with raw samples."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics
import time


def main():
    import torch
    import tiga as tg
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', default='8192,32768,131072,524288')
    parser.add_argument('--features', default='16,64')
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--seed', type=int, default=20260919)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sizes, widths = [int(n) for n in args.sizes.split(',')], [int(n) for n in args.features.split(',')]
    if min(sizes + widths + [args.repeats]) <= 0:
        parser.error('positive sizes/features/repeats required')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; do not substitute CPU measurements')

    class WeightedSum(tg.MessagePassing):
        reducer = tg.sum()

        def edge(self, src, dst, edge):
            return src.x * edge.w

    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {'schema': 'tiga.paper-scaling.v1', 'seed': args.seed,
              'device': torch.cuda.get_device_name(0), 'torch': torch.__version__,
              'dtype': 'float32', 'index_dtype': 'int32',
              'scope': 'ordinary Torch inputs; forward without gradients; CUDA-event latency including launches; no first compile, topology construction, transfer or backward; no cache flushing',
              'repeats': args.repeats, 'warmup': 5, 'rows': rows}
    for nodes in sizes:
        for features in widths:
            rng = torch.Generator(device='cuda').manual_seed(args.seed)
            degrees = torch.randint(8, 25, (nodes,), generator=rng, device='cuda', dtype=torch.int32)
            rowptr = torch.cat([torch.zeros(1, dtype=torch.int32, device='cuda'), degrees.cumsum(0, dtype=torch.int32)])
            edges = int(rowptr[-1])
            col = torch.randint(nodes, (edges,), generator=rng, device='cuda', dtype=torch.int32)
            weights = torch.randn(edges, generator=rng, device='cuda')
            x = torch.randn(nodes, features, generator=rng, device='cuda')
            destination = torch.repeat_interleave(torch.arange(nodes, device='cuda'), degrees.long())
            csr = torch.sparse_csr_tensor(rowptr, col, weights, size=(nodes, nodes))
            graph = tg.Graph.from_csr(rowptr, col, num_src=nodes)
            kernel = WeightedSum()
            gather_index = col.long()  # Prepare indexing outside the timed baseline.

            def scatter():
                output = torch.zeros_like(x)
                return output.index_add_(0, destination, x[gather_index] * weights[:, None])

            providers = {'tiga.auto': lambda: kernel(graph=graph, src={'x': x}, dst={}, edge={'w': weights}),
                         'torch.sparse.mm': lambda: torch.sparse.mm(csr, x),
                         'torch.index_add': scatter}
            oracle = scatter()
            cold, errors = {}, {}
            for name, function in providers.items():
                torch.cuda.synchronize()
                started = time.perf_counter()
                actual = function()
                torch.cuda.synchronize()
                cold[name] = (time.perf_counter() - started) * 1000
                torch.testing.assert_close(actual, oracle, rtol=3e-4, atol=3e-4)
                errors[name] = float((actual - oracle).abs().max())
                for _ in range(result['warmup']):
                    function()
            torch.cuda.synchronize()
            samples = {name: [] for name in providers}
            order_rng = random.Random(args.seed)
            for _ in range(args.repeats):
                names = list(providers)
                order_rng.shuffle(names)
                for name in names:
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    value = providers[name]()
                    end.record()
                    end.synchronize()
                    samples[name].append(start.elapsed_time(end))
                    del value
            for name in providers:
                row = {'nodes': nodes, 'edges': edges, 'features': features, 'provider': name,
                       'samples_ms': samples[name], 'median_ms': statistics.median(samples[name]),
                       'first_call_wall_ms': cold[name], 'max_abs_error': errors[name], 'correct': True}
                if name == 'tiga.auto':
                    row['diagnostics'] = str(kernel.explain())
                rows.append(row)
            args.output.write_text(json.dumps(result, indent=2) + '\n')
            print(nodes, features, {name: round(statistics.median(v), 4) for name, v in samples.items()}, flush=True)
            del actual, oracle, x, weights, csr, graph, kernel, col, gather_index, rowptr, destination, degrees
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
