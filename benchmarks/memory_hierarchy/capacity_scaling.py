"""Fresh-process CPU CSR capacity/time sweep under a native allocation budget.

This is not an OS RSS limit, a cold-disk bandwidth test or GPU out-of-core execution.
Fixtures are built separately; each configuration runs in a fresh child process.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

DEGREE = 16


def fixture(path, nodes):
    import tiga as tg
    if path.exists():
        raise FileExistsError(path)
    graph = tg.Graph.stencil((nodes,), tuple((i,) for i in range(1, DEGREE + 1)), periodic=True)
    x = tg.tensor([float(i % 97) for i in range(nodes)], dtype=tg.float32)
    tg.save(graph, path, fields={"src": {"x": x}})


def worker(args):
    import numpy as np
    import tiga as tg

    class MeanNeighbors(tg.MessagePassing):
        reducer = tg.sum()

        def edge(self, src, dst, edge):
            return src.x

        def node(self, dst, aggregate):
            return aggregate * (1.0 / DEGREE)

    result = {"nodes": args.nodes, "edges": args.nodes * DEGREE,
              "mode": args.mode, "page_rows": args.page_rows,
              "budget_bytes": args.budget_mib * 2**20,
              "prefetch": False, "samples_s": [], "status": "not_started"}
    directory = args.cache_dir / f"ring-{args.nodes}.gfg"
    result["stored_bytes"] = sum(p.stat().st_size for p in directory.rglob('*') if p.is_file())
    start = time.perf_counter()
    with tg.execution(device="cpu", memory={"ram": result["budget_bytes"]},
                      page_rows=args.page_rows, prefetch_depth=1) as scope:
        try:
            stored_graph = tg.load(directory)  # Keep its field-reader owner alive.
            graph = stored_graph
            fields = graph.fields("src")
            if args.mode == "resident":
                row, col = graph.resolve_csr()
                graph = tg.Graph.from_csr(row, col, num_src=args.nodes)
                fields["x"].realize()
            result["load_s"] = time.perf_counter() - start
            kernel = MeanNeighbors()

            def execute():
                if args.mode == "paged":
                    output = kernel(graph=graph, src=fields, dst={},
                                    page_rows=args.page_rows, prefetch=False)
                else:
                    output = kernel(graph=graph, src=fields, dst={})
                output.realize()
                return output

            start = time.perf_counter()
            output = execute()
            result["first_call_s"] = time.perf_counter() - start
            result["execution"] = {k: v for k, v in (output.execution or {}).items() if k != 'source'}
            values = output.to_numpy().copy()
            del output
            result["samples_s"] = [result["first_call_s"]]
            result["total_s"] = result["load_s"] + result["first_call_s"]
            result["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            # Independent analytic oracle for the fixed periodic stencil.
            x = (np.arange(args.nodes) % 97).astype(np.float32)
            expected = sum(np.roll(x, -i) for i in range(1, DEGREE + 1)) / DEGREE
            result["max_abs_error"] = float(np.max(np.abs(values - expected)))
            if not np.array_equal(values, expected):
                raise AssertionError(f"oracle mismatch: {result['max_abs_error']}")
            result["status"] = "ok"
        except MemoryError as error:
            result["status"] = "budget_exceeded"
            result["error"] = str(error)
        result["memory_report"] = scope.memory_report()
        result.setdefault("peak_rss_bytes", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("nodes", "mode", "page_rows", "status", "peak_rss_bytes")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nodes', type=int, default=8192)
    parser.add_argument('--sizes', default='8192,32768,131072,524288')
    parser.add_argument('--pages', default='1024,8192')
    parser.add_argument('--page-rows', type=int, default=1024)
    parser.add_argument('--budget-mib', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=3, help='independent fresh-process trials per configuration')
    parser.add_argument('--mode', choices=['resident', 'paged'], default='paged')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.nodes, args.page_rows, args.budget_mib, args.repeats) <= 0:
        parser.error('sizes, page rows, budget and repeats must be positive')
    if args.build:
        fixture(args.cache_dir / f'ring-{args.nodes}.gfg', args.nodes)
        return
    if args.worker:
        worker(args)
        return
    sizes = [int(n) for n in args.sizes.split(',')]
    pages = [int(n) for n in args.pages.split(',')]
    if min(sizes + pages) <= 0:
        parser.error('all sweep sizes must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    common = [sys.executable, '-m', 'benchmarks.memory_hierarchy.capacity_scaling',
              '--cache-dir', str(args.cache_dir), '--budget-mib', str(args.budget_mib),
              '--repeats', str(args.repeats)]
    rows = []
    for nodes in sizes:
        subprocess.run(common + ['--build', '--nodes', str(nodes), '--output', str(args.output)], check=True, timeout=300)
        for mode, page in [('resident', nodes), *[('paged', p) for p in pages]]:
            for trial in range(args.repeats):
                output = args.output / f'n{nodes}-{mode}-p{page}-r{trial}.json'
                subprocess.run(common + ['--worker', '--nodes', str(nodes), '--mode', mode,
                                        '--page-rows', str(page), '--output', str(output)], check=True, timeout=300)
                row = json.loads(output.read_text())
                row['trial'] = trial
                rows.append(row)
    summary = {'schema': 'tiga.capacity-scaling.v1', 'rows': rows, 'seed': 'deterministic ring; x[i]=i%97',
               'degree': DEGREE, 'dtype': 'float32', 'device': 'cpu',
               'scope': 'fresh process per trial; total=load+first execution including JIT; native RAM budget, not RSS; fixture ingest excluded; no forced OS cache eviction; forward only',
               'threads': {key: os.environ.get(key) for key in ('OMP_NUM_THREADS', 'TIGA_NUM_THREADS')}}
    (args.output / 'results.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
