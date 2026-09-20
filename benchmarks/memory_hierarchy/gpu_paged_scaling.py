"""Fresh-process CUDA graph capacity and recurrent execution from an NVMe store.

Native GPU memory is budgeted; host RSS and the OS cache are not. Pages travel
through host memory, not GPUDirect Storage. Ingest and oracle checks are untimed.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np
import tiga as tg

from benchmarks.distributed.paper_scaling import native


class Average(tg.MessagePassing):
    reducer = tg.sum()

    def edge(self, src, dst, edge):
        return src.x

    def node(self, dst, aggregate):
        return aggregate * (1.0 / 16)


def initial(nodes):
    return (np.arange(nodes)[:, None] % 97 + np.arange(4)[None, :]).astype(np.float32)


def build(path, nodes):
    if path.exists():
        raise FileExistsError(path)
    ids = np.arange(nodes, dtype=np.int64)
    columns = (ids[:, None] + np.arange(1, 17)) % nodes
    graph = tg.Graph.from_csr(native(np.arange(nodes+1, dtype=np.int64)*16, 'cpu'),
                             native(columns.ravel(), 'cpu'), num_src=nodes)
    tg.save(graph, path, fields={'src': {'x': native(initial(nodes), 'cpu')}})


def worker(args):
    path = args.cache_dir / f'ring-{args.nodes}.gfg'
    report = dict(nodes=args.nodes, edges=args.nodes*16, features=4, mode=args.mode,
                  page_rows=args.page_rows, steps=[], status='started',
                  stored_bytes=sum(p.stat().st_size for p in path.rglob('*') if p.is_file()),
                  gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.used,memory.total',
                                               '--format=csv,noheader'], text=True).strip())
    expected = initial(args.nodes)
    with tg.execution(device='cuda', memory={'device': args.budget_mib << 20},
                      page_rows=args.page_rows, prefetch_depth=1) as scope:
        started = time.perf_counter()
        try:
            graph = tg.load(path, device='cuda')
            x = graph.fields('src')['x']
            if args.mode == 'resident':
                # Exact same topology/values; include full loading in load_s.
                rows, columns = graph.resolve_csr()
                x.realize()
                graph = tg.Graph.from_csr(rows, columns, num_src=args.nodes)
                del rows, columns
            report['load_s'] = time.perf_counter() - started
            program = Average()
            for step in range(args.steps):
                started = time.perf_counter()
                y = program(graph=graph, src={'x': x}, dst={},
                            **({'prefetch': False} if args.mode == 'paged' else {}))
                y.realize()
                if y.ready_event is not None:
                    y.ready_event.wait()
                elapsed = time.perf_counter() - started
                backend = y.execution or {}
                if args.mode == 'paged':
                    assert backend['page_backends'] == ['cuda-ttir-triton'], backend
                else:
                    assert backend['backend'] == 'cuda-ttir-triton', backend
                # Detach history for this forward-only recurrence, retain the
                # output allocation. This is not a training/backward benchmark.
                x = tg.Tensor(y.shape, dtype=y.dtype, device=y.device, buffer=y._buffer)
                del y
                gc.collect()
                snapshot = scope.memory_report()
                expected = sum(np.roll(expected, -i, axis=0) for i in range(1,17)) / 16
                actual = x.to_numpy()
                np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-6)
                report['steps'].append(dict(step=step, seconds=elapsed,
                                           max_abs_error=float(np.max(np.abs(actual-expected))),
                                           memory=snapshot))
                del actual
            report['status'] = 'ok'
            report['max_abs_error'] = max(s['max_abs_error'] for s in report['steps'])
        except MemoryError as error:
            report['status'] = 'budget_exceeded'
            report['error'] = str(error)
        report['memory'] = scope.memory_report()
        report['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(args.nodes, args.mode, args.page_rows, report['status'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', default='32768,131072,524288')
    parser.add_argument('--nodes', type=int, default=32768)
    parser.add_argument('--pages', default='1024,8192')
    parser.add_argument('--page-rows', type=int, default=1024)
    parser.add_argument('--budget-mib', type=int, default=24)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--mode', choices=('resident','paged'), default='paged')
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.nodes,args.page_rows,args.budget_mib,args.steps,args.trials) <= 0:
        parser.error('sizes, budget, steps and trials must be positive')
    if args.build:
        build(args.cache_dir/f'ring-{args.nodes}.gfg', args.nodes)
        return
    if args.worker:
        worker(args)
        return
    sizes = [int(n) for n in args.sizes.split(',')]
    pages = [int(n) for n in args.pages.split(',')]
    if not sizes or not pages or min(sizes+pages) <= 0:
        parser.error('positive sweep sizes required')
    args.output.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    common = [sys.executable,'-m','benchmarks.memory_hierarchy.gpu_paged_scaling',
              '--cache-dir',str(args.cache_dir),'--steps',str(args.steps),'--budget-mib',str(args.budget_mib)]
    results = dict(schema='tiga.gpu-paged-scaling.v1', budget_mib=args.budget_mib,
                   steps=args.steps, trials=args.trials,
                   scope='fresh process; per-step synchronous forward including host staging; initial load separate; per-step oracle and GC excluded; native GPU budget not total VRAM/RSS; OS page cache not cleared; topology ingested separately; no gradients', rows=[])
    for n in sizes:
        subprocess.run(common+['--build','--nodes',str(n),'--output',str(args.output)],check=True,timeout=300)
        for mode,page in [('resident',n), *[('paged',p) for p in pages]]:
            for trial in range(args.trials):
                output = args.output/f'n{n}-{mode}-p{page}-r{trial}.json'
                subprocess.run(common+['--worker','--nodes',str(n),'--mode',mode,'--page-rows',str(page),
                                       '--output',str(output)],check=True,timeout=600)
                row = json.loads(output.read_text())
                row['trial'] = trial
                results['rows'].append(row)
                (args.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__ == '__main__':
    main()
