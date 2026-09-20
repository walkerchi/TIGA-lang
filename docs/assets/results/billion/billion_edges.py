"""Bounded-ingest, fully checked 10M/100M/1B-edge CUDA capacity experiment.

Run build once, then run each worker in a fresh process. The synthetic periodic
CSR has 16 edges/row and 32 FP32 features. All edges are stored and traversed;
only the independent oracle exploits periodicity. No cold-cache claim is made.
The fixture writer emits the existing v2 storage format, not a new public API.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
import tiga as tg

DEGREE = 16
FEATURES = 32
ROOT = Path(__file__).resolve().parents[2]


class Average(tg.MessagePassing):
    reducer = tg.sum()

    def edge(self, src, dst, edge):
        return src.x

    def node(self, dst, aggregate):
        return aggregate * (1.0 / DEGREE)


def values(begin, end):
    return ((np.arange(begin, end, dtype=np.int64)[:, None] % 97)
            + np.arange(FEATURES)[None, :] % 7).astype('<f4')


def oracle(begin, end, nodes):
    ids = np.arange(begin, end, dtype=np.int64)
    total = np.zeros(end - begin, dtype=np.int64)
    for offset in range(1, DEGREE + 1):
        total += ((ids + offset) % nodes) % 97
    return (total[:, None] / DEGREE + np.arange(FEATURES)[None, :] % 7).astype('<f4')


def footprint(nodes):
    topology = (nodes + 1 + nodes * DEGREE) * 8
    field = nodes * FEATURES * 4
    return dict(topology_bytes=topology, source_bytes=field, output_bytes=field,
                stored_bytes=topology + field,
                resident_lower_bound_bytes=topology + 2 * field)


def build(path, edges, chunk_rows=65536):
    if edges <= 0 or edges % DEGREE or chunk_rows <= 0:
        raise ValueError('positive edge count divisible by 16 and positive chunks required')
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(path)
    nodes = edges // DEGREE
    path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(path.parent).free < footprint(nodes)['stored_bytes'] + (1 << 30):
        raise OSError('insufficient free disk space for fixture plus 1 GiB reserve')
    temporary = Path(tempfile.mkdtemp(prefix=f'.{path.name}.', dir=path.parent))
    started = time.perf_counter()
    hashes = {name: hashlib.sha256() for name in ('row_ptr.bin', 'col_idx.bin', 'x.bin')}

    def write(file, array, name):
        payload = memoryview(np.ascontiguousarray(array)).cast('B')
        file.write(payload)
        hashes[name].update(payload)

    try:
        with (temporary/'row_ptr.bin').open('wb') as rows, \
                (temporary/'col_idx.bin').open('wb') as columns, \
                (temporary/'x.bin').open('wb') as field:
            for begin in range(0, nodes, chunk_rows):
                end = min(nodes, begin + chunk_rows)
                ids = np.arange(begin, end, dtype='<i8')
                write(rows, ids * DEGREE, 'row_ptr.bin')
                write(columns, (ids[:, None] + np.arange(1, DEGREE+1)) % nodes,
                      'col_idx.bin')
                write(field, values(begin, end), 'x.bin')
            write(rows, np.array([edges], dtype='<i8'), 'row_ptr.bin')
            for file in (rows, columns, field):
                file.flush()
                os.fsync(file.fileno())
        manifest = dict(format='tiga.gfg.csr.v2', num_src=nodes, num_dst=nodes,
                        num_edges=edges, index_dtype='int64', sorted_by_dst=True,
                        fields=[dict(role='src', name='x', dtype='float32',
                                     shape=[nodes, FEATURES], file='x.bin')])
        (temporary/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        info = dict(edges=edges, nodes=nodes, degree=DEGREE, features=FEATURES,
                    seconds=time.perf_counter()-started, chunk_rows=chunk_rows,
                    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                    payload_sha256={k: h.hexdigest() for k, h in hashes.items()},
                    **footprint(nodes))
        (temporary/'ingest.json').write_text(json.dumps(info, indent=2)+'\n')
        # Validate the public loader before publishing the completed directory.
        tg.load(temporary)
        os.rename(temporary, path)
        print(json.dumps(dict(event='built', path=str(path), **info)), flush=True)
        return info
    except BaseException:
        shutil.rmtree(temporary)
        raise


def gpu_info():
    line = subprocess.check_output(
        ['nvidia-smi', '--id=0', '--query-gpu=name,uuid,driver_version,memory.total,memory.used',
         '--format=csv,noheader,nounits'], text=True, timeout=10).strip()
    name, uuid, driver, total, used = (v.strip() for v in line.split(','))
    return dict(name=name, uuid=uuid, driver=driver, total_bytes=int(total)*2**20,
                used_bytes=int(used)*2**20)


def io_counters():
    return {key: int(value) for key, value in
            (line.split(':') for line in Path('/proc/self/io').read_text().splitlines())}


class Monitor:
    """One-second device-wide samples; not a per-process or exact peak claim."""
    def __init__(self):
        self.stop = threading.Event()
        self.samples = []
        self.errors = []
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop.is_set():
            try:
                self.samples.append(dict(seconds=time.monotonic()-self.started, **gpu_info()))
            except Exception as error:
                self.errors.append(str(error))
            self.stop.wait(1)

    def __enter__(self):
        self.started = time.monotonic()
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=15)


def sha256(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def provenance():
    sources = sorted((ROOT/'python/tiga').rglob('*.py'))
    compilers = {key: os.environ.get(key) for key in ('TIGA_OPT', 'TIGA_TRANSLATE')}
    return dict(command=sys.argv, python=sys.version, platform=platform.platform(),
                timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                benchmark_sha256=sha256(__file__),
                source_hashes={str(p.relative_to(ROOT)): sha256(p) for p in sources},
                compiler_hashes={key: sha256(path) for key, path in compilers.items() if path},
                versions={name: importlib.metadata.version(name) for name in ('numpy', 'triton')})


def resident_tensor(path, shape, dtype):
    value = tg.empty(shape, dtype=dtype, device='cuda')
    with path.open('rb') as file:
        offset = 0
        while payload := file.read(8 << 20):
            value._buffer.write(payload, offset=offset)
            offset += len(payload)
    return value


def worker(args):
    path = args.cache_dir / f'ring-{args.edges}.gfg'
    nodes = args.edges // DEGREE
    info = json.loads((path/'ingest.json').read_text())
    if (info['edges'], info['features']) != (args.edges, FEATURES):
        raise ValueError('fixture configuration mismatch')
    gpu = gpu_info()
    report = dict(schema='tiga.billion-edges.v1', status='started', edges=args.edges,
                  nodes=nodes, features=FEATURES, degree=DEGREE, mode=args.mode,
                  page_rows=args.page_rows, gpu_before=gpu, ingest=info,
                  provenance=provenance(), **footprint(nodes))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(report, indent=2)+'\n')
        temporary.replace(args.output)

    save()
    # Do not invoke the OOM killer to confirm an arithmetically impossible
    # resident case. This outcome is NOT a measured allocation failure.
    if args.mode == 'resident' and report['resident_lower_bound_bytes'] > gpu['total_bytes']:
        report.update(status='infeasible_lower_bound',
                      reason='CSR + FP32 source + output exceed physical GPU capacity; not executed')
        save()
        return
    before_io = io_counters()
    with Monitor() as monitor:
        try:
            # Preserve at least 512 MiB of currently free memory for other work.
            budget = min(args.budget_mib << 20, gpu['total_bytes']-gpu['used_bytes']-(512 << 20))
            if budget < report['output_bytes'] + (256 << 20):
                raise MemoryError('insufficient currently available GPU memory for output and page')
            report['budget_bytes'] = budget
            with tg.execution(device='cuda', memory={'device': budget},
                              page_rows=args.page_rows, prefetch_depth=1) as scope:
                started = time.perf_counter()
                if args.mode == 'resident':
                    rows = resident_tensor(path/'row_ptr.bin', (nodes+1,), tg.int64)
                    columns = resident_tensor(path/'col_idx.bin', (args.edges,), tg.int64)
                    graph = tg.Graph.from_csr(rows, columns, num_src=nodes)
                    x = resident_tensor(path/'x.bin', (nodes, FEATURES), tg.float32)
                else:
                    graph = tg.load(path, device='cuda')
                    x = graph.fields('src')['x']
                    read_page = graph.paged_page

                    def progress(begin, end):
                        if begin % (args.page_rows * 64) == 0:
                            print(json.dumps(dict(event='page', begin=begin, nodes=nodes,
                                                  seconds=time.perf_counter()-started)), flush=True)
                        return read_page(begin, end)

                    graph.paged_page = progress
                report['load_s'] = time.perf_counter()-started
                started = time.perf_counter()
                y = Average()(graph=graph, src={'x': x}, dst={},
                              **({'prefetch': False} if args.mode == 'paged' else {}))
                y.realize()
                if y.ready_event is not None:
                    y.ready_event.wait()
                report['forward_s'] = time.perf_counter()-started
                report['execution'] = y.execution
                if args.mode == 'paged':
                    assert y.execution['page_backends'] == ['cuda-ttir-triton'], y.execution
                else:
                    assert y.execution['backend'] == 'cuda-ttir-triton', y.execution
                report['forward_memory'] = scope.memory_report()
                report['forward_io_delta'] = {k: v-before_io[k] for k, v in io_counters().items()}
                report['forward_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
                save()
                started_check = time.perf_counter()
                error = 0.0
                checked = 0
                for begin in range(0, nodes, args.check_rows):
                    end = min(nodes, begin + args.check_rows)
                    actual = np.frombuffer(y._buffer.read(
                        offset=begin*FEATURES*4, bytes=(end-begin)*FEATURES*4), dtype='<f4').reshape(-1, FEATURES)
                    expected = oracle(begin, end, nodes)
                    np.testing.assert_array_equal(actual, expected)
                    error = max(error, float(np.max(np.abs(actual-expected))))
                    checked += actual.size
                report.update(status='ok', max_abs_error=error, checked_values=checked,
                              check_s=time.perf_counter()-started_check)
                del y, x, graph
                gc.collect()
                report['memory'] = scope.memory_report()
        except KeyboardInterrupt:
            report.update(status='interrupted', reason='explicitly interrupted; incomplete trial')
            raise
        except Exception as error:
            report.update(status='error', error_type=type(error).__name__, error=str(error))
            raise
        finally:
            report['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
            report['gpu_samples'] = list(monitor.samples)
            report['gpu_monitor_errors'] = list(monitor.errors)
            save()
    print(json.dumps({k: v for k, v in report.items()
                      if k not in {'provenance', 'gpu_samples', 'ingest'}}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--edges', type=int, default=1_000_000_000)
    parser.add_argument('--page-rows', type=int, default=65536)
    parser.add_argument('--check-rows', type=int, default=65536)
    parser.add_argument('--budget-mib', type=int, default=12288)
    parser.add_argument('--mode', choices=('paged', 'resident'), default='paged')
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.edges <= 0 or args.edges % DEGREE or min(args.page_rows, args.check_rows, args.budget_mib) <= 0:
        parser.error('positive edges divisible by 16 and positive row counts/budget required')
    if args.build:
        build(args.cache_dir/f'ring-{args.edges}.gfg', args.edges, args.page_rows)
    elif args.output is None:
        parser.error('--output is required for execution')
    else:
        worker(args)


if __name__ == '__main__':
    main()
