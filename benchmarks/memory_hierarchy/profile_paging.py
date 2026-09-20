"""Instrumented paging diagnostic through 1B edges, separate from timing sweeps.

Exclusive nested host timings are additive, but are not pure GPU kernel or disk
service times. Optional cudaProfilerApi capture lets Nsight measure actual CUDA
kernels and copies inside forward only (not validation or fixture ingest).
File eviction hints affect only the three benchmark fixture files, never the
system-wide page cache. /proc/self/io records whether storage was actually read.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import ctypes
import ctypes.util
import hashlib
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

import tiga as tg
from tiga.graph.storage import PagedCSRStore, PagedField
from tiga.runtime import Buffer, DeviceType
from tiga.tensor.core import Tensor
from benchmarks.memory_hierarchy import billion_edges as bench


class ExclusiveTimer:
    def __init__(self):
        self.stack = []
        self.seconds = defaultdict(float)
        self.calls = defaultdict(int)
        self.bytes = defaultdict(int)
        self.active = False

    def wrap(self, function, label, byte_count=None):
        def measured(*args, **kwargs):
            if not self.active:
                return function(*args, **kwargs)
            name = label(*args, **kwargs) if callable(label) else label
            frame = [time.perf_counter(), 0.]
            self.stack.append(frame)
            try:
                return function(*args, **kwargs)
            finally:
                duration = time.perf_counter()-frame[0]
                self.stack.pop()
                self.seconds[name] += duration-frame[1]
                self.calls[name] += 1
                if byte_count is not None:
                    self.bytes[name] += byte_count(*args, **kwargs)
                if self.stack:
                    self.stack[-1][1] += duration
        return measured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--edges', type=int, default=10_000_000)
    parser.add_argument('--evict-fixture', action='store_true')
    parser.add_argument('--cuda-capture', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.edges > 1_000_000_000 or args.edges < 16 or args.edges % 16:
        parser.error('diagnostic limited to <=1B edges, positive multiple of 16')
    if args.evict_fixture:
        for name in ('row_ptr.bin', 'col_idx.bin', 'x.bin'):
            with (args.cache_dir/f'ring-{args.edges}.gfg'/name).open('rb') as file:
                os.posix_fadvise(file.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    timer = ExclusiveTimer()
    original = bench.Average.__call__
    profiler = ctypes.CDLL(ctypes.util.find_library('cudart')) if args.cuda_capture else None
    elapsed = []

    def forward(*a, **kw):
        if profiler is not None and profiler.cudaProfilerStart() != 0:
            raise RuntimeError('cudaProfilerStart failed')
        timer.active = True
        begin = time.perf_counter()
        try:
            result = original(*a, **kw)
            result.realize()
            if result.ready_event is not None:
                result.ready_event.wait()
            return result
        finally:
            elapsed.append(time.perf_counter()-begin)
            timer.active = False
            if profiler is not None and profiler.cudaProfilerStop() != 0:
                raise RuntimeError('cudaProfilerStop failed')

    def write_label(buffer, *a, **kw):
        return 'host_pack_and_H2D' if buffer.device.type == DeviceType.CUDA else 'host_buffer_copy'

    def read_label(buffer, *a, **kw):
        return 'D2H_and_host_copy' if buffer.device.type == DeviceType.CUDA else 'host_buffer_copy'

    with patch.object(bench.Average, '__call__', forward), \
         patch.object(PagedCSRStore, 'read_page', timer.wrap(PagedCSRStore.read_page, 'topology_read_decode_validate')), \
         patch.object(PagedField, 'gather_into', timer.wrap(PagedField.gather_into, 'field_mmap_gather', lambda self, ids, address: len(ids)*self.row_bytes)), \
         patch.object(Buffer, 'write', timer.wrap(Buffer.write, write_label, lambda self, data, **kw: len(data))), \
         patch.object(Buffer, 'read', timer.wrap(Buffer.read, read_label, lambda self, **kw: kw.get('bytes', self.nbytes))), \
         patch.object(Tensor, 'realize', timer.wrap(Tensor.realize, 'realize_compile_submit_wait')):
        bench.worker(argparse.Namespace(cache_dir=args.cache_dir, edges=args.edges,
                                        page_rows=65536, check_rows=65536,
                                        budget_mib=12288, mode='paged', output=args.output))
    report = json.loads(args.output.read_text())
    assert len(elapsed) == 1 and not timer.stack
    residual = elapsed[0]-sum(timer.seconds.values())
    assert residual >= 0, 'exclusive categories must not double count'
    timer.seconds['other_host_setup_assembly'] = residual
    report['profile'] = dict(exclusive_host_seconds=dict(timer.seconds), calls=dict(timer.calls),
                             bytes=dict(timer.bytes), forward_seconds=elapsed[0],
                             fixture_eviction_hint=args.evict_fixture,
                             cuda_profiler_capture=args.cuda_capture,
                             benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                             scope='instrumented forward at recorded edge count; nested exclusive host intervals; mmap gather includes page-fault IO; no extrapolation')
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['profile'], indent=2), flush=True)


if __name__ == '__main__':
    main()
