"""Loopback regression for the exact per-host CUDA rank entry point.

The separate two-machine gate establishes physical GPU diversity; this test
uses two processes on one GPU and must not be described as a two-GPU result.
"""
import multiprocessing
from pathlib import Path
import socket
import sys
import traceback

import pytest
import tiga as tg


def _rank(rank, port, queue):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from benchmarks.distributed.multi_host_gpu_gate import run_rank
    try:
        queue.put(run_rank(rank, "127.0.0.1", port, timeout=60))
    except BaseException:
        queue.put({"error": traceback.format_exc()})


def test_host_staged_cuda_forward_backward():
    try:
        tg.runtime.cuda_compute_capability()
    except RuntimeError:
        pytest.skip("CUDA unavailable")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    workers = [context.Process(target=_rank, args=(rank, port, queue)) for rank in range(2)]
    try:
        for worker in workers:
            worker.start()
        records = [queue.get(timeout=120) for _ in workers]
        for worker in workers:
            worker.join(10)
            assert worker.exitcode == 0
        assert all(record.get("correct") for record in records), records
        for record in records:
            for launch in record["records"][1:]:
                assert launch["trace"]["interior_rows"] == 2
                assert launch["trace"]["boundary_rows"] == 2
                assert launch["trace"]["measured_overlap_ms"] is None
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            if worker.pid is not None:
                worker.join(10)
        queue.close()


def test_gate_rejects_unknown_transport_before_connecting():
    from benchmarks.distributed.multi_host_gpu_gate import run_rank
    with pytest.raises(ValueError, match="transport_kind"):
        run_rank(0, "127.0.0.1", 1, transport_kind="invalid")
