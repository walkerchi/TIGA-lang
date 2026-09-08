"""Formal latency/crash-isolation gate for the persistent TTIR worker."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time


MODULE = """module {
  tt.func public @gf_worker_copy(%x: !tt.ptr<f32>, %y: !tt.ptr<f32>)
      attributes {noinline = false} {
    %pid = tt.get_program_id x : i32
    %index = arith.extsi %pid : i32 to i64
    %source = tt.addptr %x, %index : !tt.ptr<f32>, i64
    %value = tt.load %source : !tt.ptr<f32>
    %target = tt.addptr %y, %index : !tt.ptr<f32>, i64
    tt.store %target, %value : !tt.ptr<f32>
    tt.return
  }
}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json", type=Path,
        default=Path("output/compiler/persistent_compile_worker/results.json"))
    args = parser.parse_args()
    try:
        import torch
        import triton  # noqa: F401
    except ImportError as error:
        raise SystemExit(f"CUDA and Triton are required: {error}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    with tempfile.TemporaryDirectory(prefix="tiga-worker-cache-") as cache:
        os.environ["TRITON_CACHE_DIR"] = cache
        os.environ["TIGA_COMPILE_WORKER"] = "1"
        from tiga.codegen.compile_worker import _WORKER
        from tiga.codegen.ttir import compile_ttir

        started = time.perf_counter_ns()
        cold = compile_ttir(MODULE, options={"num_warps": 1})
        cold_ready_ms = (time.perf_counter_ns() - started) / 1e6
        cold_pid = _WORKER.pid

        started = time.perf_counter_ns()
        warm = compile_ttir(MODULE, options={"num_warps": 1})
        warm_ready_ms = (time.perf_counter_ns() - started) / 1e6
        warm_pid = _WORKER.pid

        _WORKER.simulate_crash_for_test()
        started = time.perf_counter_ns()
        recovered = compile_ttir(MODULE, options={"num_warps": 1})
        recovery_ready_ms = (time.perf_counter_ns() - started) / 1e6
        recovery_pid = _WORKER.pid

    passed = bool(
        cold_pid == warm_pid
        and recovery_pid is not None
        and recovery_pid != cold_pid
        and warm.worker_compile_ms is not None
        and cold.worker_compile_ms is not None
        and warm.worker_compile_ms < cold.worker_compile_ms
        and "cubin" in recovered.artifacts
    )
    payload = {
        "operation": "persistent_vendor_compile_worker",
        "device": torch.cuda.get_device_name(),
        "target": cold.provider.target,
        "cold": {
            "worker_compile_ms": cold.worker_compile_ms,
            "runtime_cache_load_ms": cold.cache_load_ms,
            "ready_ms": cold_ready_ms,
        },
        "warm_worker": {
            "worker_compile_ms": warm.worker_compile_ms,
            "runtime_cache_load_ms": warm.cache_load_ms,
            "ready_ms": warm_ready_ms,
            "same_pid": cold_pid == warm_pid,
        },
        "post_crash_disk_hit": {
            "worker_compile_ms": recovered.worker_compile_ms,
            "runtime_cache_load_ms": recovered.cache_load_ms,
            "ready_ms": recovery_ready_ms,
            "new_pid": recovery_pid != cold_pid,
        },
        "gate": {"passed": passed},
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2) + "\n")
    from benchmarks.common.diagnostic_plotting import plot_json
    plot_json(args.json)
    report = (
        "# Persistent vendor compile worker\n\n"
        f"- gate: **{'PASS' if passed else 'FAIL'}**\n"
        f"- cold worker compile: {cold.worker_compile_ms:.3f} ms\n"
        f"- warm worker compile: {warm.worker_compile_ms:.3f} ms\n"
        f"- runtime disk-cache load: {warm.cache_load_ms:.3f} ms\n"
        f"- crash recovery/new PID: {recovery_pid != cold_pid}\n"
    )
    (args.json.parent / "REPORT.md").write_text(report)
    print(report, end="")
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
