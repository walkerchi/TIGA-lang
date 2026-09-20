"""CUDA gf_tensor -> TTIR benchmark against matched PyTorch paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import statistics
import time

import tiga as tg
from tiga.compiler.gpu_tensor import compile_tensor

import torch


def _events(function, repeat: int, warmup: int = 20) -> dict[str, object]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeat):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        function()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end))
    return {
        "median_ms": statistics.median(samples),
        "minimum_ms": min(samples),
        "samples_ms": samples,
        "warmup": warmup,
    }


def _wall(function, repeat: int, warmup: int = 20) -> dict[str, object]:
    for _ in range(warmup):
        function()
    samples = []
    for _ in range(repeat):
        begin = time.perf_counter_ns()
        function()
        samples.append((time.perf_counter_ns() - begin) / 1e6)
    return {
        "median_ms": statistics.median(samples),
        "minimum_ms": min(samples),
        "samples_ms": samples,
        "warmup": warmup,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=1024)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.quick:
        args.rows, args.cols, args.repeat = 512, 256, 20
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.rows <= 0 or args.cols <= 0 or args.repeat <= 0:
        parser.error("rows, cols and repeat must be positive")

    torch.set_grad_enabled(False)
    torch.manual_seed(7)
    x_torch = torch.randn((args.rows, args.cols), device="cuda")
    scale_torch = torch.randn((args.cols,), device="cuda")
    x = tg.from_torch(x_torch)
    scale = tg.from_torch(scale_torch)

    begin = time.perf_counter_ns()
    first = (x * scale + x).sum(axis=1)
    first_torch = first.to_torch()
    cold_result_ready_ms = (time.perf_counter_ns() - begin) / 1e6
    executable = compile_tensor(first)
    tiga_kernel = _events(lambda: executable.launch(first), args.repeat)

    def tiga_e2e():
        return (x * scale + x).sum(axis=1).to_torch()

    tiga_e2e_measurement = _wall(tiga_e2e, args.repeat)

    def eager():
        return (x_torch * scale_torch + x_torch).sum(dim=1)

    torch_eager = _events(eager, args.repeat)
    compiled_function = torch.compile(
        lambda lhs, rhs: (lhs * rhs + lhs).sum(dim=1), fullgraph=True
    )
    compile_begin = time.perf_counter_ns()
    compiled_function(x_torch, scale_torch)
    torch.cuda.synchronize()
    torch_first_ms = (time.perf_counter_ns() - compile_begin) / 1e6
    torch_compiled = _events(
        lambda: compiled_function(x_torch, scale_torch), args.repeat
    )

    expected = eager()
    max_error = (first_torch - expected).abs().max().item()
    properties = torch.cuda.get_device_properties(0)
    elements = args.rows * args.cols
    result = {
        "operation": "broadcast_mul_add_axis_sum",
        "dtype": "float32",
        "shape": [args.rows, args.cols],
        "protocol": "cuda-event-kernel-and-synchronous-python-e2e-v1",
        "system": {
            "platform": platform.platform(),
            "gpu": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "torch": torch.__version__,
        },
        "tiga": {
            "backend": first.execution["backend"],
            "cold_result_ready_ms": cold_result_ready_ms,
            "compile_ms": first.execution["compile_ms"],
            "first_launch_ms": first.execution["launch_ms"],
            "semantic_hash": first.execution["semantic_hash"],
            "artifact_kinds": sorted(first.execution["artifacts"]),
            "kernel": tiga_kernel,
            "python_e2e": tiga_e2e_measurement,
        },
        "torch_eager": torch_eager,
        "torch_compile": {
            "first_call_ms": torch_first_ms,
            **torch_compiled,
        },
        "accuracy": {"max_abs_error": max_error},
        "model": {
            "flop": 3 * elements - args.rows,
            "ideal_bytes": 4 * (elements + args.cols + args.rows),
        },
    }
    print(json.dumps(result, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n")
        from benchmarks.common.diagnostic_plotting import plot_json
        plot_json(args.json)


if __name__ == "__main__":
    main()
