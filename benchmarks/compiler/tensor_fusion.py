"""CPU Tensor fusion/JIT benchmark against semantically matched PyTorch.

The measured expression combines NumPy-style broadcast, two elementwise
operations and an axis reduction.  Graph construction, output allocation and
launch are included in warm timings; input construction is excluded.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import tempfile
import time

import tiga as tg
from tiga.compiler.cpu_tensor import compile_tensor

try:
    import torch
except ImportError:
    torch = None


def _measure(function, repeat: int, warmup: int = 12) -> dict[str, object]:
    for _ in range(warmup):
        function()
    samples = []
    for _ in range(repeat):
        start = time.perf_counter_ns()
        function()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return {
        "median_ms": statistics.median(samples),
        "samples_ms": samples,
        "warmup": warmup,
    }


def _graphforge_run(x: tg.Tensor, scale: tg.Tensor) -> tg.Tensor:
    output = (x * scale + x).sum(axis=1)
    output.realize()
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=1024)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument(
        "--dtype", choices=("float32", "complex64"), default="float32"
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument(
        "--strict", action="store_true",
        help="disable relaxed reassociation in Tiga CPU codegen",
    )
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.quick:
        args.rows, args.cols, args.repeat = 512, 256, 5
    if args.rows <= 0 or args.cols <= 0 or args.repeat <= 0:
        parser.error("rows, cols and repeat must be positive")

    if args.dtype == "complex64":
        values = [
            complex(((index % 251) - 125) / 251.0, ((index % 31) - 15) / 31.0)
            for index in range(args.rows * args.cols)
        ]
        scales = [
            complex(1.0 + (index % 17) / 17.0, (index % 7) / 19.0)
            for index in range(args.cols)
        ]
        gf_dtype = tg.complex64
    else:
        values = [
            ((index % 251) - 125) / 251.0
            for index in range(args.rows * args.cols)
        ]
        scales = [1.0 + (index % 17) / 17.0 for index in range(args.cols)]
        gf_dtype = tg.float32
    x = tg.tensor(values, dtype=gf_dtype).reshape(args.rows, args.cols)
    scale = tg.tensor(scales, dtype=gf_dtype).reshape(1, args.cols)

    old_backend = os.environ.get("TIGA_TENSOR_BACKEND")
    old_cache = os.environ.get("TIGA_CACHE_DIR")
    old_fast_math = os.environ.get("TIGA_FAST_MATH")
    with tempfile.TemporaryDirectory(prefix="tiga-tensor-benchmark-") as cache:
        os.environ["TIGA_TENSOR_BACKEND"] = "native"
        os.environ["TIGA_CACHE_DIR"] = cache
        os.environ["TIGA_FAST_MATH"] = "0" if args.strict else "1"
        start = time.perf_counter_ns()
        first = _graphforge_run(x, scale)
        cold_ms = (time.perf_counter_ns() - start) / 1e6
        warm = _measure(lambda: _graphforge_run(x, scale), args.repeat)
        executable = compile_tensor(first)
        kernel = _measure(lambda: executable.launch(first), args.repeat)
    if old_backend is None:
        os.environ.pop("TIGA_TENSOR_BACKEND", None)
    else:
        os.environ["TIGA_TENSOR_BACKEND"] = old_backend
    if old_cache is None:
        os.environ.pop("TIGA_CACHE_DIR", None)
    else:
        os.environ["TIGA_CACHE_DIR"] = old_cache
    if old_fast_math is None:
        os.environ.pop("TIGA_FAST_MATH", None)
    else:
        os.environ["TIGA_FAST_MATH"] = old_fast_math

    result = {
        "operation": "broadcast_mul_add_axis_sum",
        "dtype": args.dtype,
        "shape": [args.rows, args.cols],
        "protocol": "synchronous-result-ready-v1",
        "system": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "logical_cpus": os.cpu_count(),
        },
        "tiga": {
            "cold_ms": cold_ms,
            "version": tg.__version__,
            "warm_median_ms": warm["median_ms"],
            "warm_samples_ms": warm["samples_ms"],
            "warmup": warm["warmup"],
            "kernel_median_ms": kernel["median_ms"],
            "kernel_samples_ms": kernel["samples_ms"],
            "kernel_warmup": kernel["warmup"],
            "compile_ms": first.execution["compile_ms"],
            "backend": first.execution["backend"],
            "fast_math": first.execution["fast_math"],
        },
    }

    if torch is not None:
        torch.set_grad_enabled(False)
        torch_dtype = torch.complex64 if args.dtype == "complex64" else torch.float32
        torch_x = torch.tensor(values, dtype=torch_dtype).reshape(args.rows, args.cols)
        torch_scale = torch.tensor(scales, dtype=torch_dtype).reshape(1, args.cols)

        def torch_function(input_value, input_scale):
            return (input_value * input_scale + input_value).sum(dim=1)

        def torch_eager():
            result_value = torch_function(torch_x, torch_scale)
            # Force the public result-ready boundary.  Returning a Tensor from
            # a compiled CPU wrapper can otherwise time only submission while
            # Tiga's current CPU launch is synchronous.
            result_value[0].item()
            return result_value

        expected = torch_eager()
        torch_measurement = _measure(torch_eager, args.repeat)
        actual = first.tolist()
        max_error = max(
            abs(left - right)
            for left, right in zip(actual, expected.tolist())
        )
        result["accuracy"] = {"max_abs_error": max_error}
        result["torch_eager"] = {
            "version": torch.__version__,
            "threads": torch.get_num_threads(),
            "warm_median_ms": torch_measurement["median_ms"],
            "warm_samples_ms": torch_measurement["samples_ms"],
            "warmup": torch_measurement["warmup"],
        }
        if args.torch_compile and hasattr(torch, "compile"):
            compiled = torch.compile(torch_function, fullgraph=True)
            start = time.perf_counter_ns()
            compiled_result = compiled(torch_x, torch_scale)
            compiled_result[0].item()
            torch_compile_ms = (time.perf_counter_ns() - start) / 1e6

            def torch_compiled():
                result_value = compiled(torch_x, torch_scale)
                result_value[0].item()
                return result_value

            compiled_measurement = _measure(torch_compiled, args.repeat)
            result["torch_compile"] = {
                "first_call_ms": torch_compile_ms,
                "warm_median_ms": compiled_measurement["median_ms"],
                "warm_samples_ms": compiled_measurement["samples_ms"],
                "warmup": compiled_measurement["warmup"],
            }

    elements = args.rows * args.cols
    complex_case = args.dtype == "complex64"
    result["model"] = {
        "flop": (10 if complex_case else 3) * elements
                - (2 if complex_case else 1) * args.rows,
        "ideal_bytes": (8 if complex_case else 4)
                       * (elements + args.cols + args.rows),
    }
    print(json.dumps(result, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n")
        from benchmarks.common.diagnostic_plotting import plot_json
        plot_json(args.json)


if __name__ == "__main__":
    main()
