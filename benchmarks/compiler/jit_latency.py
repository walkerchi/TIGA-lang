"""Measure provider cold compilation, disk-cache hit, and warm execution.

The controller launches two initialized CUDA worker processes against one
isolated temporary cache.  Python import and CUDA context initialization happen
before timing.  The first worker sees an empty provider cache; the second sees
the artifact written by the first worker.  No user cache is read or deleted.

This kernel remains a handwritten lowering oracle.  The measurements establish
the protocol that a future Tiga-generated provider must implement.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def csr_spmm_jit_kernel(row_ptr, col_idx, weight, x, out,
                            n_features: tl.constexpr,
                            block_f: tl.constexpr):
        row = tl.program_id(0)
        feature = tl.program_id(1) * block_f + tl.arange(0, block_f)
        mask = feature < n_features
        start = tl.load(row_ptr + row)
        end = tl.load(row_ptr + row + 1)
        accumulator = tl.zeros((block_f,), dtype=tl.float32)
        for edge in range(start, end):
            src = tl.load(col_idx + edge)
            value = tl.load(
                x + src * n_features + feature, mask=mask, other=0.0)
            accumulator += tl.load(weight + edge) * value
        tl.store(out + row * n_features + feature, accumulator, mask=mask)


def _make_inputs(nodes: int, degree: int, features: int):
    device = torch.device("cuda")
    row_ptr = torch.arange(
        0, nodes * degree + 1, degree, device=device, dtype=torch.int64)
    offsets = torch.arange(1, degree + 1, device=device, dtype=torch.int64)
    col_idx = (
        torch.arange(nodes, device=device, dtype=torch.int64)[:, None]
        + offsets
    ).remainder(nodes).reshape(-1)
    weight = torch.rand(col_idx.numel(), device=device)
    x = torch.rand((nodes, features), device=device)
    out = torch.empty_like(x)
    torch.cuda.synchronize(device)
    return row_ptr, col_idx, weight, x, out


def _artifact_sizes(compiled) -> dict[str, int]:
    sizes = {}
    for name, artifact in compiled.asm.items():
        if isinstance(artifact, str):
            sizes[name] = len(artifact.encode())
        elif isinstance(artifact, (bytes, bytearray, memoryview)):
            sizes[name] = len(artifact)
    return sizes


def _worker(nodes: int, degree: int, features: int, repeat: int, phase: str):
    if triton is None or not torch.cuda.is_available():
        raise RuntimeError("CUDA and Triton are required")
    row_ptr, col_idx, weight, x, out = _make_inputs(nodes, degree, features)
    block_f = min(128, triton.next_power_of_2(features))
    grid = (nodes, triton.cdiv(features, block_f))

    start_ns = time.perf_counter_ns()
    compiled = csr_spmm_jit_kernel.warmup(
        row_ptr,
        col_idx,
        weight,
        x,
        out,
        n_features=features,
        block_f=block_f,
        grid=grid,
    )
    provider_ms = (time.perf_counter_ns() - start_ns) / 1e6

    start_ns = time.perf_counter_ns()
    _ = compiled.function
    module_load_ms = (time.perf_counter_ns() - start_ns) / 1e6

    start_ns = time.perf_counter_ns()
    csr_spmm_jit_kernel[grid](
        row_ptr,
        col_idx,
        weight,
        x,
        out,
        n_features=features,
        block_f=block_f,
    )
    torch.cuda.synchronize()
    first_launch_ms = (time.perf_counter_ns() - start_ns) / 1e6

    for _ in range(5):
        csr_spmm_jit_kernel[grid](
            row_ptr, col_idx, weight, x, out,
            n_features=features, block_f=block_f)
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        csr_spmm_jit_kernel[grid](
            row_ptr, col_idx, weight, x, out,
            n_features=features, block_f=block_f)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))

    return {
        "phase": phase,
        "provider_compile_or_cache_ms": provider_ms,
        "module_load_ms": module_load_ms,
        "first_launch_ms": first_launch_ms,
        "ready_to_result_ms": provider_ms + module_load_ms + first_launch_ms,
        "warm_kernel_median_ms": statistics.median(samples),
        "warm_kernel_samples_ms": samples,
        "artifact_kinds": sorted(compiled.asm),
        "artifact_bytes": _artifact_sizes(compiled),
    }


def _run_worker(args, cache_dir: Path, phase: str):
    env = os.environ.copy()
    env["TRITON_CACHE_DIR"] = str(cache_dir)
    env.pop("TRITON_ALWAYS_COMPILE", None)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--phase", phase,
        "--nodes", str(args.nodes),
        "--degree", str(args.degree),
        "--features", str(args.features),
        "--repeat", str(args.repeat),
    ]
    completed = subprocess.run(
        command, env=env, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{phase} JIT worker failed ({completed.returncode}):\n"
            f"{completed.stderr}{completed.stdout}")
    return json.loads(completed.stdout)


def _print_phase(result):
    label = "cold cache" if result["phase"] == "cold" else "disk hit"
    print(
        f"{label:10} provider={result['provider_compile_or_cache_ms']:9.3f} ms  "
        f"load={result['module_load_ms']:7.3f} ms  "
        f"first-launch={result['first_launch_ms']:7.3f} ms  "
        f"ready={result['ready_to_result_ms']:9.3f} ms  "
        f"warm={result['warm_kernel_median_ms']:8.4f} ms")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=1 << 17)
    parser.add_argument("--degree", type=int, default=16)
    parser.add_argument("--features", type=int, default=16)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--phase", choices=("cold", "disk"),
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.quick:
        args.nodes, args.repeat = 1 << 13, 5
    if args.nodes <= 0 or args.degree <= 0 or args.features <= 0:
        parser.error("nodes, degree, and features must be positive")
    if args.worker:
        if args.phase is None:
            parser.error("worker requires --phase")
        print(json.dumps(_worker(
            args.nodes, args.degree, args.features, args.repeat, args.phase)))
        return
    if triton is None or not torch.cuda.is_available():
        parser.error("CUDA and Triton are required")

    with tempfile.TemporaryDirectory(prefix="tiga-triton-cache-") as path:
        cache_dir = Path(path)
        cold = _run_worker(args, cache_dir, "cold")
        disk = _run_worker(args, cache_dir, "disk")
    payload = {
        "protocol": "isolated-cache-v1",
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "nodes": args.nodes,
        "degree": args.degree,
        "features": args.features,
        "cold": cold,
        "disk_cache_hit": disk,
    }
    print(
        f"device={payload['device']} torch={payload['torch_version']} "
        f"triton={payload['triton_version']}")
    _print_phase(cold)
    _print_phase(disk)
    print("artifacts=" + ",".join(cold["artifact_kinds"]))
    if args.json:
        args.json.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
