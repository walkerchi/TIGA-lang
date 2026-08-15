"""GraphForge Tensor matmul roofline against an external vendor baseline."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import statistics
import time

import torch

import graphforge as gf
from benchmarks.common.hardware_roofline import measure_roofs, samples_ms
from benchmarks.common.output_layout import operation_dir
from benchmarks.common.perf_protocol import evaluate_sota_gates
from benchmarks.common.plotting import plot_latency, plot_roofline, write_report
from graphforge.compiler.gpu_tensor import compile_tensor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=2048)
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fail-on-gate", action="store_true")
    parser.add_argument(
        "--provider", choices=("auto", "library", "ttir"), default="auto")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.quick:
        args.m = args.n = args.k = 512
        args.repeat = 10
    if min(args.m, args.n, args.k) <= 0:
        raise ValueError("matrix extents must be positive")

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260813)
    lhs_torch = torch.randn(
        (args.m, args.k), device=device, dtype=torch.float16,
        generator=generator)
    rhs_torch = torch.randn(
        (args.k, args.n), device=device, dtype=torch.float16,
        generator=generator)
    output_torch = torch.empty(
        (args.m, args.n), device=device, dtype=torch.float16)
    lhs = gf.from_torch(lhs_torch)
    rhs = gf.from_torch(rhs_torch)
    output = lhs @ rhs

    previous_provider = os.environ.get("GRAPHFORGE_MATMUL_PROVIDER")
    os.environ["GRAPHFORGE_MATMUL_PROVIDER"] = args.provider
    try:
        cold_started = time.perf_counter_ns()
        executable = compile_tensor(output)
        output.to_torch()
        torch.cuda.synchronize()
        cold_ms = (time.perf_counter_ns() - cold_started) / 1e6
        graphforge_launch = executable.prepare(output)
    finally:
        if previous_provider is None:
            os.environ.pop("GRAPHFORGE_MATMUL_PROVIDER", None)
        else:
            os.environ["GRAPHFORGE_MATMUL_PROVIDER"] = previous_provider

    def graphforge_run():
        graphforge_launch()
        return output_torch

    def vendor_run():
        return torch.mm(lhs_torch, rhs_torch, out=output_torch)

    torch.testing.assert_close(
        output.to_torch(), vendor_run(), rtol=2e-3, atol=2e-2)
    roof = measure_roofs(device, args.quick, args.repeat)
    flops = float(2 * args.m * args.n * args.k)
    common_bytes = float(
        (args.m * args.k + args.k * args.n + args.m * args.n) * 2)
    intensity = flops / common_bytes
    providers = (
        ("graphforge.compiler_auto", graphforge_run),
        ("torch.mm.cublas", vendor_run),
    )
    results = []
    for provider, run in providers:
        raw = samples_ms(run, device, args.repeat, None)
        median = statistics.median(raw)
        results.append({
            "topology": "dense", "locality": "contiguous", "cache": "hot",
            "provider": provider, "nodes": args.m,
            "edges": args.m * args.n, "features": args.k,
            "index_dtype": "implicit", "milliseconds": median,
            "achieved_gflops": flops / median / 1e6,
            "arithmetic_intensity_flop_per_byte": intensity,
            "ideal_cache_bytes": common_bytes,
            "semantic_common_bytes": common_bytes,
            "memory_roof": "DRAM", "samples_ms": raw,
        })
    gates = evaluate_sota_gates(
        results, {"graphforge.compiler_auto"},
        baselines={"torch.mm.cublas"}, threshold=1.0)
    gate = gates[0]
    vendor_gflops = results[1]["achieved_gflops"]
    roof_payload = asdict(roof)
    roof_payload.update({
        "compute_gflops": vendor_gflops,
        "compute_ceiling_label": "measured FP16 vendor GEMM ceiling",
        "compute_calibration": f"torch.mm FP16 {args.m}x{args.k}x{args.n}",
    })
    case = f"m{args.m}_n{args.n}_k{args.k}_fp16"
    output_dir = args.output_dir or operation_dir(
        "dense_matmul_calibration", case)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "operation": "dense_matmul_calibration",
        "workload": "C[M,N] = A[M,K] @ B[K,N]",
        "case": case,
        "roof": roof_payload,
        "results": results,
        "sota_gates": [gate.to_dict()],
        "config": {
            **vars(args), "dtype": "float16",
            "graphforge_cold_compile_ms": cold_ms,
            "graphforge_compile_ms": executable.compile_ms,
            "graphforge_lowering": (
                "gf_tensor.matmul -> cublasLtMatmul"
                if executable.backend == "cuda-cublas-library-dispatch"
                else "gf_tensor.matmul -> tt.dot"
            ),
            "provider": executable.provider.name,
            "semantic_byte_model": "A + B + C",
        },
    }
    payload["config"]["output_dir"] = str(output_dir)
    (output_dir / "roofline.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n")
    plot_roofline(payload, output_dir)
    plot_latency(payload, output_dir)
    write_report(payload, None, None, None, output_dir)
    (output_dir / "REPORT.md").write_text(
        "# Dense matmul calibration\n\n"
        "GraphForge preserves the native `gf.Tensor` contraction as "
        "`gf_tensor.matmul`. The selected provider is reported explicitly; "
        "`auto` may dispatch a legal contiguous FP16 contraction to cuBLAS, "
        "while `--provider ttir` measures compiler-emitted `tt.dot`. Neither "
        "path dispatches through Torch.\n\n"
        f"GraphForge: {results[0]['milliseconds']:.4f} ms "
        f"({results[0]['achieved_gflops']/1000:.2f} TFLOP/s).  "
        f"cuBLAS through torch.mm: {results[1]['milliseconds']:.4f} ms "
        f"({vendor_gflops/1000:.2f} TFLOP/s).  Strict SOTA gate: "
        f"{'PASS' if gate.passed else 'FAIL'}, "
        f"speedup {gate.speedup_vs_sota:.3f}x, 95% CI "
        f"[{gate.speedup_ci_low:.3f}, {gate.speedup_ci_high:.3f}].\n\n"
        f"Cold compile + first materialization: {cold_ms:.2f} ms.\n\n"
        "![Roofline](roofline.svg)\n\n"
        "![Latency](provider_latency.svg)\n")
    print(output_dir)
    if args.fail_on_gate and not gate.passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
