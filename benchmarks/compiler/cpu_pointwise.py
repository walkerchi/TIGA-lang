"""CPU vector/parallel Tiga lowering against Torch/Inductor."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import statistics
import struct
import time

import torch

import tiga as gf
from tiga.compiler.cpu_tensor import compile_tensor
from benchmarks.common.output_layout import operation_dir
from benchmarks.common.perf_protocol import evaluate_sota_gates
from benchmarks.common.plotting import plot_latency, plot_roofline


def samples_ms(function, repeat, warmup=12):
    for _ in range(warmup):
        function()
    values = []
    for _ in range(repeat):
        start = time.perf_counter_ns()
        function()
        values.append((time.perf_counter_ns() - start) / 1e6)
    return values


@dataclass
class CPURoof:
    dram_bandwidth_gbs: float
    l2_bandwidth_gbs: float
    fp32_gflops: float


def measure_roof(repeat, quick):
    dram_elements = (8 if quick else 32) * 1024 * 1024
    source = torch.ones(dram_elements)
    destination = torch.empty_like(source)
    dram = statistics.median(samples_ms(
        lambda: destination.copy_(source), max(5, repeat // 2), 4))
    dram_gbs = 2 * source.numel() * source.element_size() / dram / 1e6
    l2_source = torch.ones(256 * 1024)
    l2_destination = torch.empty_like(l2_source)
    l2 = statistics.median(samples_ms(
        lambda: l2_destination.copy_(l2_source), repeat, 8))
    l2_gbs = 2 * l2_source.numel() * 4 / l2 / 1e6
    matrix_size = 512 if quick else 1024
    lhs = torch.randn(matrix_size, matrix_size)
    rhs = torch.randn_like(lhs)
    matmul = statistics.median(samples_ms(
        lambda: torch.mm(lhs, rhs), max(5, repeat // 2), 4))
    fp32 = 2 * matrix_size**3 / matmul / 1e6
    return CPURoof(dram_gbs, l2_gbs, fp32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--elements", type=int, default=4_000_000)
    parser.add_argument("--threads", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--fail-on-gate", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.quick:
        args.elements, args.repeat = 262_144, 12
    if args.elements <= 0 or args.threads <= 0 or args.repeat <= 0:
        parser.error("elements, threads and repeat must be positive")
    torch.set_num_threads(args.threads)
    os.environ["TIGA_CPU_THREADS"] = str(args.threads)
    os.environ["TIGA_TENSOR_BACKEND"] = "native"

    # Setup/conversion is outside warm timing for both providers.
    left_values = [1.5] * args.elements
    right_values = [2.0] * args.elements
    left = gf.tensor(left_values)
    right = gf.tensor(right_values)
    output = (left * right + left).sqrt()
    cold_started = time.perf_counter_ns()
    output.realize()
    cold_ms = (time.perf_counter_ns() - cold_started) / 1e6
    executable = compile_tensor(output)

    torch_left = torch.full((args.elements,), 1.5)
    torch_right = torch.full((args.elements,), 2.0)

    def eager_run():
        return torch.sqrt(torch_left * torch_right + torch_left)

    compiled_function = torch.compile(
        lambda lhs, rhs: torch.sqrt(lhs * rhs + lhs), fullgraph=True)
    compiled_actual = compiled_function(torch_left, torch_right)
    expected_value = (1.5 * 2.0 + 1.5) ** 0.5
    for index in (0, args.elements // 2, args.elements - 1):
        found = struct.unpack(
            "=f", output._buffer.read(offset=index * 4, bytes=4)
        )[0]
        if abs(found - expected_value) > 1e-6:
            raise RuntimeError("Tiga pointwise result is incorrect")
        if abs(compiled_actual[index].item() - expected_value) > 1e-6:
            raise RuntimeError("Inductor pointwise result is incorrect")
    loops = output.generated_code("cpu_loop")
    llvm = output.generated_code("llvm")
    if "vector.load" not in loops or "tiga.cpu.vector_width" not in loops:
        raise RuntimeError("Tiga did not select vector CPU lowering")
    if "vector<16xf32>" not in llvm:
        raise RuntimeError("LLVM stage did not retain the 512-bit vector type")

    providers = (
        ("tiga.llvm.vector_parallel", lambda: executable.launch(output)),
        ("torch.compile.inductor", lambda: compiled_function(torch_left, torch_right)),
        ("torch.eager", eager_run),
    )
    useful_flops = float(3 * args.elements)
    common_bytes = float(3 * args.elements * 4)  # lhs + rhs + output
    intensity = useful_flops / common_bytes
    results = []
    for provider, function in providers:
        raw = samples_ms(function, args.repeat)
        median = statistics.median(raw)
        results.append({
            "topology": "contiguous", "locality": "local",
            "cache": "cold", "provider": provider,
            "nodes": args.elements, "edges": args.elements,
            "features": 1, "index_dtype": "implicit",
            "milliseconds": median,
            "gedges_per_second": args.elements / median / 1e6,
            "achieved_gflops": useful_flops / median / 1e6,
            "arithmetic_intensity_flop_per_byte": intensity,
            "ideal_cache_bytes": common_bytes,
            "semantic_common_bytes": common_bytes,
            "memory_roof": "DRAM", "samples_ms": raw,
        })
    gate = evaluate_sota_gates(
        results, {"tiga.llvm.vector_parallel"},
        baselines={"torch.compile.inductor", "torch.eager"}, threshold=1.0,
    )[0]
    roof = measure_roof(args.repeat, args.quick)
    case = f"n{args.elements}_f32_t{args.threads}"
    output_dir = args.output_dir or operation_dir("cpu_pointwise_fusion", case)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "operation": "cpu_pointwise_fusion",
        "workload": "sqrt(lhs*rhs+lhs)",
        "case": case, "title_prefix": "Tiga CPU",
        "roof": asdict(roof), "results": results,
        "sota_gates": [gate.to_dict()],
        "config": {
            **vars(args), "output_dir": str(output_dir),
            "dtype": "float32", "useful_flops_per_element": 3,
            "semantic_byte_model": "lhs + rhs + output",
            "graphforge_cold_compile_first_launch_ms": cold_ms,
            "graphforge_compile_ms": executable.compile_ms,
            "graphforge_vector_width": 16,
            "graphforge_worker_pool_threads": args.threads,
            "torch_threads": torch.get_num_threads(),
        },
    }
    (output_dir / "roofline.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n")
    plot_roofline(payload, output_dir)
    plot_latency(payload, output_dir)
    candidate = results[0]
    peer = min(results[1:], key=lambda item: item["milliseconds"])
    (output_dir / "REPORT.md").write_text(
        "# CPU contiguous pointwise fusion\n\n"
        "Tiga lowers the ordinary Tensor DAG to Vector IR, LLVM vectors "
        "and a persistent range-partition worker pool. The scalar tail is "
        "retained for arbitrary extents.\n\n"
        f"Tiga: {candidate['milliseconds']:.4f} ms; fastest peer "
        f"({peer['provider']}): {peer['milliseconds']:.4f} ms. Gate: "
        f"{'PASS' if gate.passed else 'FAIL'}, speedup "
        f"{gate.speedup_vs_sota:.3f}x, 95% CI "
        f"[{gate.speedup_ci_low:.3f}, {gate.speedup_ci_high:.3f}].\n\n"
        f"Cold compile + first launch: {cold_ms:.2f} ms; threads: "
        f"{args.threads}; semantic x={intensity:.4g} FLOP/byte.\n\n"
        "![Roofline](roofline.svg)\n\n![Latency](provider_latency.svg)\n"
    )
    print(output_dir)
    if args.fail_on_gate and not gate.passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
