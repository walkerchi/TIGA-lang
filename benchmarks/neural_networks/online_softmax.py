"""Segmented online-softmax compiler benchmark against matched GPU peers."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import time
import re

import torch
import triton
import triton.language as tl

import tiga as gf
from benchmarks.common.hardware_roofline import measure_roofs, samples_ms
from benchmarks.common.output_layout import operation_dir
from benchmarks.common.perf_protocol import evaluate_sota_gates
from benchmarks.common.plotting import plot_latency, plot_roofline


class StableWeightedMean(gf.Reducer):
    """User-defined stable (maximum, denominator, numerator) monoid."""

    name = "benchmark_stable_weighted_mean"
    associative = True
    commutative = True

    def identity(self):
        return -1.0e30, 0.0, 0.0

    def lift(self, score, value):
        return score, 1.0, value

    def combine(self, left, right):
        left_max, left_denominator, left_numerator = left
        right_max, right_denominator, right_numerator = right
        maximum = left_max.maximum(right_max)
        left_scale = (left_max - maximum).exp()
        right_scale = (right_max - maximum).exp()
        return (
            maximum,
            left_scale * left_denominator + right_scale * right_denominator,
            left_scale * left_numerator + right_scale * right_numerator,
        )

    def finalize(self, state):
        _, denominator, numerator = state
        return numerator / denominator


class SegmentedSoftmax(gf.MessagePassing):
    reducer = StableWeightedMean()

    def edge(self, src, dst, edge):
        return self.reducer(src.score, src.value)


@triton.jit
def _fused_segmented_softmax(
    row_ptr, col_idx, score, value, output, num_rows: tl.constexpr,
    block_m: tl.constexpr, block_d: tl.constexpr,
):
    rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
    lanes = tl.arange(0, block_d)
    row_mask = rows < num_rows
    starts = tl.load(row_ptr + rows, mask=row_mask, other=0)
    ends = tl.load(row_ptr + rows + 1, mask=row_mask, other=0)
    edges = starts[:, None] + lanes[None, :]
    mask = row_mask[:, None] & (edges < ends[:, None])
    sources = tl.load(col_idx + edges, mask=mask, other=0)
    scores = tl.load(score + sources, mask=mask, other=-float("inf"))
    values = tl.load(value + sources, mask=mask, other=0.0)
    maximum = tl.max(scores, axis=1)
    weights = tl.exp(scores - maximum[:, None])
    denominator = tl.sum(tl.where(mask, weights, 0.0), axis=1)
    numerator = tl.sum(tl.where(mask, weights * values, 0.0), axis=1)
    tl.store(output + rows, numerator / denominator, mask=row_mask)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=131072)
    parser.add_argument("--degree", type=int, default=32)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.quick:
        args.nodes, args.degree, args.repeat = 8192, 16, 20
    if args.nodes <= 0 or args.degree <= 0 or args.degree > 64:
        raise ValueError("nodes must be positive and degree must be in [1, 64]")

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260813)
    edges = args.nodes * args.degree
    row_ptr = torch.arange(
        0, edges + 1, args.degree, dtype=torch.int64, device=device)
    col_idx = torch.randint(
        args.nodes, (edges,), dtype=torch.int64, device=device,
        generator=generator)
    score = torch.randn(args.nodes, device=device, generator=generator) + 1000
    value = torch.randn(args.nodes, device=device, generator=generator)
    graph = gf.Graph.from_csr(row_ptr, col_idx, num_src=args.nodes)
    kernel = SegmentedSoftmax()
    triton_output = torch.empty_like(score)
    block_d = triton.next_power_of_2(args.degree)
    block_m = max(1, min(16, 512 // block_d))

    def graphforge_run():
        return kernel(
            graph=graph, src={"score": score, "value": value}, dst={})

    def triton_run():
        _fused_segmented_softmax[(triton.cdiv(args.nodes, block_m),)](
            row_ptr, col_idx, score, value, triton_output,
            num_rows=args.nodes, block_m=block_m, block_d=block_d,
            num_warps=4)
        return triton_output

    def torch_run():
        scores = score[col_idx].reshape(args.nodes, args.degree)
        values = value[col_idx].reshape(args.nodes, args.degree)
        return (torch.softmax(scores, dim=1) * values).sum(dim=1)

    cold_start = time.perf_counter_ns()
    actual = graphforge_run()
    torch.cuda.synchronize()
    cold_ms = (time.perf_counter_ns() - cold_start) / 1e6
    prepared_graphforge = kernel.prepare(
        graph=graph, src={"score": score, "value": value}, dst={})
    expected = triton_run()
    torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-4)
    torch.testing.assert_close(actual, torch_run(), rtol=3e-4, atol=3e-4)
    if kernel.last_variant is None or kernel.last_variant.lowering != (
        "gf-kernel-to-ttir-csr-stable-weighted-tile"
    ):
        raise RuntimeError("Tiga candidate did not use the compiler tile")
    generated_ttir = kernel.ir("gf.kernel.ttir")
    schedule = re.search(
        r"block_rows=(\d+) num_warps=(\d+)", generated_ttir)
    if schedule is None:
        raise RuntimeError("Tiga TTIR omitted its launch schedule")
    graphforge_block_rows, graphforge_num_warps = map(int, schedule.groups())

    roof = measure_roofs(device, args.quick, args.repeat)
    # Common semantic model for every provider: max, stable exponentiation,
    # denominator/numerator reductions, and final division. Transcendentals
    # count as one useful operation; exact instruction mix remains metadata.
    useful_flops = float(8 * edges + args.nodes)
    common_bytes = float(
        row_ptr.numel() * row_ptr.element_size()
        + col_idx.numel() * col_idx.element_size()
        + score.numel() * score.element_size()
        + value.numel() * value.element_size()
        + args.nodes * score.element_size())
    intensity = useful_flops / common_bytes
    providers = (
        ("tiga.compiler_ttir", prepared_graphforge),
        ("triton.handwritten_fused", triton_run),
        ("torch.explicit", torch_run),
    )
    results = []
    for provider, run in providers:
        raw = samples_ms(run, device, args.repeat, None)
        median = statistics.median(raw)
        results.append({
            "topology": "fixed-degree-csr", "locality": "random",
            "cache": "hot", "provider": provider, "nodes": args.nodes,
            "edges": edges, "features": 1, "index_dtype": "i64",
            "milliseconds": median,
            "gedges_per_second": edges / median / 1e6,
            "achieved_gflops": useful_flops / median / 1e6,
            "arithmetic_intensity_flop_per_byte": intensity,
            "ideal_cache_bytes": common_bytes,
            "semantic_common_bytes": common_bytes,
            "memory_roof": "DRAM", "samples_ms": raw,
        })
    gate = evaluate_sota_gates(
        results, {"tiga.compiler_ttir"},
        baselines={"triton.handwritten_fused", "torch.explicit"},
        threshold=1.0)[0]
    case = f"cuda_n{args.nodes}_degree{args.degree}_f1"
    output_dir = args.output_dir or operation_dir("online_softmax", case)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "operation": "online_softmax",
        "workload": "stable segmented softmax-weighted scalar reduction",
        "case": case, "roof": asdict(roof), "results": results,
        "sota_gates": [gate.to_dict()],
        "config": {
            **vars(args), "output_dir": str(output_dir),
            "graphforge_cold_compile_ms": cold_ms,
            "graphforge_lowering": kernel.last_variant.lowering,
            "graphforge_provider": kernel.last_variant.provider,
            "graphforge_block_rows": graphforge_block_rows,
            "graphforge_num_warps": graphforge_num_warps,
            "baseline_block_rows": block_m, "block_neighbors": block_d,
            "semantic_byte_model": "row_ptr + col_idx + score + value + out",
            "useful_flop_model": "8 per edge + 1 division per row",
        },
    }
    (output_dir / "roofline.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n")
    plot_roofline(payload, output_dir)
    plot_latency(payload, output_dir)
    candidate, baseline = results[0], min(
        results[1:], key=lambda result: result["milliseconds"])
    (output_dir / "REPORT.md").write_text(
        "# Segmented online softmax\n\n"
        "`StableWeightedMean` is a benchmark-defined user reducer. Tiga "
        "captures its identity/lift/combine/finalize regions, structurally "
        "proves the stable tuple algebra and emits tiled max/exp/sum reductions; "
        "no reducer class or online-softmax name is recognized.\n\n"
        f"Tiga: {candidate['milliseconds']:.4f} ms; fastest matched peer "
        f"({baseline['provider']}): {baseline['milliseconds']:.4f} ms. Strict "
        f"SOTA gate: {'PASS' if gate.passed else 'FAIL'}, speedup "
        f"{gate.speedup_vs_sota:.3f}x, 95% CI "
        f"[{gate.speedup_ci_low:.3f}, {gate.speedup_ci_high:.3f}].\n\n"
        f"Cold compile + first execution: {cold_ms:.2f} ms.\n\n"
        "![Roofline](roofline.svg)\n\n![Latency](provider_latency.svg)\n")
    print(output_dir)
    if args.fail_on_gate and not gate.passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
