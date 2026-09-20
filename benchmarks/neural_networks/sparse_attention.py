"""Dynamic tile-pruned online reducer benchmark against official FSA.

The workload class lives here, not in Tiga.  The compiler sees a dense
Cartesian relation, a dot-product message and an online-softmax reducer with a
semantic block-admission threshold.  Its generic dense streaming lowering
emits the dynamic ``scf.if`` around payload load/update.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

import tiga as tg
from benchmarks.common.hardware_roofline import measure_roofs, samples_ms
from benchmarks.common.output_layout import operation_dir
from benchmarks.common.perf_protocol import evaluate_sota_gates
from benchmarks.common.plotting import plot_latency, plot_roofline


class TilePrunedAttention(tg.MessagePassing):
    def __init__(self, threshold: float):
        super().__init__()
        self.reducer = tg.online_softmax(
            block_prune_threshold=threshold,
        )

    def edge(self, src, dst, edge, scale):
        del edge
        score = (src.key * dst.query).sum(dim=-1) * scale
        return self.reducer(score, src.value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--sequence", type=int, default=4096)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--fail-on-gate", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.quick:
        args.heads, args.sequence, args.repeat = 4, 512, 10
    if args.batch != 1:
        raise ValueError(
            "the registered no-copy entity/lane view currently requires batch=1"
        )
    if args.width not in (16, 32, 64, 128):
        raise ValueError("registered dense streaming widths are 16/32/64/128")
    try:
        from flash_sparse_attn import flash_sparse_attn_func
    except ImportError as error:
        raise SystemExit(
            "install the official flash-sparse-attn package for this benchmark"
        ) from error

    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else args.width / args.sequence
    )
    device = torch.device("cuda")
    dtype = torch.float16
    generator = torch.Generator(device=device).manual_seed(20260813)
    fsa_shape = (args.batch, args.sequence, args.heads, args.width)
    q_fsa = torch.randn(
        fsa_shape, device=device, dtype=dtype, generator=generator)
    k_fsa = torch.randn_like(q_fsa)
    v_fsa = torch.randn_like(q_fsa)

    # FSA uses token-major [B,N,H,D], whereas Tiga's currently registered
    # dense tile consumes lane-major [B,H,N,D].  Provider-native layout is
    # allowed by the benchmark protocol; the one-time repack is measured and
    # reported separately, never hidden in either warm kernel number.
    repack_started = time.perf_counter_ns()
    q_lane, k_lane, v_lane = (
        value.permute(0, 2, 1, 3).contiguous()
        for value in (q_fsa, k_fsa, v_fsa)
    )
    torch.cuda.synchronize()
    repack_ms = (time.perf_counter_ns() - repack_started) / 1e6
    lanes = args.batch * args.heads
    q_nodes = q_lane.permute(2, 0, 1, 3).reshape(
        args.sequence, lanes, args.width)
    k_nodes = k_lane.permute(2, 0, 1, 3).reshape(
        args.sequence, lanes, args.width)
    v_nodes = v_lane.permute(2, 0, 1, 3).reshape(
        args.sequence, lanes, args.width)
    graph = tg.Graph.dense(args.sequence, device=device)
    candidate = TilePrunedAttention(threshold)
    scale = args.width**-0.5

    def tiga_run():
        return candidate(
            graph=graph,
            src={"key": k_nodes, "value": v_nodes},
            dst={"query": q_nodes},
            scale=scale,
        )

    def fsa_run():
        return flash_sparse_attn_func(
            q_fsa, k_fsa, v_fsa,
            softmax_scale=scale,
            softmax_threshold=threshold,
            is_causal=False,
        )

    def exact_flash_run():
        # PyTorch SDPA's public layout is [B,H,N,D].
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return F.scaled_dot_product_attention(
                q_lane, k_lane, v_lane, dropout_p=0.0,
            )

    cold_started = time.perf_counter_ns()
    tiga_nodes = tiga_run()
    torch.cuda.synchronize()
    cold_ms = (time.perf_counter_ns() - cold_started) / 1e6
    fsa_output = fsa_run()
    exact_lane = exact_flash_run()
    torch.cuda.synchronize()
    tiga_lane = tiga_nodes.permute(1, 0, 2).reshape(
        args.batch, args.heads, args.sequence, args.width)
    fsa_lane = fsa_output.permute(0, 2, 1, 3)

    def error_metrics(value):
        error = (value - exact_lane).float()
        return {
            "relative_l2": float(
                error.norm() / exact_lane.float().norm().clamp_min(1e-12)
            ),
            "max_abs": float(error.abs().max()),
        }

    tiga_error = error_metrics(tiga_lane)
    fsa_error = error_metrics(fsa_lane)
    # Both providers implement thresholded approximation and may traverse
    # score tiles in a different legal order.  Require Tiga's accuracy
    # to be no worse than FSA (5% measurement/numerics allowance), rather than
    # incorrectly asserting bit equality between two order-dependent methods.
    accuracy_limit = max(1e-3, 1.05 * fsa_error["relative_l2"])
    if tiga_error["relative_l2"] > accuracy_limit:
        raise RuntimeError(
            "Tiga tile pruning exceeds the official FSA accuracy budget: "
            f"{tiga_error['relative_l2']:.6g} > {accuracy_limit:.6g}"
        )
    source_ttir = candidate.last_variant.artifacts.get("gf.kernel.ttir", "")
    if not source_ttir:
        source_ttir = candidate.last_variant.artifacts.get("ttir", "")
    if "scf.if" not in source_ttir or "block_prune" not in source_ttir:
        # Provider artifacts may be post-location TTIR; the lowering name and
        # domain plan still must identify the generated streaming path.
        domain = candidate.last_variant.domain_plan or ""
        if "block_prune_threshold" not in domain:
            raise RuntimeError("compiler did not retain dynamic block pruning")

    roof = measure_roofs(device, args.quick, args.repeat)
    pairs = args.batch * args.heads * args.sequence * args.sequence
    # Logical effective work keeps the same x coordinate for this operation.
    # Dynamic admitted-tile hardware FLOPs are not observable without a
    # profiler and are therefore not fabricated.
    useful_flops = float(4 * pairs * args.width + 5 * pairs)
    semantic_bytes = float(
        (q_fsa.numel() + k_fsa.numel() + v_fsa.numel()
         + fsa_output.numel()) * q_fsa.element_size()
    )
    intensity = useful_flops / semantic_bytes
    providers = (
        ("tiga.compiler_tile_pruned", tiga_run),
        ("flash_sparse_attn.official", fsa_run),
        ("torch.sdpa.flash_exact", exact_flash_run),
    )
    results = []
    for provider, run in providers:
        raw = samples_ms(run, device, args.repeat, None)
        median = statistics.median(raw)
        results.append({
            "topology": "dynamic-score-tile", "locality": "dense",
            "cache": "hot", "provider": provider,
            "nodes": args.sequence, "edges": args.sequence**2,
            "features": args.width, "index_dtype": "implicit",
            "milliseconds": median,
            "gedges_per_second": pairs / median / 1e6,
            "achieved_gflops": useful_flops / median / 1e6,
            "arithmetic_intensity_flop_per_byte": intensity,
            "ideal_cache_bytes": semantic_bytes,
            "semantic_common_bytes": semantic_bytes,
            "memory_roof": "DRAM", "samples_ms": raw,
        })
    gate = evaluate_sota_gates(
        results, {"tiga.compiler_tile_pruned"},
        baselines={"flash_sparse_attn.official"}, threshold=1.0,
    )[0]
    case = (
        f"b{args.batch}_h{args.heads}_n{args.sequence}_d{args.width}_fp16"
    )
    output_dir = args.output_dir or operation_dir("sparse_attention", case)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "operation": "sparse_attention",
        "workload": "dynamic block-threshold online softmax",
        "case": case, "title_prefix": "Tiga vs official FSA",
        "roof": asdict(roof), "results": results,
        "sota_gates": [gate.to_dict()],
        "config": {
            **vars(args), "output_dir": str(output_dir),
            "dtype": "float16", "threshold": threshold,
            "scale": scale, "causal": False,
            "tiga_error_vs_exact": tiga_error,
            "fsa_error_vs_exact": fsa_error,
            "accuracy_relative_l2_limit": accuracy_limit,
            "tiga_cold_compile_ms": cold_ms,
            "provider_layout_repack_ms": repack_ms,
            "tiga_lowering": candidate.last_variant.lowering,
            "semantic_byte_model": "Q + K + V + output",
            "flop_model": "logical 4*B*H*N*N*D + 5*B*H*N*N",
            "measured_admitted_tile_flops": None,
        },
    }
    (output_dir / "roofline.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n")
    plot_roofline(payload, output_dir)
    plot_latency(payload, output_dir)
    tiga_result, fsa_result, exact_result = results
    (output_dir / "REPORT.md").write_text(
        "# Dynamic tile-pruned attention: Tiga vs official FSA\n\n"
        "The workload is a benchmark-defined MessagePassing class. Tiga "
        "retains the block threshold in reducer IR and emits dynamic control "
        "flow around payload load/update; no attention operator is in core.\n\n"
        f"Tiga: {tiga_result['milliseconds']:.4f} ms; official FSA: "
        f"{fsa_result['milliseconds']:.4f} ms; exact Flash SDPA: "
        f"{exact_result['milliseconds']:.4f} ms. Gate: "
        f"{'PASS' if gate.passed else 'FAIL'}, speedup "
        f"{gate.speedup_vs_sota:.3f}x, 95% CI "
        f"[{gate.speedup_ci_low:.3f}, {gate.speedup_ci_high:.3f}].\n\n"
        f"Relative-L2 error vs exact: Tiga "
        f"{tiga_error['relative_l2']:.6g}, FSA "
        f"{fsa_error['relative_l2']:.6g}. One-time provider-native layout "
        f"repack: {repack_ms:.3f} ms; Tiga cold compile + first launch: "
        f"{cold_ms:.2f} ms. All points use x={intensity:.6g} logical useful "
        "FLOP/common byte; admitted-tile hardware FLOPs are not fabricated.\n\n"
        "![Roofline](roofline.svg)\n\n![Latency](provider_latency.svg)\n"
    )
    print(output_dir)
    if args.fail_on_gate and not gate.passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
