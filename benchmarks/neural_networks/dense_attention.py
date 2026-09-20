"""Matched semantic roofline for dense scaled dot-product attention."""

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

from benchmarks.common.hardware_roofline import measure_roofs, samples_ms
from benchmarks.common.output_layout import operation_dir
from benchmarks.common.perf_protocol import evaluate_sota_gates
from benchmarks.common.plotting import plot_latency, plot_roofline, write_report
import tiga as tg
from benchmarks.kernels.dense_streaming_reducer import (
    _kernel as streaming_oracle_kernel,
)


class DenseAttention(tg.MessagePassing):
    """Benchmark workload; Tiga itself contains no attention operator."""

    reducer = tg.online_softmax()

    def edge(self, src, dst, edge, scale):
        del edge
        score = (src.key * dst.query).sum(dim=-1) * scale
        return self.reducer(score, src.value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int)
    parser.add_argument("--sequence", type=int, default=4096)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--fail-on-gate", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.quick:
        args.sequence, args.repeat = 1024, 5
    if args.kv_heads is None:
        args.kv_heads = args.heads
    if args.kv_heads <= 0 or args.heads % args.kv_heads:
        parser.error("--kv-heads must be positive and divide --heads")
    device = torch.device("cuda")
    dtype = torch.float16
    shape = (args.batch, args.heads, args.sequence, args.width)
    source_shape = (args.batch, args.kv_heads, args.sequence, args.width)
    generator = torch.Generator(device=device).manual_seed(20260811)
    q = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    k = torch.randn(source_shape, device=device, dtype=dtype, generator=generator)
    v = torch.randn(source_shape, device=device, dtype=dtype, generator=generator)
    lanes = args.batch * args.heads
    source_lanes = args.batch * args.kv_heads
    # Tiga fields are entity-major views over the same physical
    # [batch*heads, sequence, width] storage consumed by the SOTA providers.
    q_nodes = q.permute(2, 0, 1, 3).reshape(
        args.sequence, lanes, args.width)
    k_nodes = k.permute(2, 0, 1, 3).reshape(
        args.sequence, source_lanes, args.width)
    v_nodes = v.permute(2, 0, 1, 3).reshape(
        args.sequence, source_lanes, args.width)
    graph = (
        tg.Graph.triangular(args.sequence, device=q.device)
        if args.causal else tg.Graph.dense(args.sequence, device=q.device)
    )
    candidate = DenseAttention()
    scale = args.width**-0.5

    def run_tiga():
        return candidate(
            graph=graph,
            src={"key": k_nodes, "value": v_nodes},
            dst={"query": q_nodes},
            scale=scale,
        )

    def run(backend):
        with sdpa_kernel(backend):
            return F.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=args.causal,
                enable_gqa=args.kv_heads != args.heads)

    expected = run(SDPBackend.MATH)
    actual = run(SDPBackend.FLASH_ATTENTION)
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
    torch.cuda.synchronize()
    compile_start = time.perf_counter()
    tiga_actual = run_tiga()
    torch.cuda.synchronize()
    tiga_cold_ms = (time.perf_counter() - compile_start) * 1000.0
    tiga_as_sdpa = tiga_actual.permute(1, 0, 2).reshape(shape)
    torch.testing.assert_close(
        tiga_as_sdpa, expected, rtol=3e-3, atol=3e-3)

    oracle_output = torch.empty_like(q.reshape(lanes, args.sequence, args.width))
    oracle_block_m = 64 if args.causal or args.width >= 64 else 128
    oracle_grid = (
        (args.sequence + oracle_block_m - 1) // oracle_block_m, lanes)
    def run_oracle():
        streaming_oracle_kernel[oracle_grid](
            q, k, v, oracle_output, scale,
            rows=args.sequence, query_lanes=lanes, source_lanes=source_lanes,
            width=args.width,
            BLOCK_M=oracle_block_m, BLOCK_N=64, CAUSAL=args.causal,
            num_warps=4, num_stages=3)
        return oracle_output
    oracle_actual = run_oracle()
    torch.testing.assert_close(
        oracle_actual.reshape(shape), expected, rtol=3e-3, atol=3e-3)
    roof = measure_roofs(device, args.quick, args.repeat)
    gemm_size = 2048 if args.quick else 4096
    left = torch.randn(
        gemm_size, gemm_size, device=device, dtype=dtype, generator=generator)
    right = torch.randn_like(left)
    gemm_samples = samples_ms(
        lambda: torch.mm(left, right), device, max(5, args.repeat // 2), None)
    fp16_gflops = 2.0 * gemm_size**3 / statistics.median(gemm_samples) / 1e6
    roof_payload = asdict(roof)
    roof_payload.update({
        "compute_gflops": fp16_gflops,
        "compute_ceiling_label": "FP16 tensor-core ceiling",
        "compute_calibration": f"torch.mm fp16 {gemm_size}x{gemm_size}",
    })

    # Convention: QK and PV FMAs plus max/sub/exp/sum/div softmax work.
    relation_pairs = (
        args.sequence * (args.sequence + 1) // 2
        if args.causal else args.sequence * args.sequence
    )
    pairs = args.batch * args.heads * relation_pairs
    useful_flops = float(
        4 * pairs * args.width + 5 * pairs)
    element_bytes = q.element_size()
    semantic_bytes = (
        q.numel() + k.numel() + v.numel() + actual.numel()) * element_bytes
    intensity = useful_flops / semantic_bytes
    optimistic = min(
        fp16_gflops, roof.dram_bandwidth_gbs * intensity)

    results = []
    providers = (
        ("tiga.direct_ttir", run_tiga),
        ("triton.streaming_oracle", run_oracle),
        ("torch.sdpa.flash", lambda: run(SDPBackend.FLASH_ATTENTION)),
        ("torch.sdpa.math", lambda: run(SDPBackend.MATH)),
    )
    for provider, provider_run in providers:
        raw = samples_ms(
            provider_run, device, args.repeat, None)
        milliseconds = statistics.median(raw)
        results.append({
            "topology": (
                "lower-triangular" if args.causal else "cartesian"),
            "locality": "dense",
            "cache": "consumer-only", "provider": provider,
            "nodes": args.sequence, "edges": relation_pairs,
            "features": args.width, "index_dtype": "implicit",
            "milliseconds": milliseconds,
            "gedges_per_second": pairs / milliseconds / 1e6,
            "achieved_gflops": useful_flops / milliseconds / 1e6,
            "algorithmic_gbs_no_reuse": semantic_bytes / milliseconds / 1e6,
            "memory_roof": "DRAM",
            "optimistic_roof_gflops": optimistic,
            "percent_of_optimistic_roof": (
                100 * useful_flops / milliseconds / 1e6 / optimistic),
            "ideal_cache_bytes": semantic_bytes,
            "semantic_common_bytes": semantic_bytes,
            "arithmetic_intensity_flop_per_byte": intensity,
            "samples_ms": raw,
        })

    case = (
        f"b{args.batch}_h{args.heads}_n{args.sequence}_d{args.width}_fp16"
        f"{'_kvh' + str(args.kv_heads) if args.kv_heads != args.heads else ''}"
        f"{'_causal' if args.causal else ''}"
    )
    output = args.output_dir or operation_dir("dense_attention", case)
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "operation": "dense_attention",
        "workload": "causal_sdpa_forward" if args.causal else "dense_sdpa_forward",
        "case": case, "title_prefix": "Tiga vs SOTA",
        "roof": roof_payload, "results": results,
        "config": {
            **vars(args), "dtype": "float16",
            "useful_flop_convention": "(4*D+5)*logical relation pairs",
            "semantic_byte_model": "Q + K + V + output",
            "tiga_status": "direct compiler-emitted TTIR",
            "tiga_cold_jit_ms": tiga_cold_ms,
            "tiga_lowering": candidate.last_variant.lowering,
            "tiga_provider": candidate.last_variant.provider,
        },
    }
    gates = evaluate_sota_gates(
        results, {"tiga.direct_ttir"},
        # Flash SDPA is the external production SOTA. The local handwritten
        # Triton kernel remains a parity/tuning oracle and is reported, but it
        # is not relabelled as an independent SOTA implementation.
        baselines={"torch.sdpa.flash"},
        threshold=1.0,
    )
    payload["sota_gates"] = [gate.to_dict() for gate in gates]
    payload["config"]["output_dir"] = str(output)
    (output / "roofline.json").write_text(json.dumps(payload, indent=2) + "\n")
    plot_roofline(payload, output)
    plot_latency(payload, output)
    write_report(payload, None, None, None, output)
    tiga_result, oracle, flash, math = results
    gate = gates[0]
    speedup = flash["milliseconds"] / tiga_result["milliseconds"]
    oracle_ratio = tiga_result["milliseconds"] / oracle["milliseconds"]
    (output / "REPORT.md").write_text(
        "# Dense attention: Tiga direct TTIR vs SOTA\n\n"
        "The Tiga point is produced by the user-defined MessagePassing "
        "program through Domain → Iter → Kernel → serialized TTIR; Tiga "
        "contains no built-in attention operator. The Triton point is a "
        "benchmark-only handwritten parity oracle.\n\n"
        f"Tiga direct TTIR: {tiga_result['milliseconds']:.4f} ms "
        f"({tiga_result['achieved_gflops'] / 1000:.2f} TFLOP/s).  "
        f"PyTorch Flash SDPA: {flash['milliseconds']:.4f} ms "
        f"({flash['achieved_gflops'] / 1000:.2f} TFLOP/s).  "
        f"PyTorch math SDPA: {math['milliseconds']:.4f} ms.  "
        f"Flash/Tiga speed ratio: {speedup:.3f}x; "
        f"Tiga/oracle latency ratio: {oracle_ratio:.3f}x.  "
        f"Strict gate: {'PASS' if gate.passed else 'FAIL'}, "
        f"{gate.speedup_vs_sota:.3f}x, 95% CI "
        f"[{gate.speedup_ci_low:.3f}, {gate.speedup_ci_high:.3f}].  "
        f"Cold capture+MLIR+provider JIT: {tiga_cold_ms:.2f} ms.  "
        f"The measured FP16 GEMM ceiling is {fp16_gflops / 1000:.2f} TFLOP/s.\n\n"
        "Both providers implement the same B/H/N/D workload and therefore use "
        f"the same x coordinate: {intensity:.6g} useful FLOP/common byte.\n\n"
        "![Dense attention roofline](roofline.svg)\n\n"
        "![Provider latency](provider_latency.svg)\n"
    )
    print(output)
    if args.fail_on_gate and not gate.passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
