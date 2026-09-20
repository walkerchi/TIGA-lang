"""Matched causal linear-recurrence benchmark against official FLA.

The Tiga candidate is written only with public Tensor algebra. The
compiler structurally fuses map(outer product) -> cumsum -> map/reduce into a
single recurrent kernel; no attention operation lives in Tiga core.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import time

import torch

import tiga as tg
from tiga.compiler.gpu_tensor import compile_tensor
from benchmarks.common.hardware_roofline import measure_roofs, samples_ms
from benchmarks.common.output_layout import operation_dir
from benchmarks.common.perf_protocol import evaluate_sota_gates
from benchmarks.common.plotting import plot_latency, plot_roofline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lanes", type=int, default=64)
    parser.add_argument("--sequence", type=int, default=512)
    parser.add_argument("--key-width", type=int, default=16)
    parser.add_argument("--value-width", type=int, default=16)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--fail-on-gate", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.quick:
        args.lanes, args.sequence, args.repeat = 8, 128, 20
    if not (1 <= args.key_width <= 16):
        raise ValueError("the registered scan-contract slice requires key width <= 16")
    try:
        from fla.ops.linear_attn import (
            chunk_linear_attn, fused_recurrent_linear_attn,
        )
    except ImportError as error:
        raise SystemExit(
            "install the official flash-linear-attention package for this benchmark"
        ) from error

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260813)
    # FLA consumes [B,T,H,D]. H=1 makes the same contiguous allocations valid
    # as Tiga's [lane,T,D] Tensor views without copies or transposes.
    q_fla = torch.randn(
        (args.lanes, args.sequence, 1, args.key_width),
        device=device, dtype=torch.float32, generator=generator,
    )
    k_fla = torch.randn_like(q_fla)
    v_fla = torch.randn(
        (args.lanes, args.sequence, 1, args.value_width),
        device=device, dtype=torch.float32, generator=generator,
    )
    q = tg.from_torch(q_fla[:, :, 0])
    k = tg.from_torch(k_fla[:, :, 0])
    v = tg.from_torch(v_fla[:, :, 0])
    state_shape = (
        args.lanes, args.sequence, args.key_width, args.value_width,
    )
    lifted = (
        k.unsqueeze(-1).broadcast_to(state_shape)
        * v.unsqueeze(-2).broadcast_to(state_shape)
    )
    output = (
        q.unsqueeze(-1).broadcast_to(state_shape) * lifted.cumsum(1)
    ).sum(axis=2)

    started = time.perf_counter_ns()
    graphforge_actual = output.to_torch()
    torch.cuda.synchronize()
    cold_ms = (time.perf_counter_ns() - started) / 1e6
    executable = compile_tensor(output)
    graphforge_run = executable.prepare(output)

    def fla_recurrent_run():
        return fused_recurrent_linear_attn(
            q_fla, k_fla, v_fla, scale=1.0, normalize=False,
        )[0]

    def fla_chunk_run():
        return chunk_linear_attn(
            q_fla, k_fla, v_fla, scale=1.0, normalize=False,
        )[0]

    expected = fla_recurrent_run()[:, :, 0]
    torch.testing.assert_close(
        graphforge_actual, expected, rtol=2e-5, atol=3e-4,
    )
    chunk_actual = fla_chunk_run()[:, :, 0]
    # FLA's chunk path evaluates the same recurrence with a different
    # reassociation/accumulation schedule. On FP32 inputs its observed error is
    # materially larger than the recurrent path, so retain the difference as
    # accuracy metadata instead of pretending bit-level equivalence.
    chunk_error = chunk_actual - expected
    chunk_max_abs_error = chunk_error.abs().max().item()
    chunk_relative_l2 = (
        chunk_error.float().norm() / expected.float().norm().clamp_min(1e-12)
    ).item()
    if chunk_relative_l2 > 1e-2:
        raise RuntimeError(
            "FLA chunk and recurrent schedules disagree beyond the registered "
            f"relative-L2 tolerance: {chunk_relative_l2:.6g}"
        )
    generated_ttir = output.generated_code("ttir")
    if "gf_tensor_scan_contract" not in generated_ttir:
        raise RuntimeError("Tiga did not select scan-contract fusion")
    if "attention" in generated_ttir.lower():
        raise RuntimeError("workload naming leaked into compiler output")

    roof = measure_roofs(device, args.quick, args.repeat)
    items = args.lanes * args.sequence
    useful_flops = float(
        4 * items * args.key_width * args.value_width
    )
    common_bytes = float(
        (q_fla.numel() + k_fla.numel() + v_fla.numel()
         + graphforge_actual.numel()) * q_fla.element_size()
    )
    intensity = useful_flops / common_bytes
    providers = (
        ("tiga.compiler_scan_contract", graphforge_run),
        ("fla.fused_recurrent", fla_recurrent_run),
        ("fla.chunk", fla_chunk_run),
    )
    results = []
    for provider, run in providers:
        raw = samples_ms(run, device, args.repeat, None)
        median = statistics.median(raw)
        results.append({
            "topology": "causal-prefix", "locality": "dense-state",
            "cache": "hot", "provider": provider,
            "nodes": items, "edges": items,
            "features": args.key_width * args.value_width,
            "index_dtype": "implicit", "milliseconds": median,
            "gedges_per_second": items / median / 1e6,
            "achieved_gflops": useful_flops / median / 1e6,
            "arithmetic_intensity_flop_per_byte": intensity,
            "ideal_cache_bytes": common_bytes,
            "semantic_common_bytes": common_bytes,
            "memory_roof": "DRAM", "samples_ms": raw,
        })
    gate = evaluate_sota_gates(
        results, {"tiga.compiler_scan_contract"},
        # Project policy is SOTA parity: the lower confidence bound must be at
        # least 1.0x.  Requiring an unrelated 5% margin would label a kernel
        # that is statistically faster than every peer as a failure.
        baselines={"fla.fused_recurrent", "fla.chunk"}, threshold=1.0,
    )[0]
    case = (
        f"l{args.lanes}_t{args.sequence}_k{args.key_width}_"
        f"v{args.value_width}_fp32"
    )
    output_dir = args.output_dir or operation_dir("linear_attention", case)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "operation": "linear_attention",
        "workload": "causal unnormalized linear-attention recurrence",
        "case": case, "title_prefix": "Tiga vs official FLA",
        "roof": asdict(roof), "results": results,
        "sota_gates": [gate.to_dict()],
        "config": {
            **vars(args), "output_dir": str(output_dir),
            "dtype": "float32", "scale": 1.0, "normalize": False,
            "graphforge_cold_compile_ms": cold_ms,
            "graphforge_lowering": "map-scan-contract structural fusion",
            "fla_chunk_max_abs_error_vs_recurrent": chunk_max_abs_error,
            "fla_chunk_relative_l2_error_vs_recurrent": chunk_relative_l2,
            "semantic_byte_model": "Q + K + V + output",
            "useful_flop_model": "4*L*T*K*V",
            "state_materialized": False,
        },
    }
    (output_dir / "roofline.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n"
    )
    plot_roofline(payload, output_dir)
    plot_latency(payload, output_dir)
    candidate = results[0]
    peer = min(results[1:], key=lambda item: item["milliseconds"])
    (output_dir / "REPORT.md").write_text(
        "# Causal linear recurrence: Tiga vs official FLA\n\n"
        "Tiga receives ordinary broadcast/multiply/cumsum/sum Tensor IR. "
        "The compiler fuses the mapped outer product, inclusive scan and "
        "contraction; the rank-four recurrent state never reaches HBM.\n\n"
        f"Tiga: {candidate['milliseconds']:.4f} ms; fastest FLA peer "
        f"({peer['provider']}): {peer['milliseconds']:.4f} ms. Gate: "
        f"{'PASS' if gate.passed else 'FAIL'}, speedup "
        f"{gate.speedup_vs_sota:.3f}x, 95% CI "
        f"[{gate.speedup_ci_low:.3f}, {gate.speedup_ci_high:.3f}].\n\n"
        f"Cold compile + first launch: {cold_ms:.2f} ms. All providers share "
        f"x={intensity:.6g} useful FLOP/common byte.\n\n"
        "![Roofline](roofline.svg)\n\n![Latency](provider_latency.svg)\n"
    )
    print(output_dir)
    if args.fail_on_gate and not gate.passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
