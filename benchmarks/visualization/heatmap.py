"""Matched GPU benchmark for GraphForge Tensor-to-RGB raster preparation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import time

import torch

import graphforge as gf
from benchmarks.common.hardware_roofline import measure_roofs, samples_ms
from benchmarks.common.output_layout import operation_dir
from benchmarks.common.perf_protocol import evaluate_sota_gates
from benchmarks.common.plotting import plot_latency, plot_roofline, write_report


LOW = (0.267, 0.005, 0.329)
HIGH = (0.993, 0.906, 0.144)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=2048)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--fail-on-gate", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.quick:
        args.height = args.width = 512
        args.repeat = 8

    source = torch.rand(
        (args.height, args.width), device="cuda", dtype=torch.float32,
        generator=torch.Generator(device="cuda").manual_seed(20260813),
    )
    gf_source = gf.from_torch(source)
    low = torch.tensor(LOW, device="cuda", dtype=torch.float32).reshape(1, 3)
    high = torch.tensor(HIGH, device="cuda", dtype=torch.float32).reshape(1, 3)

    raster = gf.visualize.heatmap(gf_source)

    @torch.compile(fullgraph=True)
    def inductor_render(value):
        normalized = value.reshape(-1, 1)
        return low + normalized * (high - low)

    expected = inductor_render(source)
    started = time.perf_counter()
    raster.realize()
    torch.cuda.synchronize()
    cold_ms = (time.perf_counter() - started) * 1000.0
    graphforge_render = raster.prepare()
    actual = raster.pixels.to_torch()
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

    roof = measure_roofs(torch.device("cuda"), args.quick, args.repeat)
    pixels = args.height * args.width
    useful_flops = float(pixels * 8)
    semantic_bytes = float((pixels + pixels * 3) * 4)
    intensity = useful_flops / semantic_bytes
    providers = (
        ("graphforge.tensor_ttir", graphforge_render),
        ("torch.inductor.fused", lambda: inductor_render(source)),
        ("torch.eager", lambda: low + source.reshape(-1, 1) * (high - low)),
    )
    results = []
    for provider, run in providers:
        raw = samples_ms(run, torch.device("cuda"), args.repeat, None)
        milliseconds = statistics.median(raw)
        results.append({
            "provider": provider, "nodes": pixels, "edges": 0,
            "features": 3, "milliseconds": milliseconds,
            "achieved_gflops": useful_flops / milliseconds / 1e6,
            "algorithmic_gbs_no_reuse": semantic_bytes / milliseconds / 1e6,
            "ideal_cache_bytes": semantic_bytes,
            "semantic_common_bytes": semantic_bytes,
            "arithmetic_intensity_flop_per_byte": intensity,
            "memory_roof": "DRAM", "samples_ms": raw,
            "topology": "image-grid", "locality": "contiguous",
            "cache": "consumer-only", "index_dtype": "implicit",
        })
    case = f"h{args.height}_w{args.width}_rgb_f32"
    output = args.output_dir or operation_dir("visualization_heatmap", case)
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "operation": "visualization_heatmap", "case": case,
        "workload": "linear_scalar_to_rgb", "title_prefix": "GraphForge visualization",
        "roof": asdict(roof), "results": results,
        "config": {**vars(args), "dtype": "float32", "output_dir": str(output),
                   "graphforge_cold_jit_ms": cold_ms,
                   "semantic_byte_model": "one scalar read + one RGB write",
                   "steady_state": "prepared executable with stable input/output storage",
                   "useful_flop_convention": "normalize(2) + RGB FMA(6)"},
    }
    gates = evaluate_sota_gates(
        results, {"graphforge.tensor_ttir"},
        baselines={"torch.inductor.fused"}, threshold=1.0)
    payload["sota_gates"] = [gate.to_dict() for gate in gates]
    (output / "roofline.json").write_text(json.dumps(payload, indent=2) + "\n")
    plot_roofline(payload, output)
    plot_latency(payload, output)
    write_report(payload, None, None, None, output)
    gate = gates[0]
    speedup = results[1]["milliseconds"] / results[0]["milliseconds"]
    gate_status = "PASS" if gate.passed else "FAIL"
    (output / "REPORT.md").write_text(
        "# GPU-native heatmap raster preparation\n\n"
        f"GraphForge: {results[0]['milliseconds']:.4f} ms; Torch Inductor: "
        f"{results[1]['milliseconds']:.4f} ms; speedup {speedup:.3f}x. "
        f"Strict gate: {gate_status}, 95% CI "
        f"[{gate.speedup_ci_low:.3f}, {gate.speedup_ci_high:.3f}]. "
        "Both points execute the same scalar-to-RGB expression. GraphForge uses "
        "its public prepared executable for stable animation buffers; cold JIT is "
        "reported separately and PNG encoding is outside the timed region.\n\n"
        "![Roofline](roofline.svg)\n\n![Latency](provider_latency.svg)\n"
    )
    if args.fail_on_gate and not gate.passed:
        raise SystemExit(
            f"SOTA gate failed: {gate_status}, "
            f"CI low={gate.speedup_ci_low:.3f}")
    print(output)


if __name__ == "__main__":
    main()
