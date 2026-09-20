"""Exact procedural kNN build roofline with matched library semantics."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics

import torch

import tiga as tg
from benchmarks.common.hardware_roofline import measure_roofs, samples_ms
from benchmarks.common.output_layout import operation_dir
from benchmarks.common.perf_protocol import evaluate_sota_gates
from benchmarks.common.plotting import plot_latency, plot_roofline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=8192)
    parser.add_argument("--dimensions", type=int, default=3)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.quick:
        args.nodes, args.k, args.repeat = 1024, 16, 10
    if args.nodes <= 1 or args.dimensions <= 0 or not 0 < args.k < args.nodes:
        raise ValueError("require nodes>1, dimensions>0 and 0<k<nodes")

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260813)
    positions = torch.rand(
        (args.nodes, args.dimensions), device=device, generator=generator)
    graph = tg.Graph.knn(positions, args.k)

    # Warm only immutable fixed-degree CSR metadata. resolve_csr still runs
    # exhaustive distance/top-k on every timed invocation and never reuses
    # col_idx.
    graph.resolve_csr()

    def tiga_run():
        return graph.resolve_csr()

    def torch_matched():
        distances = torch.cdist(positions, positions)
        distances.fill_diagonal_(float("inf"))
        columns = torch.topk(
            distances, args.k, dim=1, largest=False, sorted=True
        ).indices.reshape(-1).to(torch.int64)
        rows = torch.arange(
            0, columns.numel() + 1, args.k,
            device=device, dtype=torch.int64)
        return rows, columns

    row_ptr, col_idx = tiga_run()
    expected_row_ptr, expected = torch_matched()
    if not torch.equal(row_ptr, expected_row_ptr):
        raise RuntimeError("Tiga kNN row pointer differs from matched peer")
    # FP32 addmm and cdist can exchange neighbors whose distances are within
    # rounding error. Compare the selected distance multisets, not arbitrary
    # index tie-breaking, while still requiring exactly k exhaustive results.
    candidate_distance = torch.linalg.vector_norm(
        positions[col_idx.reshape(args.nodes, args.k)] - positions[:, None, :],
        dim=-1,
    ).sort(dim=1).values
    expected_distance = torch.linalg.vector_norm(
        positions[expected.reshape(args.nodes, args.k)] - positions[:, None, :],
        dim=-1,
    ).sort(dim=1).values
    torch.testing.assert_close(
        candidate_distance, expected_distance, rtol=2e-5, atol=2e-6)
    if row_ptr[-1].item() != args.nodes * args.k:
        raise RuntimeError("Tiga kNN row pointer is malformed")

    roof = measure_roofs(device, args.quick, args.repeat)
    candidates = args.nodes * args.nodes
    edges = args.nodes * args.k
    useful_flops = float(candidates * (3 * args.dimensions + 1))
    common_bytes = float(
        positions.numel() * positions.element_size()
        + edges * 8 + (args.nodes + 1) * 8)
    intensity = useful_flops / common_bytes
    providers = (
        ("tiga.procedural_knn", tiga_run),
        ("torch.cdist_topk", torch_matched),
    )
    results = []
    for provider, run in providers:
        raw = samples_ms(run, device, args.repeat, None)
        median = statistics.median(raw)
        results.append({
            "topology": "uniform-random", "locality": "all-pairs",
            "cache": "hot", "provider": provider, "nodes": args.nodes,
            "edges": edges, "features": args.dimensions,
            "index_dtype": "i64", "milliseconds": median,
            "gedges_per_second": edges / median / 1e6,
            "candidate_gpairs_per_second": candidates / median / 1e6,
            "achieved_gflops": useful_flops / median / 1e6,
            "arithmetic_intensity_flop_per_byte": intensity,
            "ideal_cache_bytes": common_bytes,
            "semantic_common_bytes": common_bytes,
            "memory_roof": "DRAM", "samples_ms": raw,
        })
    gate = evaluate_sota_gates(
        results, {"tiga.procedural_knn"},
        baselines={"torch.cdist_topk"}, threshold=1.0)[0]
    case = f"cuda_n{args.nodes}_d{args.dimensions}_k{args.k}"
    output_dir = args.output_dir or operation_dir("knn_graph", case)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "operation": "knn_graph",
        "workload": "exact Euclidean directed k-nearest-neighbor build",
        "case": case, "roof": asdict(roof), "results": results,
        "sota_gates": [gate.to_dict()],
        "config": {
            **vars(args), "output_dir": str(output_dir),
            "builder": graph.build_info["builder"],
            "row_ptr_policy": "persistent fixed-degree metadata",
            "candidate_pairs": candidates,
            "accepted_edges": edges,
            "useful_flop_model": "N^2 * (3D + sqrt)",
            "semantic_byte_model": "positions + CSR indices",
            "note": "cdist temporary traffic is provider-dependent and excluded",
        },
    }
    (output_dir / "roofline.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n")
    plot_roofline(payload, output_dir)
    plot_latency(payload, output_dir)
    candidate, baseline = results
    (output_dir / "REPORT.md").write_text(
        "# Exact procedural kNN build\n\n"
        "`Graph.knn` stores only positions, k and lifecycle semantics. This "
        "explicit materialization case selects an exhaustive squared-distance "
        "dense-library contraction plus top-k (sqrt is order-preserving); no "
        "workload-named kernel is embedded in compiler core.\n\n"
        f"Tiga: {candidate['milliseconds']:.4f} ms; matched Torch library "
        f"decomposition: {baseline['milliseconds']:.4f} ms. Strict gate: "
        f"{'PASS' if gate.passed else 'FAIL'}, speedup "
        f"{gate.speedup_vs_sota:.3f}x, 95% CI "
        f"[{gate.speedup_ci_low:.3f}, {gate.speedup_ci_high:.3f}].\n\n"
        "The roofline counts useful distance work; sort/select and dense "
        "temporary traffic remain explicit operational-model caveats.\n\n"
        "![Roofline](roofline.svg)\n\n![Latency](provider_latency.svg)\n")
    print(output_dir)
    if args.fail_on_gate and not gate.passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
