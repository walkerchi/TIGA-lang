"""End-to-end exact kNN build plus compiler-generated consume benchmark."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics

import torch

import tiga as tg
from benchmarks.common.hardware_roofline import (
    interleaved_samples_ms,
    measure_roofs,
)
from benchmarks.common.output_layout import operation_dir
from benchmarks.common.perf_protocol import evaluate_sota_gates
from benchmarks.common.plotting import plot_latency, plot_roofline


class NeighborSum(tg.MessagePassing):
    reducer = tg.sum()

    def edge(self, src, dst, edge):
        del dst
        return edge.weight * src.x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=8192)
    parser.add_argument("--dimensions", type=int, default=3)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--fail-on-gate", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.quick:
        args.nodes, args.k, args.repeat = 1024, 16, 10
    if args.nodes <= 1 or args.dimensions <= 0 or not 0 < args.k < args.nodes:
        raise ValueError("require nodes>1, dimensions>0 and 0<k<nodes")

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260814)
    positions = torch.rand(
        args.nodes, args.dimensions, device=device, generator=generator)
    source = torch.rand(args.nodes, device=device, generator=generator)
    weight = torch.rand(
        args.nodes * args.k, device=device, generator=generator)
    graph = tg.Graph.knn(positions, args.k)
    kernel = NeighborSum()

    def tiga_pipeline():
        return kernel(
            graph=graph, src={"x": source}, dst={},
            edge={"weight": weight})

    def torch_pipeline():
        distances = torch.cdist(positions, positions)
        distances.fill_diagonal_(float("inf"))
        index = torch.topk(
            distances, args.k, dim=1, largest=False, sorted=True).indices
        return (
            source[index] * weight.reshape(args.nodes, args.k)
        ).sum(dim=1)

    actual = tiga_pipeline()
    # Tiga's ranked metric is explicitly pairwise squared Euclidean
    # accumulation with source-index tie breaking. PyTorch's default cdist may
    # switch to a GEMM identity and perturb nearly equal boundary distances;
    # use its direct-distance mode for the one-time semantic oracle while
    # retaining default cdist as the matched SOTA performance provider.
    semantic_distance = torch.cdist(
        positions, positions,
        compute_mode="donot_use_mm_for_euclid_dist",
    )
    semantic_distance.fill_diagonal_(float("inf"))
    semantic_index = torch.topk(
        semantic_distance, args.k, dim=1, largest=False, sorted=True).indices
    expected = (
        source[semantic_index] * weight.reshape(args.nodes, args.k)
    ).sum(dim=1)
    torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-4)
    if kernel.last_variant.lowering != "gf-kernel-to-ttir-ranked-select-consume":
        raise RuntimeError(
            "kNN did not select the compiler-generated ranked build+consume TTIR")
    prepared_pipeline = kernel.prepare(
        graph=graph, src={"x": source}, dst={}, edge={"weight": weight})

    roof = measure_roofs(device, args.quick, args.repeat)
    candidate_pairs = args.nodes * args.nodes
    edges = args.nodes * args.k
    useful_flops = float(
        candidate_pairs * (3 * args.dimensions + 1) + 2 * edges)
    common_bytes = float(
        positions.numel() * positions.element_size()
        + source.numel() * source.element_size()
        + weight.numel() * weight.element_size()
        + args.nodes * source.element_size())
    intensity = useful_flops / common_bytes
    providers = (
        ("tiga.knn_build_consume_ttir", tiga_pipeline),
        ("tiga.knn_build_consume_prepared_ttir", prepared_pipeline),
        ("torch.cdist_topk_gather_sum", torch_pipeline),
    )
    raw_by_provider = interleaved_samples_ms(
        dict(providers), device, args.repeat, None)
    results = []
    for provider, _ in providers:
        raw = raw_by_provider[provider]
        milliseconds = statistics.median(raw)
        results.append({
            "topology": "uniform-random", "locality": "all-pairs",
            "cache": "hot", "provider": provider, "nodes": args.nodes,
            "edges": edges, "features": args.dimensions,
            "index_dtype": "i64", "milliseconds": milliseconds,
            "gedges_per_second": edges / milliseconds / 1e6,
            "achieved_gflops": useful_flops / milliseconds / 1e6,
            "arithmetic_intensity_flop_per_byte": intensity,
            "ideal_cache_bytes": common_bytes,
            "semantic_common_bytes": common_bytes,
            "memory_roof": "DRAM", "samples_ms": raw,
        })
    gate = evaluate_sota_gates(
        results, {"tiga.knn_build_consume_prepared_ttir"},
        baselines={"torch.cdist_topk_gather_sum"}, threshold=1.0)[0]
    case = f"cuda_n{args.nodes}_d{args.dimensions}_k{args.k}_consume"
    output = args.output_dir or operation_dir("knn_graph", case)
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "operation": "knn_graph", "case": case,
        "workload": "exact Euclidean kNN rebuild plus weighted neighbor sum",
        "roof": asdict(roof), "results": results,
        "sota_gates": [gate.to_dict()],
        "config": {
            **vars(args), "output_dir": str(output),
            "builder": "compiler-ranked-tile-select-consume",
            "consumer_lowering": kernel.last_variant.lowering,
            "consumer_provider": kernel.last_variant.provider,
            "semantic_byte_model": "positions + source + ranked-edge weight + output",
            "useful_flop_model": "N^2*(3D+sqrt) + 2*N*k",
        },
    }
    (output / "roofline.json").write_text(json.dumps(payload, indent=2) + "\n")
    plot_roofline(payload, output)
    plot_latency(payload, output)
    candidate = next(
        item for item in results
        if item["provider"] == "tiga.knn_build_consume_prepared_ttir")
    ordinary = next(
        item for item in results
        if item["provider"] == "tiga.knn_build_consume_ttir")
    baseline = next(
        item for item in results
        if item["provider"] == "torch.cdist_topk_gather_sum")
    (output / "REPORT.md").write_text(
        "# Exact kNN build + generated weighted consume\n\n"
        "`Graph.knn` compiles exact ranked selection and the edge UDF into one "
        "TTIR launch. Candidate tiles retain a masked next-power-of-two "
        "selection state and consume exactly k stable keys; no CSR or "
        "pairwise distance matrix is materialized.\n\n"
        f"Tiga prepared: {candidate['milliseconds']:.4f} ms; ordinary "
        f"lazy-JIT hot call: {ordinary['milliseconds']:.4f} ms; matched cdist/top-k/"
        f"gather-sum: {baseline['milliseconds']:.4f} ms. Strict gate: "
        f"{'PASS' if gate.passed else 'FAIL'}, {gate.speedup_vs_sota:.3f}x, "
        f"95% CI [{gate.speedup_ci_low:.3f}, {gate.speedup_ci_high:.3f}].\n\n"
        "![Roofline](roofline.svg)\n\n![Latency](provider_latency.svg)\n")
    print(output)
    if args.fail_on_gate and not gate.passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
