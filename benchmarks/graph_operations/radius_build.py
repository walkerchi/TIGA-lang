"""End-to-end Dynamic RadiusGraph build benchmark.

This measures the public ``Graph.resolve_csr`` path, including tensor dispatch,
candidate generation, exact cutoff filtering and CSR row-pointer construction.
The all-pairs provider is a correctness/performance oracle for small inputs;
it is deliberately skipped for large N rather than allowed to OOM.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch

import graphforge as gf
from benchmarks.common.hardware_roofline import measure_roofs
from benchmarks.common.output_layout import artifact_path
from benchmarks.common.plotting import plot_latency, plot_roofline, write_report


def samples_ms(function, device: torch.device, repeat: int) -> list[float]:
    for _ in range(3):
        function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    samples = []
    for _ in range(repeat):
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            function()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        else:
            begin = time.perf_counter()
            function()
            samples.append((time.perf_counter() - begin) * 1e3)
    return samples


def unit_ball_volume(dimensions: int) -> float:
    return math.pi ** (dimensions / 2) / math.gamma(dimensions / 2 + 1)


def edge_keys(graph: gf.Graph, rows: int) -> torch.Tensor:
    row_ptr, col_idx = graph.resolve_csr()
    dst = graph.destination_index(row_ptr)
    return torch.sort(dst * rows + col_idx).values


def result(
    name: str,
    graph: gf.Graph,
    samples: list[float],
    *,
    dimensions: int,
    particles: int,
    roof,
    logical_info: dict[str, object] | None = None,
    builder: str | None = None,
    implementation_bytes: int | None = None,
    materialized: bool = True,
) -> dict[str, object]:
    info = graph.build_info if logical_info is None else logical_info
    median = statistics.median(samples)
    candidates = int(info["candidate_pairs"])
    edges = int(info["accepted_edges"])
    # Provider-independent semantic work credits only the accepted logical
    # radius relation. Candidate amplification is implementation diagnostics.
    useful_flops = float(edges * (3 * dimensions + 1))
    no_reuse_bytes = (
        candidates * 2 * dimensions * 4
        + edges * 8
        + (particles + 1) * 8)
    if implementation_bytes is None:
        implementation_bytes = no_reuse_bytes
    semantic_bytes = (
        particles * dimensions * 4
        + edges * 8
        + (particles + 1) * 8)
    intensity = useful_flops / max(semantic_bytes, 1)
    achieved = useful_flops / median / 1e6
    optimistic = min(
        roof.fp32_gflops, roof.dram_bandwidth_gbs * intensity)
    operational_intensity = candidates / max(no_reuse_bytes, 1)
    achieved_gcandidate = candidates / median / 1e6
    operational_roof = roof.dram_bandwidth_gbs * operational_intensity
    return {
        "provider": name,
        "builder": builder or str(info["builder"]),
        "physical_realization": "materialized_csr" if materialized else "cell_directory",
        "materialized": materialized,
        "topology": "radius",
        "locality": "spatial",
        "cache": "build-only",
        "nodes": particles,
        "edges": edges,
        "features": 1,
        "index_dtype": "int64",
        "milliseconds": median,
        "samples_ms": samples,
        "candidate_pairs": candidates,
        "accepted_edges": edges,
        "candidate_mpair_per_second": (
            candidates / median / 1e3 if materialized else None),
        "accepted_medge_per_second": (
            edges / median / 1e3 if materialized else None),
        "candidate_to_edge": candidates / max(edges, 1),
        "achieved_gflops": achieved,
        "algorithmic_gbs_no_reuse": implementation_bytes / median / 1e6,
        "memory_roof": "DRAM",
        "optimistic_roof_gflops": optimistic,
        "percent_of_optimistic_roof": 100 * achieved / optimistic,
        "ideal_cache_bytes": semantic_bytes,
        "semantic_common_bytes": semantic_bytes,
        "implementation_modeled_bytes": implementation_bytes,
        "arithmetic_intensity_flop_per_byte": intensity,
        "operational_intensity_candidate_per_byte": (
            operational_intensity if materialized else None),
        "achieved_gcandidate_per_second": (
            achieved_gcandidate if materialized else None),
        "operational_dram_roof_gcandidate_per_second": (
            operational_roof if materialized else None),
        "percent_of_operational_roof": (
            100 * achieved_gcandidate / operational_roof
            if materialized else None),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--particles", type=int, default=1 << 15)
    parser.add_argument("--dimensions", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--target-degree", type=float, default=32.0)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--all-pairs-max", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.particles < 0 or args.repeat <= 0 or args.target_degree <= 0:
        parser.error("particles must be non-negative; repeat/target-degree must be positive")
    if args.json is None:
        case = (
            f"{args.device}_n{args.particles}_d{args.dimensions}_"
            f"degree{args.target_degree:g}")
        args.json = artifact_path(
            "radius_graph_build", case, "roofline.json")

    device = torch.device(args.device)
    roof = measure_roofs(device, args.particles <= 4096, args.repeat)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    positions = torch.rand(
        (args.particles, args.dimensions), device=device, generator=generator)
    cutoff = (
        args.target_degree
        / max(args.particles * unit_ball_volume(args.dimensions), 1.0)
    ) ** (1.0 / args.dimensions)

    def materialized_build():
        return gf.Graph.radius(positions, cutoff).resolve_csr()

    def directory_build():
        return gf.Graph.radius(positions, cutoff).generated_cell_directory()

    cell_samples = samples_ms(materialized_build, device, args.repeat)
    cell_graph = gf.Graph.radius(positions, cutoff)
    cell_graph.resolve_csr()
    logical_info = cell_graph.build_info
    directory_graph = gf.Graph.radius(positions, cutoff)
    directory_samples = samples_ms(directory_build, device, args.repeat)
    directory = directory_graph.generated_cell_directory()
    if directory is None:
        raise RuntimeError("compact cell directory is unavailable")
    directory_bytes = positions.numel() * positions.element_size() + sum(
        value.numel() * value.element_size()
        for value in (
            directory.cell_ptr, directory.particle_order,
            directory.cell_coordinates, directory.extents,
            directory.strides, directory.neighbor_offsets))
    results = [
        result(
            "graphforge.compact_cell_directory", directory_graph,
            directory_samples, dimensions=args.dimensions,
            particles=args.particles, roof=roof,
            logical_info=logical_info, builder="uniform_cell_directory",
            implementation_bytes=directory_bytes, materialized=False),
        result(
            "graphforge.materialized_cell_list_csr", cell_graph, cell_samples,
            dimensions=args.dimensions, particles=args.particles, roof=roof),
    ]
    skipped = []

    if args.particles <= args.all_pairs_max:
        def euclidean_metric(src, dst, edge):
            del src, dst
            return torch.linalg.vector_norm(edge.displacement, dim=-1)

        all_pairs_graph = gf.Graph.radius(
            positions, cutoff, metric=euclidean_metric)
        expected = edge_keys(all_pairs_graph, args.particles)
        actual = edge_keys(cell_graph, args.particles)
        torch.testing.assert_close(actual, expected)
        all_pair_samples = samples_ms(
            all_pairs_graph.resolve_csr, device, args.repeat)
        results.append(result(
            "graphforge.all_pairs_reference", all_pairs_graph, all_pair_samples,
            dimensions=args.dimensions, particles=args.particles, roof=roof))
    else:
        skipped.append({
            "provider": "graphforge.all_pairs_reference",
            "reason": f"N={args.particles} exceeds --all-pairs-max={args.all_pairs_max}",
        })

    print(
        f"device={device} N={args.particles:,} D={args.dimensions} "
        f"cutoff={cutoff:.6g}")
    print("provider                         ms candidates      edges  Mcand/s  Medge/s  cand/edge")
    for item in results:
        candidate_rate = (
            f"{item['candidate_mpair_per_second']:8.2f}"
            if item["candidate_mpair_per_second"] is not None else f"{'n/a':>8}")
        edge_rate = (
            f"{item['accepted_medge_per_second']:8.2f}"
            if item["accepted_medge_per_second"] is not None else f"{'n/a':>8}")
        print(
            f"{item['provider']:30} {item['milliseconds']:8.3f} "
            f"{item['candidate_pairs']:10,d} {item['accepted_edges']:10,d} "
            f"{candidate_rate} {edge_rate} "
            f"{item['candidate_to_edge']:9.2f}")
    for item in skipped:
        print(f"SKIP {item['provider']}: {item['reason']}")

    if args.json:
        payload = {
            "operation": "radius_graph_build",
            "workload": "radius_graph_build",
            "case": args.json.parent.name,
            "roof": {
                "dram_bandwidth_gbs": roof.dram_bandwidth_gbs,
                "l2_bandwidth_gbs": roof.l2_bandwidth_gbs,
                "fp32_gflops": roof.fp32_gflops,
                "dram_tensor_bytes": roof.dram_tensor_bytes,
                "l2_tensor_bytes": roof.l2_tensor_bytes,
                "matmul_size": roof.matmul_size,
            },
            "config": {
                "device": str(device),
                "device_name": (
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda" else None),
                "particles": args.particles,
                "dimensions": args.dimensions,
                "target_degree": args.target_degree,
                "cutoff": cutoff,
                "dtype": str(positions.dtype),
                "repeat": args.repeat,
                "seed": args.seed,
                "useful_flop_convention": (
                    "accepted_edges * (3*dimensions + 1), provider-independent"),
                "semantic_byte_model": (
                    "positions input plus accepted CSR output, provider-independent"),
                "implementation_byte_model": (
                    "no-reuse candidate endpoint positions plus accepted CSR"),
                "candidate_amplification_is_diagnostic_only": True,
            },
            "results": results,
            "skipped": skipped,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
        plot_roofline(payload, args.json.parent)
        plot_latency(payload, args.json.parent)
        write_report(payload, None, None, None, args.json.parent)


if __name__ == "__main__":
    main()
