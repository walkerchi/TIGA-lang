"""Hierarchical useful-work roofline for radius distance aggregation."""

from __future__ import annotations

from dataclasses import asdict
import argparse
import json
import math
from pathlib import Path
import statistics
import time
import warnings

import torch

import tiga as tg
from tiga.codegen import prepare_ttir_generated_radius
from tiga.interop.torch.compiler_bridge import (
    lower_kernel_to_ttir_plan,
    lower_mlir_stages,
    message_passing_domain_mlir,
)
from tiga.interop.torch.graph import from_native
from tiga.compiler.capture import WeightedSumPattern
from benchmarks.common.hardware_roofline import measure_roofs
from benchmarks.common.output_layout import operation_dir
from benchmarks.common.plotting import plot_latency, plot_roofline, write_report
from benchmarks.graph_operations.radius_build import unit_ball_volume
from benchmarks.common.perf_protocol import evaluate_sota_gates
from benchmarks.kernels.sparse_triton_oracles import (
    launch_generated_radius_distance_sum,
    launch_radius_distance_sum,
)


class DistanceAggregation(tg.MessagePassing):
    reducer = tg.sum()

    def edge(self, src, dst, edge):
        return edge.distance * src.x


def wall_samples_ms(fn, repeat: int) -> list[float]:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return samples


def tensor_bytes(*values: torch.Tensor) -> int:
    return sum(value.numel() * value.element_size() for value in values)


def result(
    provider: str,
    phase: str,
    samples: list[float],
    *,
    nodes: int,
    edges: int,
    useful_flops: float,
    ideal_bytes: int,
    implementation_bytes: int | None = None,
    roof,
) -> dict[str, object]:
    milliseconds = statistics.median(samples)
    intensity = useful_flops / ideal_bytes
    optimistic = min(
        roof.fp32_gflops, roof.dram_bandwidth_gbs * intensity)
    achieved = useful_flops / milliseconds / 1e6
    return {
        "topology": "radius",
        "locality": "spatial",
        "cache": phase,
        "provider": provider,
        "nodes": nodes,
        "edges": edges,
        "features": 1,
        "index_dtype": "int64",
        "milliseconds": milliseconds,
        "gedges_per_second": edges / milliseconds / 1e6,
        "achieved_gflops": achieved,
        "algorithmic_gbs_no_reuse": ideal_bytes / milliseconds / 1e6,
        "memory_roof": "DRAM",
        "optimistic_roof_gflops": optimistic,
        "percent_of_optimistic_roof": 100 * achieved / optimistic,
        "ideal_cache_bytes": ideal_bytes,
        "semantic_common_bytes": ideal_bytes,
        "implementation_modeled_bytes": implementation_bytes,
        "arithmetic_intensity_flop_per_byte": intensity,
        "samples_ms": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--particles", type=int, default=1 << 15)
    parser.add_argument("--dimensions", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--target-degree", type=float, default=32.0)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.quick:
        args.particles, args.repeat = 1 << 13, 5
    case = (
        f"cuda_n{args.particles}_d{args.dimensions}_"
        f"degree{args.target_degree:g}")
    output = args.output_dir or operation_dir(
        "radius_distance_aggregation", case)
    output.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator(device="cuda").manual_seed(17)
    positions = torch.rand(
        (args.particles, args.dimensions),
        device="cuda", generator=generator)
    x = torch.rand(args.particles, device="cuda", generator=generator)
    cutoff = (
        args.target_degree
        / (args.particles * unit_ball_volume(args.dimensions))
    ) ** (1.0 / args.dimensions)
    roof = measure_roofs(torch.device("cuda"), args.quick, args.repeat)

    generated_graph = tg.Graph.radius(positions, cutoff)
    directory = generated_graph.generated_cell_directory()
    if directory is None:
        raise RuntimeError("dense generated cell directory is unavailable")
    generated_launch = launch_generated_radius_distance_sum(
        directory, positions, x)
    if generated_launch is None:
        raise RuntimeError("generated radius provider is unavailable")
    domain = message_passing_domain_mlir(
        kernel=DistanceAggregation(),
        graph=generated_graph,
        directory=directory,
        src={"x": x}, dst={}, edge={}, params={},
        kernel_name="RadiusDistanceRoofline",
    )
    stages = lower_mlir_stages(domain)
    direct_manifest = lower_kernel_to_ttir_plan(stages.kernel)
    direct_plan = prepare_ttir_generated_radius(
        direct_manifest.module,
        num_rows=args.particles,
        block_rows=direct_manifest.block_rows,
        num_warps=direct_manifest.num_warps,
    )

    materialized_graph = tg.Graph.radius(positions, cutoff)
    row_ptr, col_idx = materialized_graph.resolve_csr()
    edges = col_idx.numel()
    candidate_pairs = int(materialized_graph.build_info["candidate_pairs"])
    materialized_launch = launch_radius_distance_sum(
        row_ptr, col_idx, positions, x, num_rows=args.particles)
    if materialized_launch is None:
        raise RuntimeError("materialized radius provider is unavailable")
    distance = from_native(materialized_graph).implicit_edge_fields(
        row_ptr, col_idx)["distance"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        matrix = torch.sparse_csr_tensor(
            row_ptr, col_idx, distance,
            size=(args.particles, args.particles))

    generated_consume = lambda: generated_launch.plan.run(x)
    direct_consume = lambda: direct_plan.run(directory, positions, x)
    materialized_consume = lambda: materialized_launch.plan.run(x)
    torch_consume = lambda: torch.sparse.mm(matrix, x[:, None])[:, 0]
    expected = torch_consume()
    torch.testing.assert_close(generated_consume(), expected, rtol=3e-4, atol=3e-4)
    torch.testing.assert_close(direct_consume(), expected, rtol=3e-4, atol=3e-4)
    torch.testing.assert_close(materialized_consume(), expected, rtol=3e-4, atol=3e-4)

    dynamic_graph = tg.Graph.radius(positions, cutoff)
    dynamic_kernel = DistanceAggregation()
    generated_reuse = lambda: dynamic_kernel(
        graph=dynamic_graph, src={"x": x}, dst={"x": x})
    csr_graph = tg.Graph.radius(positions, cutoff)

    def materialized_reuse():
        current_row, current_col = csr_graph.resolve_csr()
        launch = launch_radius_distance_sum(
            current_row, current_col, positions, x,
            num_rows=args.particles)
        assert launch is not None
        return launch.output

    def generated_build_consume():
        graph = tg.Graph.radius(positions, cutoff)
        return dynamic_kernel(graph=graph, src={"x": x}, dst={"x": x})

    def materialized_build_consume():
        graph = tg.Graph.radius(positions, cutoff)
        current_row, current_col = graph.resolve_csr()
        launch = launch_radius_distance_sum(
            current_row, current_col, positions, x,
            num_rows=args.particles)
        assert launch is not None
        return launch.output

    torch.testing.assert_close(generated_reuse(), expected, rtol=3e-4, atol=3e-4)
    torch.testing.assert_close(materialized_reuse(), expected, rtol=3e-4, atol=3e-4)
    torch.testing.assert_close(
        generated_build_consume(), expected, rtol=3e-4, atol=3e-4)
    torch.testing.assert_close(
        materialized_build_consume(), expected, rtol=3e-4, atol=3e-4)

    # Useful-work convention: D subtracts, D multiplies, D-1 additions,
    # one sqrt-equivalent, message multiply and reducer add = 3D+2/edge.
    useful_flops = float(edges * (3 * args.dimensions + 2))
    generated_bytes = tensor_bytes(
        positions, x, directory.cell_ptr, directory.particle_order,
        directory.cell_coordinates, directory.extents, directory.strides,
        directory.neighbor_offsets) + x.numel() * x.element_size()
    csr_bytes = tensor_bytes(
        positions, x, row_ptr, col_idx) + x.numel() * x.element_size()
    sparse_bytes = tensor_bytes(
        x, row_ptr, col_idx, distance) + x.numel() * x.element_size()
    # The main roofline compares one logical operation. Every provider is
    # therefore charged the same compulsory semantic input/output bytes.
    # Provider-specific physical traffic remains diagnostic metadata only.
    semantic_bytes = tensor_bytes(positions, x) + x.numel() * x.element_size()

    providers = {
        "tiga.direct_ttir_generated": (
            direct_consume, generated_bytes,
            candidate_pairs * (3 * args.dimensions + 1) + 2 * edges,
            "candidate distance tests plus accepted message/reduction"),
        "tiga.triton_generated_oracle": (
            generated_consume, generated_bytes,
            candidate_pairs * (3 * args.dimensions + 1) + 2 * edges,
            "candidate distance tests plus accepted message/reduction"),
        "tiga.materialized_csr": (
            materialized_consume, csr_bytes, useful_flops,
            "accepted-edge distance plus message/reduction"),
        "torch.sparse.precomputed_distance": (
            torch_consume, sparse_bytes, 2 * edges,
            "precomputed-distance multiply and reduction only"),
    }
    consumer_results = []
    for name, (fn, implementation_bytes, executed_flops, executed_model) in providers.items():
        item = result(
            name, "consumer-only", wall_samples_ms(fn, args.repeat),
            nodes=args.particles, edges=edges, useful_flops=useful_flops,
            ideal_bytes=semantic_bytes,
            implementation_bytes=implementation_bytes, roof=roof)
        executed_intensity = executed_flops / implementation_bytes
        executed_roof = min(
            roof.fp32_gflops, roof.l2_bandwidth_gbs * executed_intensity)
        item.update({
            "executed_flop_estimate": executed_flops,
            "executed_work_model": executed_model,
            "executed_intensity_flop_per_byte": executed_intensity,
            "executed_gflops": executed_flops / item["milliseconds"] / 1e6,
            "executed_memory_roof": "L2",
            "executed_optimistic_roof_gflops": executed_roof,
            "percent_of_executed_roof": (
                100 * (executed_flops / item["milliseconds"] / 1e6)
                / executed_roof),
        })
        consumer_results.append(item)
    lifecycle_results = [
        result(
            "tiga.dynamic_auto", "relation-reuse",
            wall_samples_ms(generated_reuse, args.repeat),
            nodes=args.particles, edges=edges, useful_flops=useful_flops,
            ideal_bytes=semantic_bytes, implementation_bytes=sparse_bytes,
            roof=roof),
        result(
            "tiga.materialized_reuse", "relation-reuse",
            wall_samples_ms(materialized_reuse, args.repeat),
            nodes=args.particles, edges=edges, useful_flops=useful_flops,
            ideal_bytes=semantic_bytes,
            implementation_bytes=csr_bytes, roof=roof),
        result(
            "tiga.generated_build_consume", "fresh-build+consume",
            wall_samples_ms(generated_build_consume, args.repeat),
            nodes=args.particles, edges=edges, useful_flops=useful_flops,
            ideal_bytes=semantic_bytes,
            implementation_bytes=generated_bytes, roof=roof),
        result(
            "tiga.materialized_build_consume", "fresh-build+consume",
            wall_samples_ms(materialized_build_consume, args.repeat),
            nodes=args.particles, edges=edges, useful_flops=useful_flops,
            ideal_bytes=semantic_bytes,
            implementation_bytes=csr_bytes, roof=roof),
    ]
    lifecycle_gate = evaluate_sota_gates(
        lifecycle_results,
        {"tiga.generated_build_consume"},
        baselines={"tiga.materialized_build_consume"},
        threshold=1.0,
    )[0]
    generated_peak_bytes = generated_bytes
    materialized_peak_bytes = sparse_bytes
    payload = {
        "operation": "radius_distance_aggregation",
        "workload": "radius_distance_aggregation",
        "case": case,
        "roof": asdict(roof),
        "config": {
            "particles": args.particles,
            "dimensions": args.dimensions,
            "target_degree": args.target_degree,
            "cutoff": cutoff,
            "accepted_edges": edges,
            "candidate_pairs": candidate_pairs,
            "candidate_amplification": candidate_pairs / max(edges, 1),
            "selected_auto_lowering": dynamic_kernel.last_variant.lowering,
            "useful_flop_convention": "accepted_edges * (3*dimensions + 2)",
            "sqrt_equivalent_flops": 1,
            "semantic_byte_model": (
                "provider-independent compulsory positions + source field + output"),
            "implementation_traffic_is_diagnostic_only": True,
        },
        "results": consumer_results + lifecycle_results,
        "builder_consumer_fusion": {
            "generated_edge_index_bytes": 0,
            "generated_edge_message_bytes": 0,
            "generated_peak_modeled_bytes": generated_peak_bytes,
            "materialized_peak_modeled_bytes": materialized_peak_bytes,
            "peak_byte_reduction": materialized_peak_bytes / generated_peak_bytes,
            "fresh_end_to_end_speedup": lifecycle_gate.speedup_vs_sota,
            "gate": lifecycle_gate.to_dict(),
            "proof": (
                "generated execution binds only cell directory, positions, source, "
                "and output; no row_ptr/col_idx/distance/message allocation is in "
                "the executable ABI"
            ),
        },
        "baseline_scope": (
            "consumer-only Torch receives precomputed distances and is an "
            "optimistic library bound; relation reuse and fresh build+consume "
            "are separate buckets with identical cutoff semantics."),
    }
    json_path = output / "roofline.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    plot_roofline(payload, output)
    plot_latency(payload, output)
    write_report(payload, None, None, None, output)
    print(
        f"N={args.particles:,} edges={edges:,} candidates={candidate_pairs:,} "
        f"amplification={candidate_pairs / edges:.2f}x")
    for item in payload["results"]:
        print(
            f"{item['cache']:14} {item['provider']:44} "
            f"{item['milliseconds']:.4f} ms")
    print(json_path)
    print(output / "roofline.svg")
    print(
        f"{'PASS' if lifecycle_gate.passed else 'FAIL'} generated builder-consumer "
        f"vs materialized: {lifecycle_gate.speedup_vs_sota:.3f}x, "
        f"CI low={lifecycle_gate.speedup_ci_low:.3f}x, "
        f"peak bytes={materialized_peak_bytes / generated_peak_bytes:.2f}x lower"
    )
    if args.fail_on_gate and not lifecycle_gate.passed:
        raise SystemExit("generated builder-consumer fusion gate failed")


if __name__ == "__main__":
    main()
