"""Benchmark a RadiusGraph as build, consume, and build+consume phases.

The workload is a scalar distance-weighted aggregation::

    out[dst] = sum(distance(src, dst) * x[src])

``consume-only`` freezes one CSR snapshot. ``relation-reuse`` keeps one logical
radius graph alive. ``logical-rebind`` constructs a new Graph for an unchanged
physical snapshot, while ``topology-rebuild+consume`` increments the position
version before every sample and therefore forces the builder. These phases are
never mixed in one performance gate.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import warnings
import importlib.util
from pathlib import Path

import torch

import graphforge as gf
from graphforge.interop.torch.graph import from_native
from benchmarks.common.output_layout import artifact_path
from benchmarks.common.perf_protocol import evaluate_sota_gates
from benchmarks.graph_operations.radius_build import unit_ball_volume


# "SOTA matched" means neither the median nor the bootstrap lower-confidence
# bound may be slower than the fastest semantically equivalent peer.
SOTA_MATCH_THRESHOLD = 1.0


class DistanceAggregation(gf.MessagePassing):
    reducer = gf.sum()

    def edge(self, src, dst, edge):
        return edge.distance * src.x


def wall_samples_ms(function, device: torch.device, repeat: int) -> list[float]:
    """Measure host-observed latency, including dispatch and synchronizations."""
    for _ in range(3):
        function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    samples = []
    for _ in range(repeat):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        begin = time.perf_counter()
        function()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples.append((time.perf_counter() - begin) * 1e3)
    return samples


def paired_wall_samples_ms(
    first, second, device: torch.device, repeat: int
) -> tuple[list[float], list[float]]:
    """Interleave matched providers to remove clock/thermal time-order bias."""
    for _ in range(3):
        first()
        second()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    samples = ([], [])
    for iteration in range(repeat):
        order = (0, 1) if iteration % 2 == 0 else (1, 0)
        for ordinal in order:
            function = first if ordinal == 0 else second
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            begin = time.perf_counter()
            function()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            samples[ordinal].append((time.perf_counter() - begin) * 1e3)
    return samples


def make_result(
    provider: str,
    phase: str,
    samples: list[float],
    *,
    nodes: int,
    edges: int,
    index_dtype: str,
) -> dict[str, object]:
    milliseconds = statistics.median(samples)
    return {
        "provider": provider,
        "phase": phase,
        # Gate bucket vocabulary is shared with the static roofline suite.
        "topology": "radius",
        "locality": "spatial",
        "cache": phase,
        "nodes": nodes,
        "edges": edges,
        "features": 1,
        "index_dtype": index_dtype,
        "milliseconds": milliseconds,
        "samples_ms": samples,
        "million_edges_per_second": edges / milliseconds / 1e3,
    }


def torch_csr_sum(
    row_ptr: torch.Tensor,
    col_idx: torch.Tensor,
    distance: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Sparse invariant checks.*")
        warnings.filterwarnings("ignore", message="Sparse CSR tensor support.*")
        matrix = torch.sparse_csr_tensor(
            row_ptr,
            col_idx,
            distance,
            size=(row_ptr.numel() - 1, x.numel()),
            check_invariants=False,
        )
    return torch.sparse.mm(matrix, x[:, None])[:, 0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--particles", type=int, default=1 << 15)
    parser.add_argument("--dimensions", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument(
        "--periodic", choices=("none", "box", "skew"), default="none"
    )
    parser.add_argument("--target-degree", type=float, default=32.0)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-num-neighbors", type=int, default=128)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.particles <= 0 or args.repeat <= 1 or args.target_degree <= 0:
        parser.error("particles/target-degree must be positive and repeat must exceed one")
    if args.json is None:
        case = (
            f"{args.device}_n{args.particles}_d{args.dimensions}_"
            f"degree{args.target_degree:g}_{args.periodic}")
        args.json = artifact_path(
            "radius_distance_aggregation", case, "pipeline.json")

    device = torch.device(args.device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    fractional = torch.rand(
        (args.particles, args.dimensions), device=device, generator=generator)
    periodic = None
    volume = 1.0
    if args.periodic == "box":
        periodic = torch.ones(
            args.dimensions, device=device, dtype=fractional.dtype
        )
        positions = fractional
    elif args.periodic == "skew":
        if args.dimensions == 1:
            periodic = torch.ones(1, 1, device=device, dtype=fractional.dtype)
        else:
            periodic = torch.eye(
                args.dimensions, device=device, dtype=fractional.dtype
            )
            periodic[1:, :-1] += 0.35 * torch.eye(
                args.dimensions - 1, device=device, dtype=fractional.dtype
            )
        positions = fractional @ periodic
        volume = float(torch.linalg.det(periodic).abs().item())
    else:
        positions = fractional
    x = torch.rand(args.particles, device=device, generator=generator)
    cutoff = (
        args.target_degree * volume
        / (args.particles * unit_ball_volume(args.dimensions))
    ) ** (1.0 / args.dimensions)

    # Every build-only sample owns a new logical graph. Reusing one Graph here
    # would measure its immutable physical-cache lookup, not construction.
    def build_materialized():
        return gf.Graph.radius(
            positions, cutoff, periodic=periodic
        ).resolve_csr()

    def build_directory():
        return gf.Graph.radius(positions, cutoff).generated_cell_directory()

    build_samples = wall_samples_ms(build_materialized, device, args.repeat)
    directory_build_samples = (
        wall_samples_ms(build_directory, device, args.repeat)
        if periodic is None else None
    )
    build_graph = gf.Graph.radius(positions, cutoff, periodic=periodic)
    row_ptr, col_idx = build_graph.resolve_csr()
    distance = from_native(build_graph).implicit_edge_fields(
        row_ptr, col_idx)["distance"]
    edges = col_idx.numel()
    degrees = row_ptr[1:] - row_ptr[:-1]
    degree_min = int(degrees.min().item())
    degree_max = int(degrees.max().item())

    snapshot = gf.Graph.from_csr(
        row_ptr, col_idx, num_src=args.particles, validate="basic")
    consume_kernel = DistanceAggregation()

    def graphforge_consume():
        return consume_kernel(
            graph=snapshot,
            src={"x": x},
            dst={"x": x},
            edge={"distance": distance},
        )

    def torch_consume():
        return torch_csr_sum(row_ptr, col_idx, distance, x)

    expected = torch_consume()
    torch.testing.assert_close(
        graphforge_consume(), expected, rtol=3e-4, atol=3e-4)
    consume_gf_samples, consume_torch_samples = paired_wall_samples_ms(
        graphforge_consume, torch_consume, device, args.repeat
    )

    dynamic_graph = gf.Graph.radius(positions, cutoff, periodic=periodic)
    dynamic_kernel = DistanceAggregation()

    def graphforge_reuse():
        return dynamic_kernel(
            graph=dynamic_graph, src={"x": x}, dst={"x": x})

    baseline_graph = gf.Graph.radius(positions, cutoff, periodic=periodic)
    baseline_row, baseline_col = baseline_graph.resolve_csr()
    baseline_distance = from_native(baseline_graph).implicit_edge_fields(
        baseline_row, baseline_col)["distance"]

    def torch_reuse():
        return torch_csr_sum(
            baseline_row, baseline_col, baseline_distance, x)

    def graphforge_logical_rebind():
        graph = gf.Graph.radius(positions, cutoff, periodic=periodic)
        return dynamic_kernel(
            graph=graph, src={"x": x}, dst={"x": x})

    def torch_logical_rebind():
        graph = gf.Graph.radius(positions, cutoff, periodic=periodic)
        current_row, current_col = graph.resolve_csr()
        current_distance = from_native(graph).implicit_edge_fields(
            current_row, current_col)["distance"]
        return torch_csr_sum(
            current_row, current_col, current_distance, x)

    # Rebuilding must change the physical snapshot, not merely allocate a new
    # Python Graph around the same immutable tensor.  Translating every point
    # by the same small vector preserves all pair distances/edge sets while
    # incrementing Torch's version counter.  Alternating +/- keeps coordinates
    # bounded and gives both providers the same sequence of snapshots.
    gf_rebuild_positions = positions.clone()
    torch_rebuild_positions = positions.clone()
    rebuild_shift = torch.full(
        (args.dimensions,), cutoff * 0.125,
        device=device, dtype=positions.dtype)
    gf_rebuild_sign = [1.0]
    torch_rebuild_sign = [1.0]

    def graphforge_topology_rebuild():
        gf_rebuild_positions.add_(rebuild_shift, alpha=gf_rebuild_sign[0])
        gf_rebuild_sign[0] = -gf_rebuild_sign[0]
        graph = gf.Graph.radius(
            gf_rebuild_positions, cutoff, periodic=periodic)
        return dynamic_kernel(
            graph=graph, src={"x": x}, dst={"x": x})

    def torch_topology_rebuild():
        torch_rebuild_positions.add_(
            rebuild_shift, alpha=torch_rebuild_sign[0])
        torch_rebuild_sign[0] = -torch_rebuild_sign[0]
        graph = gf.Graph.radius(
            torch_rebuild_positions, cutoff, periodic=periodic)
        current_row, current_col = graph.resolve_csr()
        current_distance = from_native(graph).implicit_edge_fields(
            current_row, current_col)["distance"]
        return torch_csr_sum(
            current_row, current_col, current_distance, x)

    torch.testing.assert_close(
        graphforge_reuse(), expected, rtol=3e-4, atol=3e-4)
    torch.testing.assert_close(
        torch_reuse(), expected, rtol=3e-4, atol=3e-4)
    torch.testing.assert_close(
        graphforge_logical_rebind(), expected, rtol=3e-4, atol=3e-4)
    torch.testing.assert_close(
        torch_logical_rebind(), expected, rtol=3e-4, atol=3e-4)
    torch.testing.assert_close(
        graphforge_topology_rebuild(), torch_topology_rebuild(),
        rtol=3e-4, atol=3e-4)
    reuse_gf_samples, reuse_torch_samples = paired_wall_samples_ms(
        graphforge_reuse, torch_reuse, device, args.repeat
    )
    rebind_gf_samples, rebind_torch_samples = paired_wall_samples_ms(
        graphforge_logical_rebind, torch_logical_rebind, device, args.repeat
    )
    rebuild_gf_samples, rebuild_torch_samples = paired_wall_samples_ms(
        graphforge_topology_rebuild, torch_topology_rebuild,
        device, args.repeat
    )

    index_dtype = str(col_idx.dtype).replace("torch.", "")
    results = [
        make_result(
            "graphforge.materialized_radius_build", "build-only", build_samples,
            nodes=args.particles, edges=edges, index_dtype=index_dtype),
        make_result(
            "graphforge.auto", "consume-only", consume_gf_samples,
            nodes=args.particles, edges=edges, index_dtype=index_dtype),
        make_result(
            "torch.sparse.mm", "consume-only", consume_torch_samples,
            nodes=args.particles, edges=edges, index_dtype=index_dtype),
        make_result(
            "graphforge.dynamic.auto", "relation-reuse", reuse_gf_samples,
            nodes=args.particles, edges=edges, index_dtype=index_dtype),
        make_result(
            "torch.sparse.precomputed_distance", "relation-reuse",
            reuse_torch_samples, nodes=args.particles, edges=edges,
            index_dtype=index_dtype),
        make_result(
            "graphforge.dynamic.auto", "logical-rebind", rebind_gf_samples,
            nodes=args.particles, edges=edges, index_dtype=index_dtype),
        make_result(
            "graphforge.builder+torch.sparse.mm", "logical-rebind",
            rebind_torch_samples, nodes=args.particles, edges=edges,
            index_dtype=index_dtype),
        make_result(
            "graphforge.dynamic.auto", "topology-rebuild+consume",
            rebuild_gf_samples, nodes=args.particles, edges=edges,
            index_dtype=index_dtype),
        make_result(
            "graphforge.builder+torch.sparse.mm", "topology-rebuild+consume",
            rebuild_torch_samples, nodes=args.particles, edges=edges,
            index_dtype=index_dtype),
    ]
    if directory_build_samples is not None:
        results.insert(1, make_result(
            "graphforge.cell_directory_build", "directory-build-only",
            directory_build_samples, nodes=args.particles, edges=edges,
            index_dtype=index_dtype))
    skipped = []
    gate_candidates = {"graphforge.auto", "graphforge.dynamic.auto"}
    if periodic is not None:
        skipped.append({
            "provider": "torch_cluster.radius_graph",
            "reason": "torch-cluster radius_graph has no periodic minimum-image contract",
        })
    elif importlib.util.find_spec("torch_cluster") is None:
        skipped.append({
            "provider": "torch_cluster.radius_graph",
            "reason": "torch_cluster is not installed in this Python environment",
        })
    elif degree_max >= args.max_num_neighbors:
        skipped.append({
            "provider": "torch_cluster.radius_graph",
            "reason": (
                f"observed maximum degree {degree_max} reaches configured "
                f"capacity {args.max_num_neighbors}; exact semantics are not proven"
            ),
        })
    else:
        from torch_cluster import radius_graph

        # GraphForge's cutoff is converted to the positions' FP32 dtype and
        # uses <=. torch-cluster uses a strict boundary test; one representable
        # FP32 step makes the two predicates identical for this registered
        # dataset. The complete edge set is checked before any timing claim.
        cluster_cutoff = torch.nextafter(
            torch.tensor(cutoff, dtype=positions.dtype),
            torch.tensor(float("inf"), dtype=positions.dtype),
        ).item()

        def torch_cluster_build():
            return radius_graph(
                positions,
                cluster_cutoff,
                loop=False,
                max_num_neighbors=args.max_num_neighbors,
                flow="source_to_target",
            )

        cluster_edges = torch_cluster_build()
        cluster_src, cluster_dst = cluster_edges
        expected_keys = torch.sort(
            snapshot.destination_index(row_ptr) * args.particles + col_idx).values
        cluster_keys = torch.sort(
            cluster_dst * args.particles + cluster_src).values
        if not torch.equal(expected_keys, cluster_keys):
            skipped.append({
                "provider": "torch_cluster.radius_graph",
                "reason": (
                    "edge-set differential failed after FP32 boundary normalization: "
                    f"GraphForge={expected_keys.numel()}, "
                    f"torch-cluster={cluster_keys.numel()}"
                ),
            })
        else:
            cluster_build_samples = wall_samples_ms(
                torch_cluster_build, device, args.repeat)

            cluster_rebuild_positions = positions.clone()
            cluster_rebuild_sign = [1.0]

            def torch_cluster_end_to_end():
                cluster_rebuild_positions.add_(
                    rebuild_shift, alpha=cluster_rebuild_sign[0])
                cluster_rebuild_sign[0] = -cluster_rebuild_sign[0]
                current_src, current_dst = radius_graph(
                    cluster_rebuild_positions,
                    cluster_cutoff,
                    loop=False,
                    max_num_neighbors=args.max_num_neighbors,
                    flow="source_to_target",
                )
                current_distance = torch.linalg.vector_norm(
                    cluster_rebuild_positions[current_src]
                    - cluster_rebuild_positions[current_dst], dim=-1)
                output = torch.zeros_like(x)
                output.index_add_(
                    0, current_dst, current_distance * x[current_src])
                return output

            torch.testing.assert_close(
                torch_cluster_end_to_end(), expected, rtol=3e-4, atol=3e-4)
            cluster_e2e_samples = wall_samples_ms(
                torch_cluster_end_to_end, device, args.repeat)
            results.extend((
                make_result(
                    "torch_cluster.radius_graph", "build-only",
                    cluster_build_samples, nodes=args.particles, edges=edges,
                    index_dtype=index_dtype),
                make_result(
                    "torch_cluster.radius_graph+index_add",
                    "topology-rebuild+consume",
                    cluster_e2e_samples, nodes=args.particles, edges=edges,
                    index_dtype=index_dtype),
            ))
            gate_candidates.add("graphforge.materialized_radius_build")

    gates = evaluate_sota_gates(
        results,
        gate_candidates,
        ignored_providers=(
            {"graphforge.materialized_radius_build",
             "graphforge.cell_directory_build"}
            if "graphforge.materialized_radius_build" not in gate_candidates
            else {"graphforge.cell_directory_build"}),
        threshold=SOTA_MATCH_THRESHOLD,
    )
    payload = {
        "config": {
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None),
            "particles": args.particles,
            "dimensions": args.dimensions,
            "periodic": args.periodic,
            "target_degree": args.target_degree,
            "cutoff": cutoff,
            "dtype": str(positions.dtype),
            "repeat": args.repeat,
            "seed": args.seed,
            "max_num_neighbors": args.max_num_neighbors,
            "timer": "synchronized_wall_clock",
        },
        "graph": {
            **build_graph.build_info,
            "minimum_degree": degree_min,
            "maximum_degree": degree_max,
        },
        "compiler": {
            "consume_provider": consume_kernel.last_variant.provider,
            "consume_lowering": consume_kernel.last_variant.lowering,
            "dynamic_provider": dynamic_kernel.last_variant.provider,
            "dynamic_lowering": dynamic_kernel.last_variant.lowering,
            "provider_key": list(dynamic_kernel.last_variant.provider_key),
            "dynamic_ir": dynamic_kernel.ir(),
        },
        "results": results,
        "sota_gates": [gate.to_dict() for gate in gates],
        "skipped": skipped,
        "baseline_scope": (
            "Every logical-rebind sample creates a new Graph around an unchanged "
            "versioned snapshot. Every topology-rebuild sample increments the "
            "position version before resolving the relation. torch-cluster is an "
            "independently implemented radius "
            "builder. The fastest measured end-to-end peer reuses GraphForge's "
            "builder only in the explicitly named GraphForge-builder peer; "
            "consumer-only and relation-reuse are separate gate buckets."
        ),
    }

    print(
        f"device={device} N={args.particles:,} D={args.dimensions} "
        f"cutoff={cutoff:.6g} edges={edges:,} degree=[{degree_min}, {degree_max}]")
    print("phase          provider                                  ms    Medge/s")
    for item in results:
        print(
            f"{item['phase']:14} {item['provider']:40} "
            f"{item['milliseconds']:8.3f} "
            f"{item['million_edges_per_second']:10.2f}")
    for gate in gates:
        status = "PASS" if gate.passed else "FAIL"
        print(
            f"{status} {gate.cache}: {gate.candidate} vs {gate.baseline} "
            f"{gate.speedup_vs_sota:.3f}x "
            f"CI=[{gate.speedup_ci_low:.3f}, {gate.speedup_ci_high:.3f}]")
    for item in skipped:
        print(f"SKIP {item['provider']}: {item['reason']}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
        from benchmarks.common.diagnostic_plotting import plot_json
        plot_json(args.json)
    if args.fail_on_gate and not all(gate.passed for gate in gates):
        raise SystemExit("one or more performance gates failed")


if __name__ == "__main__":
    main()
