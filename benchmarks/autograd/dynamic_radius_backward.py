"""Fixed-snapshot dynamic-radius geometry/source VJP performance gate.

The user writes only ``edge.distance * src.x``. GraphForge derives a packed
position/source VJP primitive, lowers it to TTIR and compares its warm backward
against Torch autograd plus a benchmark-only handwritten Triton kernel.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import statistics
import time

import torch

import graphforge as gf
from benchmarks.common.hardware_roofline import measure_roofs, samples_ms
from benchmarks.common.output_layout import artifact_path
from benchmarks.common.perf_protocol import evaluate_sota_gates
from benchmarks.common.plotting import plot_latency, plot_roofline, write_report
from benchmarks.kernels.sparse_triton_oracles import (
    prepare_radius_distance_sum_vjp,
)


class RadiusDistanceAggregation(gf.MessagePassing):
    reducer = gf.sum()

    def edge(self, src, dst, edge):
        return edge.distance * src.x


@dataclass
class Result:
    topology: str
    locality: str
    cache: str
    provider: str
    nodes: int
    edges: int
    features: int
    index_dtype: str
    milliseconds: float
    gedges_per_second: float
    achieved_gflops: float
    memory_roof: str
    optimistic_roof_gflops: float
    percent_of_optimistic_roof: float
    arithmetic_intensity_flop_per_byte: float
    samples_ms: list[float]
    compile_ms: float | None = None
    materialize_ms: float | None = None
    lowering: str | None = None
    physical_ideal_bytes: int | None = None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=32768)
    parser.add_argument("--degree", type=int, default=32)
    parser.add_argument("--dimensions", type=int, choices=(2, 3), default=3)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.nodes, args.degree, args.repeat = 4096, 16, 20
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260814)
    positions = torch.rand(
        args.nodes, args.dimensions, device=device, generator=generator,
        requires_grad=True)
    source = torch.randn(
        args.nodes, device=device, generator=generator, requires_grad=True)
    cotangent = torch.randn(args.nodes, device=device, generator=generator)
    unit_ball = torch.pi if args.dimensions == 2 else 4.0 * torch.pi / 3.0
    cutoff = float((args.degree / (args.nodes * unit_ball)) **
                   (1.0 / args.dimensions))
    graph = gf.Graph.radius(positions, cutoff=cutoff)
    kernel = RadiusDistanceAggregation()
    output = kernel(graph=graph, src={"x": source}, dst={"x": source})
    torch_graph = kernel._torch_executor._last_executable.graph

    materialize_started = time.perf_counter_ns()
    row_ptr, col_idx = torch_graph.resolve_csr()
    destination = torch_graph.destination_index(row_ptr)
    torch.cuda.synchronize()
    materialize_ms = (time.perf_counter_ns() - materialize_started) / 1e6
    edges = col_idx.numel()

    displacement = positions[col_idx] - positions[destination]
    distance = torch.linalg.vector_norm(displacement, dim=-1)
    reference_output = torch.zeros_like(source).index_add(
        0, destination, distance * source[col_idx])

    def graphforge_backward():
        return torch.autograd.grad(
            output, (positions, source), cotangent, retain_graph=True)

    def torch_backward():
        return torch.autograd.grad(
            reference_output, (positions, source), cotangent,
            retain_graph=True)

    oracle = prepare_radius_distance_sum_vjp(
        destination, col_idx, positions.detach(), source.detach())
    if oracle is None:
        raise RuntimeError("matched handwritten Triton VJP is unavailable")

    # First call includes topology binding and compiler/provider JIT; it is
    # reported separately and excluded from warm backward samples.
    compile_started = time.perf_counter_ns()
    graphforge_backward()
    torch.cuda.synchronize()
    cold_backward_ms = (time.perf_counter_ns() - compile_started) / 1e6
    prepared = torch_graph._graphforge_compiled_radius_vjp[1]
    compile_ms = prepared.executable.compile_ms

    expected = torch_backward()
    for name, provider in {
        "graphforge.generated_vjp": graphforge_backward,
        "torch.autograd": torch_backward,
        "handwritten.triton": lambda: oracle.run(cotangent),
    }.items():
        actual = provider()
        torch.testing.assert_close(
            actual[0], expected[0], rtol=4e-4, atol=4e-4,
            msg=lambda error: f"{name} position VJP: {error}")
        torch.testing.assert_close(
            actual[1], expected[1], rtol=4e-4, atol=4e-4,
            msg=lambda error: f"{name} source VJP: {error}")

    providers = {
        "graphforge.generated_vjp": graphforge_backward,
        "torch.autograd": torch_backward,
        "handwritten.triton": lambda: oracle.run(cotangent),
    }
    roof = measure_roofs(device, args.quick, args.repeat)
    # Useful FLOPs count endpoint displacement/norm, scalar adjoint and both
    # endpoint accumulations. Atomic serialization is exposed by throughput.
    flops_per_edge = 6.0 * args.dimensions + 6.0
    flops = flops_per_edge * edges
    ideal_bytes = (
        edges * (
            2 * col_idx.element_size()
            + (2 * args.dimensions + 3) * positions.element_size())
        + args.nodes * (args.dimensions + 1) * positions.element_size()
    )
    intensity = flops / ideal_bytes
    optimistic = min(roof.fp32_gflops, roof.l2_bandwidth_gbs * intensity)
    results = []
    for name, provider in providers.items():
        raw = samples_ms(provider, device, args.repeat, None)
        median = statistics.median(raw)
        results.append(Result(
            topology="radius-fixed-snapshot",
            locality="cell-list",
            cache="hot",
            provider=name,
            nodes=args.nodes,
            edges=edges,
            features=args.dimensions + 1,
            index_dtype="i64",
            milliseconds=median,
            gedges_per_second=edges / median / 1e6,
            achieved_gflops=flops / median / 1e6,
            memory_roof="L2",
            optimistic_roof_gflops=optimistic,
            percent_of_optimistic_roof=(
                100.0 * (flops / median / 1e6) / optimistic),
            arithmetic_intensity_flop_per_byte=intensity,
            samples_ms=raw,
            compile_ms=(compile_ms if name == "graphforge.generated_vjp" else None),
            materialize_ms=(
                materialize_ms if name == "graphforge.generated_vjp" else None),
            lowering=(
                "gf_tensor.csr_euclidean_distance_sum_vjp-to-ttir"
                if name == "graphforge.generated_vjp" else None),
            physical_ideal_bytes=ideal_bytes,
        ))
    gates = evaluate_sota_gates(
        results,
        ["graphforge.generated_vjp"],
        baselines={"torch.autograd", "handwritten.triton"},
        threshold=1.0,
    )
    gate = gates[0]
    for item in results:
        print(f"{item.provider:28s} {item.milliseconds:8.4f} ms  "
              f"{item.gedges_per_second:7.2f} Gedge/s")
    print(f"SOTA {'PASS' if gate.passed else 'FAIL'}: "
          f"{gate.speedup_vs_sota:.3f}x vs {gate.baseline}; "
          f"95% CI=[{gate.speedup_ci_low:.3f}, {gate.speedup_ci_high:.3f}]")

    if args.json is None:
        case = (f"radius_fixed_snapshot_cuda_n{args.nodes}_d{args.dimensions}_"
                f"degree{args.degree}")
        args.json = artifact_path("message_passing_backward", case)
    payload = {
        "operation": "message_passing_backward",
        "workload": "dynamic radius fixed-snapshot position/source backward",
        "case": args.json.parent.name,
        "roof": asdict(roof),
        "results": [asdict(item) for item in results],
        "sota_gates": [item.to_dict() for item in gates],
        "backward_contract": {
            "forward": "y[dst] = sum(||p[src]-p[dst]|| * x[src])",
            "membership": "discrete fixed forward snapshot; not differentiable",
            "gradients": "packed dpositions[N,D] and dx[N]",
            "timing": "warm backward only; topology materialization/JIT excluded",
            "topology_materialize_ms": materialize_ms,
            "cold_backward_ms": cold_backward_ms,
            "provider_compile_ms": compile_ms,
            "roofline_x_axis": "identical useful FLOPs and ideal bytes for all peers",
        },
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2) + "\n")
    plot_roofline(payload, args.json.parent)
    plot_latency(payload, args.json.parent)
    write_report(payload, None, None, None, args.json.parent)
    if args.fail_on_gate and not gate.passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
