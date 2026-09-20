"""Static CSR MessagePassing VJP roofline: generated vs handwritten kernels.

The forward UDF is ``y[dst] = sum_e weight[e] * x[src]``. Tiga derives
both source and per-edge-weight gradients from this UDF. The dweight case
reports the explicit saved-primal policy needed for an honest backward-only
comparison; benchmark-only handwritten Triton remains outside Tiga.
"""

from __future__ import annotations

import argparse
import json
import statistics
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import tiga as tg
import torch
from tiga.interop.torch.relation_vjp import compile_source_vjp

from benchmarks.common.hardware_roofline import measure_roofs, samples_ms
from benchmarks.common.output_layout import artifact_path
from benchmarks.common.perf_protocol import evaluate_sota_gates
from benchmarks.common.plotting import plot_latency, plot_roofline, write_report
from benchmarks.kernels.sparse_triton_oracles import (
    prepare_edge_weight_vjp,
    prepare_ragged_weighted_sum,
    prepare_saved_edge_weight_vjp,
    prepare_vector_edge_weight_vjp,
)
from benchmarks.sparse_compute.cases import TOPOLOGIES, make_graph


class WeightedAggregation(tg.MessagePassing):
    reducer = tg.sum()

    def edge(self, src, dst, edge):
        del dst
        return edge.weight * src.x


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
    checkpoint_policy: str | None = None
    saved_bytes: int | None = None
    physical_ideal_bytes: int | None = None
    saved_compile_ms: float | None = None
    checkpoint_planning_ms: float | None = None
    checkpoint_native_load_ms: float | None = None
    cold_compile_ms: float | None = None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=131072)
    parser.add_argument("--degree", type=int, default=16)
    parser.add_argument(
        "--topology", choices=TOPOLOGIES,
        default="regular")
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--index-dtype", choices=("i32", "i64"), default="i64")
    parser.add_argument("--locality", choices=("local", "random"), default="random")
    parser.add_argument("--gradient", choices=("dx", "dweight"), default="dx")
    parser.add_argument("--features", type=int, default=1)
    parser.add_argument(
        "--checkpoint", choices=("auto", "save", "recompute"), default="auto")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.nodes, args.repeat = 8192, 20
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.features <= 0:
        raise ValueError("features must be positive")
    if args.gradient == "dx" and args.features != 1:
        raise ValueError("the current dx gate requires --features 1")
    device = torch.device("cuda")
    index_dtype = torch.int32 if args.index_dtype == "i32" else torch.int64
    row_ptr, col_idx, dst = make_graph(
        args.nodes, args.degree, args.topology, args.locality, device, index_dtype)
    edges = col_idx.numel()
    generator = torch.Generator(device=device).manual_seed(20260813)
    weight_shape = (edges,) if args.features == 1 else (edges, 1)
    field_shape = (args.nodes,) if args.features == 1 else (
        args.nodes, args.features)
    weight = torch.randn(weight_shape, device=device, generator=generator)
    x = torch.randn(field_shape, device=device, generator=generator,
                    requires_grad=True)
    weight.requires_grad_(True)
    cotangent = torch.randn(field_shape, device=device, generator=generator)
    graph = tg.Graph.from_csr(row_ptr, col_idx, num_src=args.nodes)
    kernel = WeightedAggregation()

    if args.gradient == "dx":
        generated = compile_source_vjp(
            kernel, graph=graph, src={"x": x}, edge={"weight": weight})
    else:
        native_graph = tg.Graph.from_csr(
            tg.from_torch(row_ptr), tg.from_torch(col_idx), num_src=args.nodes)
        native_x = tg.from_torch(x, requires_grad=True)
        native_weight = tg.from_torch(weight, requires_grad=True)
        native_dy = tg.from_torch(cotangent)
        native_output = kernel(
            graph=native_graph, src={"x": native_x}, dst={},
            edge={"weight": native_weight})
        native_gradient = tg.autograd.grad(
            native_output, native_weight, grad_output=native_dy,
            checkpoint=args.checkpoint)
        from tiga.compiler.gpu_tensor import compile_tensor
        generated_executable = compile_tensor(native_gradient)
        native_gradient.to_torch()
        generated_launch = generated_executable.prepare(native_gradient)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Sparse invariant checks.*")
        warnings.filterwarnings("ignore", message="Sparse CSR tensor support.*")
        if args.gradient == "dx":
            transpose_row_ptr, transpose_col_idx = generated.graph.resolve_csr()
            transpose_csr = torch.sparse_csr_tensor(
                transpose_row_ptr, transpose_col_idx, generated.jacobian,
                size=(args.nodes, args.nodes), check_invariants=False)
        if args.features == 1:
            forward_csr = torch.sparse_csr_tensor(
                row_ptr, col_idx, weight, size=(args.nodes, args.nodes),
                check_invariants=False)
    if args.features == 1:
        forward = torch.sparse.mm(forward_csr, x[:, None])[:, 0]
    if args.gradient == "dweight":
        # Sparse-library value gradients may coalesce duplicate (dst, src)
        # entries and therefore do not preserve the per-edge field ABI. Build
        # the exact MessagePassing expression for Torch autograd instead.
        saved_source = x[col_idx]
        forward = torch.zeros_like(x).index_add(
            0, dst, weight * saved_source)

    def torch_autograd():
        differentiated = x if args.gradient == "dx" else weight
        return torch.autograd.grad(
            forward, differentiated, grad_outputs=cotangent, retain_graph=True)[0]

    if args.gradient == "dx":
        def torch_explicit():
            return torch.sparse.mm(transpose_csr, cotangent[:, None])[:, 0]

        max_degree = int((transpose_row_ptr[1:] - transpose_row_ptr[:-1]).max())
        oracle_plan = prepare_ragged_weighted_sum(
            transpose_row_ptr, transpose_col_idx, generated.jacobian, cotangent,
            num_rows=args.nodes, max_degree=max_degree)
        if oracle_plan is None:
            raise RuntimeError("benchmark-only handwritten Triton oracle is unavailable")
        providers = {
            "tiga.generated_vjp": lambda: generated(cotangent),
            "torch.autograd": torch_autograd,
            "torch.sparse.transpose": torch_explicit,
            "handwritten.triton": lambda: oracle_plan.run(
                cotangent, generated.jacobian),
        }
    else:
        def torch_explicit():
            product = cotangent[dst] * x[col_idx]
            return product if args.features == 1 else product.sum(
                dim=1, keepdim=True)

        def graphforge_generated():
            return generated_launch()

        if args.features == 1:
            oracle_plan = prepare_edge_weight_vjp(dst, col_idx, x)
            saved_oracle_plan = prepare_saved_edge_weight_vjp(dst, saved_source)
        else:
            oracle_plan = prepare_vector_edge_weight_vjp(
                dst, col_idx, x, args.features)
            saved_oracle_plan = prepare_vector_edge_weight_vjp(
                dst, None, saved_source, args.features, saved=True)
        if oracle_plan is None or saved_oracle_plan is None:
            raise RuntimeError("matched handwritten Triton oracle is unavailable")
        providers = {
            "tiga.generated_vjp": graphforge_generated,
            "torch.autograd": torch_autograd,
            "torch.explicit_gather": torch_explicit,
            "handwritten.triton.recompute": (
                (lambda: oracle_plan.run(cotangent, x))
                if args.features == 1 else
                (lambda: oracle_plan.run(cotangent))),
            "handwritten.triton.saved": lambda: saved_oracle_plan.run(cotangent),
        }
    expected = torch_explicit()
    for name, provider in providers.items():
        actual = provider()
        torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-4,
                                   msg=lambda error, name=name: f"{name}: {error}")
    del actual, expected
    torch.cuda.synchronize()

    roof = measure_roofs(device, args.quick, args.repeat)
    flops_per_edge = (
        2.0 if args.gradient == "dx" else 2.0 * args.features - 1.0)
    flops = flops_per_edge * edges
    semantic_ideal_bytes = (
        row_ptr.numel() * row_ptr.element_size()
        + edges * (col_idx.element_size() + weight.element_size())
        + 2 * args.nodes * cotangent.element_size()
        if args.gradient == "dx" else
        edges * (2 * col_idx.element_size()
                 + 2 * args.features * x.element_size()
                 + weight.element_size())
    )
    intensity = flops / semantic_ideal_bytes
    optimistic = min(roof.fp32_gflops, roof.l2_bandwidth_gbs * intensity)
    results = []
    for name, provider in providers.items():
        raw = samples_ms(provider, device, args.repeat, None)
        median = statistics.median(raw)
        if args.gradient == "dweight":
            saved_policy = (
                name in {"torch.autograd", "handwritten.triton.saved"}
                or (name == "tiga.generated_vjp"
                    and args.checkpoint != "recompute")
            )
            physical_bytes = edges * (
                dst.element_size()
                + 2 * args.features * x.element_size()
                + weight.element_size()
                if saved_policy else
                dst.element_size() + col_idx.element_size()
                + 2 * args.features * x.element_size()
                + weight.element_size())
        else:
            saved_policy = False
            physical_bytes = semantic_ideal_bytes
        results.append(Result(
            topology=args.topology, locality=args.locality, cache="hot",
            provider=name, nodes=args.nodes, edges=edges, features=args.features,
            index_dtype=args.index_dtype, milliseconds=median,
            gedges_per_second=edges / median / 1e6,
            achieved_gflops=flops / median / 1e6, memory_roof="L2",
            optimistic_roof_gflops=optimistic,
            percent_of_optimistic_roof=100 * (flops / median / 1e6) / optimistic,
            arithmetic_intensity_flop_per_byte=intensity, samples_ms=raw,
            compile_ms=((generated.compile_ms if args.gradient == "dx" else
                         generated_executable.compile_ms)
                        if name == "tiga.generated_vjp" else None),
            materialize_ms=((generated.materialize_ms if args.gradient == "dx"
                             else generated_executable.materialize_ms)
                            if name == "tiga.generated_vjp" else None),
            lowering=((generated.transform if args.gradient == "dx" else
                       "gf-tensor-vjp-to-pointwise-ttir")
                      if name == "tiga.generated_vjp" else None),
            checkpoint_policy=(args.checkpoint if args.gradient == "dweight"
                               and name == "tiga.generated_vjp" else None),
            saved_bytes=((generated_executable.saved_bytes
                          if args.gradient == "dweight" else 0)
                         if name == "tiga.generated_vjp" else
                         (saved_source.nbytes if saved_policy else 0)
                         if args.gradient == "dweight" else None),
            physical_ideal_bytes=physical_bytes,
            saved_compile_ms=((generated_executable.saved_compile_ms
                              if args.gradient == "dweight" else 0.0)
                              if name == "tiga.generated_vjp" else None),
            checkpoint_planning_ms=(
                float(generated_executable.checkpoint_plan["planning_ms"])
                if args.gradient == "dweight"
                and name == "tiga.generated_vjp"
                and generated_executable.checkpoint_plan is not None
                else None
            ),
            checkpoint_native_load_ms=(
                float(generated_executable.checkpoint_plan["native_load_ms"])
                if args.gradient == "dweight"
                and name == "tiga.generated_vjp"
                and generated_executable.checkpoint_plan is not None
                else None
            ),
            cold_compile_ms=(
                generated_executable.compile_ms
                + generated_executable.saved_compile_ms
                + float(generated_executable.checkpoint_plan["planning_ms"])
                + float(generated_executable.checkpoint_plan["native_load_ms"])
                if args.gradient == "dweight"
                and name == "tiga.generated_vjp"
                and generated_executable.checkpoint_plan is not None
                else generated.compile_ms
                if args.gradient == "dx"
                and name == "tiga.generated_vjp"
                else None
            ),
        ))
    gates = evaluate_sota_gates(
        results, ["tiga.generated_vjp"],
        baselines={"torch.autograd", "torch.sparse.transpose",
                   "torch.explicit_gather", "handwritten.triton",
                   "handwritten.triton.recompute",
                   "handwritten.triton.saved"}, threshold=1.0)
    for item in results:
        print(f"{item.provider:28s} {item.milliseconds:8.4f} ms  "
              f"{item.gedges_per_second:7.2f} Gedge/s")
    gate = gates[0]
    print(f"SOTA {'PASS' if gate.passed else 'FAIL'}: "
          f"{gate.speedup_vs_sota:.3f}x vs {gate.baseline}; "
          f"95% CI=[{gate.speedup_ci_low:.3f}, {gate.speedup_ci_high:.3f}]")
    compile_ms = generated.compile_ms if args.gradient == "dx" else generated_executable.compile_ms
    materialize_ms = generated.materialize_ms if args.gradient == "dx" else generated_executable.materialize_ms
    print(f"generated compile={compile_ms:.2f} ms, "
          f"native load="
          f"{(float(generated_executable.checkpoint_plan['native_load_ms']) if args.gradient == 'dweight' and generated_executable.checkpoint_plan is not None else 0.0):.2f} ms, "
          f"checkpoint plan="
          f"{(float(generated_executable.checkpoint_plan['planning_ms']) if args.gradient == 'dweight' and generated_executable.checkpoint_plan is not None else 0.0):.2f} ms, "
          f"checkpoint compile="
          f"{(generated_executable.saved_compile_ms if args.gradient == 'dweight' else 0.0):.2f} ms, "
          f"relation/checkpoint materialize={materialize_ms:.2f} ms")

    if args.json is None:
        feature_suffix = "" if args.features == 1 else f"_f{args.features}"
        case = (f"{args.topology}_{args.locality}_cuda_{args.index_dtype}_"
                f"n{args.nodes}_degree{args.degree}_{args.gradient}"
                f"{feature_suffix}")
        args.json = artifact_path("message_passing_backward", case)
    payload = {
        "operation": "message_passing_backward",
        "workload": f"message_passing {args.gradient} backward",
        "case": args.json.parent.name,
        "roof": asdict(roof),
        "results": [asdict(item) for item in results],
        "sota_gates": [item.to_dict() for item in gates],
        "backward_contract": {
            "forward": "y[dst,f] = sum(weight[e] * x[src,f])",
            "cotangent": "arbitrary dense dy",
            "gradient": (
                "dx = transpose(relation, weight) @ dy" if args.gradient == "dx"
                else ("dweight[e] = dy[dst(e)] * x[src(e)]"
                      if args.features == 1 else
                      "dweight[e] = sum_f dy[dst(e),f] * x[src(e),f]")),
            "features": args.features,
            "timing": "backward-only; compile and checkpoint materialization excluded",
            "checkpoint_policy": args.checkpoint if args.gradient == "dweight" else None,
            "saved_bytes": (generated_executable.saved_bytes
                            if args.gradient == "dweight" else 0),
            "saved_compile_ms": (generated_executable.saved_compile_ms
                                 if args.gradient == "dweight" else 0.0),
            "checkpoint_plan": (generated_executable.checkpoint_plan
                                if args.gradient == "dweight" else None),
            "roofline_x_axis": (
                "one semantic intensity per operation; provider-specific "
                "physical traffic is reported separately"),
            "roof_ceiling_scope": (
                "copy-calibrated diagnostic proxy; indexed load/store traffic "
                "requires profiler counters for a physical efficiency claim"),
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
