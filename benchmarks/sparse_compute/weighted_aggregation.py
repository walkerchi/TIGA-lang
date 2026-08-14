"""Measured roofline and provider comparison for weighted CSR aggregation.

The workload is ``out[dst, f] = sum_edge weight[e] * x[src, f]``.  This is a
deliberately common semantic intersection: GraphForge MessagePassing, sparse
matrix multiplication, PyG propagation, and DGL gspmm can all express it
without changing the mathematics.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

import graphforge as gf
from benchmarks.common.hardware_roofline import (
    Roof,
    interleaved_samples_ms,
    measure_roofs,
    samples_ms,
    synchronize,
)
from benchmarks.common.output_layout import artifact_path
from benchmarks.common.perf_protocol import evaluate_sota_gates
from benchmarks.common.plotting import plot_latency, plot_roofline, write_report
from benchmarks.sparse_compute.cases import make_graph

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def csr_spmm_kernel(row_ptr, col_idx, weight, x, out, n_rows,
                        n_features: tl.constexpr, block_f: tl.constexpr):
        row = tl.program_id(0)
        feature = tl.program_id(1) * block_f + tl.arange(0, block_f)
        feature_mask = feature < n_features
        start = tl.load(row_ptr + row)
        end = tl.load(row_ptr + row + 1)
        accumulator = tl.zeros((block_f,), dtype=tl.float32)
        for edge in range(start, end):
            src = tl.load(col_idx + edge)
            value = tl.load(x + src * n_features + feature,
                            mask=feature_mask, other=0.0)
            accumulator += tl.load(weight + edge) * value
        tl.store(out + row * n_features + feature, accumulator,
                 mask=feature_mask)


class WeightedAggregation(gf.MessagePassing):
    reducer = gf.sum()

    def edge(self, src, dst, edge):
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
    algorithmic_gbs_no_reuse: float
    memory_roof: str
    optimistic_roof_gflops: float
    percent_of_optimistic_roof: float
    ideal_cache_bytes: int
    arithmetic_intensity_flop_per_byte: float
    samples_ms: list[float]
    speedup_vs_torch_sparse: float | None = None
    lowering: str | None = None
    compile_ms: float | None = None


def chunked_tail_candidate(kernel, graph, row_ptr, col_idx, x, weight):
    """Build the compiler's experimental compact-tail task bundle.

    Compilation and frozen-relation worklist materialization happen here and
    are reported separately; the returned callable submits only the execute
    phase.  Returning ``None`` means the compiler did not select row splitting.
    """
    from graphforge.codegen import prepare_ttir_task_primitive
    from graphforge.interop.torch.compiler_bridge import (
        lower_mlir_stages,
        message_passing_domain_mlir,
    )
    from graphforge.interop.torch.provider import TorchCudaSubmissionProvider

    started = time.perf_counter_ns()
    module = message_passing_domain_mlir(
        kernel=kernel,
        graph=graph,
        src={"x": x},
        dst={},
        edge={"weight": weight},
        params={},
        kernel_name="WeightedAggregationPowerlawCandidate",
    )
    stages = lower_mlir_stages(module)
    if stages.task is None:
        return None
    plan = gf.compiler.translate_task_bundle(stages.task)
    if not any(
        invocation.metadata.get("row_mapping") == "worklist-chunked"
        for invocation in plan.invocations
    ):
        return None
    compiled = {
        invocation.name: prepare_ttir_task_primitive(invocation)
        for invocation in plan.invocations
        if invocation.task_kind not in {"release", "join"}
    }
    compile_ms = (time.perf_counter_ns() - started) / 1e6
    resolver = lambda invocation: compiled[invocation.name]
    provider = TorchCudaSubmissionProvider()
    output = torch.empty_like(x)
    resources = {
        "row_ptr": row_ptr,
        "col_idx": col_idx,
        "src:x": x,
        "edge:weight": weight,
        "output": output,
    }
    for requirement in plan.resources:
        if requirement.external:
            continue
        if requirement.layout == "degree-row-worklist":
            resources[requirement.name] = torch.empty(
                requirement.capacity_bytes // 8,
                device=x.device,
                dtype=torch.int64,
            )
        elif requirement.layout == "row-partial-f32":
            resources[requirement.name] = torch.empty(
                requirement.capacity_bytes // 4,
                device=x.device,
                dtype=torch.float32,
            )
        else:
            raise ValueError(
                f"unsupported candidate resource layout {requirement.layout!r}"
            )
    # Frozen CSR topology: build the degree worklist once per relation version.
    plan.bind_phase("materialize", resolver).submit(provider, resources).wait()
    execute = plan.bind_phase("execute", resolver).prepare(provider)
    resource_tuple = tuple(
        resources[name] for name in execute.bundle.required_bindings
    )

    def run():
        execute.submit_resources(resource_tuple)
        return output

    return run, compile_ms


def optional_pyg(src, dst, weight, x, nodes):
    if importlib.util.find_spec("torch_geometric") is None:
        return None, "torch_geometric is not installed"
    from torch_geometric.nn import MessagePassing

    class PyGWeighted(MessagePassing):
        def __init__(self):
            super().__init__(aggr="add")

        def forward(self, features, edge_index, edge_weight):
            return self.propagate(
                edge_index, x=features, edge_weight=edge_weight,
                size=(nodes, nodes))

        def message(self, x_j, edge_weight):
            return edge_weight[:, None] * x_j

    module = PyGWeighted().to(x.device)
    edge_index = torch.stack((src, dst))
    return lambda: module(x, edge_index, weight), None


def optional_dgl(src, dst, weight, x, nodes):
    if importlib.util.find_spec("dgl") is None:
        return None, "dgl is not installed"
    import dgl

    graph = dgl.graph((src, dst), num_nodes=nodes, device=x.device)
    return lambda: dgl.ops.gspmm(
        graph, "mul", "sum", x, weight[:, None]), None


def benchmark_case(args, roof: Roof, device: torch.device, features: int,
                   cache: str):
    row_ptr, col_idx, dst = make_graph(
        args.nodes, args.degree, args.topology, args.locality, device,
        torch.int32 if args.index_dtype == "i32" else torch.int64)
    edges = col_idx.numel()
    weight = torch.rand(edges, device=device)
    scalar = features == 1
    edge_weight = weight if scalar else weight[:, None]
    x = torch.rand(
        (args.nodes,) if scalar else (args.nodes, features), device=device)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Sparse invariant checks.*")
        warnings.filterwarnings("ignore", message="Sparse CSR tensor support.*")
        csr = torch.sparse_csr_tensor(
            row_ptr, col_idx, weight, size=(args.nodes, args.nodes),
            check_invariants=False)
    graph = gf.Graph.from_csr(row_ptr, col_idx, num_src=args.nodes)
    gf_kernel = WeightedAggregation()

    providers = {}
    compile_times = {}

    def torch_sparse():
        if scalar:
            return torch.sparse.mm(csr, x[:, None])[:, 0]
        return torch.sparse.mm(csr, x)

    providers["torch.sparse.mm"] = torch_sparse

    def torch_scatter():
        output = torch.zeros_like(x)
        message = weight * x[col_idx] if scalar else weight[:, None] * x[col_idx]
        output.index_add_(0, dst, message)
        return output

    if not args.skip_scatter:
        providers["torch.index_add"] = torch_scatter
    providers["graphforge.auto"] = lambda: gf_kernel(
        graph=graph,
        src={"x": x},
        dst={"x": x},
        edge={"weight": edge_weight},
    )
    if device.type == "cuda":
        # Public hot-loop API: ordinary lazy-JIT validates/compiles once, then
        # prepare freezes the already-proven topology and field ABI. This is
        # the matched executable boundary for scalar and vector fields alike.
        # Report first-call wall time separately from steady-state latency so
        # users can judge the JIT amortization point.
        compile_started = time.perf_counter_ns()
        prepared_auto = gf_kernel.prepare(
            graph=graph, src={"x": x}, dst={"x": x},
            edge={"weight": edge_weight})
        synchronize(device)
        compile_times["graphforge.auto"] = (
            time.perf_counter_ns() - compile_started) / 1e6
        compile_times["graphforge.prepared_auto"] = compile_times[
            "graphforge.auto"
        ]
        providers["graphforge.prepared_auto"] = prepared_auto
    if not args.skip_reference:
        providers["graphforge.reference"] = lambda: gf_kernel.reference(
            graph=graph,
            src={"x": x},
            dst={"x": x},
            edge={"weight": edge_weight},
        )
    if device.type == "cuda" and triton is not None:
        block_f = min(128, triton.next_power_of_2(features))

        def triton_csr():
            output = torch.empty_like(x)
            csr_spmm_kernel[(args.nodes, triton.cdiv(features, block_f))](
                row_ptr, col_idx, weight, x, output, args.nodes,
                features, block_f)
            return output

        providers["triton.csr"] = triton_csr

    skipped = []
    if scalar and device.type == "cuda" and args.topology == "powerlaw":
        try:
            candidate = chunked_tail_candidate(
                gf_kernel, graph, row_ptr, col_idx, x, weight)
        except Exception as error:
            candidate = None
            skipped.append((
                "graphforge.chunked_tail_candidate",
                f"compiler candidate unavailable: {error}",
            ))
        if candidate is None:
            if not any(name == "graphforge.chunked_tail_candidate"
                       for name, _ in skipped):
                skipped.append((
                    "graphforge.chunked_tail_candidate",
                    "compiler did not select compact high-degree splitting",
                ))
        else:
            providers["graphforge.chunked_tail_candidate"], compile_times[
                "graphforge.chunked_tail_candidate"
            ] = candidate
    optional_factories = () if scalar else (
        ("pyg.message_passing", optional_pyg),
        ("dgl.gspmm", optional_dgl),
    )
    for name, factory in optional_factories:
        try:
            provider, reason = factory(col_idx, dst, weight, x, args.nodes)
        except Exception as error:  # Optional frameworks often have ABI extras.
            provider, reason = None, f"initialization failed: {error}"
        if provider is None:
            skipped.append((name, reason))
        else:
            providers[name] = provider

    expected = providers["torch.sparse.mm"]()
    for name, provider in providers.items():
        actual = provider()
        torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-4,
                                   msg=lambda message: f"{name}: {message}")
    synchronize(device)

    flush = None
    if cache == "cold":
        flush_bytes = 32 if args.quick else 256
        flush = torch.zeros(
            flush_bytes * 1024 * 1024 // 4, device=device, dtype=torch.float32)

    flops = 2.0 * edges * features
    index_bytes = row_ptr.numel() * row_ptr.element_size()
    edge_bytes = edges * (col_idx.element_size() + weight.element_size())
    no_reuse_bytes = (
        index_bytes + edge_bytes
        + edges * features * x.element_size()
        + args.nodes * features * x.element_size()
    )
    ideal_cache_bytes = (
        index_bytes + edge_bytes + 2 * args.nodes * features * x.element_size())
    ideal_intensity = flops / ideal_cache_bytes
    memory_bandwidth = (
        roof.l2_bandwidth_gbs if cache == "hot" else roof.dram_bandwidth_gbs)
    memory_roof = "L2" if cache == "hot" else "DRAM"
    optimistic_roof = min(roof.fp32_gflops, memory_bandwidth * ideal_intensity)

    raw_by_provider = interleaved_samples_ms(
        providers, device, args.repeat, flush)
    results = []
    for name, provider in providers.items():
        del provider
        raw_samples = raw_by_provider[name]
        milliseconds = statistics.median(raw_samples)
        achieved = flops / milliseconds / 1e6
        results.append(Result(
            topology=args.topology,
            locality=args.locality,
            cache=cache,
            provider=name,
            nodes=args.nodes,
            edges=edges,
            features=features,
            index_dtype=args.index_dtype,
            milliseconds=milliseconds,
            gedges_per_second=edges / milliseconds / 1e6,
            achieved_gflops=achieved,
            algorithmic_gbs_no_reuse=no_reuse_bytes / milliseconds / 1e6,
            memory_roof=memory_roof,
            optimistic_roof_gflops=optimistic_roof,
            percent_of_optimistic_roof=100.0 * achieved / optimistic_roof,
            ideal_cache_bytes=ideal_cache_bytes,
            arithmetic_intensity_flop_per_byte=ideal_intensity,
            samples_ms=raw_samples,
            lowering=(
                gf_kernel.last_variant.lowering
                if name == "graphforge.auto" else None),
            compile_ms=compile_times.get(name),
        ))
    baseline = next(
        item.milliseconds for item in results
        if item.provider == "torch.sparse.mm")
    for item in results:
        item.speedup_vs_torch_sparse = baseline / item.milliseconds
    return results, skipped


def print_results(results):
    print("provider                 cache roof idx feat      ms   Gedge/s   GFLOP/s  roof%  vs sparse")
    for item in results:
        print(
            f"{item.provider:24} {item.cache:>5} {item.memory_roof:>4} "
            f"{item.index_dtype:>3} "
            f"{item.features:4d} "
            f"{item.milliseconds:8.3f} {item.gedges_per_second:9.2f} "
            f"{item.achieved_gflops:9.1f} "
            f"{item.percent_of_optimistic_roof:6.1f} "
            f"{item.speedup_vs_torch_sparse:9.2f}x"
        )


def print_gates(gates):
    for gate in gates:
        status = "PASS" if gate.passed else "FAIL"
        if gate.baseline is None or gate.speedup_ci_low is None:
            comparison = gate.reason
        else:
            comparison = (
                f"{gate.speedup_vs_sota:.3f}x vs {gate.baseline} "
                f"95%CI=[{gate.speedup_ci_low:.3f},{gate.speedup_ci_high:.3f}] "
                f"({gate.candidate_ms:.4f} vs {gate.baseline_ms:.4f} ms)")
        print(
            f"SOTA {status} {gate.candidate} cache={gate.cache} "
            f"index={gate.index_dtype} features={gate.features}: {comparison}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--nodes", type=int, default=1 << 17)
    parser.add_argument("--degree", type=int, default=16)
    parser.add_argument("--features", default="1,16,64")
    parser.add_argument("--index-dtype", choices=("i32", "i64"), default="i64")
    parser.add_argument("--topology", choices=(
        "regular", "irregular", "skewed", "powerlaw"),
                        default="regular")
    parser.add_argument("--locality", choices=("local", "random"), default="local")
    parser.add_argument("--cache", choices=("hot", "cold", "both"), default="both")
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--skip-reference", action="store_true",
        help="omit the eager semantic reference when edge-message materialization is too large")
    parser.add_argument(
        "--skip-scatter", action="store_true",
        help="omit eager index_add when edge-message materialization is too large")
    parser.add_argument("--json", type=Path)
    parser.add_argument(
        "--gate-provider", action="append", default=[],
        help="provider to compare with the fastest eligible measured baseline")
    parser.add_argument(
        "--gate-baselines",
        help="comma-separated eligible baseline providers; default: all peers")
    parser.add_argument("--gate-threshold", type=float, default=1.0)
    parser.add_argument(
        "--fail-on-gate", action="store_true",
        help="exit with status 2 if any requested SOTA gate fails")
    args = parser.parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.quick:
        args.nodes, args.repeat = 1 << 13, 5
    if args.json is None:
        case = (
            f"{args.topology}_{args.locality}_{args.device}_{args.index_dtype}_"
            f"n{args.nodes}_degree{args.degree}_f{args.features}")
        args.json = artifact_path("weighted_aggregation", case)
    features = [int(value) for value in args.features.split(",")]
    if any(value <= 0 for value in features):
        parser.error("all feature widths must be positive")
    device = torch.device(args.device)
    print(f"device={device} torch={torch.__version__}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")
    roof = measure_roofs(device, args.quick, args.repeat)
    print(
        f"measured roofs: DRAM={roof.dram_bandwidth_gbs:.1f} GB/s, "
        f"L2={roof.l2_bandwidth_gbs:.1f} GB/s, "
        f"FP32={roof.fp32_gflops:.1f} GFLOP/s"
    )

    caches = ("hot", "cold") if args.cache == "both" else (args.cache,)
    all_results = []
    skipped = set()
    for cache in caches:
        for width in features:
            results, case_skipped = benchmark_case(
                args, roof, device, width, cache)
            all_results.extend(results)
            skipped.update(case_skipped)
    print_results(all_results)
    baseline_names = None
    if args.gate_baselines:
        baseline_names = {
            name.strip() for name in args.gate_baselines.split(",")
            if name.strip()
        }
    try:
        gates = evaluate_sota_gates(
            all_results,
            args.gate_provider,
            baselines=baseline_names,
            threshold=args.gate_threshold,
        )
    except ValueError as error:
        parser.error(str(error))
    print_gates(gates)
    for name, reason in sorted(skipped):
        print(f"SKIP {name}: {reason}")
    if args.json:
        payload = {
            "operation": "weighted_aggregation",
            "case": args.json.parent.name,
            "roof": asdict(roof),
            "results": [asdict(item) for item in all_results],
            "sota_gates": [item.to_dict() for item in gates],
            "skipped": [{"provider": name, "reason": reason}
                        for name, reason in sorted(skipped)],
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
        plot_roofline(payload, args.json.parent)
        plot_latency(payload, args.json.parent)
        write_report(payload, None, None, None, args.json.parent)
    if args.fail_on_gate and any(not item.passed for item in gates):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
