"""Matched horizontal-fusion benchmark for two CSR reductions.

Tiga's measured candidate is produced by the complete compiler pipeline:
two typed ``gf.apply`` operations, horizontal-fusion, Domain→Iter→Kernel,
schedule selection, Kernel→TTIR, and the active vendor provider.  The Triton
kernel below is deliberately only a handwritten SOTA/oracle comparator.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

import tiga as tg
from benchmarks.common.hardware_roofline import measure_roofs, samples_ms
from benchmarks.common.output_layout import artifact_path
from benchmarks.common.perf_protocol import evaluate_sota_gates
from benchmarks.common.plotting import plot_latency, plot_roofline, write_report
from tiga.codegen import prepare_ttir_csr_product
from tiga.compiler.toolchain import find_gf_opt, find_gf_translate
from tiga.interop.torch.compiler_bridge import (
    lower_kernel_to_ttir,
    parse_kernel_ttir_plan,
)

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def fused_oracle(x, col, w0, w1, out0, out1, degree: tl.constexpr,
                     block_m: tl.constexpr, block_d: tl.constexpr,
                     rows: tl.constexpr):
        row = tl.program_id(0) * block_m + tl.arange(0, block_m)
        neighbor = tl.arange(0, block_d)
        edge = row[:, None] * degree + neighbor[None, :]
        mask = (row[:, None] < rows) & (neighbor[None, :] < degree)
        src = tl.load(col + edge, mask=mask, other=0)
        value = tl.load(x + src, mask=mask, other=0.0)
        a = tl.load(w0 + edge, mask=mask, other=0.0)
        b = tl.load(w1 + edge, mask=mask, other=0.0)
        tl.store(out0 + row, tl.sum(value * a, axis=1), mask=row < rows)
        tl.store(out1 + row, tl.sum(value * b, axis=1), mask=row < rows)


class WeightedAggregation(tg.MessagePassing):
    reducer = tg.sum()

    def edge(self, src, dst, edge):
        return src.x * edge.weight


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
    achieved_gflops: float
    arithmetic_intensity_flop_per_byte: float
    ideal_cache_bytes: int
    memory_roof: str
    optimistic_roof_gflops: float
    percent_of_optimistic_roof: float
    samples_ms: list[float]
    compile_ms: float | None = None
    lowering: str | None = None


def domain_program(nodes: int, degree: int, index: str) -> str:
    """A typed semantic compiler input, never provider code."""
    return f'''"gf.reducer"() <{{sym_name = "sum", kind = "algebraic",
  message_types = [f32], state_types = [f32], result_types = [f32],
  associative = true, commutative = true}}> ({{
  %zero = arith.constant 0.0 : f32
  "gf.reducer_yield"(%zero) : (f32) -> ()
}}, {{
^bb0(%message: f32):
  "gf.reducer_yield"(%message) : (f32) -> ()
}}, {{
^bb0(%left: f32, %right: f32):
  %sum = arith.addf %left, %right : f32
  "gf.reducer_yield"(%sum) : (f32) -> ()
}}, {{
^bb0(%state: f32):
  "gf.reducer_yield"(%state) : (f32) -> ()
}}) : () -> ()
func.func @horizontal(%row: tensor<?x{index}>, %col: tensor<?x{index}>,
                      %x: tensor<?xf32>, %w0: tensor<?xf32>,
                      %w1: tensor<?xf32>)
    -> (tensor<?xf32>, tensor<?xf32>) {{
  %r = "gf.relation"(%row, %col) {{origin = "external", lifecycle = "frozen",
    realization = "materialized", relation_id = "fusion", version = 0 : i64,
    num_src = {nodes} : i64, num_dst = {nodes} : i64,
    degree_min = {degree} : i64, degree_max = {degree} : i64}} :
    (tensor<?x{index}>, tensor<?x{index}>) -> !gf.relation
  %o0 = "gf.apply"(%r, %x, %w0) ({{
  ^bb0(%src: f32, %weight: f32):
    %m = arith.mulf %src, %weight : f32
    "gf.yield"(%m) : (f32) -> ()
  }}) {{reducers = [@sum], region_kinds = array<i64: 0>,
    input_segment_sizes = array<i64: 2>, snapshot_versions = array<i64: 0, 0>,
    input_roles = ["src", "edge"], input_names = ["x", "w0"],
    effects = ["read", "read"], deterministic = false}} :
    (!gf.relation, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %o1 = "gf.apply"(%r, %x, %w1) ({{
  ^bb0(%src: f32, %weight: f32):
    %m = arith.mulf %src, %weight : f32
    "gf.yield"(%m) : (f32) -> ()
  }}) {{reducers = [@sum], region_kinds = array<i64: 0>,
    input_segment_sizes = array<i64: 2>, snapshot_versions = array<i64: 0, 0>,
    input_roles = ["src", "edge"], input_names = ["x", "w1"],
    effects = ["read", "read"], deterministic = false}} :
    (!gf.relation, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %o0, %o1 : tensor<?xf32>, tensor<?xf32>
}}'''


def compile_fused(module: str, row, col, nodes: int):
    gf_opt = find_gf_opt()
    gf_translate = find_gf_translate()
    if gf_opt is None or gf_translate is None:
        raise RuntimeError("built gf-opt and gf-translate are required")
    started = time.perf_counter_ns()
    optimized = subprocess.run(
        [gf_opt, "-gf-fuse-compatible-applies", "-gf-lower-domain-to-iter",
         "-gf-lower-iter-to-kernel", "-gf-select-kernel-schedule"],
        input=module, text=True, capture_output=True, check=True,
    ).stdout
    ttir = lower_kernel_to_ttir(optimized, gf_translate=gf_translate)
    manifest = parse_kernel_ttir_plan(ttir)
    if manifest.entry != "gf_csr_product_additive_tile":
        raise RuntimeError(f"compiler did not form product launch: {manifest}")
    plan = prepare_ttir_csr_product(
        ttir, row_ptr=row, col_idx=col, num_rows=nodes,
        block_rows=manifest.block_rows, num_results=2,
        num_warps=manifest.num_warps,
    )
    return plan, (time.perf_counter_ns() - started) / 1e6, optimized, ttir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=1 << 17)
    parser.add_argument("--degree", type=int, default=16)
    parser.add_argument("--index-dtype", choices=("i32", "i64"), default="i64")
    parser.add_argument("--repeat", type=int, default=40)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.nodes, args.repeat = 1 << 14, 8
    if not torch.cuda.is_available() or triton is None:
        raise SystemExit("CUDA and Triton are required")
    device = torch.device("cuda")
    dtype = torch.int32 if args.index_dtype == "i32" else torch.int64
    nodes, degree = args.nodes, args.degree
    row = torch.arange(nodes + 1, device=device, dtype=dtype) * degree
    col = ((torch.arange(nodes, device=device, dtype=dtype)[:, None] +
            torch.arange(1, degree + 1, device=device, dtype=dtype)) % nodes).flatten()
    edges = col.numel()
    x = torch.randn(nodes, device=device)
    w0 = torch.randn(edges, device=device)
    w1 = torch.randn(edges, device=device)
    graph = tg.Graph.from_csr(row, col, num_src=nodes, validate="basic")
    leaf = WeightedAggregation()
    compiler_plan, compile_ms, kernel_ir, ttir = compile_fused(
        domain_program(nodes, degree, args.index_dtype), row, col, nodes)
    out0 = torch.empty_like(x)
    out1 = torch.empty_like(x)
    block_m = 16
    block_d = triton.next_power_of_2(degree)

    providers = {
        "tiga.compiler_fused": lambda: compiler_plan.run(x, w0, x, w1),
        "tiga.unfused": lambda: (
            leaf(graph=graph, src={"x": x}, dst={}, edge={"weight": w0}),
            leaf(graph=graph, src={"x": x}, dst={}, edge={"weight": w1}),
        ),
        "triton.fused_oracle": lambda: (
            fused_oracle[(triton.cdiv(nodes, block_m),)](
                x, col, w0, w1, out0, out1, degree, block_m, block_d, nodes),
            (out0, out1),
        )[1],
    }
    csr0 = torch.sparse_csr_tensor(
        row, col, w0, (nodes, nodes), check_invariants=False)
    csr1 = torch.sparse_csr_tensor(
        row, col, w1, (nodes, nodes), check_invariants=False)
    providers["torch.sparse.mm_x2"] = lambda: (
        torch.sparse.mm(csr0, x[:, None])[:, 0],
        torch.sparse.mm(csr1, x[:, None])[:, 0],
    )

    expected = providers["torch.sparse.mm_x2"]()
    for name, provider in providers.items():
        actual = provider()
        torch.cuda.synchronize()
        for lhs, rhs in zip(actual, expected):
            torch.testing.assert_close(lhs, rhs, rtol=3e-4, atol=3e-4,
                                       msg=lambda message: f"{name}: {message}")

    roof = measure_roofs(device, args.quick, args.repeat)
    flops = 4.0 * edges
    ideal_bytes = (
        row.numel() * row.element_size() + col.numel() * col.element_size() +
        x.numel() * x.element_size() + w0.numel() * w0.element_size() +
        w1.numel() * w1.element_size() + 2 * x.numel() * x.element_size()
    )
    intensity = flops / ideal_bytes
    optimistic = min(roof.fp32_gflops, roof.l2_bandwidth_gbs * intensity)
    results = []
    for name, provider in providers.items():
        raw = samples_ms(provider, device, args.repeat, None)
        ms = statistics.median(raw)
        achieved = flops / ms / 1e6
        results.append(Result(
            topology="regular", locality="local", cache="hot", provider=name,
            nodes=nodes, edges=edges, features=1, index_dtype=args.index_dtype,
            milliseconds=ms, achieved_gflops=achieved,
            arithmetic_intensity_flop_per_byte=intensity,
            ideal_cache_bytes=ideal_bytes, memory_roof="L2",
            optimistic_roof_gflops=optimistic,
            percent_of_optimistic_roof=100.0 * achieved / optimistic,
            samples_ms=raw,
            compile_ms=compile_ms if name == "tiga.compiler_fused" else None,
            lowering=("gf.apply-product→Kernel→TTIR"
                      if name == "tiga.compiler_fused" else None),
        ))
    gates = evaluate_sota_gates(
        results, {"tiga.compiler_fused"}, threshold=1.0)
    case = f"cuda_n{nodes}_degree{degree}_{args.index_dtype}"
    output = args.json or artifact_path("horizontal_fusion", case)
    payload = {
        "operation": "horizontal_fusion", "workload": "horizontal fusion",
        "case": output.parent.name, "roof": asdict(roof),
        "results": [asdict(item) for item in results],
        "sota_gates": [item.to_dict() for item in gates],
        "compiler_artifacts": {
            "kernel_ir": "kernel.mlir", "provider_ttir": "kernel.ttir"
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    (output.parent / "kernel.mlir").write_text(kernel_ir)
    (output.parent / "kernel.ttir").write_text(ttir)
    plot_roofline(payload, output.parent)
    plot_latency(payload, output.parent)
    write_report(payload, None, None, None, output.parent)
    for item in results:
        print(f"{item.provider:28} {item.milliseconds:.5f} ms")
    for gate in gates:
        print(f"SOTA {'PASS' if gate.passed else 'FAIL'}: "
              f"{gate.speedup_vs_sota:.3f}x vs {gate.baseline}, "
              f"95%CI low={gate.speedup_ci_low:.3f}")
    if args.fail_on_gate and any(not gate.passed for gate in gates):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
