"""Captured product-reducer VJP: compiler TTIR versus handwritten Triton.

The UDF has no custom backward.  Tiga proves the reducer's product
monoid from ``identity/lift/combine/finalize``, emits a zero-safe CSR VJP and
compares the warm backward kernel with a benchmark-only equivalent.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import statistics

import torch

import tiga as gf
from benchmarks.common.hardware_roofline import measure_roofs, samples_ms
from benchmarks.common.output_layout import artifact_path
from benchmarks.common.perf_protocol import evaluate_sota_gates
from benchmarks.common.plotting import plot_latency, plot_roofline, write_report
from benchmarks.kernels.sparse_triton_oracles import prepare_csr_product_vjp


class Product(gf.Reducer):
    associative = True
    commutative = True

    def identity(self):
        return 1.0

    def lift(self, value):
        return value

    def combine(self, left, right):
        return left * right


class EdgeProduct(gf.MessagePassing):
    reducer = Product()

    def edge(self, src, dst, edge):
        del src, dst
        return self.reducer(edge.value)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=131072)
    parser.add_argument("--degree", type=int, default=16)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--index-dtype", choices=("i32", "i64"), default="i64")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.nodes, args.repeat = 8192, 20
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.nodes <= 0 or args.degree <= 0 or args.degree > 64:
        raise ValueError("nodes must be positive and degree must be in [1, 64]")

    device = torch.device("cuda")
    index_dtype = torch.int32 if args.index_dtype == "i32" else torch.int64
    edges = args.nodes * args.degree
    row_ptr = torch.arange(
        0, edges + 1, args.degree, device=device, dtype=index_dtype)
    col_idx = torch.arange(edges, device=device, dtype=index_dtype) % args.nodes
    destination = torch.arange(
        args.nodes, device=device, dtype=index_dtype).repeat_interleave(args.degree)
    generator = torch.Generator(device=device).manual_seed(20260813)
    # Values close to one keep the formal large-degree forward finite while
    # still including exact zeros in correctness probes outside timing.
    message = 0.98 + 0.04 * torch.rand(
        edges, device=device, generator=generator)
    message.requires_grad_(True)
    cotangent = torch.randn(args.nodes, device=device, generator=generator)

    graph = gf.Graph.from_csr(
        gf.from_torch(row_ptr), gf.from_torch(col_idx), num_src=args.nodes)
    native_message = gf.from_torch(message, requires_grad=True)
    native_cotangent = gf.from_torch(cotangent)
    output = EdgeProduct()(
        graph=graph, src={}, dst={}, edge={"value": native_message})
    gradient = gf.autograd.grad(
        output, native_message, grad_output=native_cotangent)
    from tiga.compiler.gpu_tensor import compile_tensor

    executable = compile_tensor(gradient)
    generated_output = gradient.to_torch()
    generated_launch = executable.prepare(gradient)

    def generated():
        generated_launch()
        return generated_output
    oracle = prepare_csr_product_vjp(
        row_ptr, destination, message, max_degree=args.degree)
    if oracle is None:
        raise RuntimeError("benchmark-only product VJP oracle is unavailable")

    torch_forward = message.reshape(args.nodes, args.degree).prod(dim=1)

    def torch_autograd():
        return torch.autograd.grad(
            torch_forward, message, grad_outputs=cotangent,
            retain_graph=True)[0]

    providers = {
        "tiga.generated_vjp": generated,
        "handwritten.triton": lambda: oracle.run(message, cotangent),
        "torch.autograd": torch_autograd,
    }
    expected = torch_autograd()
    for name, provider in providers.items():
        torch.testing.assert_close(
            provider(), expected, rtol=4e-4, atol=4e-4,
            msg=lambda error, name=name: f"{name}: {error}")
    # Explicitly verify the no-division rule where the row product is zero.
    zero_message = torch.tensor(
        [2.0, 0.0, 3.0], device=device, requires_grad=True)
    zero_row = torch.tensor([0, 3], device=device, dtype=index_dtype)
    zero_dst = torch.zeros(3, device=device, dtype=index_dtype)
    zero_oracle = prepare_csr_product_vjp(
        zero_row, zero_dst, zero_message, max_degree=3)
    zero_actual = zero_oracle.run(
        zero_message, torch.tensor([7.0], device=device))
    torch.testing.assert_close(
        zero_actual, torch.tensor([0.0, 42.0, 0.0], device=device))
    torch.cuda.synchronize()

    roof = measure_roofs(device, args.quick, args.repeat)
    flops = float(edges * args.degree)
    semantic_bytes = (
        message.nbytes + row_ptr.nbytes + destination.nbytes
        + cotangent.nbytes + message.nbytes)
    intensity = flops / semantic_bytes
    optimistic = min(roof.fp32_gflops, roof.l2_bandwidth_gbs * intensity)
    results = []
    for name, provider in providers.items():
        raw = samples_ms(provider, device, args.repeat, None)
        median = statistics.median(raw)
        achieved = flops / median / 1e6
        results.append(Result(
            topology="regular", locality="edge-local", cache="hot",
            provider=name, nodes=args.nodes, edges=edges, features=1,
            index_dtype=args.index_dtype, milliseconds=median,
            gedges_per_second=edges / median / 1e6,
            achieved_gflops=achieved, memory_roof="L2",
            optimistic_roof_gflops=optimistic,
            percent_of_optimistic_roof=100 * achieved / optimistic,
            arithmetic_intensity_flop_per_byte=intensity,
            samples_ms=raw,
            compile_ms=(executable.compile_ms
                        if name == "tiga.generated_vjp" else None),
            materialize_ms=(executable.materialize_ms
                            if name == "tiga.generated_vjp" else None),
            lowering=("captured-product→gf_tensor.csr_segment_product_vjp→TTIR"
                      if name == "tiga.generated_vjp" else None),
        ))
    gates = evaluate_sota_gates(
        results, ["tiga.generated_vjp"],
        baselines={"handwritten.triton", "torch.autograd"}, threshold=1.0)
    gate = gates[0]
    for item in results:
        print(f"{item.provider:28s} {item.milliseconds:8.4f} ms  "
              f"{item.gedges_per_second:7.2f} Gedge/s")
    print(f"SOTA {'PASS' if gate.passed else 'FAIL'}: "
          f"{gate.speedup_vs_sota:.3f}x vs {gate.baseline}; "
          f"95% CI=[{gate.speedup_ci_low:.3f}, {gate.speedup_ci_high:.3f}]")

    if args.json is None:
        case = (f"regular_edge_local_cuda_{args.index_dtype}_n{args.nodes}_"
                f"degree{args.degree}_product")
        args.json = artifact_path("message_passing_backward", case)
    payload = {
        "operation": "message_passing_backward",
        "workload": "captured product reducer backward",
        "case": args.json.parent.name,
        "roof": asdict(roof),
        "results": [asdict(item) for item in results],
        "sota_gates": [item.to_dict() for item in gates],
        "backward_contract": {
            "forward": "y[row] = product(message[edges(row)])",
            "gradient": "dmessage[e] = dy[row(e)] * product(other row edges)",
            "zero_semantics": "exclude current edge; never total_product / message[e]",
            "timing": "warm backward-only; compiler and topology derivation excluded",
            "roofline_x_axis": "one semantic intensity shared by every provider",
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
