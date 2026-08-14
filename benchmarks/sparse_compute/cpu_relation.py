"""CPU LLVM relation lowering against provider-native CSR SpMV peers."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import statistics
import struct
import time
import warnings

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse._sparsetools import csr_matvec
import torch

import graphforge as gf
from graphforge.compiler.cpu_tensor import compile_tensor
from benchmarks.common.output_layout import operation_dir
from benchmarks.common.perf_protocol import evaluate_sota_gates
from benchmarks.common.plotting import plot_latency, plot_roofline
from benchmarks.compiler.cpu_pointwise import measure_roof, samples_ms


class WeightedAggregation(gf.MessagePassing):
    reducer = gf.sum()

    def edge(self, src, dst, edge):
        return edge.weight * src.x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=131_072)
    parser.add_argument("--degree", type=int, default=16)
    parser.add_argument("--threads", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--fail-on-gate", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.quick:
        args.nodes, args.degree, args.repeat = 16_384, 16, 12
    if min(args.nodes, args.degree, args.threads, args.repeat) <= 0:
        parser.error("nodes, degree, threads and repeat must be positive")

    os.environ["GRAPHFORGE_TENSOR_BACKEND"] = "native"
    os.environ["GRAPHFORGE_CPU_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)
    nodes, degree = args.nodes, args.degree
    edges = nodes * degree

    # Deterministic non-contiguous source access. Conversion and CSR creation
    # are setup costs for every provider and stay outside warm measurements.
    row_ptr_np = np.arange(0, edges + 1, degree, dtype=np.int32)
    rows = np.arange(nodes, dtype=np.int64)[:, None]
    offsets = np.arange(degree, dtype=np.int64)[None, :]
    col_idx_np = ((17 * rows + 13 * offsets) % nodes).astype(np.int32).reshape(-1)
    weight_np = np.full(edges, 0.5, dtype=np.float32)
    x_np = np.full(nodes, 1.25, dtype=np.float32)

    row_ptr = gf.tensor(row_ptr_np.tolist(), dtype=gf.int32)
    col_idx = gf.tensor(col_idx_np.tolist(), dtype=gf.int32)
    weight = gf.tensor(weight_np.tolist(), dtype=gf.float32)
    x = gf.tensor(x_np.tolist(), dtype=gf.float32)
    graph = gf.Graph.from_csr(row_ptr, col_idx, num_src=nodes)
    kernel = WeightedAggregation()
    output = kernel(graph=graph, src={"x": x}, dst={}, edge={"weight": weight})
    cold_started = time.perf_counter_ns()
    output.realize()
    cold_ms = (time.perf_counter_ns() - cold_started) / 1e6
    executable = compile_tensor(output)

    scipy_csr = csr_matrix(
        (weight_np, col_idx_np, row_ptr_np), shape=(nodes, nodes), copy=False)
    scipy_output = np.zeros(nodes, dtype=np.float32)

    def scipy_run():
        scipy_output.fill(0.0)
        csr_matvec(
            nodes, nodes, scipy_csr.indptr, scipy_csr.indices,
            scipy_csr.data, x_np, scipy_output,
        )
        return scipy_output

    torch_row_ptr = torch.from_numpy(row_ptr_np)
    torch_col_idx = torch.from_numpy(col_idx_np)
    torch_weight = torch.from_numpy(weight_np)
    torch_x = torch.from_numpy(x_np)[:, None]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Sparse invariant checks.*")
        warnings.filterwarnings("ignore", message="Sparse CSR tensor support.*")
        torch_csr = torch.sparse_csr_tensor(
            torch_row_ptr, torch_col_idx, torch_weight,
            size=(nodes, nodes), check_invariants=False,
        )

    def torch_run():
        return torch.sparse.mm(torch_csr, torch_x)

    expected = degree * 0.5 * 1.25
    for index in (0, nodes // 2, nodes - 1):
        found = struct.unpack(
            "=f", output._buffer.read(offset=index * 4, bytes=4)
        )[0]
        if abs(found - expected) > 1e-5:
            raise RuntimeError(f"GraphForge CSR result mismatch at row {index}")
    if not np.allclose(scipy_run()[[0, nodes // 2, nodes - 1]], expected):
        raise RuntimeError("SciPy CSR result mismatch")
    if not torch.allclose(
        torch_run()[[0, nodes // 2, nodes - 1], 0],
        torch.full((3,), expected), rtol=1e-5, atol=1e-5,
    ):
        raise RuntimeError("Torch CSR result mismatch")
    cpu_loop = output.generated_code("cpu_loop")
    if '"gf_tensor.' in cpu_loop or cpu_loop.count("scf.for") < 2:
        raise RuntimeError("relation DAG was not lowered to fused nested CPU loops")
    if kernel.last_variant.lowering != "gf-tensor-relation-autograd":
        raise RuntimeError("MessagePassing did not use the native compiler path")

    providers = (
        ("graphforge.llvm.parallel_relation", lambda: executable.launch(output)),
        ("scipy.csr_matvec", scipy_run),
        ("torch.sparse.mm", torch_run),
    )
    useful_flops = float(2 * edges)
    common_bytes = float(
        row_ptr_np.nbytes + col_idx_np.nbytes + weight_np.nbytes
        + x_np.nbytes + nodes * 4
    )
    intensity = useful_flops / common_bytes
    results = []
    for provider, function in providers:
        raw = samples_ms(function, args.repeat)
        median = statistics.median(raw)
        results.append({
            "topology": "regular", "locality": "permuted", "cache": "hot",
            "provider": provider, "nodes": nodes, "edges": edges,
            "features": 1, "index_dtype": "i32", "milliseconds": median,
            "gedges_per_second": edges / median / 1e6,
            "achieved_gflops": useful_flops / median / 1e6,
            "arithmetic_intensity_flop_per_byte": intensity,
            "ideal_cache_bytes": common_bytes,
            "semantic_common_bytes": common_bytes,
            "memory_roof": "L2", "samples_ms": raw,
        })
    gate = evaluate_sota_gates(
        results, {"graphforge.llvm.parallel_relation"},
        baselines={"scipy.csr_matvec", "torch.sparse.mm"}, threshold=1.0,
    )[0]
    roof = measure_roof(args.repeat, args.quick)
    case = f"regular_permuted_i32_n{nodes}_degree{degree}_t{args.threads}"
    output_dir = args.output_dir or operation_dir("cpu_relation", case)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "operation": "cpu_relation", "workload": "weighted CSR SpMV",
        "case": case, "title_prefix": "GraphForge CPU relation",
        "roof": asdict(roof), "results": results,
        "sota_gates": [gate.to_dict()],
        "config": {
            **vars(args), "output_dir": str(output_dir),
            "dtype": "float32", "index_dtype": "int32",
            "graphforge_cold_compile_first_launch_ms": cold_ms,
            "graphforge_compile_ms": executable.compile_ms,
            "graphforge_worker_pool_threads": args.threads,
            "semantic_byte_model": "row_ptr + col_idx + weight + x + output",
        },
    }
    (output_dir / "roofline.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n")
    plot_roofline(payload, output_dir)
    plot_latency(payload, output_dir)
    candidate = results[0]
    peer = min(results[1:], key=lambda item: item["milliseconds"])
    (output_dir / "REPORT.md").write_text(
        "# CPU weighted CSR relation\n\n"
        "The native MessagePassing DAG is lowered to fused nested LLVM loops; "
        "the persistent worker pool partitions independent destination ranges. "
        "CSR construction and framework conversion are outside warm timing.\n\n"
        f"GraphForge: {candidate['milliseconds']:.4f} ms; fastest peer "
        f"({peer['provider']}): {peer['milliseconds']:.4f} ms. Gate: "
        f"{'PASS' if gate.passed else 'FAIL'}, speedup "
        f"{gate.speedup_vs_sota:.3f}x, 95% CI "
        f"[{gate.speedup_ci_low:.3f}, {gate.speedup_ci_high:.3f}].\n\n"
        f"Cold compile + first launch: {cold_ms:.2f} ms; threads: "
        f"{args.threads}; semantic x={intensity:.4g} FLOP/byte.\n\n"
        "![Roofline](roofline.png)\n\n![Latency](provider_latency.png)\n"
    )
    print(output_dir)
    if args.fail_on_gate and not gate.passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
