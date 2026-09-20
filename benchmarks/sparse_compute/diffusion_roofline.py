"""Measured roofline/provider comparison for sparse graph diffusion."""

from __future__ import annotations

import argparse
import importlib.util
import json
import warnings
from dataclasses import asdict
from pathlib import Path

import tiga as tg
import torch

from benchmarks.common.hardware_roofline import (
    measure_roofs,
    samples_ms,
    synchronize,
)
from benchmarks.common.output_layout import artifact_path
from benchmarks.common.perf_protocol import evaluate_sota_gates
from benchmarks.common.plotting import plot_latency, plot_roofline, write_report
from benchmarks.sparse_compute.cases import TOPOLOGIES, degree_statistics, make_graph

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def csr_diffusion_kernel(row_ptr, col_idx, weight, x, out, n_rows):
        row = tl.program_id(0)
        start = tl.load(row_ptr + row)
        end = tl.load(row_ptr + row + 1)
        center = tl.load(x + row)
        accumulator = 0.0
        for edge in range(start, end):
            src = tl.load(col_idx + edge)
            accumulator += tl.load(weight + edge) * (tl.load(x + src) - center)
        tl.store(out + row, accumulator)


class Diffusion(tg.MessagePassing):
    reducer = tg.sum()

    def edge(self, src, dst, edge):
        return edge.weight * (src.u - dst.u)


def optional_pyg(src, dst, weight, x, nodes):
    if importlib.util.find_spec("torch_geometric") is None:
        return None, "torch_geometric is not installed"
    from torch_geometric.nn import MessagePassing

    class PyGDiffusion(MessagePassing):
        def __init__(self):
            super().__init__(aggr="add")

        def forward(self, features, edge_index, edge_weight):
            return self.propagate(
                edge_index, x=features, edge_weight=edge_weight,
                size=(nodes, nodes))

        def message(self, x_j, x_i, edge_weight):
            return edge_weight[:, None] * (x_j - x_i)

    module = PyGDiffusion().to(x.device)
    edge_index = torch.stack((src, dst))
    return lambda: module(x[:, None], edge_index, weight)[:, 0], None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--nodes", type=int, default=1 << 17)
    parser.add_argument("--degree", type=int, default=16)
    parser.add_argument("--topology", choices=TOPOLOGIES,
                        default="regular")
    parser.add_argument("--locality", choices=("local", "random"), default="local")
    parser.add_argument("--cache", choices=("hot", "cold", "both"), default="both")
    parser.add_argument("--index-dtype", choices=("i32", "i64"), default="i64")
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.quick:
        args.nodes, args.repeat = 1 << 13, 5
    if args.json is None:
        case = (
            f"{args.topology}_{args.locality}_{args.device}_{args.index_dtype}_"
            f"n{args.nodes}_degree{args.degree}")
        args.json = artifact_path("diffusion", case)
    device = torch.device(args.device)
    roof = measure_roofs(device, args.quick, args.repeat)
    row_ptr, col_idx, dst = make_graph(
        args.nodes, args.degree, args.topology, args.locality, device,
        torch.int32 if args.index_dtype == "i32" else torch.int64)
    degree_stats = degree_statistics(row_ptr)
    edges = col_idx.numel()
    weight = torch.rand(edges, device=device)
    x = torch.rand(args.nodes, device=device)
    graph = tg.Graph.from_csr(row_ptr, col_idx, num_src=args.nodes)
    gf_kernel = Diffusion()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Sparse invariant checks.*")
        warnings.filterwarnings("ignore", message="Sparse CSR tensor support.*")
        csr = torch.sparse_csr_tensor(
            row_ptr, col_idx, weight, size=(args.nodes, args.nodes),
            check_invariants=False)
    row_weight = torch.zeros_like(x)
    row_weight.index_add_(0, dst, weight)

    providers = {
        "torch.sparse_laplacian": lambda: (
            torch.sparse.mm(csr, x[:, None])[:, 0] - row_weight * x),
    }

    def torch_scatter():
        output = torch.zeros_like(x)
        output.index_add_(0, dst, weight * (x[col_idx] - x[dst]))
        return output

    providers["torch.index_add"] = torch_scatter
    providers["tiga.auto"] = lambda: gf_kernel(
        graph=graph,
        src={"u": x},
        dst={"u": x},
        edge={"weight": weight},
    )
    providers["tiga.reference"] = lambda: gf_kernel.reference(
        graph=graph,
        src={"u": x},
        dst={"u": x},
        edge={"weight": weight},
    )
    if device.type == "cuda" and triton is not None:
        def triton_csr():
            output = torch.empty_like(x)
            csr_diffusion_kernel[(args.nodes,)](
                row_ptr, col_idx, weight, x, output, args.nodes)
            return output
        providers["triton.csr"] = triton_csr

    skipped = []
    try:
        pyg, reason = optional_pyg(col_idx, dst, weight, x, args.nodes)
    except Exception as error:  # noqa: BLE001 - optional framework/ABI probe
        pyg, reason = None, f"initialization failed: {error}"
    if pyg is None:
        skipped.append(("pyg.message_passing", reason))
    else:
        providers["pyg.message_passing"] = pyg
    skipped.append((
        "dgl.gspmm",
        "the exact message reads both src and dst; one binary gspmm op does not",
    ))

    expected = providers["torch.index_add"]()
    for name, provider in providers.items():
        torch.testing.assert_close(
            provider(), expected, rtol=3e-4, atol=3e-4,
            msg=lambda message, name=name: f"{name}: {message}")
    synchronize(device)

    flops = 3.0 * edges
    edge_bytes = edges * (col_idx.element_size() + weight.element_size())
    index_bytes = row_ptr.numel() * row_ptr.element_size()
    no_reuse_bytes = index_bytes + edge_bytes + 2 * edges * x.element_size() \
        + args.nodes * x.element_size()
    ideal_bytes = index_bytes + edge_bytes + 2 * args.nodes * x.element_size()
    print(f"device={device} topology={args.topology} locality={args.locality}")
    print(f"nodes={args.nodes:,} edges={edges:,}")
    print(f"measured roofs: DRAM={roof.dram_bandwidth_gbs:.1f} GB/s, "
          f"L2={roof.l2_bandwidth_gbs:.1f} GB/s, "
          f"FP32={roof.fp32_gflops:.1f} GFLOP/s")
    print("provider                  cache      ms   Gedge/s  GFLOP/s  alg.GB/s  roof%")
    all_results = []
    caches = ("hot", "cold") if args.cache == "both" else (args.cache,)
    for cache in caches:
        flush = None
        if cache == "cold":
            flush_mib = 32 if args.quick else 256
            flush = torch.zeros(flush_mib * 1024 * 1024 // 4, device=device)
        memory_bandwidth = (
            roof.l2_bandwidth_gbs if cache == "hot"
            else roof.dram_bandwidth_gbs)
        memory_roof = "L2" if cache == "hot" else "DRAM"
        intensity = flops / ideal_bytes
        optimistic_roof = min(
            roof.fp32_gflops, memory_bandwidth * intensity)
        for name, provider in providers.items():
            raw = samples_ms(provider, device, args.repeat, flush)
            elapsed = sorted(raw)[len(raw) // 2]
            achieved = flops / elapsed / 1e6
            item = {
                "topology": args.topology,
                "locality": args.locality,
                "cache": cache,
                "provider": name,
                "nodes": args.nodes,
                "edges": edges,
                "features": 1,
                "index_dtype": args.index_dtype,
                "milliseconds": elapsed,
                "gedges_per_second": edges / elapsed / 1e6,
                "achieved_gflops": achieved,
                "algorithmic_gbs_no_reuse": no_reuse_bytes / elapsed / 1e6,
                "memory_roof": memory_roof,
                "optimistic_roof_gflops": optimistic_roof,
                "percent_of_optimistic_roof": 100 * achieved / optimistic_roof,
                "ideal_cache_bytes": ideal_bytes,
                "arithmetic_intensity_flop_per_byte": intensity,
                "samples_ms": raw,
                "degree_min": int(degree_stats["minimum"]),
                "degree_mean": float(degree_stats["mean"]),
                "degree_p50": float(degree_stats["p50"]),
                "degree_p95": float(degree_stats["p95"]),
                "degree_p99": float(degree_stats["p99"]),
                "degree_max": int(degree_stats["maximum"]),
                "degree_zero_fraction": float(degree_stats["zero_fraction"]),
                "degree_coefficient_of_variation": float(
                    degree_stats["coefficient_of_variation"]),
            }
            all_results.append(item)
            print(
                f"{name:26} {cache:>5} {elapsed:8.3f} "
                f"{item['gedges_per_second']:9.2f} {achieved:8.1f} "
                f"{item['algorithmic_gbs_no_reuse']:9.1f} "
                f"{item['percent_of_optimistic_roof']:6.1f}")
    gates = evaluate_sota_gates(all_results, ["tiga.auto"])
    for gate in gates:
        print(
            f"SOTA {'PASS' if gate.passed else 'FAIL'} tiga.auto "
            f"cache={gate.cache}: {gate.speedup_vs_sota:.3f}x vs "
            f"{gate.baseline}, 95%CI=[{gate.speedup_ci_low:.3f},"
            f"{gate.speedup_ci_high:.3f}]")
    for name, reason in skipped:
        print(f"SKIP {name}: {reason}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "operation": "diffusion",
            "case": args.json.parent.name,
            "workload": "diffusion",
            "roof": asdict(roof),
            "results": all_results,
            "sota_gates": [gate.to_dict() for gate in gates],
            "skipped": [
                {"provider": name, "reason": reason} for name, reason in skipped],
        }
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
        plot_roofline(payload, args.json.parent)
        plot_latency(payload, args.json.parent)
        write_report(payload, None, None, None, args.json.parent)
    if args.fail_on_gate and any(not gate.passed for gate in gates):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
