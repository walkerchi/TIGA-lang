"""Matched fixed-iteration PageRank benchmark and roofline artifact.

GraphForge captures the repeated MessagePassing body once and lowers each
iteration to one fused CSR-message/reduction/node-epilogue kernel.  The peer is
Torch's sparse CSR matvec followed by its ordinary damping/base operations;
both execute the same number of iterations from the same initial rank.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import graphforge as gf  # noqa: E402
from benchmarks.common.hardware_roofline import (  # noqa: E402
    interleaved_samples_ms,
    measure_roofs,
)
from benchmarks.common.output_layout import operation_dir  # noqa: E402
from benchmarks.common.perf_protocol import evaluate_sota_gates  # noqa: E402
from benchmarks.common.plotting import provider_color  # noqa: E402


class PageRankStep(gf.MessagePassing):
    reducer = gf.sum()

    def edge(self, src, dst, edge):
        del dst
        return src.rank * edge.inverse_out_degree

    def node(self, dst, incoming, damping, base):
        del dst
        return base + damping * incoming


def _case(nodes: int, degree: int, iterations: int, samples: int, roof):
    row = torch.arange(
        0, nodes * degree + 1, degree, dtype=torch.int64, device="cuda")
    offsets = torch.arange(degree, dtype=torch.int64, device="cuda")
    column = (
        torch.arange(nodes, device="cuda")[:, None] + offsets[None, :]
    ).remainder(nodes).flatten()
    weight = torch.full(
        (nodes * degree,), 1.0 / degree, dtype=torch.float32, device="cuda")
    initial = torch.full(
        (nodes,), 1.0 / nodes, dtype=torch.float32, device="cuda")
    base = 0.15 / nodes
    graph = gf.Graph.from_csr(
        gf.from_torch(row), gf.from_torch(column), num_src=nodes)
    native_weight = gf.from_torch(weight)
    # Do not charge asynchronous graph construction performed before this
    # scope to JIT compilation. The reported wall clock starts from a settled
    # input snapshot and still includes degree analysis, IR construction,
    # vendor compilation, allocation and prepared-launch binding.
    torch.cuda.synchronize()
    started = time.perf_counter_ns()
    output = gf.repeat(
        gf.from_torch(initial),
        lambda current: PageRankStep()(
            graph=graph,
            src={"rank": current},
            dst={},
            edge={"inverse_out_degree": native_weight},
            damping=0.85,
            base=base,
        ),
        iterations=iterations,
    )
    graphforge_submit = output.prepare()
    torch.cuda.synchronize()
    compile_ms = (time.perf_counter_ns() - started) / 1e6
    actual = output.to_torch()
    torch.testing.assert_close(actual, initial, rtol=2e-5, atol=2e-7)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Sparse invariant checks.*")
        warnings.filterwarnings("ignore", message="Sparse CSR tensor support.*")
        matrix = torch.sparse_csr_tensor(
            row, column, weight, size=(nodes, nodes), check_invariants=False)

    def torch_sparse():
        rank = initial
        for _ in range(iterations):
            rank = 0.85 * torch.mv(matrix, rank) + base
        return rank

    timings = interleaved_samples_ms(
        {
            "graphforge.control": graphforge_submit,
            "torch.sparse.mm": torch_sparse,
        },
        torch.device("cuda"),
        samples,
        flush=None,
    )
    edges = nodes * degree
    useful_flops = iterations * (2 * edges + 2 * nodes)
    # Provider-independent semantic bytes: topology/weights, one ideally
    # cached rank read and one rank write per iteration.
    semantic_bytes = iterations * (
        (nodes + 1) * 8 + edges * (8 + 4) + 2 * nodes * 4)
    intensity = useful_flops / semantic_bytes
    results = []
    for provider, raw in timings.items():
        milliseconds = statistics.median(raw)
        achieved = useful_flops / milliseconds / 1e6
        results.append({
            "provider": provider,
            "topology": f"regular-degree-{degree}",
            "locality": "cyclic",
            "nodes": nodes,
            "edges": edges,
            "degree": degree,
            "iterations": iterations,
            "features": 1,
            "cache": "hot",
            "index_dtype": "i64",
            "milliseconds": milliseconds,
            "samples_ms": raw,
            "achieved_gflops": achieved,
            "gedges_per_second": iterations * edges / milliseconds / 1e6,
            "arithmetic_intensity_flop_per_byte": intensity,
            "ideal_cache_bytes": semantic_bytes,
            "optimistic_roof_gflops": min(
                roof.fp32_gflops, roof.l2_bandwidth_gbs * intensity),
        })
    execution = output.execution or {}
    return results, {
        "compile_wall_ms": compile_ms,
        "compiler_reported_ms": execution.get("compile_ms"),
        "backend": execution.get("backend"),
        "semantic_hash": execution.get("semantic_hash"),
        "loop_buffers": 2,
        "kernel_launches_per_iteration": 1,
        "entry": "gf_tensor_csr_sum_epilogue",
        "block_manifest": next(
            (line for line in str(
                output.generated_code("provider_ttir") or "").splitlines()
             if "graphforge.tensor entry=" in line),
            "compiled TTIR artifact carries the launch manifest",
        ),
    }


def _plots(payload: dict, output: Path) -> None:
    results = payload["results"]
    degrees = sorted({item["degree"] for item in results})
    fig, axes = plt.subplots(
        len(degrees), 1, figsize=(8.2, 3.0 * len(degrees)),
        constrained_layout=True, sharex=True)
    axes = np.atleast_1d(axes)
    for axis, degree in zip(axes, degrees):
        chosen = [item for item in results if item["degree"] == degree]
        grouped = {}
        for item in chosen:
            grouped.setdefault(item["provider"], []).append(item)
        nodes = sorted({item["nodes"] for item in chosen})
        x = np.arange(len(nodes))
        width = 0.36
        for ordinal, provider in enumerate(sorted(grouped)):
            by_nodes = {item["nodes"]: item for item in grouped[provider]}
            values = [by_nodes[node]["milliseconds"] for node in nodes]
            offset = (ordinal - (len(grouped) - 1) / 2) * width
            bars = axis.bar(
                x + offset, values, width, label=provider,
                color=provider_color(provider), edgecolor="white")
            for bar, value in zip(bars, values):
                axis.text(
                    bar.get_x() + bar.get_width() / 2, value,
                    f"{value:.3f}", ha="center", va="bottom", fontsize=7)
        axis.set_title(f"degree={degree} · {payload['config']['iterations']} iterations")
        axis.set_ylabel("latency (ms)")
        axis.grid(True, axis="y", alpha=0.35)
        axis.legend(fontsize=8)
        axis.set_xticks(x, [f"N={node:,}" for node in nodes])
    axes[-1].set_xlabel("graph size")
    fig.suptitle("PageRank matched provider latency (lower is better)", fontweight="bold")
    fig.savefig(output / "latency.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _roofline_plot(payload: dict, output: Path) -> None:
    results = payload["results"]
    degrees = sorted({item["degree"] for item in results})
    roof = payload["roof"]
    intensities = [item["arithmetic_intensity_flop_per_byte"] for item in results]
    xmin, xmax = min(intensities) / 1.7, max(intensities) * 1.7
    xs = np.logspace(np.log10(xmin), np.log10(xmax), 300)
    fig, axes = plt.subplots(
        len(degrees), 1, figsize=(9.2, 3.7 * len(degrees)),
        constrained_layout=True, sharex=True)
    axes = np.atleast_1d(axes)
    providers = sorted({item["provider"] for item in results})
    ids = {provider: str(index + 1) for index, provider in enumerate(providers)}
    for panel, (axis, degree) in enumerate(zip(axes, degrees)):
        axis.plot(
            xs,
            np.minimum(
                roof["fp32_gflops"], roof["dram_bandwidth_gbs"] * xs),
            "--", color="#64748b", label="DRAM roof")
        axis.plot(
            xs,
            np.minimum(
                roof["fp32_gflops"], roof["l2_bandwidth_gbs"] * xs),
            "-", color="#111827", label="measured L2 reference")
        selected = [item for item in results if item["degree"] == degree]
        minimum_node = min(item["nodes"] for item in selected)
        for item in selected:
            axis.scatter(
                item["arithmetic_intensity_flop_per_byte"],
                item["achieved_gflops"],
                marker=f"${ids[item['provider']]}$", s=125,
                color=provider_color(item["provider"]),
                label=(item["provider"]
                       if panel == 0 and item["nodes"] == minimum_node
                       else None))
            horizontal = 7 if item["provider"].startswith("graphforge") else -7
            alignment = "left" if horizontal > 0 else "right"
            vertical = 5 if item["nodes"] == minimum_node else -11
            axis.annotate(
                f"N={item['nodes'] // 1024}K",
                (item["arithmetic_intensity_flop_per_byte"],
                 item["achieved_gflops"]),
                xytext=(horizontal, vertical), textcoords="offset points",
                ha=alignment, fontsize=7)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_ylabel("useful GFLOP/s")
        axis.set_title(f"degree={degree}")
        axis.grid(True, which="both", alpha=0.35)
        if panel == 0:
            axis.legend(fontsize=8, ncol=2)
    axes[-1].set_xlabel(
        "useful FLOP / provider-independent semantic byte")
    fig.suptitle(
        "PageRank hierarchical roofline · numeric markers expose overlap",
        fontweight="bold")
    fig.savefig(output / "roofline.svg", bbox_inches="tight")
    fig.savefig(output / "roofline.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _report(payload: dict, output: Path) -> None:
    lines = [
        "# Fixed-iteration PageRank compiler probe",
        "",
        "The GraphForge candidate is one captured `gf_control.repeat` whose "
        "MessagePassing body lowers to one compiler-generated TTIR kernel per "
        "iteration. The peer executes the same recurrence and iteration count "
        "with `torch.sparse.mm`; this is not a convergence benchmark.",
        "",
        "![Matched latency](latency.png)",
        "",
        "![Hierarchical roofline](roofline.svg)",
        "",
        "| N | Degree | GraphForge (ms) | Peer (ms) | Speedup | 95% CI | Gate |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for gate in payload["sota_gates"]:
        lines.append(
            f"| {gate['nodes']:,} | "
            f"{int(str(gate['topology']).rsplit('-', 1)[-1])} | "
            f"{gate['candidate_ms']:.4f} | {gate['baseline_ms']:.4f} | "
            f"{gate['speedup_vs_sota']:.3f}x | "
            f"[{gate['speedup_ci_low']:.3f}, {gate['speedup_ci_high']:.3f}] | "
            f"{'PASS' if gate['passed'] else 'FAIL'} |"
        )
    lines.extend([
        "",
        "Cold compilation is recorded separately in `roofline.json`; it is "
        "never folded into steady-state kernel latency. Device-side convergence "
        "and a cuGraph/GraphBLAS peer remain outside this fixed-iteration claim.",
    ])
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    nodes = (1 << 16, 1 << 18) if not args.quick else (1 << 15, 1 << 16)
    degrees = (4, 16, 32)
    if args.quick:
        args.samples = min(args.samples, 8)
    output = args.output_dir or operation_dir(
        "pagerank", f"cuda_fp32_iter{args.iterations}")
    output.mkdir(parents=True, exist_ok=True)
    roof = measure_roofs(torch.device("cuda"), args.quick, args.samples)
    results, compiles = [], []
    for node_count in nodes:
        for degree in degrees:
            measured, compiled = _case(
                node_count, degree, args.iterations, args.samples, roof)
            results.extend(measured)
            compiles.append({"nodes": node_count, "degree": degree, **compiled})
    payload = {
        "operation": "pagerank",
        "workload": "fixed_iteration_pagerank",
        "config": {"nodes": nodes, "degrees": degrees,
                   "iterations": args.iterations, "dtype": "float32"},
        "roof": roof.__dict__,
        "results": results,
        "compilation": compiles,
    }
    payload["sota_gates"] = [
        gate.to_dict() for gate in evaluate_sota_gates(
            results,
            ["graphforge.control"],
            baselines={"torch.sparse.mm"},
        )
    ]
    (output / "roofline.json").write_text(
        json.dumps(payload, indent=2) + "\n")
    _plots(payload, output)
    _roofline_plot(payload, output)
    _report(payload, output)
    print(output)
    for node_count in nodes:
        for degree in degrees:
            case = [item for item in results
                    if item["nodes"] == node_count and item["degree"] == degree]
            gf_ms = next(item["milliseconds"] for item in case
                         if item["provider"] == "graphforge.control")
            peer_ms = next(item["milliseconds"] for item in case
                           if item["provider"] == "torch.sparse.mm")
            print(f"N={node_count} d={degree}: {peer_ms / gf_ms:.3f}x")
    failed = [gate for gate in payload["sota_gates"] if not gate["passed"]]
    if args.fail_on_gate and failed:
        raise SystemExit(
            f"{len(failed)} PageRank performance bucket(s) failed the "
            "95% confidence gate")


if __name__ == "__main__":
    main()
