"""Gate the direct gf_kernel -> TTIR path against scalar CSR peers.

The benchmark keeps compilation and steady-state execution separate.  The
runtime gate is per degree bucket: a candidate passes only when it is no more
than five percent slower than the fastest in-tree Triton template or
torch.sparse.mm peer.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import time
import warnings

import torch

import graphforge as gf
from benchmarks.kernels.sparse_triton_oracles import prepare_fixed_weighted_sum


class WeightedAggregation(gf.MessagePassing):
    reducer = gf.sum()

    def edge(self, src, dst, edge):
        return edge.weight * src.x


def samples_ms(fn, repeat: int) -> list[float]:
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeat):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end))
    return samples


def median_ms(fn, repeat: int) -> tuple[float, list[float]]:
    samples = samples_ms(fn, repeat)
    return statistics.median(samples), samples


def plot(results: list[dict[str, object]], path: Path) -> None:
    import matplotlib.pyplot as plt

    providers = ("graphforge.kernel_ttir", "triton.template", "torch.sparse.mm")
    colors = ("#7c3aed", "#059669", "#dc2626")
    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    for provider, color in zip(providers, colors):
        axis.plot(
            [item["degree"] for item in results],
            [item["runtime_ms"][provider] for item in results],
            marker="o",
            linewidth=2,
            label=provider,
            color=color,
        )
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xlabel("Fixed CSR degree")
    axis.set_ylabel("Median runtime (ms; lower is better)")
    axis.set_title("GraphForge direct Kernel IR → TTIR performance gate")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=1 << 17)
    parser.add_argument("--degrees", type=int, nargs="+", default=(4, 16, 64))
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--json", type=Path,
        default=Path(
            "output/roofline/weighted_aggregation/kernel_ttir/results.json"))
    parser.add_argument(
        "--plot", type=Path,
        default=Path(
            "output/roofline/weighted_aggregation/kernel_ttir/runtime.png"))
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.nodes, args.repeat = 1 << 14, 20
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if not os.environ.get("GRAPHFORGE_OPT"):
        raise SystemExit("set GRAPHFORGE_OPT to the built gf-opt")
    if not os.environ.get("GRAPHFORGE_TRANSLATE"):
        raise SystemExit("set GRAPHFORGE_TRANSLATE to the built gf-translate")

    results = []
    all_passed = True
    for degree in args.degrees:
        nodes = args.nodes
        row_ptr = torch.arange(
            0, nodes * degree + 1, degree,
            device="cuda", dtype=torch.int64)
        col_idx = torch.randint(
            nodes, (nodes * degree,), device="cuda", dtype=torch.int64)
        x = torch.randn(nodes, device="cuda")
        weight = torch.randn(nodes * degree, device="cuda")
        graph = gf.Graph.from_csr(row_ptr, col_idx, num_src=nodes)

        kernel = WeightedAggregation()
        torch.cuda.synchronize()
        compile_begin = time.perf_counter()
        direct_output = kernel(
            graph=graph, src={"x": x}, dst={"x": x},
            edge={"weight": weight})
        torch.cuda.synchronize()
        jit_wall_ms = (time.perf_counter() - compile_begin) * 1e3
        if kernel.last_variant.lowering != (
            "gf-kernel-to-ttir-fixed-csr-weighted-sum"
        ):
            raise RuntimeError(
                f"direct path was not selected: {kernel.explain()}")

        template = prepare_fixed_weighted_sum(
            col_idx, weight, x, num_rows=nodes, degree=degree)
        if template is None:
            raise RuntimeError("in-tree Triton template is unavailable")
        template_output = template.run(x, weight)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            csr = torch.sparse_csr_tensor(
                row_ptr, col_idx, weight, size=(nodes, nodes))
        torch_output = torch.sparse.mm(csr, x[:, None])[:, 0]
        torch.cuda.synchronize()
        torch.testing.assert_close(direct_output, torch_output)
        torch.testing.assert_close(template_output, torch_output)

        calls = {
            "graphforge.kernel_ttir": lambda: kernel(
                graph=graph, src={"x": x}, dst={"x": x},
                edge={"weight": weight}),
            "triton.template": lambda: template.run(x, weight),
            "torch.sparse.mm": lambda: torch.sparse.mm(csr, x[:, None]),
        }
        runtime = {}
        raw_samples = {}
        for name, call in calls.items():
            runtime[name], raw_samples[name] = median_ms(call, args.repeat)
        fastest_peer = min(runtime["triton.template"], runtime["torch.sparse.mm"])
        ratio = runtime["graphforge.kernel_ttir"] / fastest_peer
        passed = ratio <= 1.05
        all_passed &= passed
        results.append({
            "nodes": nodes,
            "edges": nodes * degree,
            "degree": degree,
            "runtime_ms": runtime,
            "samples_ms": raw_samples,
            "jit_wall_ms_process_cache_aware": jit_wall_ms,
            "ratio_vs_fastest_peer": ratio,
            "gate_limit": 1.05,
            "gate_passed": passed,
            "lowering": kernel.last_variant.lowering,
            "provider": kernel.last_variant.provider,
        })
        print(
            f"degree={degree:>2} direct={runtime['graphforge.kernel_ttir']:.6f} "
            f"template={runtime['triton.template']:.6f} "
            f"torch={runtime['torch.sparse.mm']:.6f} ms "
            f"ratio={ratio:.3f} {'PASS' if passed else 'FAIL'}")

    payload = {
        "benchmark": "fixed-scalar-weighted-sum-kernel-to-ttir",
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "results": results,
        "all_gates_passed": all_passed,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2))
    plot(results, args.plot)
    print(f"wrote {args.json}")
    print(f"wrote {args.plot}")
    if args.fail_on_gate and not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
