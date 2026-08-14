"""Two-process automatic MessagePassing overlap benchmark.

This benchmark exercises the public ``Graph.halo()`` path rather than timing
pack/exchange primitives in isolation.  Every rank owns interior rows and a
remote-source boundary, so the runtime can execute useful interior work while
the stdlib transport progresses the halo.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from pathlib import Path
import statistics
import time

import graphforge as gf
from graphforge.distributed import DistributedRuntime, PipeTransport, owned_range


class NeighborSum(gf.MessagePassing):
    reducer = gf.sum()

    def edge(self, src, dst, edge):
        del dst, edge
        value = src.x
        # A representative nonlinear UDF makes interior work substantial
        # enough for overlap to matter without changing the graph topology.
        for _ in range(8):
            value = value * src.x + src.x
        return value


class DelayedPipeTransport(PipeTransport):
    """Two-process pipe with an explicit deployment-latency model."""

    prefer_compute_overlap = True

    def __init__(self, rank, world_size, connections, *, delay_ms: float):
        super().__init__(rank, world_size, connections)
        self.prefer_compute_overlap = True
        self.delay_seconds = delay_ms / 1e3

    def receive(self, peer):
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        return super().receive(peer)


def _topology(entities: int, degree: int, boundary_fraction: float):
    half = entities // 2
    boundary_rows = max(1, int(half * boundary_fraction))
    rows = [degree * row for row in range(entities + 1)]
    columns: list[int] = []
    for destination in range(entities):
        owner = destination // half
        begin = owner * half
        local = destination - begin
        source_begin = (1 - owner) * half if local < boundary_rows else begin
        columns.extend(
            source_begin + (local + neighbor) % half
            for neighbor in range(degree)
        )
    return rows, columns


def _worker(
    rank, endpoint, entities, degree, features, boundary_fraction,
    warmup, repeats, transport_delay_ms, barrier, queue,
):
    os.environ["GRAPHFORGE_TENSOR_BACKEND"] = "native"
    rows, columns = _topology(entities, degree, boundary_fraction)
    begin, end = owned_range(entities, 2, rank)
    graph = gf.Graph.from_csr(
        gf.tensor(rows, dtype=gf.int64),
        gf.tensor(columns, dtype=gf.int64),
        num_src=entities, validate="full",
    ).halo(gf.DeviceMesh("cpu", 2), depth=1)
    source = gf.tensor([1.0] * ((end - begin) * features)).reshape(
        end - begin, features)
    transport = DelayedPipeTransport(
        rank, 2, {1 - rank: endpoint}, delay_ms=transport_delay_ms)
    kernel = NeighborSum()
    samples = {"graphforge.auto": [], "graphforge.serialized": []}
    last = {}
    with DistributedRuntime(transport) as runtime:
        for iteration in range(warmup + repeats):
            order = (
                ("graphforge.auto", "graphforge.serialized")
                if iteration % 2 == 0
                else ("graphforge.serialized", "graphforge.auto")
            )
            for method in order:
                runtime._force_serialized = (
                    True if method == "graphforge.serialized" else None)
                barrier.wait()
                started = time.perf_counter_ns()
                output = kernel(graph=graph, src={"x": source}, dst={})
                output.realize()
                elapsed_ms = (time.perf_counter_ns() - started) / 1e6
                trace = runtime.last_execution_trace
                if trace is None:
                    raise RuntimeError(
                        "automatic runtime did not emit an execution trace")
                if iteration >= warmup:
                    execution = output.execution or {}
                    samples[method].append({
                        **trace,
                        "end_to_end_ms": elapsed_ms,
                        "final_backend": execution.get("backend"),
                        "final_compile_ms": execution.get("compile_ms"),
                        "final_launch_ms": execution.get("launch_ms"),
                        "final_cache_hit": execution.get("cache_hit"),
                    })
                last[method] = output
        # Normalize every output element before reducing.  The synthetic UDF
        # produces ``degree * 9`` exactly; summing raw values across tens of
        # millions of FP32 elements would measure serial accumulation error
        # rather than MessagePassing correctness.
        checksums = {
            method: float((output / float(degree * 9)).sum().tolist())
            for method, output in last.items()
        }
    expected = float((end - begin) * features)
    queue.put({
        "rank": rank,
        "samples": samples,
        "checksums": checksums,
        "expected_checksum": expected,
        "correct": all(
            abs(checksum - expected) <= max(1.0, expected) * 1e-5
            for checksum in checksums.values()
        ),
    })


def _plot(result: dict[str, object], path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from benchmarks.common.plotting import provider_color

    records = result["ranks"]
    selected = []
    for record in records:
        samples = record["samples"]["graphforge.auto"]
        median = statistics.median(item["end_to_end_ms"] for item in samples)
        selected.append(min(samples, key=lambda item: abs(
            item["end_to_end_ms"] - median)))

    fig, axes = plt.subplots(
        len(selected) + 1, 1,
        figsize=(10, 2.4 * (len(selected) + 1)), squeeze=False,
        constrained_layout=True,
    )
    latency_axis = axes[0, 0]
    methods = ("graphforge.auto", "graphforge.serialized")
    latencies = [result["median_end_to_end_ms"][method] for method in methods]
    bars = latency_axis.barh(
        range(len(methods)), latencies,
        color=[provider_color(method) for method in methods],
    )
    latency_axis.set_yticks(range(len(methods)), methods)
    latency_axis.invert_yaxis()
    latency_axis.set_xlabel("Median rank end-to-end latency (ms, lower is better)")
    latency_axis.set_title(
        f"Automatic overlap speedup: {result['speedup_vs_serialized']:.3f}×")
    latency_axis.grid(axis="x", alpha=0.25)
    latency_axis.bar_label(bars, fmt="%.3f ms", padding=3)
    colors = {"communication": "#E45756", "interior": "#4C78A8"}
    for rank, (axis, sample) in enumerate(zip(axes[1:, 0], selected)):
        origin = min(
            sample["communication_started_ns"], sample["interior_started_ns"])
        intervals = {
            "communication": (
                (sample["communication_started_ns"] - origin) / 1e6,
                (sample["communication_finished_ns"]
                 - sample["communication_started_ns"]) / 1e6,
            ),
            "interior": (
                (sample["interior_started_ns"] - origin) / 1e6,
                (sample["interior_finished_ns"]
                 - sample["interior_started_ns"]) / 1e6,
            ),
        }
        for row, name in enumerate(("communication", "interior")):
            start, width = intervals[name]
            axis.barh(row, width, left=start, height=0.56,
                      color=colors[name], label=name)
        overlap = float(sample["measured_overlap_ms"])
        axis.set_yticks((0, 1), ("halo", "interior compute"))
        axis.set_xlabel("Time from first activity (ms)")
        axis.set_title(
            f"Rank {rank}: measured overlap {overlap:.3f} ms · "
            f"end-to-end {sample['end_to_end_ms']:.3f} ms")
        axis.grid(axis="x", alpha=0.25)
    fig.suptitle("GraphForge automatic interior ∥ halo → boundary execution")
    fig.legend(
        handles=[Patch(color=colors[name], label=name)
                 for name in ("communication", "interior")],
        loc="outside upper right",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entities", type=int, default=65536)
    parser.add_argument("--degree", type=int, default=16)
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--boundary-fraction", type=float, default=0.25)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--transport-delay-ms", type=float, default=5.0,
        help="controlled per-receive latency used to model an inter-node link",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--output", type=Path,
        default=Path("output/distributed/automatic_cpu_overlap/results.json"),
    )
    args = parser.parse_args()
    if args.quick:
        args.entities, args.degree, args.features = 32768, 16, 64
        # A useful partition keeps the cut small; this still leaves 2,048
        # remote boundary rows across both ranks and enough interior work to
        # make the overlap decision measurable above process jitter.
        args.boundary_fraction = 0.125
        args.warmup, args.repeats = 1, 3
    if (args.entities < 4 or args.entities % 2 or args.degree <= 0
            or args.features <= 0 or args.repeats <= 0
            or args.transport_delay_ms < 0.0
            or not 0.0 < args.boundary_fraction < 1.0):
        raise ValueError("invalid automatic-overlap benchmark domain")

    context = multiprocessing.get_context("spawn")
    first, second = context.Pipe(duplex=True)
    barrier = context.Barrier(2)
    queue = context.Queue()
    processes = (
        context.Process(
            target=_worker,
            args=(0, first, args.entities, args.degree, args.features,
                  args.boundary_fraction, args.warmup, args.repeats,
                  args.transport_delay_ms, barrier, queue),
        ),
        context.Process(
            target=_worker,
            args=(1, second, args.entities, args.degree, args.features,
                  args.boundary_fraction, args.warmup, args.repeats,
                  args.transport_delay_ms, barrier, queue),
        ),
    )
    for process in processes:
        process.start()
    records = [queue.get(timeout=300) for _ in processes]
    for process in processes:
        process.join(timeout=30)
        if process.exitcode != 0:
            raise RuntimeError(f"overlap worker exited with {process.exitcode}")
    records.sort(key=lambda item: item["rank"])
    automatic_samples = [
        sample for record in records
        for sample in record["samples"]["graphforge.auto"]
    ]
    overlaps = [sample["measured_overlap_ms"] for sample in automatic_samples]
    end_to_end = {
        method: [
            sample["end_to_end_ms"]
            for record in records for sample in record["samples"][method]
        ]
        for method in ("graphforge.auto", "graphforge.serialized")
    }
    median_end_to_end = {
        method: statistics.median(samples)
        for method, samples in end_to_end.items()
    }
    result = {
        "schema": "graphforge.distributed-automatic-overlap.v1",
        "transport": "multiprocessing.Connection",
        "world_size": 2,
        "entities": args.entities,
        "degree": args.degree,
        "features": args.features,
        "boundary_fraction": args.boundary_fraction,
        "controlled_transport_delay_ms": args.transport_delay_ms,
        "warmup": args.warmup,
        "repeats_per_rank": args.repeats,
        "median_measured_overlap_ms": statistics.median(overlaps),
        "median_end_to_end_ms": median_end_to_end,
        "speedup_vs_serialized": (
            median_end_to_end["graphforge.serialized"]
            / median_end_to_end["graphforge.auto"]
        ),
        "ranks": records,
    }
    result["gate"] = (
        "PASS" if all(record["correct"] for record in records)
        and all(overlap > 0.0 for overlap in overlaps)
        and result["speedup_vs_serialized"] >= 1.0 else "FAIL"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _plot(result, args.output.with_name("timeline.png"))
    args.output.with_name("REPORT.md").write_text(
        "# Automatic CPU halo overlap\n\n"
        f"Two ranks, {args.entities:,} entities, degree {args.degree}, feature "
        f"width {args.features}, boundary fraction {args.boundary_fraction:.2f}. "
        f"The controlled transport model adds {args.transport_delay_ms:.2f} ms "
        "per receive. "
        f"Median measured communication/interior overlap is "
        f"{result['median_measured_overlap_ms']:.4f} ms; median rank end-to-end "
        f"automatic latency is "
        f"{result['median_end_to_end_ms']['graphforge.auto']:.4f} ms versus "
        f"{result['median_end_to_end_ms']['graphforge.serialized']:.4f} ms "
        f"serialized ({result['speedup_vs_serialized']:.3f}x). Gate: "
        f"**{result['gate']}**.\n\n"
        "This proves the automatic CPU runtime ordering over a real two-process "
        "transport under an explicit latency model. It is not measured "
        "multi-GPU or inter-node performance evidence.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    if result["gate"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
