"""Torch-free two-process halo exchange correctness/latency benchmark."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from pathlib import Path
import statistics
import struct
import time

import tiga as gf
from tiga.distributed import (
    PipeTransport, exchange_packed, pack_halo, unpack_halo,
)


def worker(rank, endpoint, halo, repeats, features, ready, start, queue):
    transport = PipeTransport(rank, 2, {1 - rank: endpoint})
    element_bytes = 4 * features
    owned = bytearray(halo.owned_entities * element_bytes)
    for local, entity in enumerate(range(halo.owned_begin, halo.owned_end)):
        struct.pack_into("f", owned, local * element_bytes, float(entity))
    ready.put(rank)
    start.wait()
    samples = {"pack": [], "exchange": [], "unpack": [], "total": []}
    last = None
    for _ in range(repeats):
        total_begin = time.perf_counter_ns()
        begin = time.perf_counter_ns()
        packed = pack_halo(halo, owned, element_bytes=element_bytes)
        samples["pack"].append((time.perf_counter_ns() - begin) / 1e6)
        begin = time.perf_counter_ns()
        received = exchange_packed(halo, packed, transport=transport)
        samples["exchange"].append((time.perf_counter_ns() - begin) / 1e6)
        begin = time.perf_counter_ns()
        last = unpack_halo(halo, received)
        samples["unpack"].append((time.perf_counter_ns() - begin) / 1e6)
        samples["total"].append((time.perf_counter_ns() - total_begin) / 1e6)
    correct = all(
        struct.unpack_from("f", last.value_bytes(entity))[0] == float(entity)
        for entity in last.ids
    )
    queue.put((rank, samples, correct, len(last.data)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entities", type=int, default=65536)
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument(
        "--quick", action="store_true",
        help="reduced size/repeats for smoke runs")
    parser.add_argument(
        "--output", type=Path,
        default=Path("output/distributed/two_process_halo/results.json"),
    )
    args = parser.parse_args()
    if args.quick:
        args.entities, args.features, args.repeats = 8192, 16, 10
    if args.entities < 4 or args.features <= 0 or args.repeats <= 0:
        raise ValueError("entities/features/repeats are outside benchmark domain")
    # A half-ring permutation makes every source remote while keeping CSR
    # construction O(N); the union halo is the complete peer partition.
    degree = 1
    row_ptr = [degree * row for row in range(args.entities + 1)]
    col_idx = [
        (destination + args.entities // 2) % args.entities
        for destination in range(args.entities)
    ]
    halos = gf.collective_halo_maps(
        row_ptr, col_idx, num_entities=args.entities, world_size=2
    )
    context = multiprocessing.get_context("spawn")
    first, second = context.Pipe(duplex=True)
    ready = context.Queue()
    start = context.Event()
    queue = context.Queue()
    processes = (
        context.Process(
            target=worker,
            args=(0, first, halos[0], args.repeats, args.features,
                  ready, start, queue),
        ),
        context.Process(
            target=worker,
            args=(1, second, halos[1], args.repeats, args.features,
                  ready, start, queue),
        ),
    )
    for process in processes:
        process.start()
    for _ in processes:
        ready.get(timeout=30)
    start.set()
    results = [queue.get(timeout=120) for _ in processes]
    for process in processes:
        process.join(timeout=30)
        if process.exitcode != 0:
            raise RuntimeError(f"halo worker exited with {process.exitcode}")
    all_samples = {
        phase: [
            sample
            for _rank, samples, _correct, _bytes in results
            for sample in samples[phase]
        ]
        for phase in ("pack", "exchange", "unpack", "total")
    }
    payload = sum(item[3] for item in results)
    medians = {
        phase: statistics.median(samples)
        for phase, samples in all_samples.items()
    }
    result = {
        "schema": "tiga.distributed-halo.v1",
        "world_size": 2,
        "entities": args.entities,
        "features": args.features,
        "degree": degree,
        "repeats_per_rank": args.repeats,
        "payload_bytes_per_iteration": payload,
        "median_rank_pack_ms": medians["pack"],
        "median_rank_transport_ms": medians["exchange"],
        "median_rank_unpack_ms": medians["unpack"],
        "median_rank_total_ms": medians["total"],
        "aggregate_payload_GBps": payload / medians["total"] / 1e6,
        "samples_ms": all_samples,
        "correct": all(item[2] for item in results),
        "transport": "multiprocessing.Connection",
    }
    result["gate"] = (
        "PASS" if result["correct"] and medians["total"] > 0 else "FAIL")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    from benchmarks.common.diagnostic_plotting import plot_json
    plot_json(args.output)
    print(json.dumps(result, indent=2))
    if result["gate"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
