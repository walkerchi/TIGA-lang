"""Real two-rank MPI halo transport latency/bandwidth benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import struct
import time

from mpi4py import MPI

import tiga as gf
from tiga.distributed import (
    create_transport, exchange_packed, pack_halo, unpack_halo,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entities", type=int, default=65536)
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument(
        "--output", type=Path,
        default=Path("output/distributed/mpi_two_process_halo/results.json"))
    args = parser.parse_args()
    rank, world = MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size()
    if world != 2:
        raise SystemExit("launch with mpiexec -n 2")
    if args.entities < 4 or args.entities % 2 or args.features <= 0 or args.repeats <= 0:
        raise ValueError("entities must be even >=4; features/repeats must be positive")
    rows = list(range(args.entities + 1))
    columns = [
        (destination + args.entities // 2) % args.entities
        for destination in range(args.entities)
    ]
    halo = gf.collective_halo_maps(
        rows, columns, num_entities=args.entities, world_size=world)[rank]
    element_bytes = 4 * args.features
    owned = bytearray(halo.owned_entities * element_bytes)
    for local, entity in enumerate(range(halo.owned_begin, halo.owned_end)):
        struct.pack_into("f", owned, local * element_bytes, float(entity))
    owned = bytes(owned)
    transport = create_transport("mpi", communicator=MPI.COMM_WORLD, tag=29)
    MPI.COMM_WORLD.Barrier()
    phases = {name: [] for name in ("pack", "transport", "unpack", "total")}
    last = None
    for _ in range(args.repeats):
        total = time.perf_counter_ns()
        started = time.perf_counter_ns()
        packed = pack_halo(halo, owned, element_bytes=element_bytes)
        phases["pack"].append((time.perf_counter_ns() - started) / 1e6)
        started = time.perf_counter_ns()
        received = exchange_packed(halo, packed, transport=transport)
        phases["transport"].append((time.perf_counter_ns() - started) / 1e6)
        started = time.perf_counter_ns()
        last = unpack_halo(halo, received)
        phases["unpack"].append((time.perf_counter_ns() - started) / 1e6)
        phases["total"].append((time.perf_counter_ns() - total) / 1e6)
    correct = all(
        struct.unpack_from("f", last.value_bytes(entity))[0] == float(entity)
        for entity in last.ids)
    local = {
        "rank": rank, "correct": correct, "payload_bytes": len(last.data),
        "samples_ms": phases,
    }
    records = MPI.COMM_WORLD.gather(local, root=0)
    if rank:
        return
    combined = {
        phase: [sample for record in records for sample in record["samples_ms"][phase]]
        for phase in phases
    }
    medians = {phase: statistics.median(samples) for phase, samples in combined.items()}
    payload = sum(record["payload_bytes"] for record in records)
    result = {
        "schema": "tiga.distributed-halo.v1",
        "transport": "mpi4py/MPICH",
        "mpi_library": MPI.Get_library_version().splitlines()[0],
        "world_size": world, "entities": args.entities,
        "features": args.features, "degree": 1,
        "repeats_per_rank": args.repeats,
        "payload_bytes_per_iteration": payload,
        "median_rank_pack_ms": medians["pack"],
        "median_rank_transport_ms": medians["transport"],
        "median_rank_unpack_ms": medians["unpack"],
        "median_rank_total_ms": medians["total"],
        "aggregate_payload_GBps": payload / medians["total"] / 1e6,
        "samples_ms": combined,
        "correct": all(record["correct"] for record in records),
    }
    result["gate"] = "PASS" if result["correct"] and medians["total"] > 0 else "FAIL"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    from benchmarks.common.diagnostic_plotting import plot_json
    plot_json(args.output)
    (args.output.parent / "REPORT.md").write_text(
        "# Two-rank MPI halo exchange\n\n"
        f"Provider: `{result['transport']}` ({result['mpi_library']}).  "
        f"Aggregate payload per iteration: {payload / (1 << 20):.2f} MiB.  "
        f"Median rank pack/transport/unpack/total: {medians['pack']:.4f} / "
        f"{medians['transport']:.4f} / {medians['unpack']:.4f} / "
        f"{medians['total']:.4f} ms.  Effective aggregate end-to-end payload "
        f"bandwidth: {result['aggregate_payload_GBps']:.3f} GB/s.  "
        f"Correctness gate: {result['gate']}.\n\n"
        "This is two processes on one host, not a multi-node or GPU-direct claim.\n",
        encoding="utf-8")
    print(args.output)
    if result["gate"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
