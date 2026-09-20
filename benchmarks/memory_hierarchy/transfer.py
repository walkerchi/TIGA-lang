"""Formal native-runtime HBM/pinned transfer and NVMe spill benchmark."""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
import statistics
import time

import tiga as tg


def median_ms(operation, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        operation()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bytes", type=int, default=64 << 20)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--quick", action="store_true",
        help="reduced size/repeats for smoke runs")
    parser.add_argument(
        "--output", type=Path,
        default=Path("output/memory_hierarchy/hbm_pinned_nvme/results.json"),
    )
    args = parser.parse_args()
    if args.quick:
        args.bytes, args.repeats = 4 << 20, 5
    runtime = tg.runtime.HierarchyRuntime()
    try:
        host = runtime.allocate(
            "tile", 1, tier="host-pinned", capacity_bytes=args.bytes
        )
        restored = runtime.allocate(
            "tile", 1, tier="host-pinned", capacity_bytes=args.bytes
        )
        device = runtime.allocate(
            "tile", 1, tier="device", capacity_bytes=args.bytes,
            device="cuda:0",
        )
        ctypes.memset(host.buffer.host_address, 0x5A, args.bytes)
        stream = tg.runtime.Stream("cuda:0")

        def h2d():
            host.buffer.copy_to(device.buffer, stream=stream).wait()

        def d2h():
            device.buffer.copy_to(restored.buffer, stream=stream).wait()

        h2d(); d2h()
        h2d_ms = median_ms(h2d, args.repeats)
        d2h_ms = median_ms(d2h, args.repeats)
        correctness = restored.buffer.read(offset=0, bytes=4096) == bytes([0x5A]) * 4096

        ram = runtime.allocate("spill", 2, tier="ram", capacity_bytes=args.bytes)
        ctypes.memset(ram.buffer.host_address, 0xA5, args.bytes)
        nvme = runtime.allocate("spill", 2, tier="nvme", capacity_bytes=args.bytes)
        spill_ms = median_ms(lambda: runtime.transfer(ram, nvme).wait(), 3)
        restored_ram = runtime.allocate(
            "spill", 2, tier="ram", capacity_bytes=args.bytes
        )
        restore_ms = median_ms(lambda: runtime.transfer(nvme, restored_ram).wait(), 3)
        correctness = correctness and (
            restored_ram.buffer.read(offset=0, bytes=4096) == bytes([0xA5]) * 4096
        )
        result = {
            "schema": "tiga.memory-hierarchy.v1",
            "bytes": args.bytes,
            "repeats": args.repeats,
            "correct": correctness,
            "h2d_ms": h2d_ms,
            "d2h_ms": d2h_ms,
            "h2d_GBps": args.bytes / h2d_ms / 1e6,
            "d2h_GBps": args.bytes / d2h_ms / 1e6,
            "nvme_spill_ms": spill_ms,
            "nvme_restore_ms": restore_ms,
            "nvme_spill_GBps": args.bytes / spill_ms / 1e6,
            "nvme_restore_GBps": args.bytes / restore_ms / 1e6,
            "peak_live_bytes": {
                tier.value: value
                for tier, value in runtime.peak_live_bytes.items() if value
            },
            "gate": "PASS" if correctness and h2d_ms > 0 and d2h_ms > 0 else "FAIL",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        from benchmarks.common.diagnostic_plotting import plot_json
        plot_json(args.output)
        print(json.dumps(result, indent=2))
        if result["gate"] != "PASS":
            raise SystemExit(1)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
