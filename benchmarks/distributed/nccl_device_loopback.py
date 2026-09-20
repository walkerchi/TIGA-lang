"""Single-rank NCCL-provider device-buffer conformance diagnostic.

This intentionally is not an NCCL communication or bandwidth claim. It proves
that the optional provider creates a real communicator and preserves its
device-buffer/stream/completion contract through the rank-local D2D fast path.
Only the separate two-device gate exercises ncclSend/ncclRecv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import tiga as tg
from tiga.distributed import DeviceBufferSlice, create_transport, nccl_unique_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[1024, 1 << 20, 16 << 20])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output", type=Path,
        default=Path("output/distributed/nccl_single_rank_device_direct/results.json"),
    )
    args = parser.parse_args()
    if any(size <= 0 for size in args.sizes) or args.warmup < 0 or args.repeats <= 0:
        raise ValueError("sizes/repeats must be positive and warmup non-negative")

    transport = create_transport(
        "nccl", rank=0, world_size=1,
        communicator_id=nccl_unique_id(), device=args.device,
    )
    stream = tg.runtime.Stream(args.device)
    cases = []
    try:
        for byte_count in args.sizes:
            source = tg.runtime.Buffer(byte_count, device=args.device)
            destination = tg.runtime.Buffer(byte_count, device=args.device)
            payload = bytes(index % 251 for index in range(byte_count))
            source.write(payload)
            send = ((0, DeviceBufferSlice(source, 0, byte_count)),)
            receive = ((0, DeviceBufferSlice(destination, 0, byte_count)),)
            try:
                for _ in range(args.warmup):
                    transport.exchange_device(send, receive, stream=stream).wait()
                samples = []
                for _ in range(args.repeats):
                    started = time.perf_counter_ns()
                    transport.exchange_device(send, receive, stream=stream).wait()
                    samples.append((time.perf_counter_ns() - started) / 1e6)
                median_ms = statistics.median(samples)
                # Poison the target and perform one separately validated
                # exchange so a stale warmup result cannot satisfy the gate.
                destination.write(bytes(byte_count))
                transport.exchange_device(send, receive, stream=stream).wait()
                cases.append({
                    "bytes": byte_count,
                    "median_ms": median_ms,
                    "diagnostic_payload_GBps": byte_count / median_ms / 1e6,
                    "minimum_ms": min(samples),
                    "maximum_ms": max(samples),
                    "correct": destination.read() == payload,
                })
            finally:
                source.close()
                destination.close()
    finally:
        stream.close()
        version = transport.version
        library = transport.library_path
        transport.close()

    gate = "PASS" if all(case["correct"] for case in cases) else "FAIL"
    result = {
        "schema": "tiga.nccl-device-conformance.v1",
        "scope": "single-rank-device-pointer-conformance",
        "device": args.device,
        "world_size": 1,
        "nccl_version": version,
        "nccl_library": library,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "cases": cases,
        "gate": gate,
        "performance_claim": False,
        "transfer_path": "rank-local-device-copy",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    from benchmarks.common.diagnostic_plotting import plot_json
    plot_json(args.output)
    rows = "\n".join(
        f"| {case['bytes']} | {case['median_ms']:.4f} | "
        f"{case['diagnostic_payload_GBps']:.3f} | {case['correct']} |"
        for case in cases
    )
    (args.output.parent / "REPORT.md").write_text(
        "# NCCL device-pointer conformance\n\n"
        f"Device: `{args.device}`. NCCL version: `{version}`. Gate: **{gate}**.\n\n"
        "| bytes | median ms | diagnostic payload GB/s | correct |\n"
        "|---:|---:|---:|:---:|\n" + rows + "\n\n"
        "This creates a real NCCL communicator, but a rank cannot communicate "
        "with itself through NCCL point-to-point. The provider therefore uses "
        "its native stream-ordered D2D fast path. The throughput column is only "
        "a local diagnostic; it is not NCCL, peer-link, multi-GPU, overlap, or "
        "production performance evidence.\n",
        encoding="utf-8",
    )
    print(args.output)
    if gate != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
