"""Two-GPU NCCL byte bandwidth plus sharded MessagePassing/VJP gate.

The script fails before writing an artifact unless two CUDA devices and NCCL
are genuinely available. It is the external-hardware closure command for X0;
the single-GPU development host must not manufacture its result.
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
from graphforge.distributed import (
    DeviceBufferSlice, DistributedRuntime, NCCLTransport, nccl_unique_id,
    owned_range,
)


class NeighborSum(gf.MessagePassing):
    reducer = gf.sum()

    def edge(self, src, dst, edge):
        del dst, edge
        return src.x


def _worker(rank, communicator_id, byte_count, repeats, queue):
    device = f"cuda:{rank}"
    os.environ["GRAPHFORGE_TENSOR_BACKEND"] = "native"
    transport = NCCLTransport(
        rank, 2, communicator_id, device=device,
    )
    stream = gf.runtime.Stream(device)
    source = gf.runtime.Buffer(byte_count, device=device)
    destination = gf.runtime.Buffer(byte_count, device=device)
    source.write(bytes([rank + 17]) * byte_count)
    send = ((1 - rank, DeviceBufferSlice(source, 0, byte_count)),)
    receive = ((1 - rank, DeviceBufferSlice(destination, 0, byte_count)),)
    try:
        for _ in range(20):
            transport.exchange_device(send, receive, stream=stream).wait()
        samples = []
        for _ in range(repeats):
            started = time.perf_counter_ns()
            transport.exchange_device(send, receive, stream=stream).wait()
            samples.append((time.perf_counter_ns() - started) / 1e6)
        byte_correct = destination.read() == bytes([1 - rank + 17]) * byte_count

        entities = 8
        rows = [3 * row for row in range(entities + 1)]
        columns = [
            source_id for destination_id in range(entities)
            for source_id in (
                (destination_id - 1) % entities,
                destination_id,
                (destination_id + 1) % entities,
            )
        ]
        graph = gf.Graph.from_csr(
            gf.tensor(rows, dtype=gf.int64, device=device),
            gf.tensor(columns, dtype=gf.int64, device=device),
            num_src=entities, validate="full",
        ).halo(gf.DeviceMesh("cuda", 2), depth=1)
        begin, end = owned_range(entities, 2, rank)
        local_x = gf.tensor(
            [float(entity) for entity in range(begin, end)],
            device=device, requires_grad=True,
        )
        with DistributedRuntime(transport):
            output = NeighborSum()(graph=graph, src={"x": local_x}, dst={})
            output_values = output.tolist()
            gradient = gf.autograd.grad(output.sum(), local_x)
            gradient_values = gradient.tolist()
        queue.put({
            "rank": rank,
            "device": device,
            "byte_correct": byte_correct,
            "samples_ms": samples,
            "output": output_values,
            "gradient": gradient_values,
            "forward_backend": output.execution["backend"],
            "backward_backend": gradient.execution["backend"],
            "nccl_version": transport.version,
        })
    finally:
        stream.close()
        source.close()
        destination.close()
        transport.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bytes", type=int, default=16 << 20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument(
        "--output", type=Path,
        default=Path("output/distributed/nccl_two_gpu/results.json"),
    )
    args = parser.parse_args()
    if args.bytes <= 0 or args.repeats <= 0:
        raise ValueError("bytes and repeats must be positive")
    try:
        capabilities = [
            gf.runtime.cuda_compute_capability(f"cuda:{ordinal}")
            for ordinal in range(2)
        ]
    except RuntimeError as error:
        raise SystemExit(
            "X0 two-GPU gate requires two real CUDA devices; no artifact was written: "
            + str(error)
        ) from error

    communicator_id = nccl_unique_id()
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    processes = [
        context.Process(
            target=_worker,
            args=(rank, communicator_id, args.bytes, args.repeats, queue),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    records = [queue.get(timeout=180) for _ in processes]
    for process in processes:
        process.join(timeout=30)
        if process.exitcode:
            raise SystemExit(f"NCCL rank exited with code {process.exitcode}")
    records.sort(key=lambda record: record["rank"])
    expected_outputs = ([8.0, 3.0, 6.0, 9.0], [12.0, 15.0, 18.0, 13.0])
    correct = all(
        record["byte_correct"]
        and record["output"] == expected_outputs[record["rank"]]
        and record["gradient"] == [3.0] * 4
        and record["forward_backend"] == "cuda-ttir-triton"
        and record["backward_backend"] == "cuda-ttir-triton"
        for record in records
    )
    combined = [sample for record in records for sample in record["samples_ms"]]
    median_ms = statistics.median(combined)
    result = {
        "schema": "graphforge.nccl-two-gpu.v1",
        "world_size": 2,
        "capabilities": capabilities,
        "payload_bytes_per_rank": args.bytes,
        "repeats_per_rank": args.repeats,
        "median_rank_exchange_ms": median_ms,
        "aggregate_unidirectional_payload_GBps": 2 * args.bytes / median_ms / 1e6,
        "records": records,
        "correct": correct,
        "gate": "PASS" if correct else "FAIL",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    from benchmarks.common.diagnostic_plotting import plot_json
    plot_json(args.output)
    (args.output.parent / "REPORT.md").write_text(
        "# Two-GPU NCCL GraphForge gate\n\n"
        f"Gate: **{result['gate']}**. Median rank exchange: {median_ms:.4f} ms. "
        f"Aggregate unidirectional payload bandwidth: "
        f"{result['aggregate_unidirectional_payload_GBps']:.3f} GB/s.\n\n"
        "Both ranks also execute ordinary `Graph.halo()` MessagePassing forward "
        "and compiler-generated reverse halo VJP with native TTIR backends.\n",
        encoding="utf-8",
    )
    print(args.output)
    if not correct:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
