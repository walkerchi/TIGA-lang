"""Matched radius index/reuse/rebuild comparison; one provider per process.

Both providers compute sum_j distance(i,j)*x[j], with exact cutoff and no
self edges. No CSR caching, truncation, or topology construction hidden in the
rebuild phase. Warp is an optional benchmark dependency, not a Tiga dependency.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
from datetime import datetime, timezone
import subprocess

import torch
import tiga as tg


class DistanceSum(tg.MessagePassing):
    reducer = tg.sum()

    def edge(self, src, dst, edge):
        return src.x * edge.distance


def warp_kernel(wp):
    # Module-global name is required by Warp's code generator.
    globals()["wp"] = wp

    @wp.kernel
    def radius_sum(grid: wp.uint64, points: wp.array(dtype=wp.vec3),
                   x: wp.array(dtype=float), cutoff: float,
                   output: wp.array(dtype=float)):
        i = wp.hash_grid_point_id(grid, wp.tid())
        p = points[i]
        total = float(0.)
        query = wp.hash_grid_query(grid, p, cutoff)
        j = int(0)
        while wp.hash_grid_query_next(query, j):
            delta = p - points[j]
            squared = wp.dot(delta, delta)
            if j != i and squared <= cutoff * cutoff:
                total += wp.sqrt(squared) * x[j]
        output[i] = total

    return radius_sum


def samples(fn, prepare, repeats):
    for step in range(5):
        prepare(step)
        fn()
    torch.cuda.synchronize()
    wall, gpu = [], []
    for step in range(repeats):
        prepare(step)
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin = time.perf_counter()
        start.record()
        fn()
        end.record()
        end.synchronize()
        wall.append((time.perf_counter() - begin) * 1000)
        gpu.append(start.elapsed_time(end))
    return {"wall_ms": statistics.median(wall), "gpu_ms": statistics.median(gpu),
            "wall_samples_ms": wall, "gpu_samples_ms": gpu}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("tiga", "tiga-dense", "warp"), required=True)
    parser.add_argument("--nodes", type=int, default=32768)
    parser.add_argument("--degree", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--distribution", choices=("uniform", "clustered", "sparse"), default="uniform")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.nodes, args.degree, args.repeats) < 1:
        parser.error("nodes, degree and repeats must be positive")
    # Keep both methods matrix-free even in the fixed-position phase.
    os.environ["TIGA_RADIUS_CACHE_BYTES"] = "0"
    torch.set_num_threads(1)
    rng = torch.Generator(device="cuda").manual_seed(20260920)
    snapshots = [(torch.rand(args.nodes, 3, generator=rng, device="cuda") * 1024).floor() / 1024 for _ in range(2)]
    if args.distribution == "clustered":
        snapshots = [p * .25 for p in snapshots]
    elif args.distribution == "sparse":
        # Disconnected translated clusters: large bounds, unchanged local scale.
        groups = (torch.arange(args.nodes, device="cuda") % 8).float()[:, None]
        snapshots = [p + groups * 64 for p in snapshots]
    cutoff = (args.degree / (args.nodes * 4 * math.pi / 3)) ** (1/3)
    if args.distribution == "clustered":
        cutoff *= .25
    if args.distribution == "sparse":
        cutoff *= 2
    # Put the cutoff between representable squared-distance levels.
    quantum = 4096 if args.distribution == "clustered" else 1024
    cutoff = math.sqrt(math.floor((cutoff * quantum)**2) + .5) / quantum
    positions = snapshots[0].clone()
    x = torch.rand(args.nodes, generator=rng, device="cuda")
    digests = [hashlib.sha256(p.cpu().numpy().tobytes()).hexdigest() for p in snapshots]

    # Full reference outside the timed/memory regions; no neighbor count cap.
    expected, edge_counts = [], []
    for p in snapshots:
        reference = tg.Graph.radius(p, cutoff)
        rows, cols = reference.resolve_csr()
        dst = reference.destination_index(rows)
        values = (p[cols] - p[dst]).norm(dim=-1) * x[cols]
        expected.append(torch.zeros_like(x).index_add_(0, dst, values).cpu())
        edge_counts.append(cols.numel())
        del reference, rows, cols, dst, values
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch_baseline = torch.cuda.memory_allocated()

    warp_version = None
    warp_pool_initial_high = None
    if args.provider == "warp":
        import warp as wp
        wp.init()
        warp_version = wp.__version__
        wp.set_stream(wp.stream_from_torch(torch.cuda.current_stream()))
        warp_pool_initial_high = wp.get_mempool_used_mem_high("cuda")
        points, features = wp.from_torch(positions, dtype=wp.vec3), wp.from_torch(x)
        side = 4
        while side ** 3 < max(1, (args.nodes + 7)//8):
            side *= 2
        grid = wp.HashGrid(side, side, side, device="cuda")
        output = wp.empty(args.nodes, dtype=float, device="cuda")
        kernel = warp_kernel(wp)

        def build():
            grid.build(points, cutoff)

        def consume():
            wp.launch(kernel, args.nodes, inputs=[grid.id, points, features, cutoff], outputs=[output])
            return wp.to_torch(output)
    else:
        from tiga.interop.torch.graph import from_native
        graph = tg.Graph.radius(positions, cutoff)
        view = from_native(graph)
        view._directory_mode = "dense" if args.provider == "tiga-dense" else "auto"
        kernel = DistanceSum()

        def build():
            directory = view.generated_cell_directory()
            if directory is None:
                raise RuntimeError("no matrix-free directory for this distribution")

        def consume():
            return kernel(graph=graph, src={"x": x}, dst={})

    def prepare(step):
        positions.copy_(snapshots[step % 2])

    def rebuild_consume():
        build()
        return consume()

    errors = []
    for step in range(2):
        prepare(step)
        actual = rebuild_consume().detach().cpu()
        torch.testing.assert_close(actual, expected[step], rtol=2e-5, atol=2e-5)
        errors.append(float((actual - expected[step]).abs().max()))
    if args.provider != "warp":
        if "generated-radius" not in kernel.last_variant.lowering:
            raise RuntimeError(f"expected matrix-free lowering, got {kernel.last_variant.lowering}")
        if view._radius_csr_cache is not None:
            raise RuntimeError("benchmark unexpectedly materialized CSR")
    result = {
        "protocol": "radius-warp-v1", "provider": args.provider,
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "driver": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip(),
        "warp": warp_version, "nodes": args.nodes, "cutoff": cutoff,
        "seed": 20260920, "target_degree": args.degree, "repeats": args.repeats,
        "distribution": args.distribution, "edges": edge_counts,
        "snapshot_sha256": digests, "max_abs_error": errors,
        "semantics": "sum(distance * x), FP32, exclude self, <= cutoff, no CSR cache",
        "timing": "warm synchronized wall and CUDA events; coordinate copy outside timed region; both query in bucket order",
        "phases": {},
    }
    root = Path(__file__).resolve().parents[2]
    result["source_sha256"] = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in (
            "benchmarks/graph_operations/warp_radius.py",
            "python/tiga/interop/torch/radius_grid.py",
            "python/tiga/interop/torch/graph.py",
            "python/tiga/interop/torch/message_passing.py",
            "lib/Target/Triton/Translate.cpp",
            "lib/Transforms/SelectKernelSchedule.cpp",
        )
    }
    for name, fn, setup in (
        ("index_build", build, prepare),
        ("fixed_index_compute", consume, lambda _: None),
        ("rebuild_and_compute", rebuild_consume, prepare),
    ):
        # Previous phase ends on a valid index; fixed reuse never rebuilds it.
        result["phases"][name] = samples(fn, setup, args.repeats)
    torch.cuda.synchronize()
    torch_peak = torch.cuda.max_memory_allocated()
    if args.provider == "warp":
        warp_peak = wp.get_mempool_used_mem_high("cuda")
        result["memory"] = {
            "torch_peak_bytes": torch_peak, "warp_pool_peak_bytes": warp_peak,
            "combined_peak_upper_bound_bytes": torch_peak + warp_peak,
            "warp_pool_initial_high_bytes": warp_pool_initial_high,
            "torch_inputs_constant": torch_peak == torch_baseline,
        }
        # Warp initializes its allocator with a one-byte probe, already
        # included in the high-water total; reject any unrelated pool history.
        if warp_pool_initial_high > 1 or torch_peak != torch_baseline:
            raise RuntimeError(f"fresh-process memory protocol violated: {result['memory']}, baseline={torch_baseline}")
    else:
        result["memory"] = {"torch_peak_bytes": torch_peak,
                            "combined_peak_upper_bound_bytes": torch_peak}
    result["memory_note"] = (
        "Allocated buffers from provider initialization through all phases, including inputs and scratch; "
        "excludes driver/module memory and reserved free blocks. Warp uses its CUDA-pool high-water "
        "counter plus constant Torch-owned inputs, not Torch's counter alone."
    )
    if args.provider != "warp":
        directory = view.generated_cell_directory()
        result["index_kind"] = "modular_hash_grid" if directory.hash_grid else "dense_cell_directory"
        result["lowering"] = kernel.last_variant.lowering
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
