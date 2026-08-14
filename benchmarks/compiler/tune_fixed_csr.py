"""Tune the benchmark-only fixed-degree CSR oracle on the local CUDA device.

This is a lowering experiment, not an autotuner exposed to users.  It searches
row × neighbor × feature tiles for the regular-degree specialization that the
planner can guard with graph statistics.
"""

from __future__ import annotations

import argparse
import statistics

import torch
import triton
import triton.language as tl


@triton.jit
def fixed_csr_spmm_kernel(
    col_idx,
    weight,
    x,
    out,
    n_rows: tl.constexpr,
    degree: tl.constexpr,
    n_features: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    neighbors = tl.arange(0, BLOCK_D)
    features = tl.program_id(1) * BLOCK_F + tl.arange(0, BLOCK_F)

    edge = rows[:, None] * degree + neighbors[None, :]
    edge_mask = (rows[:, None] < n_rows) & (neighbors[None, :] < degree)
    source = tl.load(col_idx + edge, mask=edge_mask, other=0)
    edge_weight = tl.load(weight + edge, mask=edge_mask, other=0.0)

    value_offset = (
        source[:, :, None] * n_features + features[None, None, :])
    value_mask = edge_mask[:, :, None] & (features[None, None, :] < n_features)
    value = tl.load(x + value_offset, mask=value_mask, other=0.0)
    result = tl.sum(value * edge_weight[:, :, None], axis=1)

    output_offset = rows[:, None] * n_features + features[None, :]
    output_mask = (rows[:, None] < n_rows) & (features[None, :] < n_features)
    tl.store(out + output_offset, result, mask=output_mask)


def ring_csr(nodes: int, degree: int, features: int):
    device = torch.device("cuda")
    row_ptr = torch.arange(
        0, nodes * degree + 1, degree, device=device, dtype=torch.int64)
    offsets = torch.arange(1, degree + 1, device=device, dtype=torch.int64)
    col_idx = (
        torch.arange(nodes, device=device, dtype=torch.int64)[:, None]
        + offsets
    ).remainder(nodes).reshape(-1)
    weight = torch.rand(col_idx.numel(), device=device)
    x = torch.rand((nodes, features), device=device)
    csr = torch.sparse_csr_tensor(
        row_ptr, col_idx, weight, size=(nodes, nodes), check_invariants=False)
    return row_ptr, col_idx, weight, x, csr


def samples_ms(function, repeat: int):
    for _ in range(5):
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def configs_for(features: int):
    block_fs = sorted({min(features, value) for value in (1, 8, 16, 32, 64)})
    for block_m in (1, 2, 4, 8, 16, 32):
        for block_f in block_fs:
            if block_m * 16 * block_f > 16384:
                continue
            for num_warps in (1, 2, 4, 8):
                yield block_m, block_f, num_warps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=1 << 17)
    parser.add_argument("--degree", type=int, default=16)
    parser.add_argument("--features", default="1,16,64")
    parser.add_argument("--repeat", type=int, default=30)
    args = parser.parse_args()
    device_name = torch.cuda.get_device_name()
    print(f"device={device_name} nodes={args.nodes:,} degree={args.degree}")

    for features in (int(value) for value in args.features.split(",")):
        _, col_idx, weight, x, csr = ring_csr(
            args.nodes, args.degree, features)
        expected = torch.sparse.mm(csr, x)
        baseline_samples = samples_ms(lambda: torch.sparse.mm(csr, x), args.repeat)
        baseline = statistics.median(baseline_samples)
        candidates = []
        for block_m, block_f, num_warps in configs_for(features):
            output = torch.empty_like(x)
            grid = (triton.cdiv(args.nodes, block_m),
                    triton.cdiv(features, block_f))

            def launch():
                fixed_csr_spmm_kernel[grid](
                    col_idx, weight, x, output,
                    n_rows=args.nodes,
                    degree=args.degree,
                    n_features=features,
                    BLOCK_M=block_m,
                    BLOCK_D=triton.next_power_of_2(args.degree),
                    BLOCK_F=block_f,
                    num_warps=num_warps,
                )

            try:
                launch()
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    output, expected, rtol=3e-4, atol=3e-4)
                elapsed = statistics.median(samples_ms(launch, args.repeat))
            except Exception as error:
                print(
                    f"F={features:3d} M={block_m:2d} BF={block_f:2d} "
                    f"W={num_warps}: SKIP {type(error).__name__}: {error}")
                continue
            candidates.append((elapsed, block_m, block_f, num_warps))
        candidates.sort()
        print(f"F={features:3d} torch.sparse={baseline:.4f} ms")
        for elapsed, block_m, block_f, num_warps in candidates[:8]:
            print(
                f"  {elapsed:.4f} ms  {baseline / elapsed:.3f}x  "
                f"BLOCK_M={block_m:2d} BLOCK_F={block_f:2d} warps={num_warps}")


if __name__ == "__main__":
    main()
