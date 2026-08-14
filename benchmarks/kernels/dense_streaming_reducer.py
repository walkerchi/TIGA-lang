"""Benchmark-only oracle for a tiled Cartesian streaming reducer.

This deliberately lives outside ``python/graphforge``. It provides a target
shape and provider-TTIR inspection oracle while the compiler's generic
``DenseLaunch + reducer regions`` lowering is developed.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _kernel(
    lhs, rhs, payload, output, scale,
    rows: tl.constexpr, query_lanes: tl.constexpr, source_lanes: tl.constexpr,
    width: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    tile = tl.program_id(0)
    lane = tl.program_id(1)
    lane_base = lane * rows * width
    source_lane = lane // (query_lanes // source_lanes)
    source_lane_base = source_lane * rows * width
    lhs_ptr = tl.make_block_ptr(
        base=lhs + lane_base, shape=(rows, width), strides=(width, 1),
        offsets=(tile * BLOCK_M, 0), block_shape=(BLOCK_M, width),
        order=(1, 0),
    )
    lhs_tile = tl.load(lhs_ptr, boundary_check=(0,), padding_option="zero")
    maximum = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    normalizer = tl.zeros((BLOCK_M,), tl.float32)
    state = tl.zeros((BLOCK_M, width), tl.float32)
    log2_scale = scale * 1.4426950408889634
    loop_end = rows
    if CAUSAL:
        loop_end = tl.minimum((tile + 1) * BLOCK_M, rows)
    for start in tl.range(0, loop_end, BLOCK_N):
        rhs_ptr = tl.make_block_ptr(
            base=rhs + source_lane_base, shape=(rows, width), strides=(width, 1),
            offsets=(start, 0), block_shape=(BLOCK_N, width), order=(1, 0),
        )
        rhs_tile = tl.load(
            rhs_ptr, boundary_check=(0,), padding_option="zero")
        message = tl.dot(lhs_tile, tl.trans(rhs_tile)) * log2_scale
        sources = start + tl.arange(0, BLOCK_N)
        valid = sources[None, :] < rows
        if CAUSAL:
            queries = tile * BLOCK_M + tl.arange(0, BLOCK_M)
            valid &= sources[None, :] <= queries[:, None]
        message = tl.where(valid, message, -float("inf"))
        next_maximum = tl.maximum(maximum, tl.max(message, axis=1))
        weight = tl.math.exp2(message - next_maximum[:, None])
        correction = tl.math.exp2(maximum - next_maximum)
        normalizer = normalizer * correction + tl.sum(weight, axis=1)
        value_ptr = tl.make_block_ptr(
            base=payload + source_lane_base, shape=(rows, width), strides=(width, 1),
            offsets=(start, 0), block_shape=(BLOCK_N, width), order=(1, 0),
        )
        value_tile = tl.load(
            value_ptr, boundary_check=(0,), padding_option="zero")
        state *= correction[:, None]
        state = tl.dot(weight.to(tl.float16), value_tile, state)
        maximum = next_maximum
    result = state / normalizer[:, None]
    out_ptr = tl.make_block_ptr(
        base=output + lane_base, shape=(rows, width), strides=(width, 1),
        offsets=(tile * BLOCK_M, 0), block_shape=(BLOCK_M, width),
        order=(1, 0),
    )
    tl.store(out_ptr, result.to(tl.float16), boundary_check=(0,))


def launch(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    payload: torch.Tensor,
    scale: float,
    *,
    block_m: int = 128,
    block_n: int = 64,
    causal: bool = False,
):
    """Inputs are physical ``[lanes, rows, width]`` tensors."""
    lanes, rows, width = lhs.shape
    source_lanes = rhs.shape[0]
    if payload.shape != rhs.shape or lanes % source_lanes:
        raise ValueError("source K/V lanes must match and divide query lanes")
    output = torch.empty_like(lhs)
    compiled = _kernel[(triton.cdiv(rows, block_m), lanes)](
        lhs, rhs, payload, output, scale,
        rows=rows, query_lanes=lanes, source_lanes=source_lanes, width=width,
        BLOCK_M=block_m, BLOCK_N=block_n,
        CAUSAL=causal,
        num_warps=4, num_stages=3,
    )
    return output, compiled
