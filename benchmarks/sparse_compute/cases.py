"""Sparse graph generators shared by sparse-compute workloads."""

from __future__ import annotations

import torch


def make_graph(
    nodes: int,
    degree: int,
    topology: str,
    locality: str,
    device: torch.device,
    index_dtype: torch.dtype = torch.int64,
):
    generator = torch.Generator(device=device).manual_seed(20260807)
    if topology == "regular":
        degrees = torch.full((nodes,), degree, device=device, dtype=index_dtype)
    elif topology == "irregular":
        degrees = torch.randint(
            0, 2 * degree + 1, (nodes,), device=device,
            generator=generator, dtype=index_dtype)
    elif topology == "skewed":
        bucket = torch.rand((nodes,), device=device, generator=generator)
        degrees = torch.where(
            bucket < 0.75,
            torch.full_like(bucket, max(1, degree // 2), dtype=index_dtype),
            torch.where(
                bucket < 0.95,
                torch.full_like(bucket, 2 * degree, dtype=index_dtype),
                torch.full_like(bucket, 4 * degree, dtype=index_dtype),
            ),
        )
    elif topology == "powerlaw":
        # A compact social-network-like tail: the default degree=16 case is
        # 90% degree 8, 9% degree 64, and 1% degree 256.  This deliberately
        # exercises bounded rows and hub splitting in one relation.
        bucket = torch.rand((nodes,), device=device, generator=generator)
        degrees = torch.where(
            bucket < 0.90,
            torch.full_like(bucket, max(1, degree // 2), dtype=index_dtype),
            torch.where(
                bucket < 0.99,
                torch.full_like(bucket, 4 * degree, dtype=index_dtype),
                torch.full_like(bucket, 16 * degree, dtype=index_dtype),
            ),
        )
    else:
        raise ValueError(topology)
    row_ptr = torch.empty(nodes + 1, device=device, dtype=index_dtype)
    row_ptr[0] = 0
    torch.cumsum(degrees, 0, out=row_ptr[1:])
    edges = int(row_ptr[-1].item())
    dst = torch.repeat_interleave(
        torch.arange(nodes, device=device, dtype=index_dtype), degrees)
    if locality == "local":
        ordinal = torch.arange(edges, device=device, dtype=index_dtype)
        src = (dst + 1 + ordinal.remainder(max(1, degree))) % nodes
    elif locality == "random":
        src = torch.randint(
            nodes, (edges,), device=device, generator=generator,
            dtype=index_dtype)
    else:
        raise ValueError(locality)
    return row_ptr, src, dst
