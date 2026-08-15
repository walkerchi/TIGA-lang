"""Sparse graph generators shared by sparse-compute workloads."""

from __future__ import annotations

import torch

TOPOLOGIES = (
    "regular", "irregular", "skewed", "powerlaw", "lognormal",
    "exponential",
)


def degree_statistics(row_ptr: torch.Tensor) -> dict[str, float | int | str]:
    """Return portable distribution metadata for a generated CSR relation."""
    degrees = (row_ptr[1:] - row_ptr[:-1]).to(torch.float64)
    if degrees.numel() == 0:
        return {
            "family": "empty", "minimum": 0, "mean": 0.0,
            "p50": 0.0, "p95": 0.0, "p99": 0.0, "maximum": 0,
            "zero_fraction": 0.0, "coefficient_of_variation": 0.0,
        }
    quantiles = torch.quantile(
        degrees, torch.tensor(
            [0.5, 0.95, 0.99], device=degrees.device,
            dtype=degrees.dtype,
        ),
    )
    mean = float(degrees.mean().item())
    std = float(degrees.std(correction=0).item())
    return {
        "minimum": int(degrees.min().item()),
        "mean": mean,
        "p50": float(quantiles[0].item()),
        "p95": float(quantiles[1].item()),
        "p99": float(quantiles[2].item()),
        "maximum": int(degrees.max().item()),
        "zero_fraction": float((degrees == 0).to(torch.float64).mean().item()),
        "coefficient_of_variation": std / mean if mean else 0.0,
    }


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
    elif topology in {"lognormal", "exponential"}:
        # Continuous, reproducible families complement the deliberately
        # discrete power-law regression bucket above.  Both are rescaled to
        # the requested mean degree before rounding, and their explicit cap
        # prevents one sample from making a benchmark allocation unbounded.
        cap = max(8, 16 * degree)
        if topology == "lognormal":
            raw = torch.exp(torch.randn(
                (nodes,), device=device, generator=generator,
                dtype=torch.float64,
            ) * 1.25)
        else:
            uniform = torch.rand(
                (nodes,), device=device, generator=generator,
                dtype=torch.float64,
            ).clamp_(min=torch.finfo(torch.float64).eps, max=1.0 - 1e-12)
            raw = -torch.log1p(-uniform)
        scaled = raw * (float(degree) / float(raw.mean().item()))
        degrees = scaled.round().clamp_(0, cap).to(index_dtype)
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
