"""Differential accuracy checks for sparse MessagePassing programs."""

from __future__ import annotations

import argparse

import torch

import tiga as tg


class Diffusion(tg.MessagePassing):
    reducer = tg.sum()

    def edge(self, src, dst, edge):
        return edge.weight * (src.u - dst.u)


class NonlinearVector(tg.MessagePassing):
    reducer = tg.sum()

    def edge(self, src, dst, edge, eps):
        delta = src.u - dst.u
        scale = edge.weight / torch.sqrt((delta * delta).sum(-1, keepdim=True) + eps)
        return scale * delta

    def node(self, dst, force, dt):
        return dst.u + dt * force


def irregular_csr(nodes: int, max_degree: int, device: str):
    generator = torch.Generator(device=device).manual_seed(20260807)
    degree = torch.randint(
        0, max_degree + 1, (nodes,), device=device, generator=generator)
    row_ptr = torch.empty(nodes + 1, device=device, dtype=torch.int64)
    row_ptr[0] = 0
    torch.cumsum(degree, 0, out=row_ptr[1:])
    edges = int(row_ptr[-1].item())
    col_idx = torch.randint(
        0, nodes, (edges,), device=device, generator=generator)
    return row_ptr, col_idx


def check(device: str, nodes: int, max_degree: int):
    row_ptr, col_idx = irregular_csr(nodes, max_degree, device)
    graph = tg.Graph.from_csr(row_ptr, col_idx, num_src=nodes)
    dst_index = graph.destination_index()
    weight = torch.randn(col_idx.numel(), device=device)

    scalar = torch.randn(nodes, device=device)
    actual = Diffusion()(
        graph=graph,
        src={"u": scalar},
        dst={"u": scalar},
        edge={"weight": weight},
    )
    expected = torch.zeros_like(scalar)
    expected.index_add_(
        0, dst_index, weight * (scalar[col_idx] - scalar[dst_index]))
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

    vector = torch.randn(nodes, 3, device=device)
    delta = vector[col_idx] - vector[dst_index]
    message = weight[:, None] * delta / torch.sqrt(
        (delta * delta).sum(-1, keepdim=True) + 1e-5)
    force = torch.zeros_like(vector)
    force.index_add_(0, dst_index, message)
    expected_vector = vector + 0.01 * force
    actual_vector = NonlinearVector()(
        graph=graph,
        src={"u": vector},
        dst={"u": vector},
        edge={"weight": weight[:, None]},
        eps=1e-5,
        dt=0.01,
    )
    torch.testing.assert_close(
        actual_vector, expected_vector, rtol=2e-5, atol=2e-5)

    max_abs = max(
        (actual - expected).abs().max().item(),
        (actual_vector - expected_vector).abs().max().item(),
    )
    print(
        f"{device:>4}: nodes={nodes:,} edges={col_idx.numel():,} "
        f"scalar+nonlinear-vector passed, max_abs={max_abs:.3e}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=4096)
    parser.add_argument("--max-degree", type=int, default=32)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.nodes, args.max_degree = 1024, 12
    check("cpu", args.nodes, args.max_degree)
    if torch.cuda.is_available():
        check("cuda", args.nodes, args.max_degree)
    else:
        print("cuda: skipped (CUDA unavailable)")


if __name__ == "__main__":
    main()
