"""Sparse CPU/GPU benchmark for y[dst] += w * (x[src] - x[dst])."""

from __future__ import annotations

import argparse
import statistics
import time

import torch

try:
    import graphforge as gf
except ImportError:
    gf = None

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only development is supported.
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def edge_atomic_kernel(x, src, dst, weight, out, n_edges: tl.constexpr,
                           block: tl.constexpr):
        e = tl.program_id(0) * block + tl.arange(0, block)
        mask = e < n_edges
        s = tl.load(src + e, mask=mask, other=0)
        d = tl.load(dst + e, mask=mask, other=0)
        w = tl.load(weight + e, mask=mask, other=0.0)
        msg = w * (tl.load(x + s, mask=mask, other=0.0)
                   - tl.load(x + d, mask=mask, other=0.0))
        tl.atomic_add(out + d, msg, mask=mask)


    @triton.jit
    def csr_node_kernel(x, col, weight, out, degree: tl.constexpr,
                        block: tl.constexpr):
        row = tl.program_id(0)
        lane = tl.arange(0, block)
        mask = lane < degree
        e = row * degree + lane
        s = tl.load(col + e, mask=mask, other=0)
        w = tl.load(weight + e, mask=mask, other=0.0)
        center = tl.load(x + row)
        msg = w * (tl.load(x + s, mask=mask, other=0.0) - center)
        tl.store(out + row, tl.sum(msg, axis=0))


def ring_graph(n: int, degree: int, device: str):
    dst = torch.arange(n, device=device, dtype=torch.int64).repeat_interleave(degree)
    offsets = torch.arange(1, degree + 1, device=device, dtype=torch.int64)
    src = (torch.arange(n, device=device, dtype=torch.int64)[:, None] + offsets) % n
    src = src.reshape(-1)
    weight = torch.rand(n * degree, device=device, dtype=torch.float32)
    return src, dst, weight


def reference(x, src, dst, weight):
    out = torch.zeros_like(x)
    out.index_add_(0, dst, weight * (x[src] - x[dst]))
    return out


if gf is not None:

    class GraphForgeDiffusion(gf.MessagePassing):
        reducer = gf.sum()

        def edge(self, src, dst, edge):
            return edge.weight * (src.u - dst.u)


def graphforge_case(x, src, weight, degree):
    nodes = x.numel()
    row_ptr = torch.arange(
        0, src.numel() + 1, degree, device=x.device, dtype=src.dtype)
    graph = gf.Graph.from_csr(row_ptr, src, num_src=nodes)
    kernel = GraphForgeDiffusion()
    return graph, kernel


def csr_reference(x, col, weight, degree):
    rows = torch.arange(x.numel(), device=x.device)[:, None]
    cols = col.view(x.numel(), degree)
    weights = weight.view(x.numel(), degree)
    return (weights * (x[cols] - x[rows])).sum(dim=1)


def median_ms(fn, warmup=5, repeat=20, cuda=False):
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
    samples = []
    for _ in range(repeat):
        start = time.perf_counter_ns()
        fn()
        if cuda:
            torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return statistics.median(samples)


def run_cpu(n: int, degree: int, repeat: int):
    x = torch.rand(n)
    src, dst, weight = ring_graph(n, degree, "cpu")
    expected = reference(x, src, dst, weight)
    torch.testing.assert_close(csr_reference(x, src, weight, degree), expected)
    scatter_ms = median_ms(lambda: reference(x, src, dst, weight), repeat=repeat)
    csr_ms = median_ms(lambda: csr_reference(x, src, weight, degree), repeat=repeat)
    edges = n * degree
    print(f"CPU torch.index_add: {scatter_ms:8.3f} ms  {edges / scatter_ms / 1e6:7.3f} Gedge/s")
    print(f"CPU torch CSR:       {csr_ms:8.3f} ms  {edges / csr_ms / 1e6:7.3f} Gedge/s")
    if gf is not None:
        graph, kernel = graphforge_case(x, src, weight, degree)
        actual = kernel(
            graph=graph, src={"u": x}, dst={"u": x}, edge={"weight": weight})
        torch.testing.assert_close(actual, expected)
        graphforge_ms = median_ms(
            lambda: kernel(
                graph=graph,
                src={"u": x},
                dst={"u": x},
                edge={"weight": weight},
            ),
            repeat=repeat,
        )
        print(f"CPU GraphForge ref:  {graphforge_ms:8.3f} ms  "
              f"{edges / graphforge_ms / 1e6:7.3f} Gedge/s")
    return expected


def run_gpu(n: int, degree: int, repeat: int):
    if not torch.cuda.is_available() or triton is None:
        print("GPU skipped: CUDA and Triton are required")
        return
    x = torch.rand(n, device="cuda")
    src, dst, weight = ring_graph(n, degree, "cuda")
    expected = reference(x, src, dst, weight)
    out_atomic = torch.zeros_like(x)
    out_csr = torch.empty_like(x)
    block_edge = 256
    block_row = triton.next_power_of_2(degree)

    def atomic():
        out_atomic.zero_()
        edge_atomic_kernel[(triton.cdiv(src.numel(), block_edge),)](
            x, src, dst, weight, out_atomic, src.numel(), block_edge)

    def node():
        csr_node_kernel[(n,)](x, src, weight, out_csr, degree, block_row)

    atomic()
    node()
    torch.cuda.synchronize()
    torch.testing.assert_close(out_atomic, expected, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(out_csr, expected, rtol=2e-5, atol=2e-5)

    edges = n * degree
    ref_ms = median_ms(lambda: reference(x, src, dst, weight), repeat=repeat, cuda=True)
    atomic_ms = median_ms(atomic, repeat=repeat, cuda=True)
    node_ms = median_ms(node, repeat=repeat, cuda=True)
    print(f"GPU torch.index_add: {ref_ms:8.3f} ms  {edges / ref_ms / 1e6:7.3f} Gedge/s")
    print(f"GPU Triton atomic:  {atomic_ms:8.3f} ms  {edges / atomic_ms / 1e6:7.3f} Gedge/s")
    print(f"GPU Triton CSR:     {node_ms:8.3f} ms  {edges / node_ms / 1e6:7.3f} Gedge/s")
    if gf is not None:
        graph, kernel = graphforge_case(x, src, weight, degree)
        actual = kernel(
            graph=graph, src={"u": x}, dst={"u": x}, edge={"weight": weight})
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
        graphforge_ms = median_ms(
            lambda: kernel(
                graph=graph,
                src={"u": x},
                dst={"u": x},
                edge={"weight": weight},
            ),
            repeat=repeat,
            cuda=True,
        )
        print(f"GPU GraphForge ref: {graphforge_ms:8.3f} ms  "
              f"{edges / graphforge_ms / 1e6:7.3f} Gedge/s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=1 << 18)
    parser.add_argument("--degree", type=int, default=16)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.nodes, args.repeat = 1 << 16, 8
    print(f"nodes={args.nodes:,} degree={args.degree} edges={args.nodes * args.degree:,}")
    run_cpu(args.nodes, args.degree, args.repeat)
    run_gpu(args.nodes, args.degree, args.repeat)


if __name__ == "__main__":
    main()
