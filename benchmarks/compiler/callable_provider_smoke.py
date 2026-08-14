"""Benchmark-only provider smoke for callable message specialization."""

import torch
import triton
import triton.language as tl


@triton.jit
def diffusion_msg(src, dst, weight):
    return weight * (src - dst)


@triton.jit
def square_msg(src, dst, weight):
    return weight * src * src


@triton.jit
def edge_skeleton(x, src, dst, weight, out, n_edges: tl.constexpr,
                  message_fn: tl.constexpr, block: tl.constexpr):
    edge = tl.program_id(0) * block + tl.arange(0, block)
    mask = edge < n_edges
    source = tl.load(src + edge, mask=mask, other=0)
    destination = tl.load(dst + edge, mask=mask, other=0)
    source_value = tl.load(x + source, mask=mask, other=0.0)
    destination_value = tl.load(x + destination, mask=mask, other=0.0)
    edge_weight = tl.load(weight + edge, mask=mask, other=0.0)
    message = message_fn(source_value, destination_value, edge_weight)
    tl.atomic_add(out + destination, message, mask=mask)


def main():
    num_nodes, degree, block = 4096, 8, 256
    x = torch.rand(num_nodes, device="cuda")
    dst = torch.arange(num_nodes, device="cuda", dtype=torch.int64).repeat_interleave(degree)
    offset = torch.arange(dst.numel(), device="cuda") % degree + 1
    src = (dst + offset) % num_nodes
    weight = torch.rand(src.numel(), device="cuda")

    cases = (
        (diffusion_msg, weight * (x[src] - x[dst])),
        (square_msg, weight * x[src] * x[src]),
    )
    for message_fn, messages in cases:
        out = torch.zeros_like(x)
        edge_skeleton[(triton.cdiv(src.numel(), block),)](
            x, src, dst, weight, out, src.numel(), message_fn, block)
        expected = torch.zeros_like(x).index_add_(0, dst, messages)
        torch.testing.assert_close(out, expected, rtol=2e-5, atol=2e-5)
        print(message_fn.fn.__name__, "ok")


if __name__ == "__main__":
    main()
