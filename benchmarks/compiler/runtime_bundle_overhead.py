"""Measure provider-neutral bundle submission overhead against direct launch."""

from __future__ import annotations

import argparse
import time

import torch

import tiga as tg
from tiga.interop.torch.message_passing import _csr_bundle_runner


class Diffusion(tg.MessagePassing):
    reducer = tg.sum()

    def edge(self, src, dst, edge):
        return edge.weight * (src.x - dst.x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=1 << 17)
    parser.add_argument("--degree", type=int, default=16)
    parser.add_argument("--repeat", type=int, default=500)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    n, degree = args.nodes, args.degree
    row_ptr = torch.arange(
        0, n * degree + 1, degree, device="cuda", dtype=torch.int64)
    col_idx = torch.arange(
        n * degree, device="cuda", dtype=torch.int64).remainder(n)
    graph = tg.Graph.from_csr(row_ptr, col_idx, num_src=n)
    x = torch.rand(n, device="cuda")
    weight = torch.rand(n * degree, device="cuda")
    kernel = Diffusion()
    kernel(
        graph=graph, src={"x": x}, dst={"x": x}, edge={"weight": weight})
    executable = kernel._torch_executor._last_executable
    plan = executable.runner.__closure__[0].cell_contents
    # The closure contains either the bundle or plan depending on cell order;
    # locate the TTIR plan structurally instead of relying on that order.
    for cell in executable.runner.__closure__ or ():
        candidate = cell.cell_contents
        if hasattr(candidate, "launch_into") and hasattr(candidate, "acquire_output"):
            plan = candidate
            break
    output = plan.acquire_output(x, x, weight)
    _, bundle_run = _csr_bundle_runner(plan)

    def device_ms(function) -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        for _ in range(args.repeat):
            function()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / args.repeat

    direct = lambda: plan.launch_into(output=output, inputs=(x, x, weight))
    bundled = lambda: bundle_run(x, x, weight)
    for _ in range(50):
        direct(); bundled()
    torch.cuda.synchronize()
    print(f"direct CUDA elapsed: {device_ms(direct):.6f} ms")
    print(f"bundle CUDA elapsed: {device_ms(bundled):.6f} ms")
    for name, function in (("direct", direct), ("bundle", bundled)):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        for _ in range(args.repeat):
            function()
        elapsed = (time.perf_counter_ns() - start) / args.repeat / 1e3
        torch.cuda.synchronize()
        print(f"{name} host submit: {elapsed:.3f} us")


if __name__ == "__main__":
    main()
