"""Automatic graph offload: a RAM budget pages a giant graph transparently."""

from __future__ import annotations

import time

import tiga as tg


# --8<-- [start:core]
class Smoothing(tg.MessagePassing):
    reducer = tg.sum()

    def edge(self, src, dst, edge):
        return src.x  # staged edge item: neighbor value, (E_page,)

    def node(self, dst, aggregate):
        return 0.25 * aggregate  # Jacobi average over the 4 neighbors, (N_page,)


def main(grid=(1000, 1000), budget: int = 8 << 20):
    # The 1000x1000 stencil CSR is ~48MB — over the 8MB budget. from_csr
    # persists the topology as .gfg and returns a paged graph instead.
    with tg.runtime.auto_offload(ram=budget):
        graph = tg.Graph.stencil(grid, ((-1, 0), (1, 0), (0, -1), (0, 1)))
        assert graph.schema.realization == "paged_csr"
        x = tg.tensor(                      # (N,) node field — fields stay in RAM
            [float(node % 977) for node in range(graph.schema.num_dst)])
        started = time.perf_counter()
        out = Smoothing()(graph=graph, src={"x": x}, dst={})  # (1_000_000,)
        elapsed = time.perf_counter() - started
# --8<-- [end:core]
    print(
        f"auto-offloaded {graph.schema.num_dst:,} nodes / "
        f"{graph.num_edges:,} edges in {elapsed:.1f}s; "
        f"checksum {sum(out.tolist()):.0f}")


# Inspect:
#   TIGA_GRAPH_RAM_BUDGET=8388608 python examples/auto_offload.py
#   Smoothing()(graph=..., page_rows=50_000, prefetch_depth=4)


if __name__ == "__main__":
    main()
