"""PageRank as a compiler probe, not a core Tiga algorithm."""

from __future__ import annotations

import tiga as gf


def run(iterations: int = 20, damping: float = 0.85):
    # Incoming CSR for 0 <- 2, 1 <- 0, 2 <- 1 and dangling vertex 3.
    row_ptr = gf.tensor([0, 1, 2, 3, 3], dtype=gf.int64)      # (N+1,)
    col_idx = gf.tensor([2, 0, 1], dtype=gf.int64)            # (E,)
    graph = gf.Graph.from_csr(
        row_ptr, col_idx, num_src=4, validate="full")

    rank = gf.tensor([0.25] * 4, dtype=gf.float32)            # (N,)
    inverse_out_degree = gf.tensor([1.0, 1.0, 1.0], dtype=gf.float32)  # (E,)
    dangling = gf.tensor([0.0, 0.0, 0.0, 1.0], dtype=gf.float32)       # (N,)

    # --8<-- [start:core]
    class PageRankStep(gf.MessagePassing):
        reducer = gf.sum()

        def edge(self, src, dst, edge):
            return src.rank * edge.inverse_out_degree

        def node(self, dst, incoming, damping, base):
            return base + damping * incoming

    step = PageRankStep()

    @gf.jit
    def iterate(rank, inverse_out_degree, dangling, damping, iterations):
        for _ in range(iterations):
            dangling_mass = (rank * dangling).sum()
            base = (1.0 - damping) / 4.0 + damping * dangling_mass / 4.0
            rank = step(
                graph=graph,
                src={"rank": rank},
                dst={},
                edge={"inverse_out_degree": inverse_out_degree},
                damping=damping,
                base=base,
            )
        return rank

    rank = iterate(rank, inverse_out_degree, dangling, damping, iterations)
    # --8<-- [end:core]

    canonical_ir = rank.mlir()
    actual = rank.tolist()

    expression_nodes = sum(
        line.startswith("%") for line in rank.expression().splitlines())
    return actual, expression_nodes, canonical_ir


if __name__ == "__main__":
    run()

# Inspect: rank.expression(), rank.mlir()  (via run()'s return values)
