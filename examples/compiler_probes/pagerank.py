"""PageRank as a compiler probe, not a core GraphForge algorithm.

The body is captured once in ``gf_control.repeat``. CPU lowering uses two
loop-carried buffers, so changing the iteration count does not unroll the
MessagePassing expression or grow live Tensor storage.
"""

from __future__ import annotations

import graphforge as gf


class PageRankStep(gf.MessagePassing):
    reducer = gf.sum()

    def edge(self, src, dst, edge):
        del dst
        return src.rank * edge.inverse_out_degree

    def node(self, dst, incoming, damping, base):
        del dst
        return base + damping * incoming


def reference(iterations: int, damping: float) -> list[float]:
    """Small framework-independent oracle for the graph used below."""
    rank = [0.25] * 4
    incoming = ((2,), (0,), (1,), ())
    out_degree = (1, 1, 1, 0)
    for _ in range(iterations):
        dangling_mass = rank[3]
        base = (1.0 - damping) / 4.0 + damping * dangling_mass / 4.0
        rank = [
            base + damping * sum(
                rank[source] / out_degree[source] for source in sources
            )
            for sources in incoming
        ]
    return rank


def run(iterations: int = 20, damping: float = 0.85):
    # Incoming CSR for 0 <- 2, 1 <- 0, 2 <- 1 and dangling vertex 3.
    row_ptr = gf.tensor([0, 1, 2, 3, 3], dtype=gf.int64)
    col_idx = gf.tensor([2, 0, 1], dtype=gf.int64)
    graph = gf.Graph.from_csr(
        row_ptr, col_idx, num_src=4, validate="full")

    rank = gf.tensor([0.25] * 4, dtype=gf.float32)
    inverse_out_degree = gf.tensor([1.0, 1.0, 1.0], dtype=gf.float32)
    dangling = gf.tensor([0.0, 0.0, 0.0, 1.0], dtype=gf.float32)
    step = PageRankStep()

    def body(current):
        dangling_mass = (current * dangling).sum()
        base = (1.0 - damping) / 4.0 + damping * dangling_mass / 4.0
        return step(
            graph=graph,
            src={"rank": current},
            dst={},
            edge={"inverse_out_degree": inverse_out_degree},
            damping=damping,
            base=base,
        )

    rank = gf.repeat(rank, body, iterations=iterations)

    canonical_ir = rank.mlir()
    actual = rank.tolist()
    expected = reference(iterations, damping)
    for got, want in zip(actual, expected):
        if abs(got - want) > 2.0e-6:
            raise AssertionError(f"PageRank mismatch: {actual} != {expected}")
    if abs(sum(actual) - 1.0) > 2.0e-6:
        raise AssertionError(f"PageRank mass is not conserved: {sum(actual)}")

    expression_nodes = sum(
        line.startswith("%") for line in rank.expression().splitlines())
    return actual, expression_nodes, canonical_ir


if __name__ == "__main__":
    values, nodes, canonical_ir = run()
    print("rank:", values)
    print("outer expression nodes:", nodes)
    print("canonical repeat ops:", canonical_ir.count("gf_control.repeat"))
