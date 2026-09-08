"""Matrix-free 1D P1 FEM Poisson solve through MessagePassing and CG."""

from __future__ import annotations

import tiga as gf
from solvers import linear_solve


# --8<-- [start:core]
class StiffnessApply(gf.MessagePassing):
    reducer = gf.sum()

    def edge(self, src, dst, edge):
        return edge.value * src.u


def solve(
    interior_nodes: int = 8,
    *,
    iterations: int | None = None,
    tolerance: float | None = 1.0e-6,
    max_iterations: int = 100,
):
    """Solve Au = h·1 without assembling A.

    Default is a residual-driven ``gf_control.while``; pass ``iterations=k``
    for a fixed ``gf_control.repeat`` loop instead.
    """
    kernel, graph, stiffness, spacing = poisson_operator(interior_nodes)
    load = gf.tensor(
        [spacing] * interior_nodes, dtype=gf.float32, requires_grad=True)  # (N,)
    if iterations is not None:  # fixed repeat and bounded while are exclusive
        tolerance = max_iterations = None
    solution = linear_solve(
        kernel,
        load,
        method="cg",
        graph=graph,
        field="u",
        edge={"value": stiffness},
        iterations=iterations,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )  # (N,)
    coordinates = [(index + 1) * spacing for index in range(interior_nodes)]
    exact = [0.5 * x * (1.0 - x) for x in coordinates]
    error = max(abs(actual - expected) for actual, expected in zip(
        solution.tolist(), exact, strict=True))
    return solution, exact, error


def poisson_operator(interior_nodes: int):
    """P1 stiffness stencil [-1/h, 2/h, -1/h] as a Graph + edge values."""
    if interior_nodes < 1:
        raise ValueError("interior_nodes must be positive")
    spacing = 1.0 / (interior_nodes + 1)
    graph = gf.Graph.stencil((interior_nodes,), ((-1,), (0,), (1,)))
    row_ptr, col_idx = graph.resolve_csr()     # (N+1,), (E,) CSR, boundary-truncated
    destination = graph.destination_index(row_ptr)
    values = [                                 # edge values follow the CSR order:
        (2.0 if source == target else -1.0) / spacing  # diagonal 2/h, off-diagonal -1/h
        for source, target in zip(
            col_idx.tolist(), destination.tolist(), strict=True)
    ]
    stiffness = gf.tensor(values, dtype=gf.float32, requires_grad=True)  # (E,)
    return StiffnessApply(), graph, stiffness, spacing
# --8<-- [end:core]


def load_gradient(interior_nodes: int = 4):
    """Differentiate sum(u) w.r.t. the load through the captured iterations."""
    kernel, graph, stiffness, spacing = poisson_operator(interior_nodes)
    load = gf.tensor(
        [spacing] * interior_nodes, dtype=gf.float32, requires_grad=True)  # (N,)
    solution = linear_solve(
        kernel,
        load,
        method="cg",
        graph=graph,
        field="u",
        edge={"value": stiffness},
        iterations=(interior_nodes + 1) // 2,
    )  # (N,)
    return gf.autograd.grad(solution.sum(), load)                          # (N,)


# Inspect: solve()[0].mlir() (bounded gf_control.while loop),
#   solve(iterations=4)[0].mlir() (fixed gf_control.repeat loop)

if __name__ == "__main__":
    solve()
    load_gradient()
