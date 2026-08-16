"""Matrix-free 1D P1 FEM Poisson solve through MessagePassing and CG.

The mesh topology is a Graph, stiffness application is a user UDF, and the
four-state solver loop is captured once as gf_control.repeat. No sparse matrix
is assembled and the UDF participates in automatic algorithmic differentiation.
Run with ``PYTHONPATH=python python examples/fem_poisson.py``.
"""

from __future__ import annotations

import graphforge as gf
from solvers import cg


class StiffnessApply(gf.MessagePassing):
    reducer = gf.sum()

    def edge(self, src, dst, edge):
        del dst
        return edge.value * src.u


def poisson_operator(interior_nodes: int):
    """Build the P1 stiffness relation for a uniform unit-interval mesh."""
    if interior_nodes < 1:
        raise ValueError("interior_nodes must be positive")
    spacing = 1.0 / (interior_nodes + 1)
    row_ptr = [0]
    col_idx: list[int] = []
    values: list[float] = []
    for destination in range(interior_nodes):
        if destination > 0:
            col_idx.append(destination - 1)
            values.append(-1.0 / spacing)
        col_idx.append(destination)
        values.append(2.0 / spacing)
        if destination + 1 < interior_nodes:
            col_idx.append(destination + 1)
            values.append(-1.0 / spacing)
        row_ptr.append(len(col_idx))

    graph = gf.Graph.from_csr(
        gf.tensor(row_ptr, dtype=gf.int64),
        gf.tensor(col_idx, dtype=gf.int64),
        num_src=interior_nodes,
    )
    stiffness = gf.tensor(values, dtype=gf.float32, requires_grad=True)
    return StiffnessApply(), graph, stiffness, spacing


def run(interior_nodes: int = 8, iterations: int | None = None):
    kernel, graph, stiffness, spacing = poisson_operator(interior_nodes)
    load = gf.tensor(
        [spacing] * interior_nodes,
        dtype=gf.float32,
        requires_grad=True,
    )
    # The constant load and symmetric mesh occupy ceil(n/2) eigenmodes. Stop
    # there: fixed-count CG has no residual guard and intentionally exposes
    # exact-convergence breakdown instead of hiding a host-side tolerance test.
    steps = (interior_nodes + 1) // 2 if iterations is None else iterations
    solution = cg(
        kernel,
        load,
        graph=graph,
        field="u",
        edge={"value": stiffness},
        iterations=steps,
    )
    coordinates = [(index + 1) * spacing for index in range(interior_nodes)]
    exact = [0.5 * x * (1.0 - x) for x in coordinates]
    error = max(abs(actual - expected) for actual, expected in zip(
        solution.tolist(), exact, strict=True))
    return solution, exact, error


def run_until_converged(
    interior_nodes: int = 8,
    *,
    tolerance: float = 1.0e-6,
    max_iterations: int = 100,
):
    """Solve with a residual condition captured as bounded device control."""
    kernel, graph, stiffness, spacing = poisson_operator(interior_nodes)
    load = gf.tensor([spacing] * interior_nodes, dtype=gf.float32)
    solution = cg(
        kernel,
        load,
        graph=graph,
        field="u",
        edge={"value": stiffness},
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    coordinates = [(index + 1) * spacing for index in range(interior_nodes)]
    exact = [0.5 * x * (1.0 - x) for x in coordinates]
    error = max(abs(actual - expected) for actual, expected in zip(
        solution.tolist(), exact, strict=True))
    return solution, exact, error


def load_gradient(interior_nodes: int = 4):
    """Differentiate through the captured iterations (algorithmic VJP)."""
    kernel, graph, stiffness, spacing = poisson_operator(interior_nodes)
    load = gf.tensor(
        [spacing] * interior_nodes, dtype=gf.float32, requires_grad=True)
    solution = cg(
        kernel,
        load,
        graph=graph,
        field="u",
        edge={"value": stiffness},
        iterations=(interior_nodes + 1) // 2,
    )
    return gf.autograd.grad(solution.sum(), load)


if __name__ == "__main__":
    result, expected, maximum_error = run()
    print("solution:", [round(value, 6) for value in result.tolist()])
    print("exact:   ", [round(value, 6) for value in expected])
    print(f"max error: {maximum_error:.3e}")
    print("d sum(u) / d load:", load_gradient().tolist())
    converged, _, converged_error = run_until_converged()
    print(f"bounded-while max error: {converged_error:.3e}")
    print("uses gf_control.while:", "gf_control.while" in converged.mlir())
    print(result.mlir())
