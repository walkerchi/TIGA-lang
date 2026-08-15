"""Matrix-free 1D P1 FEM Poisson solve through MessagePassing.

The mesh topology is a Graph, stiffness application is a user UDF, and the
fixed solver loop is captured once as gf_control.repeat. No sparse matrix is
assembled. Run with ``PYTHONPATH=python python examples/fem_poisson.py``.
"""

from __future__ import annotations

import graphforge as gf


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
    kernel = StiffnessApply()

    def matvec(displacement: gf.Tensor) -> gf.Tensor:
        return kernel(
            graph=graph,
            src={"u": displacement},
            dst={},
            edge={"value": stiffness},
        )

    return gf.linalg.LinearOperator(
        (interior_nodes, interior_nodes),
        matvec=matvec,
        parameters=(stiffness,),
        symmetric=True,
        name="P1PoissonStiffness",
    ), spacing


def run(interior_nodes: int = 8, iterations: int = 240):
    operator, spacing = poisson_operator(interior_nodes)
    load = gf.tensor(
        [spacing] * interior_nodes,
        dtype=gf.float32,
        requires_grad=True,
    )
    # lambda_max(A) < 4 / h, so omega=h/2 is a stable stationary step.
    solution = gf.linalg.richardson(
        operator,
        load,
        iterations=iterations,
        relaxation=0.5 * spacing,
    )
    coordinates = [(index + 1) * spacing for index in range(interior_nodes)]
    exact = [0.5 * x * (1.0 - x) for x in coordinates]
    error = max(abs(actual - expected) for actual, expected in zip(
        solution.tolist(), exact, strict=True))
    return solution, exact, error


if __name__ == "__main__":
    result, expected, maximum_error = run()
    print("solution:", [round(value, 6) for value in result.tolist()])
    print("exact:   ", [round(value, 6) for value in expected])
    print(f"max error: {maximum_error:.3e}")
    print(result.mlir())
