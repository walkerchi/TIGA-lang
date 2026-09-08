"""Dynamic-radius matrix-free solve with a bounded device-side CG loop."""

from __future__ import annotations

import tiga as gf
from solvers import linear_solve, vector_norm


# --8<-- [start:core]
class ShiftedRadiusLaplacian(gf.MessagePassing):
    """Apply ``mass * u + sum_neighbour(u_dst - u_src)``."""

    reducer = gf.sum()

    def edge(self, src, dst, edge):
        return dst.u - src.u

    def node(self, dst, laplacian, mass):
        return mass * dst.u + laplacian


def solve(
    points: int = 8,
    *,
    tolerance: float = 1.0e-5,
    max_iterations: int = 32,
):
    """Solve a positive-definite operator on a generated path relation."""
    if points < 2:
        raise ValueError("points must be at least two")
    spacing = 1.0 / (points - 1)
    positions = gf.tensor(
        [[index * spacing, 0.0] for index in range(points)],
        dtype=gf.float32,
    )  # (N, 2)
    # The logical relation remains procedural. Its current physical snapshot
    # may be built, streamed, cached, or rebuilt by the selected provider.
    graph = gf.Graph.radius(positions, cutoff=1.01 * spacing)
    # A MessagePassing kernel bound to a Graph is the linear operator; the
    # solver binds the iterated vector to the "u" field each iteration.
    apply_laplacian = ShiftedRadiusLaplacian()
    rhs = gf.tensor(
        [1.0 + float(index % 3) for index in range(points)],
        dtype=gf.float32,
    )  # (N,)
    solution = linear_solve(
        apply_laplacian,
        rhs,
        method="cg",
        graph=graph,
        field="u",
        params={"mass": 1.0},
        tolerance=tolerance,
        max_iterations=max_iterations,
    )  # (N,)
    residual_norm = vector_norm(
        apply_laplacian(
            graph=graph,
            src={"u": solution},
            dst={"u": solution},
            mass=1.0,
        )
        - rhs
    )
    return solution, residual_norm, graph, apply_laplacian
# --8<-- [end:core]


# Inspect: solve()[2].explain(), solve()[0].mlir(),
#   solve()[0].generated_code("cpu_loop")

if __name__ == "__main__":
    solve()
