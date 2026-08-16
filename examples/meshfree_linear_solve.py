"""Dynamic-radius matrix-free solve with a bounded device-side CG loop.

The radius relation is generated from positions, the shifted graph Laplacian
is a normal MessagePassing UDF, and CG captures its convergence test and four
carried states in one ``gf_control.while``.  No adjacency or sparse matrix is
written by the user.
"""

from __future__ import annotations

import graphforge as gf
from solvers import cg, vector_norm


class ShiftedRadiusLaplacian(gf.MessagePassing):
    """Apply ``mass * u + sum_neighbour(u_dst - u_src)``."""

    reducer = gf.sum()

    def edge(self, src, dst, edge):
        del edge
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
    )
    # The logical relation remains procedural. Its current physical snapshot
    # may be built, streamed, cached, or rebuilt by the selected provider.
    graph = gf.Graph.radius(positions, cutoff=1.01 * spacing)
    # A MessagePassing kernel bound to a Graph is the linear operator; the
    # solver binds the iterated vector to the "u" field each iteration.
    apply_laplacian = ShiftedRadiusLaplacian()
    rhs = gf.tensor(
        [1.0 + float(index % 3) for index in range(points)],
        dtype=gf.float32,
    )
    solution = cg(
        apply_laplacian,
        rhs,
        graph=graph,
        field="u",
        params={"mass": 1.0},
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
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


if __name__ == "__main__":
    x, residual, relation, kernel = solve()
    print("solution:", [round(value, 6) for value in x.tolist()])
    print("residual norm:", residual.tolist())
    print("relation:", relation.explain())
    print("solver uses gf_control.while:", "gf_control.while" in x.mlir())
    print("operator provider:", kernel.last_variant.provider)
    print(x.generated_code("cpu_loop"))
