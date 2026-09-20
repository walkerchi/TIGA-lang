"""Matrix-free 1D nonlinear diffusion solve through MessagePassing and Picard."""

from __future__ import annotations

import tiga as tg
from solvers import dot


# --8<-- [start:core]
class NonlinearDiffusionApply(tg.MessagePassing):
    """Assemble F(u) = A(u)·u for -∇·(k(u)∇u) with k(u) = 1 + u².

    Edge (j -> i) sends the conductance-scaled difference
    k_e·(u_i - u_j)/h with the endpoint average k_e = 1 + (u_i² + u_j²)/2;
    the sum reducer assembles one nonlinear stiffness row per node. The two
    boundary elements (u fixed to 0) contribute the diagonal term
    (1 + u_i²/2)·u_i/h, added by the node UDF where dst.boundary = 1:

        F(u)_i = (1/h)·Σ_j k_e·(u_i - u_j) + (1/h)·δ_i·(1 + u_i²/2)·u_i,
        u₀ = u_{N+1} = 0.
    """

    reducer = tg.sum()

    def edge(self, src, dst, edge, inverse_spacing):
        del edge
        # src.u, dst.u: (N,) endpoint values of the current iterate;
        # the message is one scalar per edge.
        conductance = 1.0 + 0.5 * (src.u * src.u + dst.u * dst.u)  # k_e at u
        return conductance * (dst.u - src.u) * inverse_spacing

    def node(self, dst, incoming, inverse_spacing):
        # Boundary-element diagonal at the two end nodes (dst.boundary = 1);
        # k_b = 1 + u_i²/2 averages k(u_i) with k(0) = 1 at the fixed end.
        conductance = 1.0 + 0.5 * dst.u * dst.u
        return incoming + dst.boundary * conductance * dst.u * inverse_spacing


@tg.jit
def _picard_fixed_loop(apply, omega, state, rhs, iterations):
    for _ in range(iterations):
        state = state + omega * (rhs - apply(state))
    return state


@tg.jit
def _picard_tolerance_loop(
    apply, omega, state, rhs, threshold_squared, max_iterations,
):
    residual = rhs - apply(state)
    for _ in range(max_iterations):
        if dot(residual, residual) <= threshold_squared:
            break
        state = state + omega * residual
        residual = rhs - apply(state)
    return state


def nonlinear_solve(
    operator,
    rhs: tg.Tensor,
    *,
    omega: float,
    iterations: int | None = None,
    tolerance: float | None = 1.0e-6,
    max_iterations: int = 2_000,
    initial: tg.Tensor | None = None,
) -> tg.Tensor:
    """Solve F(u) = rhs by damped Picard (nonlinear Richardson) iteration.

    The fixed-point step u ← u + ω·(rhs − F(u)) runs in ONE flat loop:
    ``iterations=k`` captures a fixed ``gf_control.repeat`` (differentiable
    through ``tg.autograd.grad``; the VJP unrolls the loop, so keep k small
    when differentiating); ``tolerance=eps`` with ``max_iterations``
    captures a bounded device-side ``gf_control.while`` instead (forward-only
    — the while form has no VJP yet). The contracts are exclusive.
    """
    if not callable(operator):
        raise TypeError("operator must be a Tensor -> Tensor callable")
    if not isinstance(rhs, tg.Tensor) or rhs.ndim != 1:
        raise TypeError(
            "nonlinear_solve rhs must be a rank-one tiga.Tensor")
    if rhs.dtype.kind != "float":
        raise TypeError("nonlinear_solve requires a real floating Tensor")
    fixed = iterations is not None
    if fixed == (tolerance is not None):
        raise ValueError(
            "nonlinear_solve requires exactly one stopping contract: "
            "iterations, or tolerance with max_iterations")
    if fixed:
        if not isinstance(iterations, int) or isinstance(iterations, bool):
            raise TypeError("nonlinear_solve iterations must be an integer")
        if iterations < 0:
            raise ValueError("nonlinear_solve iterations must be non-negative")
    else:
        if not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool):
            raise TypeError("nonlinear_solve tolerance must be a real number")
        if tolerance < 0:
            raise ValueError("nonlinear_solve tolerance must be non-negative")
        if (not isinstance(max_iterations, int) or
                isinstance(max_iterations, bool)):
            raise TypeError("nonlinear_solve max_iterations must be an integer")
        if max_iterations < 0:
            raise ValueError(
                "nonlinear_solve max_iterations must be non-negative")
    state = tg.zeros_like(rhs) if initial is None else initial
    if not isinstance(state, tg.Tensor) or state.shape != rhs.shape:
        raise ValueError("nonlinear_solve initial value must match rhs shape")
    if state.dtype is not rhs.dtype or state.device != rhs.device:
        raise ValueError(
            "nonlinear_solve initial value must match rhs dtype and device")
    if fixed:
        return _picard_fixed_loop(operator, omega, state, rhs, iterations)
    threshold = tg.tensor(
        tolerance * tolerance, dtype=rhs.dtype, device=rhs.device)
    return _picard_tolerance_loop(
        operator, omega, state, rhs, threshold, max_iterations)


def nonlinear_operator(interior_nodes: int):
    """P1 stencil grid + kernel for -∇·((1+u²)∇u) on the unit interval."""
    if interior_nodes < 1:
        raise ValueError("interior_nodes must be positive")
    spacing = 1.0 / (interior_nodes + 1)
    graph = tg.Graph.stencil((interior_nodes,), ((-1,), (0,), (1,)))
    flags = (                                   # one entry per boundary element
        [2.0] if interior_nodes == 1
        else [1.0] + [0.0] * (interior_nodes - 2) + [1.0])
    boundary = tg.tensor(flags, dtype=tg.float32)  # (N,)
    return NonlinearDiffusionApply(), graph, spacing, boundary


def solve(
    interior_nodes: int = 8,
    *,
    omega: float = 0.05,
    iterations: int | None = None,
    tolerance: float | None = 1.0e-6,
    max_iterations: int = 2_000,
):
    """Solve F(u) = h·1 for the nonlinear diffusion operator.

    Default is a residual-driven ``gf_control.while``; pass ``iterations=k``
    for a fixed ``gf_control.repeat`` loop instead. ω must stay below
    2/λ_max of the Jacobian — λ_max ≈ 4·max(k)/h here, so ω = 0.05 is
    stable for both grid sizes used below.
    """
    kernel, graph, spacing, boundary = nonlinear_operator(interior_nodes)
    load = tg.tensor(
        [spacing] * interior_nodes, dtype=tg.float32, requires_grad=True)  # (N,)

    def apply(state):  # F(u): (N,) -> (N,), conductance from current iterate
        return kernel(
            graph=graph,
            src={"u": state},
            dst={"u": state, "boundary": boundary},
            inverse_spacing=1.0 / spacing,
        )

    if iterations is not None:  # fixed repeat and bounded while are exclusive
        tolerance = None
    solution = nonlinear_solve(
        apply,
        load,
        omega=omega,
        iterations=iterations,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )  # (N,)
    residual = load - apply(solution)                        # (N,)
    error = max(abs(value) for value in residual.tolist())
    return solution, residual, error
# --8<-- [end:core]


def load_gradient(
    interior_nodes: int = 4, *, omega: float = 0.05, iterations: int = 3,
):
    """Differentiate sum(u) w.r.t. the load through the fixed Picard repeat.

    The fixed-repeat VJP unrolls the loop, so its compile time grows quickly
    with the iteration count — keep it in the low single digits (the
    forward-only drivers above run hundreds of iterations cheaply).
    """
    kernel, graph, spacing, boundary = nonlinear_operator(interior_nodes)
    load = tg.tensor(
        [spacing] * interior_nodes, dtype=tg.float32, requires_grad=True)  # (N,)

    def apply(state):  # F(u): (N,) -> (N,)
        return kernel(
            graph=graph,
            src={"u": state},
            dst={"u": state, "boundary": boundary},
            inverse_spacing=1.0 / spacing,
        )

    solution = nonlinear_solve(
        apply, load, omega=omega, iterations=iterations, tolerance=None)  # (N,)
    return tg.autograd.grad(solution.sum(), load)              # (N,)


# Inspect: solve()[0].mlir() (bounded gf_control.while loop),
#   solve(iterations=200)[0].mlir() (fixed gf_control.repeat loop)

if __name__ == "__main__":
    solve()
    load_gradient()
