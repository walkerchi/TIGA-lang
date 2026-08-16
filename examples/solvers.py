"""Matrix-free stationary solvers composed from GraphForge primitives.

This module is grammar sugar, not compiler surface, and deliberately lives in
``examples/`` rather than the core package: every solver is ordinary Python
whose loop body is captured once into ``gf_control.repeat`` or
``gf_control.while`` regions (written as natural ``for``/``while`` under
``@gf.jit``); the primitives are Tensor algebra, relation application and
bounded device control.

A MessagePassing kernel bound to a Graph already *is* a linear operator, so
solvers accept the kernel directly::

    solution = cg(
        ShiftedRadiusLaplacian(),
        rhs,
        graph=graph,
        field="u",
        params={"mass": 1.0},
        tolerance=1.0e-5,
        max_iterations=32,
    )

``field`` names the unknown: each iteration binds it as
``src={field: value}, dst={field: value}``. Constant edge/node data and UDF
parameters are bound once through ``src=``/``dst=``/``edge=``/``params=``.
Pure Tensor algebra (a diagonal scaling, a Jacobi sweep) can be passed as a
plain ``Tensor -> Tensor`` callable instead. No matrix is ever materialized.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import graphforge as gf

Matvec = Callable[[gf.Tensor], gf.Tensor]
Preconditioner = Callable[[gf.Tensor], gf.Tensor]


def dot(left: gf.Tensor, right: gf.Tensor) -> gf.Tensor:
    """Capture a rank-one Hermitian dot product as Tensor algebra."""
    if not isinstance(left, gf.Tensor) or not isinstance(right, gf.Tensor):
        raise TypeError("dot operands must be graphforge.Tensor values")
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape:
        raise ValueError("dot requires rank-one operands with identical shapes")
    if left.dtype is not right.dtype or left.device != right.device:
        raise ValueError("dot operands must have identical dtype and device")
    return (left.conj() * right).sum()


def vector_norm(value: gf.Tensor) -> gf.Tensor:
    """Capture the Euclidean norm of one real vector.

    Complex norm needs a real-valued projection dtype that the minimal Tensor
    frontend does not expose yet, so it fails instead of returning a complex
    square root with misleading semantics.
    """
    if not isinstance(value, gf.Tensor) or value.ndim != 1:
        raise TypeError("vector_norm expects one rank-one graphforge.Tensor")
    if value.dtype.kind != "float":
        raise TypeError("vector_norm currently requires a real floating Tensor")
    return dot(value, value).sqrt()


def _bind_operator(
    operator,
    *,
    graph: gf.Graph | None,
    field: str | None,
    src: Mapping[str, gf.Tensor] | None,
    dst: Mapping[str, gf.Tensor] | None,
    edge: Mapping[str, gf.Tensor] | None,
    params: Mapping[str, object] | None,
) -> tuple[Matvec, int | None]:
    """Return ``(matvec, extent)`` for a kernel or plain callable operator."""
    if isinstance(operator, gf.Kernel):
        if graph is None or field is None:
            raise TypeError(
                "a MessagePassing operator requires graph= and field= so the "
                "solver can bind the iterated vector")
        if not isinstance(field, str) or not field:
            raise TypeError("field must be a non-empty field name string")
        schema = graph.schema
        if schema.num_src != schema.num_dst:
            raise ValueError(
                "solvers require a square relation: graph has "
                f"{schema.num_src} sources and {schema.num_dst} destinations")
        constant_src = dict(src) if src is not None else {}
        constant_dst = dict(dst) if dst is not None else {}
        if field in constant_src or field in constant_dst:
            raise ValueError(
                f"field {field!r} is the iterated unknown and must not also "
                "be bound as a constant")
        constant_edge = dict(edge) if edge is not None else {}
        bound_params = dict(params) if params is not None else {}

        def apply(vector: gf.Tensor) -> gf.Tensor:
            return operator(
                graph=graph,
                src={**constant_src, field: vector},
                dst={**constant_dst, field: vector},
                edge=constant_edge,
                **bound_params,
            )

        return apply, schema.num_dst
    bindings = {
        "graph": graph, "field": field, "src": src,
        "dst": dst, "edge": edge, "params": params,
    }
    given = [name for name, value in bindings.items() if value is not None]
    if given:
        names = ", ".join(given)
        raise TypeError(
            f"{names}= only apply to MessagePassing operators; a plain "
            "callable closes over its own constants")
    if not callable(operator):
        raise TypeError(
            "operator must be a graphforge.MessagePassing kernel or a "
            "Tensor -> Tensor callable")
    return operator, None


def _checked_apply(
    matvec: Matvec,
    vector: gf.Tensor,
    extent: int | None,
) -> gf.Tensor:
    """Apply one matrix-free product and validate the contract."""
    if extent is not None and vector.shape != (extent,):
        raise ValueError(
            f"operator expected input shape {(extent,)}, got {vector.shape}")
    output = matvec(vector)
    if not isinstance(output, gf.Tensor):
        raise TypeError("operator must return a graphforge.Tensor")
    if output.shape != vector.shape:
        raise ValueError(
            f"operator must be square: input shape {vector.shape} produced "
            f"output shape {output.shape}")
    if output.dtype is not vector.dtype or output.device != vector.device:
        raise ValueError(
            "operator input and output must have identical dtype and device")
    return output


@gf.jit
def _richardson_loop(matvec, omega, state, rhs, iterations):
    for _ in range(iterations):
        state = state + omega * (rhs - matvec(state))
    return state


@gf.jit
def _cg_fixed_loop(
    matvec, prepare, solution, residual, direction, product, iterations,
):
    for _ in range(iterations):
        applied = matvec(direction)
        alpha = product / dot(direction, applied)
        solution = solution + alpha * direction
        residual = residual - alpha * applied
        preconditioned = prepare(residual)
        updated = dot(residual, preconditioned)
        direction = preconditioned + (updated / product) * direction
        product = updated
    return solution


@gf.jit
def _cg_tolerance_loop(
    matvec, prepare, solution, residual, direction, product,
    threshold_squared, max_iterations,
):
    for _ in range(max_iterations):
        if dot(residual, residual) <= threshold_squared:
            break
        applied = matvec(direction)
        alpha = product / dot(direction, applied)
        solution = solution + alpha * direction
        residual = residual - alpha * applied
        preconditioned = prepare(residual)
        updated = dot(residual, preconditioned)
        direction = preconditioned + (updated / product) * direction
        product = updated
    return solution


def richardson(
    operator,
    rhs: gf.Tensor,
    *,
    graph: gf.Graph | None = None,
    field: str | None = None,
    src: Mapping[str, gf.Tensor] | None = None,
    dst: Mapping[str, gf.Tensor] | None = None,
    edge: Mapping[str, gf.Tensor] | None = None,
    params: Mapping[str, object] | None = None,
    iterations: int,
    relaxation: gf.Tensor | float = 1.0,
    initial: gf.Tensor | None = None,
) -> gf.Tensor:
    """Capture a bounded matrix-free Richardson iteration.

    The body is captured once inside ``gf_control.repeat`` and can contain
    MessagePassing. It is intentionally fixed-count; use ``cg`` with
    ``tolerance`` for a device-side convergence test.
    """
    matvec, extent = _bind_operator(
        operator, graph=graph, field=field, src=src, dst=dst,
        edge=edge, params=params)
    if not isinstance(rhs, gf.Tensor) or rhs.ndim != 1:
        raise TypeError("richardson rhs must be a rank-one graphforge.Tensor")
    if extent is not None and rhs.shape != (extent,):
        raise ValueError(
            f"richardson rhs shape must be {(extent,)}, got {rhs.shape}")
    if not isinstance(iterations, int) or isinstance(iterations, bool):
        raise TypeError("richardson iterations must be an integer")
    if iterations < 0:
        raise ValueError("richardson iterations must be non-negative")
    state = gf.zeros_like(rhs) if initial is None else initial
    if not isinstance(state, gf.Tensor) or state.shape != rhs.shape:
        raise ValueError("richardson initial value must match rhs shape")
    if state.dtype is not rhs.dtype or state.device != rhs.device:
        raise ValueError("richardson initial value must match rhs dtype and device")
    omega = relaxation
    return _richardson_loop(
        lambda value: _checked_apply(matvec, value, extent),
        omega,
        state,
        rhs,
        iterations,
    )


def cg(
    operator,
    rhs: gf.Tensor,
    *,
    graph: gf.Graph | None = None,
    field: str | None = None,
    src: Mapping[str, gf.Tensor] | None = None,
    dst: Mapping[str, gf.Tensor] | None = None,
    edge: Mapping[str, gf.Tensor] | None = None,
    params: Mapping[str, object] | None = None,
    iterations: int | None = None,
    tolerance: gf.Tensor | float | None = None,
    max_iterations: int | None = None,
    initial: gf.Tensor | None = None,
    preconditioner: Preconditioner | None = None,
) -> gf.Tensor:
    """Capture bounded matrix-free (preconditioned) conjugate gradient.

    The operator contract is symmetric positive-definite, as in scipy; the
    sugar layer does not attempt a symmetry proof. Four values (solution,
    residual, search direction, residual inner product) are carried through
    one ``gf_control.repeat`` or ``gf_control.while`` region, so the operator
    and optional preconditioner remain ordinary compiler-visible
    Tensor/MessagePassing programs.

    Pass ``iterations=k`` for a fixed ``gf_control.repeat`` or pass both
    ``tolerance=eps`` and ``max_iterations=k`` for a device-side bounded
    ``gf_control.while``. The convergence condition is the absolute Euclidean
    residual norm and never synchronizes a scalar through Python.
    """
    matvec, extent = _bind_operator(
        operator, graph=graph, field=field, src=src, dst=dst,
        edge=edge, params=params)
    if not isinstance(rhs, gf.Tensor) or rhs.ndim != 1:
        raise TypeError("cg rhs must be a rank-one graphforge.Tensor")
    if rhs.dtype.kind != "float":
        raise TypeError("cg currently requires a real floating Tensor")
    if extent is not None and rhs.shape != (extent,):
        raise ValueError(f"cg rhs shape must be {(extent,)}, got {rhs.shape}")
    fixed = iterations is not None
    convergent = tolerance is not None or max_iterations is not None
    if fixed == convergent:
        raise ValueError(
            "cg requires exactly one stopping contract: iterations, or "
            "tolerance with max_iterations")
    if fixed:
        if not isinstance(iterations, int) or isinstance(iterations, bool):
            raise TypeError("cg iterations must be an integer")
        if iterations < 0:
            raise ValueError("cg iterations must be non-negative")
    else:
        if tolerance is None or max_iterations is None:
            raise ValueError(
                "tolerance-driven cg requires tolerance and max_iterations")
        if (not isinstance(max_iterations, int) or
                isinstance(max_iterations, bool)):
            raise TypeError("cg max_iterations must be an integer")
        if max_iterations < 0:
            raise ValueError("cg max_iterations must be non-negative")
    if preconditioner is not None and not callable(preconditioner):
        raise TypeError("cg preconditioner must be a Tensor -> Tensor callable")
    solution = gf.zeros_like(rhs) if initial is None else initial
    if (not isinstance(solution, gf.Tensor) or solution.shape != rhs.shape or
            solution.dtype is not rhs.dtype or solution.device != rhs.device):
        raise ValueError("cg initial value must match rhs shape, dtype, and device")

    def apply_preconditioner(value: gf.Tensor) -> gf.Tensor:
        if preconditioner is None:
            return value
        output = preconditioner(value)
        if (not isinstance(output, gf.Tensor) or output.shape != value.shape or
                output.dtype is not value.dtype or output.device != value.device):
            raise ValueError(
                "cg preconditioner must preserve residual shape, dtype, and device")
        return output

    residual = rhs - _checked_apply(matvec, solution, extent)
    preconditioned = apply_preconditioner(residual)
    residual_product = dot(residual, preconditioned)

    def checked(value: gf.Tensor) -> gf.Tensor:
        return _checked_apply(matvec, value, extent)

    if fixed:
        assert iterations is not None
        return _cg_fixed_loop(
            checked, apply_preconditioner, solution, residual, preconditioned,
            residual_product, iterations)
    assert tolerance is not None and max_iterations is not None
    if isinstance(tolerance, gf.Tensor):
        threshold = tolerance
        if (threshold.shape != () or threshold.dtype is not rhs.dtype or
                threshold.device != rhs.device):
            raise ValueError(
                "cg Tensor tolerance must be scalar and match rhs dtype/device")
    else:
        if not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool):
            raise TypeError("cg tolerance must be a scalar Tensor or real number")
        if tolerance < 0:
            raise ValueError("cg tolerance must be non-negative")
        threshold = gf.tensor(tolerance, dtype=rhs.dtype, device=rhs.device)
    threshold_squared = threshold * threshold
    return _cg_tolerance_loop(
        checked, apply_preconditioner, solution, residual, preconditioned,
        residual_product, threshold_squared, max_iterations)


__all__ = [
    "Matvec", "Preconditioner", "cg", "dot", "richardson", "vector_norm",
]
