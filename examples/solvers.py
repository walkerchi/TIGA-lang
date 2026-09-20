"""Matrix-free stationary solvers composed from Tiga primitives."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import tiga as tg

Matvec = Callable[[tg.Tensor], tg.Tensor]
Preconditioner = Callable[[tg.Tensor], tg.Tensor]


def dot(left: tg.Tensor, right: tg.Tensor) -> tg.Tensor:
    """Capture a rank-one Hermitian dot product as Tensor algebra."""
    if not isinstance(left, tg.Tensor) or not isinstance(right, tg.Tensor):
        raise TypeError("dot operands must be tiga.Tensor values")
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape:
        raise ValueError("dot requires rank-one operands with identical shapes")
    if left.dtype is not right.dtype or left.device != right.device:
        raise ValueError("dot operands must have identical dtype and device")
    return (left.conj() * right).sum()


def vector_norm(value: tg.Tensor) -> tg.Tensor:
    """Capture the Euclidean norm of one real vector.

    Complex norm needs a real-valued projection dtype that the minimal Tensor
    frontend does not expose yet, so it fails instead of returning a complex
    square root with misleading semantics.
    """
    if not isinstance(value, tg.Tensor) or value.ndim != 1:
        raise TypeError("vector_norm expects one rank-one tiga.Tensor")
    if value.dtype.kind != "float":
        raise TypeError("vector_norm currently requires a real floating Tensor")
    return dot(value, value).sqrt()


def _bind_operator(
    operator,
    *,
    graph: tg.Graph | None,
    field: str | None,
    src: Mapping[str, tg.Tensor] | None,
    dst: Mapping[str, tg.Tensor] | None,
    edge: Mapping[str, tg.Tensor] | None,
    params: Mapping[str, object] | None,
) -> tuple[Matvec, int | None]:
    """Return ``(matvec, extent)`` for a kernel or plain callable operator."""
    if isinstance(operator, tg.Kernel):
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

        def apply(vector: tg.Tensor) -> tg.Tensor:
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
            "operator must be a tiga.MessagePassing kernel or a "
            "Tensor -> Tensor callable")
    return operator, None


def _checked_apply(
    matvec: Matvec,
    vector: tg.Tensor,
    extent: int | None,
) -> tg.Tensor:
    """Apply one matrix-free product and validate the contract."""
    if extent is not None and vector.shape != (extent,):
        raise ValueError(
            f"operator expected input shape {(extent,)}, got {vector.shape}")
    output = matvec(vector)
    if not isinstance(output, tg.Tensor):
        raise TypeError("operator must return a tiga.Tensor")
    if output.shape != vector.shape:
        raise ValueError(
            f"operator must be square: input shape {vector.shape} produced "
            f"output shape {output.shape}")
    if output.dtype is not vector.dtype or output.device != vector.device:
        raise ValueError(
            "operator input and output must have identical dtype and device")
    return output


@tg.jit
def _richardson_loop(matvec, omega, state, rhs, iterations):
    for _ in range(iterations):
        state = state + omega * (rhs - matvec(state))
    return state


@tg.jit
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


@tg.jit
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


@tg.jit
def _bicgstab_fixed_loop(
    matvec, solution, residual, shadow, direction, velocity,
    rho, alpha, omega, iterations,
):
    for _ in range(iterations):
        rho_new = dot(shadow, residual)
        beta = (rho_new / rho) * (alpha / omega)
        direction = residual + beta * (direction - omega * velocity)
        velocity = matvec(direction)
        alpha = rho_new / dot(shadow, velocity)
        intermediate = residual - alpha * velocity
        second = matvec(intermediate)
        omega = dot(second, intermediate) / dot(second, second)
        solution = solution + alpha * direction + omega * intermediate
        residual = intermediate - omega * second
        rho = rho_new
    return solution


@tg.jit
def _bicgstab_tolerance_loop(
    matvec, solution, residual, shadow, direction, velocity,
    rho, alpha, omega, threshold_squared, max_iterations,
):
    for _ in range(max_iterations):
        if dot(residual, residual) <= threshold_squared:
            break
        rho_new = dot(shadow, residual)
        beta = (rho_new / rho) * (alpha / omega)
        direction = residual + beta * (direction - omega * velocity)
        velocity = matvec(direction)
        alpha = rho_new / dot(shadow, velocity)
        intermediate = residual - alpha * velocity
        second = matvec(intermediate)
        omega = dot(second, intermediate) / dot(second, second)
        solution = solution + alpha * direction + omega * intermediate
        residual = intermediate - omega * second
        rho = rho_new
    return solution


def richardson(
    operator,
    rhs: tg.Tensor,
    *,
    graph: tg.Graph | None = None,
    field: str | None = None,
    src: Mapping[str, tg.Tensor] | None = None,
    dst: Mapping[str, tg.Tensor] | None = None,
    edge: Mapping[str, tg.Tensor] | None = None,
    params: Mapping[str, object] | None = None,
    iterations: int,
    relaxation: tg.Tensor | float = 1.0,
    initial: tg.Tensor | None = None,
) -> tg.Tensor:
    """Capture a bounded matrix-free Richardson iteration.

    The body is captured once inside ``gf_control.repeat`` and can contain
    MessagePassing. It is intentionally fixed-count; use ``cg`` with
    ``tolerance`` for a device-side convergence test.
    """
    matvec, extent = _bind_operator(
        operator, graph=graph, field=field, src=src, dst=dst,
        edge=edge, params=params)
    if not isinstance(rhs, tg.Tensor) or rhs.ndim != 1:
        raise TypeError("richardson rhs must be a rank-one tiga.Tensor")
    if extent is not None and rhs.shape != (extent,):
        raise ValueError(
            f"richardson rhs shape must be {(extent,)}, got {rhs.shape}")
    if not isinstance(iterations, int) or isinstance(iterations, bool):
        raise TypeError("richardson iterations must be an integer")
    if iterations < 0:
        raise ValueError("richardson iterations must be non-negative")
    state = tg.zeros_like(rhs) if initial is None else initial
    if not isinstance(state, tg.Tensor) or state.shape != rhs.shape:
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
    rhs: tg.Tensor,
    *,
    graph: tg.Graph | None = None,
    field: str | None = None,
    src: Mapping[str, tg.Tensor] | None = None,
    dst: Mapping[str, tg.Tensor] | None = None,
    edge: Mapping[str, tg.Tensor] | None = None,
    params: Mapping[str, object] | None = None,
    iterations: int | None = None,
    tolerance: tg.Tensor | float | None = None,
    max_iterations: int | None = None,
    initial: tg.Tensor | None = None,
    preconditioner: Preconditioner | None = None,
) -> tg.Tensor:
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
    if not isinstance(rhs, tg.Tensor) or rhs.ndim != 1:
        raise TypeError("cg rhs must be a rank-one tiga.Tensor")
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
    solution = tg.zeros_like(rhs) if initial is None else initial
    if (not isinstance(solution, tg.Tensor) or solution.shape != rhs.shape or
            solution.dtype is not rhs.dtype or solution.device != rhs.device):
        raise ValueError("cg initial value must match rhs shape, dtype, and device")

    def apply_preconditioner(value: tg.Tensor) -> tg.Tensor:
        if preconditioner is None:
            return value
        output = preconditioner(value)
        if (not isinstance(output, tg.Tensor) or output.shape != value.shape or
                output.dtype is not value.dtype or output.device != value.device):
            raise ValueError(
                "cg preconditioner must preserve residual shape, dtype, and device")
        return output

    residual = rhs - _checked_apply(matvec, solution, extent)
    preconditioned = apply_preconditioner(residual)
    residual_product = dot(residual, preconditioned)

    def checked(value: tg.Tensor) -> tg.Tensor:
        return _checked_apply(matvec, value, extent)

    if fixed:
        assert iterations is not None
        return _cg_fixed_loop(
            checked, apply_preconditioner, solution, residual, preconditioned,
            residual_product, iterations)
    assert tolerance is not None and max_iterations is not None
    if isinstance(tolerance, tg.Tensor):
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
        threshold = tg.tensor(tolerance, dtype=rhs.dtype, device=rhs.device)
    threshold_squared = threshold * threshold
    return _cg_tolerance_loop(
        checked, apply_preconditioner, solution, residual, preconditioned,
        residual_product, threshold_squared, max_iterations)


def bicgstab(
    operator,
    rhs: tg.Tensor,
    *,
    graph: tg.Graph | None = None,
    field: str | None = None,
    src: Mapping[str, tg.Tensor] | None = None,
    dst: Mapping[str, tg.Tensor] | None = None,
    edge: Mapping[str, tg.Tensor] | None = None,
    params: Mapping[str, object] | None = None,
    iterations: int | None = None,
    tolerance: tg.Tensor | float | None = None,
    max_iterations: int | None = None,
    initial: tg.Tensor | None = None,
) -> tg.Tensor:
    """Capture bounded matrix-free BiCGStab for nonsymmetric operators.

    Same stopping contracts as ``cg``: ``iterations=k`` for a fixed
    ``gf_control.repeat``, or ``tolerance=eps`` plus ``max_iterations=k`` for
    a bounded device-side ``gf_control.while``. Seven values (solution,
    residual, search direction, operator images, and the scalar recurrences)
    are carried through one control region.

    The convergence test runs at the top of each iteration, as in ``cg``.
    Landing exactly on the solution at a half step (residual surrogate
    ``s = 0``) divides by ``dot(A·s, A·s)`` — the same exact-breakdown hazard
    fixed-count CG has; a nonzero tolerance exits before it in practice.
    """
    matvec, extent = _bind_operator(
        operator, graph=graph, field=field, src=src, dst=dst,
        edge=edge, params=params)
    if not isinstance(rhs, tg.Tensor) or rhs.ndim != 1:
        raise TypeError("bicgstab rhs must be a rank-one tiga.Tensor")
    if rhs.dtype.kind != "float":
        raise TypeError("bicgstab currently requires a real floating Tensor")
    if extent is not None and rhs.shape != (extent,):
        raise ValueError(f"bicgstab rhs shape must be {(extent,)}, got {rhs.shape}")
    fixed = iterations is not None
    convergent = tolerance is not None or max_iterations is not None
    if fixed == convergent:
        raise ValueError(
            "bicgstab requires exactly one stopping contract: iterations, or "
            "tolerance with max_iterations")
    if fixed:
        if not isinstance(iterations, int) or isinstance(iterations, bool):
            raise TypeError("bicgstab iterations must be an integer")
        if iterations < 0:
            raise ValueError("bicgstab iterations must be non-negative")
    else:
        if tolerance is None or max_iterations is None:
            raise ValueError(
                "tolerance-driven bicgstab requires tolerance and max_iterations")
        if (not isinstance(max_iterations, int) or
                isinstance(max_iterations, bool)):
            raise TypeError("bicgstab max_iterations must be an integer")
        if max_iterations < 0:
            raise ValueError("bicgstab max_iterations must be non-negative")
    solution = tg.zeros_like(rhs) if initial is None else initial
    if (not isinstance(solution, tg.Tensor) or solution.shape != rhs.shape or
            solution.dtype is not rhs.dtype or solution.device != rhs.device):
        raise ValueError(
            "bicgstab initial value must match rhs shape, dtype, and device")

    residual = rhs - _checked_apply(matvec, solution, extent)
    shadow = residual                       # r̂ = r₀ stays constant
    direction = tg.zeros_like(rhs)
    velocity = tg.zeros_like(rhs)
    one = tg.tensor(1.0, dtype=rhs.dtype, device=rhs.device)
    rho = alpha = omega = one

    def checked(value: tg.Tensor) -> tg.Tensor:
        return _checked_apply(matvec, value, extent)

    if fixed:
        assert iterations is not None
        return _bicgstab_fixed_loop(
            checked, solution, residual, shadow, direction, velocity,
            rho, alpha, omega, iterations)
    assert tolerance is not None and max_iterations is not None
    if isinstance(tolerance, tg.Tensor):
        threshold = tolerance
        if (threshold.shape != () or threshold.dtype is not rhs.dtype or
                threshold.device != rhs.device):
            raise ValueError(
                "bicgstab Tensor tolerance must be scalar and match rhs")
    else:
        if not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool):
            raise TypeError(
                "bicgstab tolerance must be a scalar Tensor or real number")
        if tolerance < 0:
            raise ValueError("bicgstab tolerance must be non-negative")
        threshold = tg.tensor(tolerance, dtype=rhs.dtype, device=rhs.device)
    threshold_squared = threshold * threshold
    return _bicgstab_tolerance_loop(
        checked, solution, residual, shadow, direction, velocity,
        rho, alpha, omega, threshold_squared, max_iterations)


def linear_solve(
    operator,
    rhs: tg.Tensor,
    *,
    method: str = "cg",
    graph: tg.Graph | None = None,
    field: str | None = None,
    src: Mapping[str, tg.Tensor] | None = None,
    dst: Mapping[str, tg.Tensor] | None = None,
    edge: Mapping[str, tg.Tensor] | None = None,
    params: Mapping[str, object] | None = None,
    iterations: int | None = None,
    tolerance: tg.Tensor | float | None = None,
    max_iterations: int | None = None,
    initial: tg.Tensor | None = None,
    preconditioner: Preconditioner | None = None,
    relaxation: tg.Tensor | float = 1.0,
) -> tg.Tensor:
    """Solve ``A·x = b`` matrix-free; ``method`` selects the iteration.

    - ``"cg"`` (default) — conjugate gradient; operator contract is symmetric
      positive-definite, optional ``preconditioner``.
    - ``"bicgstab"`` — BiCGStab for nonsymmetric operators.
    - ``"richardson"`` — fixed-count damped iteration; requires
      ``iterations=k`` and accepts ``relaxation``.

    All methods share the stopping contracts: ``iterations=k`` for a fixed
    ``gf_control.repeat``, or ``tolerance=eps`` with ``max_iterations=k`` for
    a bounded device-side ``gf_control.while``.
    """
    if method == "cg":
        return cg(
            operator, rhs, graph=graph, field=field, src=src, dst=dst,
            edge=edge, params=params, iterations=iterations,
            tolerance=tolerance, max_iterations=max_iterations,
            initial=initial, preconditioner=preconditioner)
    if method == "bicgstab":
        if preconditioner is not None:
            raise ValueError(
                "linear_solve method='bicgstab' takes no preconditioner in "
                "this sugar layer")
        return bicgstab(
            operator, rhs, graph=graph, field=field, src=src, dst=dst,
            edge=edge, params=params, iterations=iterations,
            tolerance=tolerance, max_iterations=max_iterations,
            initial=initial)
    if method == "richardson":
        if tolerance is not None or max_iterations is not None:
            raise ValueError(
                "linear_solve method='richardson' is fixed-count: pass "
                "iterations, not tolerance")
        if preconditioner is not None:
            raise ValueError(
                "linear_solve method='richardson' takes no preconditioner")
        return richardson(
            operator, rhs, graph=graph, field=field, src=src, dst=dst,
            edge=edge, params=params, iterations=iterations,
            relaxation=relaxation, initial=initial)
    raise ValueError(
        f"unknown linear_solve method {method!r}: "
        "choose 'cg', 'bicgstab' or 'richardson'")


__all__ = [
    "Matvec", "Preconditioner", "bicgstab", "cg", "dot", "linear_solve",
    "richardson", "vector_norm",
]
