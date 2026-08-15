"""Matrix-free linear algebra over GraphForge semantic programs.

This module intentionally starts with the operator boundary and one bounded
stationary solver.  It does not materialize matrices or hide Python iteration
behind a solver-shaped convenience function.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..control import repeat
from ..tensor import Tensor, zeros_like


Matvec = Callable[[Tensor], Tensor]
Preconditioner = Callable[[Tensor], Tensor]


def dot(left: Tensor, right: Tensor) -> Tensor:
    """Capture a rank-one Hermitian dot product as Tensor algebra."""
    if not isinstance(left, Tensor) or not isinstance(right, Tensor):
        raise TypeError("dot operands must be graphforge.Tensor values")
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape:
        raise ValueError("dot requires rank-one operands with identical shapes")
    if left.dtype is not right.dtype or left.device != right.device:
        raise ValueError("dot operands must have identical dtype and device")
    return (left.conj() * right).sum()


def vector_norm(value: Tensor) -> Tensor:
    """Capture the Euclidean norm of one real vector.

    Complex norm needs a real-valued projection dtype that the minimal Tensor
    frontend does not expose yet, so it fails instead of returning a complex
    square root with misleading semantics.
    """
    if not isinstance(value, Tensor) or value.ndim != 1:
        raise TypeError("vector_norm expects one rank-one graphforge.Tensor")
    if value.dtype.kind != "float":
        raise TypeError("vector_norm currently requires a real floating Tensor")
    return dot(value, value).sqrt()


@dataclass(frozen=True)
class LinearOperator:
    """A matrix-free linear map whose application remains compiler-visible.

    ``matvec`` may call ordinary Tensor programs or MessagePassing kernels over
    static, generated, paged, or halo relations. ``parameters`` identifies
    differentiable operator data for future implicit-solve VJP lowering; it is
    metadata today and does not replace normal expression dependencies.
    """

    shape: tuple[int, int]
    matvec: Matvec
    rmatvec: Matvec | None = None
    parameters: tuple[Tensor, ...] = ()
    symmetric: bool = False
    name: str = "LinearOperator"

    def __post_init__(self) -> None:
        shape = tuple(self.shape)
        if (len(shape) != 2 or any(
                not isinstance(extent, int) or isinstance(extent, bool)
                or extent < 0 for extent in shape)):
            raise ValueError("LinearOperator shape must contain two non-negative integers")
        if not callable(self.matvec):
            raise TypeError("LinearOperator matvec must be callable")
        if self.rmatvec is not None and not callable(self.rmatvec):
            raise TypeError("LinearOperator rmatvec must be callable")
        parameters = tuple(self.parameters)
        if any(not isinstance(value, Tensor) for value in parameters):
            raise TypeError("LinearOperator parameters must be graphforge.Tensor values")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("LinearOperator name must be a non-empty string")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "parameters", parameters)

    def __call__(self, value: Tensor) -> Tensor:
        return self.apply(value)

    def apply(self, value: Tensor) -> Tensor:
        """Capture one application without materializing a matrix."""
        if not isinstance(value, Tensor):
            raise TypeError("LinearOperator input must be a graphforge.Tensor")
        if value.shape != (self.shape[1],):
            raise ValueError(
                f"LinearOperator expected input shape {(self.shape[1],)}, "
                f"got {value.shape}"
            )
        output = self.matvec(value)
        if not isinstance(output, Tensor):
            raise TypeError("LinearOperator matvec must return a graphforge.Tensor")
        if output.shape != (self.shape[0],):
            raise ValueError(
                f"LinearOperator matvec returned shape {output.shape}, "
                f"expected {(self.shape[0],)}"
            )
        if output.dtype is not value.dtype or output.device != value.device:
            raise ValueError(
                "LinearOperator input and output must have identical dtype and device"
            )
        return output

    def adjoint_apply(self, value: Tensor) -> Tensor:
        """Capture one adjoint application for solver and VJP construction."""
        if self.rmatvec is None:
            if not self.symmetric:
                raise ValueError(
                    "LinearOperator requires rmatvec unless symmetric=True")
            return self.apply(value)
        if not isinstance(value, Tensor):
            raise TypeError("LinearOperator adjoint input must be a graphforge.Tensor")
        if value.shape != (self.shape[0],):
            raise ValueError(
                f"LinearOperator adjoint expected shape {(self.shape[0],)}, "
                f"got {value.shape}"
            )
        output = self.rmatvec(value)
        if not isinstance(output, Tensor) or output.shape != (self.shape[1],):
            raise ValueError(
                "LinearOperator rmatvec must return a graphforge.Tensor with "
                f"shape {(self.shape[1],)}"
            )
        if output.dtype is not value.dtype or output.device != value.device:
            raise ValueError(
                "LinearOperator adjoint input and output must have identical "
                "dtype and device"
            )
        return output


def richardson(
    operator: LinearOperator,
    rhs: Tensor,
    *,
    iterations: int,
    relaxation: Tensor | float = 1.0,
    initial: Tensor | None = None,
) -> Tensor:
    """Capture a bounded matrix-free Richardson iteration.

    This is the executable bootstrap solver: the body is captured once inside
    ``gf_control.repeat`` and can contain MessagePassing. It is intentionally
    fixed-count. A convergence-driven CG API must wait for first-class
    multi-state ``gf_control.while`` rather than silently returning to Python.
    """
    if not isinstance(operator, LinearOperator):
        raise TypeError("richardson operator must be a LinearOperator")
    if not isinstance(rhs, Tensor):
        raise TypeError("richardson rhs must be a graphforge.Tensor")
    if operator.shape[0] != operator.shape[1]:
        raise ValueError("richardson requires a square LinearOperator")
    if rhs.shape != (operator.shape[0],):
        raise ValueError(
            f"richardson rhs shape must be {(operator.shape[0],)}, got {rhs.shape}"
        )
    if not isinstance(iterations, int) or isinstance(iterations, bool):
        raise TypeError("richardson iterations must be an integer")
    if iterations < 0:
        raise ValueError("richardson iterations must be non-negative")
    state = zeros_like(rhs) if initial is None else initial
    if not isinstance(state, Tensor) or state.shape != rhs.shape:
        raise ValueError("richardson initial value must match rhs shape")
    if state.dtype is not rhs.dtype or state.device != rhs.device:
        raise ValueError("richardson initial value must match rhs dtype and device")
    omega = relaxation if isinstance(relaxation, Tensor) else relaxation
    return repeat(
        state,
        lambda value: value + omega * (rhs - operator(value)),
        iterations=iterations,
    )


def cg(
    operator: LinearOperator,
    rhs: Tensor,
    *,
    iterations: int,
    initial: Tensor | None = None,
    preconditioner: LinearOperator | Preconditioner | None = None,
) -> Tensor:
    """Capture fixed-count matrix-free (preconditioned) conjugate gradient.

    Four values (solution, residual, search direction, residual inner product)
    are carried through one ``gf_control.repeat`` region. ``operator`` and an
    optional ``preconditioner`` therefore remain ordinary compiler-visible
    Tensor/MessagePassing programs and can be fused or scheduled by providers.

    This fixed-count API never performs a host synchronization to inspect a
    residual. A future tolerance-driven overload will lower to bounded
    ``gf_control.while``; callers should not infer convergence merely because
    this routine returned.
    """
    if not isinstance(operator, LinearOperator):
        raise TypeError("cg operator must be a LinearOperator")
    if not operator.symmetric:
        raise ValueError("cg requires operator.symmetric=True")
    if not isinstance(rhs, Tensor):
        raise TypeError("cg rhs must be a graphforge.Tensor")
    if operator.shape[0] != operator.shape[1]:
        raise ValueError("cg requires a square LinearOperator")
    if rhs.shape != (operator.shape[0],):
        raise ValueError(f"cg rhs shape must be {(operator.shape[0],)}")
    if rhs.dtype.kind != "float":
        raise TypeError("cg currently requires a real floating Tensor")
    if not isinstance(iterations, int) or isinstance(iterations, bool):
        raise TypeError("cg iterations must be an integer")
    if iterations < 0:
        raise ValueError("cg iterations must be non-negative")
    if (preconditioner is not None and
            not isinstance(preconditioner, LinearOperator) and
            not callable(preconditioner)):
        raise TypeError("cg preconditioner must be a LinearOperator or callable")
    solution = zeros_like(rhs) if initial is None else initial
    if (not isinstance(solution, Tensor) or solution.shape != rhs.shape or
            solution.dtype is not rhs.dtype or solution.device != rhs.device):
        raise ValueError("cg initial value must match rhs shape, dtype, and device")

    def apply_preconditioner(value: Tensor) -> Tensor:
        if preconditioner is None:
            return value
        output = preconditioner(value)
        if (not isinstance(output, Tensor) or output.shape != value.shape or
                output.dtype is not value.dtype or output.device != value.device):
            raise ValueError(
                "cg preconditioner must preserve residual shape, dtype, and device")
        return output

    residual = rhs - operator(solution)
    preconditioned = apply_preconditioner(residual)
    residual_product = dot(residual, preconditioned)

    def step(
        current_solution: Tensor,
        current_residual: Tensor,
        direction: Tensor,
        current_product: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        applied = operator(direction)
        alpha = current_product / dot(direction, applied)
        next_solution = current_solution + alpha * direction
        next_residual = current_residual - alpha * applied
        next_preconditioned = apply_preconditioner(next_residual)
        next_product = dot(next_residual, next_preconditioned)
        beta = next_product / current_product
        next_direction = next_preconditioned + beta * direction
        return next_solution, next_residual, next_direction, next_product

    result = repeat(
        (solution, residual, preconditioned, residual_product),
        step,
        iterations=iterations,
    )
    assert isinstance(result, tuple)
    return result[0]


__all__ = [
    "LinearOperator", "Matvec", "Preconditioner", "cg", "dot",
    "richardson", "vector_norm",
]
