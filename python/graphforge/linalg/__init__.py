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


__all__ = ["LinearOperator", "Matvec", "dot", "richardson", "vector_norm"]
