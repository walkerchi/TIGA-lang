"""Structured compiler control flow over GraphForge semantic values."""

from __future__ import annotations

from collections.abc import Callable

from ..tensor import Tensor
from ..tensor.core import _Expr, _Region


def _captures(output: Tensor, state: Tensor) -> tuple[Tensor, ...]:
    ordered: list[Tensor] = []
    visited: set[int] = set()

    def visit(value: Tensor) -> None:
        if value is state or id(value) in visited:
            return
        visited.add(id(value))
        expression = value._expr
        if expression is None:
            ordered.append(value)
            return
        if expression.op == "loop_argument":
            raise ValueError("repeat body used a loop argument from another region")
        if expression.region is not None:
            raise NotImplementedError("nested control regions are not supported yet")
        for operand in expression.operands:
            visit(operand)

    visit(output)
    return tuple(ordered)


def repeat(
    initial: Tensor,
    body: Callable[[Tensor], Tensor],
    *,
    iterations: int,
) -> Tensor:
    """Capture a fixed-count loop with one Tensor-carried state.

    ``body`` is traced once. Its Tensor leaves become explicit immutable
    region captures; the returned Tensor stays lazy and triggers ordinary JIT
    at observation. This is compiler control flow, not a Python execution
    loop and not an algorithm-specific operator.
    """
    if not isinstance(initial, Tensor):
        raise TypeError("repeat initial state must be a graphforge.Tensor")
    if not callable(body):
        raise TypeError("repeat body must be callable")
    if not isinstance(iterations, int) or isinstance(iterations, bool):
        raise TypeError("repeat iterations must be an integer")
    if iterations < 0:
        raise ValueError("repeat iterations must be non-negative")

    state = Tensor(
        initial.shape,
        dtype=initial.dtype,
        device=initial.device,
        requires_grad=initial.requires_grad,
        expression=_Expr("loop_argument", ()),
        version=initial.version,
    )
    output = body(state)
    if not isinstance(output, Tensor):
        raise TypeError("repeat body must return one graphforge.Tensor")
    if (output.shape != initial.shape or output.dtype is not initial.dtype or
            output.device != initial.device):
        raise ValueError(
            "repeat body must preserve state shape, dtype, and device")
    captures = _captures(output, state)
    return Tensor(
        initial.shape,
        dtype=initial.dtype,
        device=initial.device,
        requires_grad=initial.requires_grad or output.requires_grad,
        expression=_Expr(
            "repeat",
            (initial, *captures),
            (("iterations", iterations),),
            _Region((state, *captures), output),
        ),
        version=max(initial.version, output.version),
    )


__all__ = ["repeat"]
