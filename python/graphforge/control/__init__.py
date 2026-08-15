"""Structured compiler control flow over GraphForge semantic values."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from ..tensor import Tensor
from ..tensor.core import _Expr, _Region


def _captures(
    outputs: tuple[Tensor, ...], states: tuple[Tensor, ...]
) -> tuple[Tensor, ...]:
    ordered: list[Tensor] = []
    visited: set[int] = set()
    state_ids = {id(state) for state in states}

    def visit(value: Tensor) -> None:
        if id(value) in state_ids or id(value) in visited:
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

    for output in outputs:
        visit(output)
    return tuple(ordered)


def repeat(
    initial: Tensor | Sequence[Tensor],
    body: Callable[..., Tensor | Sequence[Tensor]],
    *,
    iterations: int,
) -> Tensor | tuple[Tensor, ...]:
    """Capture a fixed-count loop with one or more Tensor-carried states.

    ``body`` is traced once and receives one positional argument per carried
    state. Its other Tensor leaves become explicit immutable region captures;
    returned Tensors stay lazy and trigger ordinary JIT at observation. A
    single initial Tensor preserves the original single-result API, while a
    tuple/list returns a tuple. This is compiler control flow, not a Python
    execution loop and not an algorithm-specific operator.
    """
    single = isinstance(initial, Tensor)
    initials = (initial,) if single else tuple(initial)
    if not initials or any(not isinstance(item, Tensor) for item in initials):
        raise TypeError(
            "repeat initial state must be a Tensor or non-empty Tensor sequence")
    if not callable(body):
        raise TypeError("repeat body must be callable")
    if not isinstance(iterations, int) or isinstance(iterations, bool):
        raise TypeError("repeat iterations must be an integer")
    if iterations < 0:
        raise ValueError("repeat iterations must be non-negative")

    states = tuple(
        Tensor(
            item.shape,
            dtype=item.dtype,
            device=item.device,
            requires_grad=item.requires_grad,
            expression=_Expr("loop_argument", ()),
            version=item.version,
        )
        for item in initials
    )
    returned = body(*states)
    outputs = (returned,) if isinstance(returned, Tensor) else tuple(returned)
    if len(outputs) != len(initials) or any(
        not isinstance(item, Tensor) for item in outputs
    ):
        raise TypeError(
            "repeat body must return one Tensor per loop-carried state")
    for index, (output, item) in enumerate(zip(outputs, initials, strict=True)):
        if (output.shape != item.shape or output.dtype is not item.dtype or
                output.device != item.device):
            raise ValueError(
                f"repeat body result {index} must preserve its state shape, "
                "dtype, and device")
    captures = _captures(outputs, states)
    region = _Region((*states, *captures), outputs)
    results = tuple(
        Tensor(
            item.shape,
            dtype=item.dtype,
            device=item.device,
            requires_grad=item.requires_grad or output.requires_grad,
            expression=_Expr(
                "repeat",
                (*initials, *captures),
                (
                    ("iterations", iterations),
                    ("num_carried", len(initials)),
                    ("result_index", index),
                ),
                region,
            ),
            version=max(item.version, output.version),
        )
        for index, (item, output) in enumerate(
            zip(initials, outputs, strict=True)
        )
    )
    region.results = results
    return results[0] if single else results


__all__ = ["repeat"]
