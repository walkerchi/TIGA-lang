"""Structured compiler control flow over Tiga semantic values."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextvars import ContextVar

from ..tensor import Tensor
from ..tensor.core import _Expr, _Region


_STAGING: ContextVar[bool] = ContextVar(
    "tiga_control_staging", default=False
)


def staging_active() -> bool:
    """True while a control region body or condition is being traced.

    Kernel calls made while staging have per-iteration semantics, so they
    must take the inline Tensor IR path instead of registering as top-level
    leaves of an enclosing GraphProgram.
    """
    return _STAGING.get()


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
        if expression is None or expression.op == "program_value":
            # A program SSA leaf referenced from a region is an immutable
            # external capture, exactly like a plain input Tensor.
            ordered.append(value)
            return
        if expression.op == "loop_argument":
            raise ValueError("control region used a loop argument from another region")
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
    staging_token = _STAGING.set(True)
    try:
        returned = body(*states)
    finally:
        _STAGING.reset(staging_token)
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


def while_loop(
    initial: Tensor | Sequence[Tensor],
    condition: Callable[..., Tensor],
    body: Callable[..., Tensor | Sequence[Tensor]],
    *,
    max_iterations: int,
) -> Tensor | tuple[Tensor, ...]:
    """Capture bounded data-dependent control without a host scalar check.

    ``condition`` and ``body`` are each traced once over the same carried
    Tensor placeholders. The condition must return a rank-zero boolean Tensor.
    ``max_iterations`` is mandatory and remains in semantic IR as a finite
    resource/specialization guard even when the loop converges earlier.
    """
    single = isinstance(initial, Tensor)
    initials = (initial,) if single else tuple(initial)
    if not initials or any(not isinstance(item, Tensor) for item in initials):
        raise TypeError(
            "while_loop initial state must be a Tensor or non-empty Tensor sequence")
    if not callable(condition) or not callable(body):
        raise TypeError("while_loop condition and body must be callable")
    if not isinstance(max_iterations, int) or isinstance(max_iterations, bool):
        raise TypeError("while_loop max_iterations must be an integer")
    if max_iterations < 0:
        raise ValueError("while_loop max_iterations must be non-negative")

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
    staging_token = _STAGING.set(True)
    try:
        predicate = condition(*states)
    finally:
        _STAGING.reset(staging_token)
    if not isinstance(predicate, Tensor):
        raise TypeError("while_loop condition must return a Tensor")
    if predicate.shape != () or predicate.dtype.name != "bool":
        raise ValueError(
            "while_loop condition must return a rank-zero boolean Tensor")
    if predicate.device != initials[0].device:
        raise ValueError("while_loop condition must use the state device")
    staging_token = _STAGING.set(True)
    try:
        returned = body(*states)
    finally:
        _STAGING.reset(staging_token)
    outputs = (returned,) if isinstance(returned, Tensor) else tuple(returned)
    if len(outputs) != len(initials) or any(
        not isinstance(item, Tensor) for item in outputs
    ):
        raise TypeError(
            "while_loop body must return one Tensor per loop-carried state")
    for index, (output, item) in enumerate(zip(outputs, initials, strict=True)):
        if (output.shape != item.shape or output.dtype is not item.dtype or
                output.device != item.device):
            raise ValueError(
                f"while_loop body result {index} must preserve its state "
                "shape, dtype, and device")
    if any(item.device != initials[0].device for item in initials):
        raise ValueError("while_loop carried states must use one device")
    captures = _captures((predicate, *outputs), states)
    region = _Region((*states, *captures), outputs, condition=predicate)
    results = tuple(
        Tensor(
            item.shape,
            dtype=item.dtype,
            device=item.device,
            requires_grad=item.requires_grad or output.requires_grad,
            expression=_Expr(
                "while",
                (*initials, *captures),
                (
                    ("max_iterations", max_iterations),
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


class Repeat:
    """Class-based fixed-count control: subclass and override ``body``.

    The class form is the primary spelling, matching the MessagePassing
    builder convention: instances carry their captured constants as ordinary
    attributes, can be named, reused and inspected, while the functional
    ``repeat`` remains the anonymous shorthand. Both lower to the same
    ``gf_control.repeat`` op; ``body`` is traced exactly once either way.

    ```python
    class Integrate(tg.control.Repeat):
        def __init__(self, rate):
            self.rate = rate

        def body(self, value):
            return value + self.rate * value

    result = Integrate(0.1)(initial, iterations=100)
    ```
    """

    def body(self, *states: Tensor) -> Tensor | Sequence[Tensor]:
        raise NotImplementedError

    def __call__(
        self,
        initial: Tensor | Sequence[Tensor],
        *,
        iterations: int,
    ) -> Tensor | tuple[Tensor, ...]:
        return repeat(initial, self.body, iterations=iterations)


class While:
    """Class-based bounded data-dependent control.

    ``condition`` and ``body`` are each traced once over the same carried
    Tensor placeholders; the condition must return a rank-zero boolean
    Tensor. ``max_iterations`` is mandatory and remains in semantic IR as a
    finite resource/specialization guard even when the loop converges
    earlier. Lowers to the same ``gf_control.while`` op as ``while_loop``.
    """

    def condition(self, *states: Tensor) -> Tensor:
        raise NotImplementedError

    def body(self, *states: Tensor) -> Tensor | Sequence[Tensor]:
        raise NotImplementedError

    def __call__(
        self,
        initial: Tensor | Sequence[Tensor],
        *,
        max_iterations: int,
    ) -> Tensor | tuple[Tensor, ...]:
        return while_loop(
            initial, self.condition, self.body, max_iterations=max_iterations)


__all__ = ["Repeat", "While", "repeat", "while_loop"]
