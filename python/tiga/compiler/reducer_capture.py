"""Typed Python UDF capture for framework-independent reducer kernels."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

from .capture import Expr, as_expr
from ..tensor import DType, float32


@dataclass(frozen=True)
class ReducerRegion:
    arguments: int
    results: tuple[Expr, ...]


@dataclass(frozen=True)
class ReducerDescriptor:
    symbol: str
    name: str
    kind: str
    associative: bool
    commutative: bool
    message_dtypes: tuple[str, ...]
    state_dtypes: tuple[str, ...]
    result_dtypes: tuple[str, ...]
    identity: ReducerRegion
    lift: ReducerRegion
    combine: ReducerRegion
    finalize: ReducerRegion
    block_prune_threshold: float | None = None


def _flatten(value) -> tuple[object, ...]:
    if isinstance(value, tuple):
        result: list[object] = []
        for item in value:
            result.extend(_flatten(item))
        return tuple(result)
    if isinstance(value, list):
        return _flatten(tuple(value))
    return (value,)


def _expressions(value) -> tuple[Expr, ...]:
    return tuple(as_expr(item) for item in _flatten(value))


def _state(values: Sequence[Expr]):
    return values[0] if len(values) == 1 else tuple(values)


def capture_reducer(
    reducer,
    *,
    message_dtypes: Sequence[DType | str] = (float32,),
    symbol: str | None = None,
) -> ReducerDescriptor:
    """Trace one Reducer subclass into four typed algebra regions.

    The initial native slice supports scalar messages/state. The descriptor is
    typed data consumed by C++ OpBuilder; it is not MLIR source code.
    """
    dtypes = tuple(message_dtypes)
    if not dtypes or any(not isinstance(dtype, (DType, str)) for dtype in dtypes):
        raise TypeError(
            "message_dtypes must contain Tiga DType objects or vector spellings"
        )
    spellings = tuple(
        dtype.name if isinstance(dtype, DType) else dtype for dtype in dtypes
    )
    if any(
        spelling != "float32"
        and not re.fullmatch(r"vector:[1-9][0-9]*:float32", spelling)
        for spelling in spellings
    ):
        raise NotImplementedError(
            "initial UDF reducer capture supports scalar/vector f32"
        )
    if len(set(spellings)) != 1:
        raise NotImplementedError(
            "initial UDF reducer capture requires one common message type"
        )
    if not reducer.associative:
        raise ValueError(
            "parallel Reducer subclasses must explicitly set associative = True"
        )

    message = tuple(Expr("reducer_arg", (index,)) for index in range(len(dtypes)))
    identity = _expressions(reducer.identity())
    lifted = _expressions(reducer.lift(*message))
    if len(identity) != len(lifted):
        raise TypeError("Reducer.identity() and lift() must produce the same state")
    left = tuple(
        Expr("reducer_arg", (index,)) for index in range(len(identity))
    )
    right = tuple(
        Expr("reducer_arg", (len(identity) + index,))
        for index in range(len(identity))
    )
    combined = _expressions(reducer.combine(_state(left), _state(right)))
    if len(combined) != len(identity):
        raise TypeError("Reducer.combine() must preserve the state structure")
    final_state = tuple(
        Expr("reducer_arg", (index,)) for index in range(len(identity))
    )
    finalized = _expressions(reducer.finalize(_state(final_state)))
    if not finalized:
        raise TypeError("Reducer.finalize() must produce at least one result")
    raw_symbol = symbol or f"{type(reducer).__module__}.{type(reducer).__qualname__}"
    clean_symbol = re.sub(r"[^A-Za-z0-9_$.-]", "_", raw_symbol)
    if not clean_symbol or clean_symbol[0].isdigit():
        clean_symbol = "reducer_" + clean_symbol
    return ReducerDescriptor(
        symbol=clean_symbol,
        name=reducer.name,
        kind="algebraic",
        associative=bool(reducer.associative),
        commutative=bool(reducer.commutative),
        message_dtypes=spellings,
        state_dtypes=tuple(spellings[0] for _ in identity),
        result_dtypes=tuple(spellings[0] for _ in finalized),
        identity=ReducerRegion(0, identity),
        lift=ReducerRegion(len(message), lifted),
        combine=ReducerRegion(2 * len(identity), combined),
        finalize=ReducerRegion(len(identity), finalized),
    )


def capture_online_softmax(
    *, width: int, symbol: str, block_prune_threshold: float | None = None
) -> ReducerDescriptor:
    """Instantiate the built-in stable tuple algebra for one payload width.

    This returns typed expression data.  The native frontend constructs all
    reducer operations with OpBuilder; no MLIR syntax is emitted in Python.
    """
    if width not in (16, 32, 64, 128):
        raise ValueError("online softmax payload width must be 16, 32, 64, or 128")
    vector16 = f"vector:{width}:float16"
    vector32 = f"vector:{width}:float32"

    def arg(index: int) -> Expr:
        return Expr("reducer_arg", (index,))

    def constant(value: float, dtype: str = "float32") -> Expr:
        return Expr("typed_constant", (value, dtype))

    def cast(value: Expr, dtype: str) -> Expr:
        return Expr("cast", (value, dtype))

    def broadcast(value: Expr, dtype: str) -> Expr:
        return Expr("broadcast", (value, dtype))

    m0, l0, o0, m1, l1, o1 = (arg(index) for index in range(6))
    maximum = m0.maximum(m1)
    a0 = (m0 - maximum).exp()
    a1 = (m1 - maximum).exp()
    denominator = a0 * l0 + a1 * l1
    numerator = broadcast(a0, vector32) * o0 + broadcast(a1, vector32) * o1
    m, denominator_arg, numerator_arg = arg(0), arg(1), arg(2)
    normalized = numerator_arg / broadcast(denominator_arg, vector32)
    return ReducerDescriptor(
        symbol=symbol,
        name="online_softmax",
        # Scheduling is selected by the typed algebra below, not this label.
        kind="algebraic",
        associative=True,
        commutative=True,
        message_dtypes=("float32", vector16),
        state_dtypes=("float32", "float32", vector32),
        result_dtypes=(vector16,),
        identity=ReducerRegion(
            0,
            (constant(-1.0e30), constant(0.0), constant(0.0, vector32)),
        ),
        lift=ReducerRegion(
            2,
            (arg(0), constant(1.0), cast(arg(1), vector32)),
        ),
        combine=ReducerRegion(6, (maximum, denominator, numerator)),
        finalize=ReducerRegion(3, (cast(normalized, vector16),)),
        block_prune_threshold=block_prune_threshold,
    )


def reducer_mlir(reducer, *, message_dtypes=(float32,), symbol=None) -> str:
    from .native import reducer_ir

    return reducer_ir(
        capture_reducer(
            reducer, message_dtypes=message_dtypes, symbol=symbol
        )
    )


__all__ = [
    "ReducerDescriptor",
    "ReducerRegion",
    "capture_online_softmax",
    "capture_reducer",
    "reducer_mlir",
]
