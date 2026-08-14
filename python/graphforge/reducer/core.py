"""Framework-independent reducer kernels.

A reducer is executable compiler input, not an eager tensor operation.  Its
four methods are captured into ``gf.reducer`` regions and fused with the
relation traversal before provider lowering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..kernel import Kernel


@dataclass(frozen=True)
class ReducerCall:
    """A staged binding between one reducer and its edge-local messages."""

    reducer: "Reducer"
    messages: tuple[Any, ...]


class Reducer(Kernel):
    """Base class for user-defined reducer kernels.

    Subclasses implement a pure scalar/vector algebra. ``identity`` creates
    state, ``lift`` maps one edge message to state, ``combine`` merges two
    states, and ``finalize`` maps state to the node result.  State may be a
    scalar or a tuple. Parallel lowering requires an explicit associativity
    declaration; GraphForge never attempts to prove it from Python source.
    """

    name = "custom"
    associative = False
    commutative = False
    deterministic = False

    def __init__(self, *, deterministic: bool = False) -> None:
        super().__init__()
        self.deterministic = bool(deterministic)

    def identity(self):
        raise NotImplementedError

    def lift(self, *messages):
        return messages[0] if len(messages) == 1 else tuple(messages)

    def combine(self, left, right):
        raise NotImplementedError

    def finalize(self, state):
        return state

    def __call__(self, *messages: Any) -> ReducerCall:
        if not messages:
            raise TypeError("a reducer binding requires at least one message")
        return ReducerCall(self, tuple(messages))

    def specialization_key(self) -> tuple[object, ...]:
        configuration = tuple(
            sorted(
                (name, repr(value))
                for name, value in vars(self).items()
                if not name.startswith("_") and name != "deterministic"
            )
        )
        return (
            type(self).__module__,
            type(self).__qualname__,
            self.name,
            self.associative,
            self.commutative,
            self.deterministic,
            configuration,
        )

    def mlir(self, *, message_dtypes=None, symbol: str | None = None) -> str:
        """Capture this UDF kernel as a verified native ``gf.reducer`` op."""
        from ..compiler.reducer_capture import reducer_mlir
        from ..tensor import float32

        return reducer_mlir(
            self,
            message_dtypes=(float32,) if message_dtypes is None else message_dtypes,
            symbol=symbol,
        )


class SumReducer(Reducer):
    name = "sum"
    associative = True
    commutative = True

    def __init__(
        self, *, identity: float | int = 0, deterministic: bool = False
    ) -> None:
        super().__init__(deterministic=deterministic)
        self.identity_value = identity

    def identity(self):
        return self.identity_value

    def combine(self, left, right):
        return left + right

    def specialization_key(self) -> tuple[object, ...]:
        # Preserve the compact public key used by existing cache files.
        return self.name, self.identity_value, self.deterministic


@dataclass(frozen=True)
class OnlineSoftmaxItem:
    """Staged score/payload pair accepted by ``OnlineSoftmaxReducer``."""

    score: Any
    value: Any


class OnlineSoftmaxReducer(Reducer):
    """Stable tuple-state softmax-weighted reducer kernel.

    The typed compiler instantiates its state as ``(maximum, denominator,
    numerator)`` after it knows score lanes and payload width.
    """

    name = "online_softmax"
    associative = True
    commutative = True

    def __init__(
        self,
        *,
        accumulation_dtype: object | None = None,
        deterministic: bool = False,
        block_prune_threshold: float | None = None,
    ) -> None:
        super().__init__(deterministic=deterministic)
        if block_prune_threshold is not None and not (
            0.0 < float(block_prune_threshold) <= 1.0
        ):
            raise ValueError("block_prune_threshold must be in (0, 1]")
        self.accumulation_dtype = accumulation_dtype
        # This is a semantic approximation policy, not a placement/schedule
        # hint.  A dense streaming implementation may skip a whole score tile
        # when its maximum softmax weight is below this ratio relative to the
        # running block maximum.  None retains exact online softmax.
        self.block_prune_threshold = (
            None if block_prune_threshold is None
            else float(block_prune_threshold)
        )

    def __call__(self, score: Any, value: Any) -> OnlineSoftmaxItem:
        return OnlineSoftmaxItem(score=score, value=value)

    def specialization_key(self) -> tuple[object, ...]:
        return (
            self.name,
            str(self.accumulation_dtype),
            self.deterministic,
            self.block_prune_threshold,
        )


def sum(
    *, identity: float | int = 0, deterministic: bool = False
) -> SumReducer:
    return SumReducer(identity=identity, deterministic=deterministic)


def online_softmax(
    *,
    accumulation_dtype: object | None = None,
    deterministic: bool = False,
    block_prune_threshold: float | None = None,
) -> OnlineSoftmaxReducer:
    return OnlineSoftmaxReducer(
        accumulation_dtype=accumulation_dtype,
        deterministic=deterministic,
        block_prune_threshold=block_prune_threshold,
    )


__all__ = [
    "OnlineSoftmaxItem",
    "OnlineSoftmaxReducer",
    "Reducer",
    "ReducerCall",
    "SumReducer",
    "online_softmax",
    "sum",
]
