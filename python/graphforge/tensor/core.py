"""Minimal Torch-independent Tensor frontend for GraphForge.

Tensor operations capture a shape-aware expression DAG.  Native CPU storage
supports strided zero-copy views; the Python evaluator remains a correctness
oracle while canonical MLIR/code generation is brought up.
"""

from __future__ import annotations

import builtins
import ctypes
import math
import os
import struct
import time
from dataclasses import dataclass
from typing import Iterable, Sequence

from ..runtime import Buffer, Device, Event
from ..runtime import DeviceType


class _CComplex64(ctypes.Structure):
    _fields_ = [("real", ctypes.c_float), ("imag", ctypes.c_float)]


class _CComplex128(ctypes.Structure):
    _fields_ = [("real", ctypes.c_double), ("imag", ctypes.c_double)]


@dataclass(frozen=True)
class DType:
    name: str
    itemsize: int
    ctype: type
    kind: str

    def __repr__(self) -> str:
        return f"gf.{self.name}"


float16 = DType("float16", 2, ctypes.c_uint16, "float")
float32 = DType("float32", 4, ctypes.c_float, "float")
float64 = DType("float64", 8, ctypes.c_double, "float")
int32 = DType("int32", 4, ctypes.c_int32, "int")
int64 = DType("int64", 8, ctypes.c_int64, "int")
complex64 = DType("complex64", 8, _CComplex64, "complex")
complex128 = DType("complex128", 16, _CComplex128, "complex")
bool = DType("bool", 1, ctypes.c_bool, "bool")

_DTYPES = {
    item.name: item
    for item in (
        float16, float32, float64, int32, int64, complex64, complex128, bool
    )
}


def _ieee_divide(left, right):
    """Match floating provider division instead of Python's zero exception."""
    try:
        return left / right
    except ZeroDivisionError:
        if isinstance(left, complex) or isinstance(right, complex):
            return complex(math.nan, math.nan)
        left_value = float(left)
        right_value = float(right)
        if left_value == 0.0 or math.isnan(left_value):
            return math.nan
        sign = math.copysign(1.0, left_value) * math.copysign(1.0, right_value)
        return math.copysign(math.inf, sign)


@dataclass(frozen=True)
class _Expr:
    op: str
    operands: tuple[Tensor, ...]
    attrs: tuple[tuple[str, object], ...] = ()
    region: _Region | None = None

    def attr(self, name: str) -> object:
        for key, value in self.attrs:
            if key == name:
                return value
        raise KeyError(name)


@dataclass(frozen=True)
class _Region:
    """One captured semantic region with explicit Tensor arguments."""

    arguments: tuple[Tensor, ...]
    output: Tensor


def _normalize_shape(shape: int | Sequence[int]) -> tuple[int, ...]:
    normalized = (shape,) if isinstance(shape, int) else tuple(shape)
    for extent in normalized:
        if not isinstance(extent, int) or isinstance(extent, builtins.bool) or extent < 0:
            raise ValueError("Tensor shape must contain non-negative integers")
    return normalized


def _numel(shape: tuple[int, ...]) -> int:
    return math.prod(shape)


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides: list[int] = []
    running = 1
    for extent in reversed(shape):
        strides.append(running)
        running *= extent
    return tuple(reversed(strides))


def _resolve_reshape(
    arguments: tuple[int | Sequence[int], ...], numel: int
) -> tuple[int, ...]:
    if len(arguments) == 1 and isinstance(arguments[0], Sequence):
        raw = tuple(arguments[0])
    else:
        raw = tuple(arguments)
    inferred = [index for index, extent in enumerate(raw) if extent == -1]
    if len(inferred) > 1:
        raise ValueError("reshape accepts at most one inferred dimension")
    if any(
        not isinstance(extent, int)
        or isinstance(extent, builtins.bool)
        or extent < -1
        for extent in raw
    ):
        raise ValueError("reshape dimensions must be non-negative or -1")
    resolved = list(raw)
    if inferred:
        known = math.prod(extent for extent in raw if extent != -1)
        if known == 0 or numel % known:
            raise ValueError("cannot infer a reshape dimension for this Tensor")
        resolved[inferred[0]] = numel // known
    normalized = tuple(resolved)
    if _numel(normalized) != numel:
        raise ValueError("reshape cannot change the number of elements")
    return normalized


def _normalize_axes(
    axis: int | Sequence[int] | None, rank: int
) -> tuple[int, ...]:
    if axis is None:
        return tuple(range(rank))
    raw = (axis,) if isinstance(axis, int) else tuple(axis)
    normalized: list[int] = []
    for item in raw:
        if not isinstance(item, int) or isinstance(item, builtins.bool):
            raise TypeError("reduction axes must be integers")
        candidate = item + rank if item < 0 else item
        if candidate < 0 or candidate >= rank:
            raise ValueError(f"axis {item} is out of range for rank {rank}")
        if candidate in normalized:
            raise ValueError("reduction axes must be unique")
        normalized.append(candidate)
    return tuple(sorted(normalized))


def _normalize_permutation(axes: Sequence[int], rank: int) -> tuple[int, ...]:
    if len(axes) != rank:
        raise ValueError("permute requires exactly one axis per Tensor dimension")
    normalized = tuple(axis + rank if axis < 0 else axis for axis in axes)
    if sorted(normalized) != list(range(rank)):
        raise ValueError("permute axes must be a permutation of Tensor dimensions")
    return normalized


def _broadcast_shape(lhs: tuple[int, ...], rhs: tuple[int, ...]) -> tuple[int, ...]:
    rank = max(len(lhs), len(rhs))
    left = (1,) * (rank - len(lhs)) + lhs
    right = (1,) * (rank - len(rhs)) + rhs
    output: list[int] = []
    for left_extent, right_extent in zip(left, right):
        if left_extent == right_extent:
            output.append(left_extent)
        elif left_extent == 1:
            output.append(right_extent)
        elif right_extent == 1:
            output.append(left_extent)
        else:
            raise ValueError(f"shapes {lhs} and {rhs} are not broadcast-compatible")
    return tuple(output)


def _broadcast_strides(
    shape: tuple[int, ...], strides: tuple[int, ...], target: tuple[int, ...]
) -> tuple[int, ...]:
    if len(shape) > len(target):
        raise ValueError(f"cannot broadcast shape {shape} to {target}")
    padded_shape = (1,) * (len(target) - len(shape)) + shape
    padded_strides = (0,) * (len(target) - len(shape)) + strides
    output: list[int] = []
    for source_extent, target_extent, stride in zip(
        padded_shape, target, padded_strides
    ):
        if source_extent == target_extent:
            output.append(stride)
        elif source_extent == 1:
            output.append(0)
        else:
            raise ValueError(f"cannot broadcast shape {shape} to {target}")
    return tuple(output)


def _flatten_data(value: object) -> tuple[tuple[int, ...], list[object]]:
    if isinstance(value, (list, tuple)):
        if not value:
            return (0,), []
        children = [_flatten_data(item) for item in value]
        child_shape = children[0][0]
        if any(shape != child_shape for shape, _ in children[1:]):
            raise ValueError("Tensor input must be rectangular")
        flattened: list[object] = []
        for _, items in children:
            flattened.extend(items)
        return (len(value), *child_shape), flattened
    if isinstance(value, (builtins.bool, int, float, complex)):
        return (), [value]
    raise TypeError("Tensor data must be a scalar or nested list/tuple")


def _infer_dtype(values: Iterable[object]) -> DType:
    seen = list(values)
    if not seen:
        return float32
    if any(isinstance(value, complex) for value in seen):
        return complex64
    if any(isinstance(value, float) for value in seen):
        return float32
    if any(
        isinstance(value, int) and not isinstance(value, builtins.bool)
        for value in seen
    ):
        return int64
    return bool


def _logical_offsets(shape: tuple[int, ...], strides: tuple[int, ...]):
    """Yield storage offsets in row-major logical order."""
    if _numel(shape) == 0:
        return
    if not shape:
        yield 0
        return
    coordinate = [0] * len(shape)
    for _ in range(_numel(shape)):
        yield sum(index * stride for index, stride in zip(coordinate, strides))
        for axis in range(len(shape) - 1, -1, -1):
            coordinate[axis] += 1
            if coordinate[axis] < shape[axis]:
                break
            coordinate[axis] = 0


def _broadcast_indices(
    source_shape: tuple[int, ...], target_shape: tuple[int, ...]
):
    source_strides = _contiguous_strides(source_shape)
    projected = _broadcast_strides(source_shape, source_strides, target_shape)
    return _logical_offsets(target_shape, projected)


class Tensor:
    """A logical strided value backed by a GraphForge runtime allocation."""

    def __init__(
        self,
        shape: int | Sequence[int],
        *,
        dtype: DType = float32,
        device: Device | str = "cpu",
        buffer: Buffer | None = None,
        offset: int = 0,
        strides: Sequence[int] | None = None,
        requires_grad: bool = False,
        expression: _Expr | None = None,
        version: int = 0,
        ready_event: Event | None = None,
    ) -> None:
        self.shape = _normalize_shape(shape)
        if dtype.name not in _DTYPES or _DTYPES[dtype.name] is not dtype:
            raise TypeError("dtype must be a GraphForge DType")
        self.dtype = dtype
        self.device = Device.parse(device)
        self.offset = offset
        self.strides = (
            _contiguous_strides(self.shape) if strides is None else tuple(strides)
        )
        if len(self.strides) != len(self.shape):
            raise ValueError("strides must match Tensor rank")
        if offset < 0 or any(stride < 0 for stride in self.strides):
            raise ValueError("negative offsets/strides are not supported yet")
        if not isinstance(version, int) or version < 0:
            raise ValueError("Tensor version must be non-negative")
        self.requires_grad = builtins.bool(requires_grad)
        if self.requires_grad and dtype.kind not in {"float", "complex"}:
            raise TypeError("only floating-point or complex Tensors can require gradients")
        self.version = version
        self.ready_event = ready_event
        self._expr = expression
        self._buffer = buffer
        self._execution_info: dict[str, object] | None = None
        self._prepared_launch = None
        if expression is None:
            self._jit_key = (
                "input", id(self), self.shape, self.strides, self.offset,
                self.dtype.name, str(self.device),
            )
        else:
            region_key = None
            if expression.region is not None:
                region_key = (
                    tuple(argument._jit_key for argument in expression.region.arguments),
                    expression.region.output._jit_key,
                )
            self._jit_key = (
                expression.op, self.shape, self.dtype.name, expression.attrs,
                tuple(operand._jit_key for operand in expression.operands),
                region_key,
            )
        if expression is None and buffer is None and self.device.type == DeviceType.CPU:
            self._buffer = Buffer(self.nbytes, device=self.device, _pooled=True)
        if expression is None and buffer is None and self.device.type != DeviceType.CPU:
            raise RuntimeError(
                f"direct Tensor construction for {self.device} requires a provider "
                "allocation; use gf.empty() or gf.from_torch()"
            )
        if (self._buffer is not None and self.ready_event is None and
                self.device.type == DeviceType.CPU):
            self.ready_event = Event(self.device, _pooled=True)
        if self._buffer is not None and self._required_bytes() > self._buffer.nbytes:
            raise ValueError("Tensor view exceeds its physical buffer")

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def numel(self) -> int:
        return _numel(self.shape)

    @property
    def nbytes(self) -> int:
        return self.numel * self.dtype.itemsize

    @property
    def is_leaf(self) -> bool:
        return self._expr is None

    @property
    def is_contiguous(self) -> bool:
        return self.strides == _contiguous_strides(self.shape)

    @property
    def T(self) -> Tensor:
        if self.ndim < 2:
            return self
        return self.transpose(-2, -1)

    @classmethod
    def scalar(cls, value, *, dtype: DType, device: Device | str) -> Tensor:
        return tensor(value, dtype=dtype, device=device)

    def _required_bytes(self) -> int:
        if self.numel == 0:
            return self.offset * self.dtype.itemsize
        last = self.offset + sum(
            (extent - 1) * stride
            for extent, stride in zip(self.shape, self.strides)
        )
        return (last + 1) * self.dtype.itemsize

    def reshape(self, *shape: int | Sequence[int]) -> Tensor:
        normalized = _resolve_reshape(shape, self.numel)
        shared = self._buffer is not None and self.is_contiguous
        return Tensor(
            normalized,
            dtype=self.dtype,
            device=self.device,
            buffer=self._buffer if shared else None,
            offset=self.offset if shared else 0,
            requires_grad=self.requires_grad,
            expression=_Expr("reshape", (self,), (("shape", normalized),)),
            version=self.version,
            ready_event=self.ready_event,
        )

    def permute(self, *axes: int | Sequence[int]) -> Tensor:
        raw = tuple(axes[0]) if len(axes) == 1 and isinstance(axes[0], Sequence) \
            else tuple(axes)
        permutation = _normalize_permutation(raw, self.ndim)
        shape = tuple(self.shape[axis] for axis in permutation)
        shared = self._buffer is not None
        strides = tuple(self.strides[axis] for axis in permutation) \
            if shared else _contiguous_strides(shape)
        return Tensor(
            shape,
            dtype=self.dtype,
            device=self.device,
            buffer=self._buffer,
            offset=self.offset,
            strides=strides,
            requires_grad=self.requires_grad,
            expression=_Expr("permute", (self,), (("axes", permutation),)),
            version=self.version,
            ready_event=self.ready_event,
        )

    def transpose(self, dim0: int, dim1: int) -> Tensor:
        rank = self.ndim
        first = dim0 + rank if dim0 < 0 else dim0
        second = dim1 + rank if dim1 < 0 else dim1
        if first < 0 or first >= rank or second < 0 or second >= rank:
            raise ValueError("transpose dimension is out of range")
        axes = list(range(rank))
        axes[first], axes[second] = axes[second], axes[first]
        return self.permute(axes)

    swapaxes = transpose

    def unsqueeze(self, dim: int) -> Tensor:
        axis = dim + self.ndim + 1 if dim < 0 else dim
        if axis < 0 or axis > self.ndim:
            raise ValueError("unsqueeze dimension is out of range")
        shape = list(self.shape)
        shape.insert(axis, 1)
        return self.reshape(shape)

    def squeeze(self, dim: int | Sequence[int] | None = None) -> Tensor:
        if dim is None:
            axes = tuple(
                axis for axis, extent in enumerate(self.shape) if extent == 1
            )
        else:
            axes = _normalize_axes(dim, self.ndim)
            axes = tuple(axis for axis in axes if self.shape[axis] == 1)
        shape = tuple(
            extent for axis, extent in enumerate(self.shape) if axis not in axes
        )
        return self.reshape(shape)

    def broadcast_to(self, shape: int | Sequence[int]) -> Tensor:
        target = _normalize_shape(shape)
        view_strides = _broadcast_strides(self.shape, self.strides, target)
        shared = self._buffer is not None
        return Tensor(
            target,
            dtype=self.dtype,
            device=self.device,
            buffer=self._buffer,
            offset=self.offset,
            strides=view_strides if shared else _contiguous_strides(target),
            requires_grad=self.requires_grad,
            expression=_Expr("broadcast", (self,), (("shape", target),)),
            version=self.version,
            ready_event=self.ready_event,
        )

    def expand(self, *shape: int | Sequence[int]) -> Tensor:
        target = tuple(shape[0]) if len(shape) == 1 and isinstance(shape[0], Sequence) \
            else tuple(shape)
        return self.broadcast_to(target)

    def __add__(self, other: Tensor | int | float | complex) -> Tensor:
        return self._binary("add", other)

    def __radd__(self, other: Tensor | int | float | complex) -> Tensor:
        return self._binary("add", other)

    def __sub__(self, other: Tensor | int | float | complex) -> Tensor:
        rhs = other if isinstance(other, Tensor) else tensor(
            other, dtype=self.dtype, device=self.device)
        return self + (-rhs)

    def __rsub__(self, other: Tensor | int | float | complex) -> Tensor:
        lhs = other if isinstance(other, Tensor) else tensor(
            other, dtype=self.dtype, device=self.device)
        return lhs + (-self)

    def __ne__(self, other: object) -> Tensor:  # type: ignore[override]
        if not isinstance(other, (Tensor, int, float, complex, builtins.bool)):
            return NotImplemented
        rhs = other if isinstance(other, Tensor) else tensor(
            other, dtype=self.dtype, device=self.device)
        if self.device != rhs.device or self.dtype is not rhs.dtype:
            raise ValueError("comparison operands require identical device and dtype")
        return Tensor(
            _broadcast_shape(self.shape, rhs.shape),
            dtype=bool,
            device=self.device,
            requires_grad=False,
            expression=_Expr("not_equal", (self, rhs)),
            version=max(self.version, rhs.version),
        )

    def __mul__(self, other: Tensor | int | float | complex) -> Tensor:
        return self._binary("mul", other)

    def __rmul__(self, other: Tensor | int | float | complex) -> Tensor:
        return self._binary("mul", other)

    def __truediv__(self, other: Tensor | int | float | complex) -> Tensor:
        if self.dtype.kind not in {"float", "complex"}:
            raise TypeError("Tensor division requires floating-point or complex dtype")
        return self._binary("div", other)

    def __rtruediv__(self, other: Tensor | int | float | complex) -> Tensor:
        if self.dtype.kind not in {"float", "complex"}:
            raise TypeError("Tensor division requires floating-point or complex dtype")
        lhs = other if isinstance(other, Tensor) else tensor(
            other, dtype=self.dtype, device=self.device)
        return lhs._binary("div", self)

    def __matmul__(self, other: Tensor) -> Tensor:
        return self.matmul(other)

    def matmul(self, other: Tensor) -> Tensor:
        """Matrix product captured as a first-class Tensor IR operation.

        The initial compiler contract is deliberately rank-two.  Keeping the
        contraction visible (instead of expanding it into broadcast/mul/sum)
        lets CPU lowering generate a reduction loop and GPU lowering select
        tensor-core ``tt.dot`` tiles.
        """
        if not isinstance(other, Tensor):
            raise TypeError("Tensor.matmul requires another graphforge.Tensor")
        if self.ndim != 2 or other.ndim != 2:
            raise ValueError("Tensor.matmul currently requires rank-two operands")
        if self.shape[1] != other.shape[0]:
            raise ValueError(
                "Tensor.matmul contraction dimensions must match: "
                f"{self.shape[1]} != {other.shape[0]}"
            )
        if self.device != other.device or self.dtype is not other.dtype:
            raise ValueError("Tensor.matmul operands require identical device and dtype")
        if self.dtype.kind not in {"float", "complex"}:
            raise TypeError("Tensor.matmul requires floating-point or complex dtype")
        return Tensor(
            (self.shape[0], other.shape[1]),
            dtype=self.dtype,
            device=self.device,
            requires_grad=self.requires_grad or other.requires_grad,
            expression=_Expr("matmul", (self, other)),
            version=max(self.version, other.version),
        )

    def __neg__(self) -> Tensor:
        return Tensor(
            self.shape,
            dtype=self.dtype,
            device=self.device,
            requires_grad=self.requires_grad,
            expression=_Expr("neg", (self,)),
            version=self.version,
        )

    def exp(self) -> Tensor:
        """Elementwise exponential captured in the Tensor IR."""
        if self.dtype.kind != "float":
            raise TypeError("Tensor.exp requires a floating-point dtype")
        return Tensor(
            self.shape,
            dtype=self.dtype,
            device=self.device,
            requires_grad=self.requires_grad,
            expression=_Expr("exp", (self,)),
            version=self.version,
        )

    def sqrt(self) -> Tensor:
        """Elementwise square root captured in the Tensor IR."""
        if self.dtype.kind != "float":
            raise TypeError("Tensor.sqrt requires a floating-point dtype")
        return Tensor(
            self.shape,
            dtype=self.dtype,
            device=self.device,
            requires_grad=self.requires_grad,
            expression=_Expr("sqrt", (self,)),
            version=self.version,
        )

    def cumsum(self, dim: int, *, reverse: bool = False) -> Tensor:
        """Inclusive additive scan along ``dim``.

        The scan remains a first-class Tensor IR operation so providers can
        choose serial, warp, block or hierarchical implementations.
        ``reverse=True`` computes an inclusive suffix sum in the original
        physical result layout.
        """
        axis = dim + self.ndim if dim < 0 else dim
        if self.ndim == 0 or axis < 0 or axis >= self.ndim:
            raise ValueError("cumsum dimension is out of range")
        if self.dtype.kind not in {"int", "uint", "float", "complex"}:
            raise TypeError("Tensor.cumsum requires an arithmetic dtype")
        return Tensor(
            self.shape,
            dtype=self.dtype,
            device=self.device,
            requires_grad=self.requires_grad,
            expression=_Expr(
                "cumsum", (self,),
                (("axis", axis), ("reverse", builtins.bool(reverse)))
            ),
            version=self.version,
        )

    def _binary(self, op: str, other: Tensor | int | float | complex) -> Tensor:
        rhs = other if isinstance(other, Tensor) else tensor(
            other, dtype=self.dtype, device=self.device
        )
        if self.device != rhs.device or self.dtype is not rhs.dtype:
            raise ValueError("binary Tensor operands require identical device and dtype")
        shape = _broadcast_shape(self.shape, rhs.shape)
        return Tensor(
            shape,
            dtype=self.dtype,
            device=self.device,
            requires_grad=self.requires_grad or rhs.requires_grad,
            expression=_Expr(op, (self, rhs)),
            version=max(self.version, rhs.version),
        )

    def conj(self) -> Tensor:
        if self.dtype.kind != "complex":
            return self
        return Tensor(
            self.shape,
            dtype=self.dtype,
            device=self.device,
            requires_grad=self.requires_grad,
            expression=_Expr("conj", (self,)),
            version=self.version,
        )

    conjugate = conj

    def checkpoint(self) -> Tensor:
        """Preserve this value for a later compiled backward program.

        ``checkpoint`` is an identity in the logical Tensor IR. A physical
        memory-planning pass may turn it into a separately materialized ABI
        input, making the save/recompute tradeoff explicit and inspectable.
        """
        return Tensor(
            self.shape,
            dtype=self.dtype,
            device=self.device,
            requires_grad=self.requires_grad,
            expression=_Expr("checkpoint", (self,)),
            version=self.version,
        )

    def _checkpoint_candidate(self) -> Tensor:
        """Offer this value to the compiler's save/recompute planner."""
        return Tensor(
            self.shape,
            dtype=self.dtype,
            device=self.device,
            requires_grad=self.requires_grad,
            expression=_Expr("checkpoint_candidate", (self,)),
            version=self.version,
        )

    def sum(
        self,
        axis: int | Sequence[int] | None = None,
        *,
        keepdims: bool = False,
    ) -> Tensor:
        axes = _normalize_axes(axis, self.ndim)
        if keepdims:
            shape = tuple(
                1 if index in axes else extent
                for index, extent in enumerate(self.shape)
            )
        else:
            shape = tuple(
                extent for index, extent in enumerate(self.shape) if index not in axes
            )
        return Tensor(
            shape,
            dtype=self.dtype,
            device=self.device,
            requires_grad=self.requires_grad,
            expression=_Expr(
                "sum", (self,), (("axes", axes), ("keepdims", builtins.bool(keepdims)))
            ),
            version=self.version,
        )

    def gather(self, index: Tensor) -> Tensor:
        """Gather leading-dimension rows through a relation index Tensor."""
        if not isinstance(index, Tensor) or index.ndim != 1:
            raise TypeError("gather index must be a rank-one graphforge.Tensor")
        if index.dtype.kind != "int":
            raise TypeError("gather index must have an integer dtype")
        if self.ndim == 0:
            raise ValueError("cannot gather a scalar Tensor")
        if self.device != index.device:
            raise ValueError("gather input and index must share a device")
        return Tensor(
            (index.shape[0], *self.shape[1:]),
            dtype=self.dtype,
            device=self.device,
            requires_grad=self.requires_grad,
            expression=_Expr("gather", (self, index)),
            version=max(self.version, index.version),
        )

    def segment_sum(self, index: Tensor, num_segments: int) -> Tensor:
        """Sum leading-dimension messages into destination segments."""
        if not isinstance(index, Tensor) or index.ndim != 1:
            raise TypeError("segment index must be a rank-one graphforge.Tensor")
        if index.dtype.kind != "int":
            raise TypeError("segment index must have an integer dtype")
        if self.ndim == 0 or self.shape[0] != index.shape[0]:
            raise ValueError("message and segment index leading extents must match")
        if self.device != index.device:
            raise ValueError("message and segment index must share a device")
        if (not isinstance(num_segments, int) or
                isinstance(num_segments, builtins.bool) or num_segments < 0):
            raise ValueError("num_segments must be a non-negative integer")
        return Tensor(
            (num_segments, *self.shape[1:]),
            dtype=self.dtype,
            device=self.device,
            requires_grad=self.requires_grad,
            expression=_Expr(
                "segment_sum", (self, index),
                (("num_segments", num_segments),),
            ),
            version=max(self.version, index.version),
        )

    def _scatter_rows(
        self, destination: Tensor, inverse: Tensor, num_rows: int
    ) -> Tensor:
        """Compiler primitive for a proved disjoint row placement.

        ``destination`` maps input rows to output rows and ``inverse`` maps
        output rows back to inputs, using -1 for rows not written by this
        partition. The redundant maps make both forward and VJP linear.
        """
        if self.ndim == 0:
            raise ValueError("cannot scatter rows from a scalar Tensor")
        if (not isinstance(destination, Tensor) or destination.ndim != 1
                or destination.dtype.kind != "int"):
            raise TypeError("scatter destination must be a rank-one integer Tensor")
        if (not isinstance(inverse, Tensor) or inverse.ndim != 1
                or inverse.dtype.kind != "int"):
            raise TypeError("scatter inverse must be a rank-one integer Tensor")
        if destination.shape[0] != self.shape[0]:
            raise ValueError("scatter destination extent must match input rows")
        if (not isinstance(num_rows, int)
                or isinstance(num_rows, builtins.bool) or num_rows < 0
                or inverse.shape[0] != num_rows):
            raise ValueError("scatter inverse extent must equal num_rows")
        if self.device != destination.device or self.device != inverse.device:
            raise ValueError("scatter input and maps must share a device")
        return Tensor(
            (num_rows, *self.shape[1:]),
            dtype=self.dtype,
            device=self.device,
            requires_grad=self.requires_grad,
            expression=_Expr(
                "scatter_rows", (self, destination, inverse),
                (("num_rows", num_rows),),
            ),
            version=max(self.version, destination.version, inverse.version),
        )

    def csr_expand_rows(self, row_ptr: Tensor, num_edges: int) -> Tensor:
        """Expand destination rows over their CSR edge ranges."""
        if not isinstance(row_ptr, Tensor) or row_ptr.ndim != 1 or \
                row_ptr.dtype.kind != "int":
            raise TypeError("row_ptr must be a rank-one integer gf.Tensor")
        if self.ndim == 0 or row_ptr.shape[0] != self.shape[0] + 1:
            raise ValueError("row_ptr extent must equal node rows plus one")
        if self.device != row_ptr.device:
            raise ValueError("input and row_ptr must share a device")
        if (not isinstance(num_edges, int) or
                isinstance(num_edges, builtins.bool) or num_edges < 0):
            raise ValueError("num_edges must be a non-negative integer")
        return Tensor(
            (num_edges, *self.shape[1:]), dtype=self.dtype, device=self.device,
            requires_grad=self.requires_grad,
            expression=_Expr(
                "csr_expand_rows", (self, row_ptr), (("num_edges", num_edges),)),
            version=max(self.version, row_ptr.version),
        )

    def csr_segment_sum(
        self,
        row_ptr: Tensor,
        num_rows: int,
        *,
        degree_bounds: tuple[int, int] | None = None,
    ) -> Tensor:
        """Sum edge messages within each CSR destination row."""
        if not isinstance(row_ptr, Tensor) or row_ptr.ndim != 1 or \
                row_ptr.dtype.kind != "int":
            raise TypeError("row_ptr must be a rank-one integer gf.Tensor")
        if self.ndim == 0 or row_ptr.shape[0] != num_rows + 1:
            raise ValueError("row_ptr extent must equal num_rows plus one")
        if self.device != row_ptr.device:
            raise ValueError("message and row_ptr must share a device")
        degree_min, degree_max = degree_bounds or (0, 0)
        if (not isinstance(degree_min, int) or
                not isinstance(degree_max, int) or
                degree_min < 0 or degree_max < degree_min):
            raise ValueError("degree_bounds must be non-negative (minimum, maximum)")
        return Tensor(
            (num_rows, *self.shape[1:]), dtype=self.dtype, device=self.device,
            requires_grad=self.requires_grad,
            expression=_Expr(
                "csr_segment_sum",
                (self, row_ptr),
                (("num_rows", num_rows),
                 ("degree_min", degree_min),
                 ("degree_max", degree_max)),
            ),
            version=max(self.version, row_ptr.version),
        )

    def _csr_segment_product(self, row_ptr: Tensor, num_rows: int) -> Tensor:
        """Compiler primitive for a structurally proven CSR product reducer."""
        if self.dtype.kind != "float":
            raise TypeError("CSR segment product requires floating-point messages")
        if not isinstance(row_ptr, Tensor) or row_ptr.ndim != 1 or \
                row_ptr.dtype.kind != "int":
            raise TypeError("row_ptr must be a rank-one integer gf.Tensor")
        if self.ndim == 0 or row_ptr.shape[0] != num_rows + 1:
            raise ValueError("row_ptr extent must equal num_rows plus one")
        if self.device != row_ptr.device:
            raise ValueError("message and row_ptr must share a device")
        rows = tuple(int(value) for value in row_ptr.tolist())
        if rows[0] != 0 or rows[-1] != self.shape[0] or any(
            right < left for left, right in zip(rows, rows[1:])
        ):
            raise ValueError("row_ptr does not describe the message edge domain")
        max_degree = max(
            (right - left for left, right in zip(rows, rows[1:])), default=0)
        uniform_degree = all(
            right - left == max_degree for left, right in zip(rows, rows[1:]))
        destination = getattr(
            row_ptr, "_graphforge_csr_destination_index", None)
        if destination is None or destination.shape != (self.shape[0],):
            if getattr(
                getattr(row_ptr, "_buffer", None),
                "_graphforge_torch_buffer", False,
            ) and row_ptr.device.type.name == "CUDA":
                from ..interop.torch.provider import cuda_destination_index

                destination = cuda_destination_index(row_ptr, num_rows)
            else:
                from . import tensor

                destination = tensor(
                    [row for row, (begin, end) in enumerate(zip(rows, rows[1:]))
                     for _ in range(end - begin)],
                    dtype=row_ptr.dtype, device=row_ptr.device,
                )
            row_ptr._graphforge_csr_destination_index = destination
        return Tensor(
            (num_rows, *self.shape[1:]), dtype=self.dtype, device=self.device,
            requires_grad=self.requires_grad,
            expression=_Expr(
                "csr_segment_product", (self, row_ptr, destination),
                (("num_rows", num_rows), ("max_degree", max_degree),
                 ("uniform_degree", uniform_degree))),
            version=max(self.version, row_ptr.version, destination.version),
        )

    def _csr_segment_product_vjp(
        self, row_ptr: Tensor, destination: Tensor, upstream: Tensor,
        num_rows: int, max_degree: int, uniform_degree: bool,
    ) -> Tensor:
        """Internal zero-safe VJP generated from a CSR product primitive."""
        if upstream.shape != (num_rows, *self.shape[1:]):
            raise ValueError("product cotangent shape disagrees with CSR rows")
        if destination.shape != (self.shape[0],):
            raise ValueError("product destination index disagrees with edge domain")
        if any(value.device != self.device
               for value in (row_ptr, destination, upstream)):
            raise ValueError("product VJP operands must share a device")
        return Tensor(
            self.shape, dtype=self.dtype, device=self.device,
            requires_grad=self.requires_grad,
            expression=_Expr(
                "csr_segment_product_vjp",
                (self, row_ptr, destination, upstream),
                (("num_rows", num_rows), ("max_degree", max_degree),
                 ("uniform_degree", uniform_degree))),
            version=max(
                self.version, row_ptr.version, destination.version,
                upstream.version),
        )

    def _csr_euclidean_distance_sum_vjp(
        self,
        source_value: Tensor,
        destination: Tensor,
        source_index: Tensor,
        upstream: Tensor,
        lattice: Tensor,
        inverse_lattice: Tensor,
        *,
        periodic: bool,
    ) -> Tensor:
        """Internal packed VJP for a fixed Euclidean CSR snapshot.

        ``self`` is positions ``[N,D]``. The result ``[N,D+1]`` contains
        position adjoints followed by the scalar source-value adjoint.
        """
        if self.dtype is not float32 or self.ndim != 2 or self.shape[1] not in (2, 3):
            raise TypeError("Euclidean CSR VJP requires FP32 positions [N,2|3]")
        nodes, dimensions = self.shape
        if source_value.shape != (nodes,) or upstream.shape != (nodes,):
            raise ValueError("source_value and upstream must have shape [N]")
        if source_value.dtype is not self.dtype or upstream.dtype is not self.dtype:
            raise TypeError("Euclidean CSR VJP floating operands must be FP32")
        if destination.shape != source_index.shape or destination.ndim != 1:
            raise ValueError("destination/source_index must be matching vectors")
        if destination.dtype is not int64 or source_index.dtype is not int64:
            raise TypeError("Euclidean CSR VJP indices must be gf.int64")
        if lattice.shape != (dimensions, dimensions) or \
                inverse_lattice.shape != lattice.shape:
            raise ValueError("lattice and inverse_lattice must have shape [D,D]")
        operands = (
            self, source_value, destination, source_index, upstream,
            lattice, inverse_lattice,
        )
        if any(value.device != self.device for value in operands[1:]):
            raise ValueError("Euclidean CSR VJP operands must share a device")
        return Tensor(
            (nodes, dimensions + 1),
            dtype=self.dtype,
            device=self.device,
            requires_grad=False,
            expression=_Expr(
                "csr_euclidean_distance_sum_vjp",
                operands,
                (("dimensions", dimensions),
                 ("periodic", builtins.bool(periodic))),
            ),
            version=max(value.version for value in operands),
        )

    def _csr_segment_max_stop_gradient(
        self, row_ptr: Tensor, num_rows: int
    ) -> Tensor:
        """Stable-reducer row shift with an explicit stop-gradient contract.

        This private primitive is used only as the translation-invariant
        numerical shift in online softmax.  It is deliberately distinct from
        a user-visible differentiable max reduction.
        """
        if self.dtype.kind != "float":
            raise TypeError("CSR segment maximum requires floating-point messages")
        if not isinstance(row_ptr, Tensor) or row_ptr.ndim != 1 or \
                row_ptr.dtype.kind != "int":
            raise TypeError("row_ptr must be a rank-one integer gf.Tensor")
        if self.ndim == 0 or row_ptr.shape[0] != num_rows + 1:
            raise ValueError("row_ptr extent must equal num_rows plus one")
        if self.device != row_ptr.device:
            raise ValueError("message and row_ptr must share a device")
        return Tensor(
            (num_rows, *self.shape[1:]), dtype=self.dtype, device=self.device,
            requires_grad=False,
            expression=_Expr(
                "csr_segment_max_stop_gradient", (self, row_ptr),
                (("num_rows", num_rows),)),
            version=max(self.version, row_ptr.version),
        )

    def _distributed_halo_reverse(self, halo, owned_rows: int) -> Tensor:
        """Internal adjoint of a compiler-installed ``owned || ghost`` view."""
        if self.ndim == 0 or self.shape[0] != owned_rows + len(halo.ghost_ids):
            raise ValueError("reverse halo input shape disagrees with halo map")
        return Tensor(
            (owned_rows, *self.shape[1:]), dtype=self.dtype,
            device=self.device, requires_grad=self.requires_grad,
            expression=_Expr(
                "distributed_halo_reverse", (self,),
                (("halo", halo), ("owned_rows", owned_rows)),
            ), version=self.version,
        )

    def realize(self) -> Tensor:
        if self._buffer is not None:
            return self
        requested_backend = os.environ.get(
            "GRAPHFORGE_TENSOR_BACKEND", "auto"
        ).lower()
        if requested_backend not in {"auto", "native", "python"}:
            raise ValueError(
                "GRAPHFORGE_TENSOR_BACKEND must be auto, native or python"
            )
        largest_value = max(
            (value.numel for value in self._expression_values()), default=self.numel
        )
        use_native = requested_backend == "native" or (
            requested_backend == "auto" and largest_value >= 4096
        )
        if use_native:
            try:
                if self.device.type == DeviceType.CPU:
                    from ..compiler.cpu_tensor import compile_tensor
                    executable = compile_tensor(self)
                    self._buffer = Buffer(
                        self.nbytes, device=self.device, _pooled=True
                    )
                elif self.device.type == DeviceType.CUDA:
                    from ..compiler.gpu_tensor import compile_tensor
                    executable = compile_tensor(self)
                    if getattr(executable, "aliases_output", False):
                        self._buffer = None
                    elif executable.uses_torch_storage:
                        from ..interop.torch.tensor import allocate_buffer

                        self._buffer = allocate_buffer(
                            self.shape, self.dtype, self.device
                        )
                    else:
                        self._buffer = Buffer(self.nbytes, device=self.device)
                else:
                    raise NotImplementedError(
                        f"Tensor codegen does not support {self.device}"
                    )
                self.strides = _contiguous_strides(self.shape)
                self.offset = 0
                start = time.perf_counter_ns()
                completion = executable.launch(self)
                if completion is not None:
                    completion.wait()
                elif self.device.type == DeviceType.CUDA:
                    from ..interop.torch.tensor import synchronize

                    synchronize(self.device)
                launch_ms = (time.perf_counter_ns() - start) / 1e6
                self.ready_event = (
                    Event(self.device, _pooled=True)
                    if self.device.type == DeviceType.CPU else None
                )
                self._execution_info = {
                    "backend": executable.backend,
                    "compile_ms": executable.compile_ms,
                    "materialize_ms": getattr(executable, "materialize_ms", 0.0),
                    "saved_bytes": getattr(executable, "saved_bytes", 0),
                    "saved_compile_ms": getattr(
                        executable, "saved_compile_ms", 0.0),
                    "checkpoint_plan": getattr(
                        executable, "checkpoint_plan", None),
                    "cache_hit": executable.cache_hit,
                    "fast_math": executable.fast_math,
                    "launch_ms": launch_ms,
                    "artifact": str(executable.artifact),
                    "source": executable.source,
                    "ir": executable.ir,
                    "semantic_hash": executable.semantic_hash,
                }
                if hasattr(executable, "artifacts"):
                    self._execution_info["artifacts"] = executable.artifacts
                prepare = getattr(executable, "prepare", None)
                if prepare is not None:
                    self._prepared_launch = prepare(self)
                return self
            except Exception as error:
                if (requested_backend == "native" or
                        (self._expr is not None and self._expr.op == "repeat" and
                         self.device.type == DeviceType.CUDA)):
                    raise
                native_error = str(error)
        else:
            native_error = None
        values = self._evaluate_flat({})
        self._buffer = Buffer(self.nbytes, device=self.device, _pooled=True)
        self.strides = _contiguous_strides(self.shape)
        self.offset = 0
        self._write_flat(values)
        self.ready_event = Event(self.device, _pooled=True)
        self._execution_info = {
            "backend": "python-oracle",
            "native_error": native_error,
        }
        return self

    def prepare(self):
        """Return a hot callable for repeatedly submitting this compiled DAG.

        Input and output storage stay bound. This is useful for simulations,
        interactive rendering, and any loop that updates external zero-copy
        inputs in place. Cold capture/JIT remains visible on the first
        ``realize()`` and is never hidden inside the returned callable.
        """
        self.realize()
        launch = self._prepared_launch
        if launch is None:
            raise RuntimeError(
                f"prepared submission is unavailable for "
                f"{(self.execution or {}).get('backend', 'this backend')}"
            )

        def submit():
            completion = launch()
            if isinstance(completion, Event):
                self.ready_event = completion
            return self

        return submit

    def _expression_values(self) -> list[Tensor]:
        ordered: list[Tensor] = []
        visited: set[int] = set()

        def visit(value: Tensor) -> None:
            if id(value) in visited:
                return
            visited.add(id(value))
            if value._expr is not None:
                for operand in value._expr.operands:
                    visit(operand)
                if value._expr.region is not None:
                    visit(value._expr.region.output)
            ordered.append(value)

        visit(self)
        return ordered

    @property
    def execution(self) -> dict[str, object] | None:
        """Return codegen/launch diagnostics after realization."""
        if self._execution_info is None:
            return None
        return dict(self._execution_info)

    def generated_code(self, kind: str | None = None) -> str | bytes | None:
        """Return generated source or a named provider artifact."""
        if self._execution_info is None:
            return None
        artifacts = self._execution_info.get("artifacts")
        if isinstance(artifacts, dict):
            selected = kind
            if selected is None:
                selected = next(
                    (
                        stage
                        for stage in ("ptx", "llvm", "ttir", "cpu_loop", "gf_tensor")
                        if stage in artifacts
                    ),
                    None,
                )
            if selected is None:
                return None
            return artifacts.get(selected)
        source = self._execution_info.get("source")
        return source if isinstance(source, str) else None

    def mlir(self, *, verify: bool = False) -> str:
        """Return canonical target-independent Tensor MLIR.

        Set ``verify=True`` to round-trip the program through the native
        ``gf-opt`` parser and C++ operation verifiers.
        """
        if verify:
            from ..compiler.tensor_mlir import verify_tensor_mlir

            return verify_tensor_mlir(self)
        from ..compiler.tensor_mlir import tensor_mlir

        return tensor_mlir(self)

    def _evaluate_flat(self, memo: dict[int, list[object]] | None = None) -> list[object]:
        cache = {} if memo is None else memo
        key = id(self)
        if key in cache:
            return cache[key]
        if self._buffer is not None:
            result = self._read_flat()
            cache[key] = result
            return result
        if self._expr is None:
            raise RuntimeError("Tensor has neither storage nor an expression")

        expression = self._expr
        values = [operand._evaluate_flat(cache) for operand in expression.operands]
        if expression.op == "repeat":
            region = expression.region
            if region is None or not region.arguments:
                raise RuntimeError("repeat expression is missing its body region")
            iterations = int(expression.attr("iterations"))
            current = values[0]
            captured_values = values[1:]
            for _ in range(iterations):
                iteration_cache = {
                    id(region.arguments[0]): current,
                    **{
                        id(argument): value
                        for argument, value in zip(
                            region.arguments[1:], captured_values
                        )
                    },
                }
                current = region.output._evaluate_flat(iteration_cache)
            result = current
        elif expression.op == "loop_argument":
            raise RuntimeError("a loop argument escaped its control region")
        elif expression.op == "reshape":
            result = values[0]
        elif expression.op == "broadcast":
            source = expression.operands[0]
            result = [values[0][index] for index in _broadcast_indices(source.shape, self.shape)]
        elif expression.op == "permute":
            source = expression.operands[0]
            axes = expression.attr("axes")
            inverse = [0] * self.ndim
            for output_axis, input_axis in enumerate(axes):
                inverse[input_axis] = output_axis
            source_strides = _contiguous_strides(source.shape)
            output_coordinate = [0] * self.ndim
            result = []
            for _ in range(self.numel):
                source_index = sum(
                    output_coordinate[inverse[axis]] * source_strides[axis]
                    for axis in range(self.ndim)
                )
                result.append(values[0][source_index])
                for axis in range(self.ndim - 1, -1, -1):
                    output_coordinate[axis] += 1
                    if output_coordinate[axis] < self.shape[axis]:
                        break
                    output_coordinate[axis] = 0
        elif expression.op == "sum":
            source = expression.operands[0]
            axes = set(expression.attr("axes"))
            keepdims = expression.attr("keepdims")
            result = [0] * self.numel
            source_coordinate = [0] * source.ndim
            output_strides = _contiguous_strides(self.shape)
            for value in values[0]:
                if keepdims:
                    output_coordinate = tuple(
                        0 if axis in axes else coordinate
                        for axis, coordinate in enumerate(source_coordinate)
                    )
                else:
                    output_coordinate = tuple(
                        coordinate
                        for axis, coordinate in enumerate(source_coordinate)
                        if axis not in axes
                    )
                output_index = sum(
                    coordinate * stride
                    for coordinate, stride in zip(output_coordinate, output_strides)
                )
                result[output_index] += value
                for axis in range(source.ndim - 1, -1, -1):
                    source_coordinate[axis] += 1
                    if source_coordinate[axis] < source.shape[axis]:
                        break
                    source_coordinate[axis] = 0
        elif expression.op == "gather":
            source = expression.operands[0]
            index_values = values[1]
            row_width = _numel(source.shape[1:])
            result = []
            for raw_index in index_values:
                row = int(raw_index)
                if row < 0 or row >= source.shape[0]:
                    raise IndexError("gather index is out of range")
                begin = row * row_width
                result.extend(values[0][begin:begin + row_width])
        elif expression.op == "segment_sum":
            segments = int(expression.attr("num_segments"))
            index_values = values[1]
            row_width = _numel(self.shape[1:])
            result = [0] * self.numel
            for source_row, raw_index in enumerate(index_values):
                destination = int(raw_index)
                if destination < 0 or destination >= segments:
                    raise IndexError("segment index is out of range")
                source_begin = source_row * row_width
                destination_begin = destination * row_width
                for column in range(row_width):
                    result[destination_begin + column] += values[0][
                        source_begin + column]
        elif expression.op == "scatter_rows":
            source = expression.operands[0]
            inverse_values = values[2]
            row_width = _numel(source.shape[1:])
            result = [0] * self.numel
            for destination, raw_source in enumerate(inverse_values):
                source_row = int(raw_source)
                if source_row < 0:
                    continue
                if source_row >= source.shape[0]:
                    raise IndexError("scatter inverse index is out of range")
                source_begin = source_row * row_width
                destination_begin = destination * row_width
                result[destination_begin:destination_begin + row_width] = values[0][
                    source_begin:source_begin + row_width]
        elif expression.op == "csr_expand_rows":
            rows = [int(value) for value in values[1]]
            row_width = _numel(self.shape[1:])
            result = []
            for row, (begin, end) in enumerate(zip(rows, rows[1:])):
                item = values[0][row * row_width:(row + 1) * row_width]
                for _ in range(end - begin):
                    result.extend(item)
            if len(result) != self.numel:
                raise ValueError("row_ptr endpoint does not match num_edges")
        elif expression.op == "csr_segment_sum":
            rows = [int(value) for value in values[1]]
            row_width = _numel(self.shape[1:])
            result = [0] * self.numel
            for row, (begin, end) in enumerate(zip(rows, rows[1:])):
                destination_begin = row * row_width
                for edge_row in range(begin, end):
                    source_begin = edge_row * row_width
                    for column in range(row_width):
                        result[destination_begin + column] += values[0][
                            source_begin + column]
        elif expression.op == "csr_segment_product":
            rows = [int(value) for value in values[1]]
            row_width = _numel(self.shape[1:])
            result = [1] * self.numel
            for row, (begin, end) in enumerate(zip(rows, rows[1:])):
                destination_begin = row * row_width
                for edge_row in range(begin, end):
                    source_begin = edge_row * row_width
                    for column in range(row_width):
                        result[destination_begin + column] *= values[0][
                            source_begin + column]
        elif expression.op == "csr_segment_product_vjp":
            rows = [int(value) for value in values[1]]
            destinations = [int(value) for value in values[2]]
            row_width = _numel(self.shape[1:])
            result = [0] * self.numel
            for edge_row, row in enumerate(destinations):
                begin, end = rows[row], rows[row + 1]
                for column in range(row_width):
                    product = values[3][row * row_width + column]
                    for other in range(begin, end):
                        if other != edge_row:
                            product *= values[0][other * row_width + column]
                    result[edge_row * row_width + column] = product
        elif expression.op == "csr_euclidean_distance_sum_vjp":
            dimensions = int(expression.attr("dimensions"))
            periodic = builtins.bool(expression.attr("periodic"))
            positions, source_values, destinations, sources, upstream, \
                lattice, inverse = values
            stride = dimensions + 1
            result = [0.0] * self.numel
            for destination, source in zip(destinations, sources):
                destination = int(destination)
                source = int(source)
                delta = [
                    positions[source * dimensions + axis]
                    - positions[destination * dimensions + axis]
                    for axis in range(dimensions)
                ]
                if periodic:
                    fractional = [
                        sum(delta[component] * inverse[
                            component * dimensions + axis]
                            for component in range(dimensions))
                        for axis in range(dimensions)
                    ]
                    centered = [value - round(value) for value in fractional]
                    delta = [
                        sum(centered[component] * lattice[
                            component * dimensions + axis]
                            for component in range(dimensions))
                        for axis in range(dimensions)
                    ]
                distance = math.sqrt(sum(value * value for value in delta))
                dy = upstream[destination]
                scale = (
                    dy * source_values[source] / distance
                    if distance > 0.0 else 0.0
                )
                for axis, value in enumerate(delta):
                    contribution = scale * value
                    result[source * stride + axis] += contribution
                    result[destination * stride + axis] -= contribution
                result[source * stride + dimensions] += dy * distance
        elif expression.op == "csr_segment_max_stop_gradient":
            rows = [int(value) for value in values[1]]
            row_width = _numel(self.shape[1:])
            result = [-math.inf] * self.numel
            for row, (begin, end) in enumerate(zip(rows, rows[1:])):
                destination_begin = row * row_width
                for edge_row in range(begin, end):
                    source_begin = edge_row * row_width
                    for column in range(row_width):
                        result[destination_begin + column] = max(
                            result[destination_begin + column],
                            values[0][source_begin + column])
        elif expression.op == "distributed_halo_snapshot":
            raise RuntimeError(
                "a distributed halo snapshot must retain installed storage")
        elif expression.op == "distributed_halo_reverse":
            from ..distributed.transport import (
                current_distributed_runtime, reverse_halo_values,
            )

            runtime = current_distributed_runtime()
            if runtime is None:
                raise RuntimeError(
                    "distributed halo VJP must execute inside DistributedRuntime")
            result = reverse_halo_values(
                expression.attr("halo"), values[0],
                row_width=_numel(self.shape[1:]),
                transport=runtime.transport,
            )
        elif expression.op in {"checkpoint", "checkpoint_candidate"}:
            result = values[0]
        elif expression.op == "conj":
            result = [value.conjugate() for value in values[0]]
        elif expression.op == "neg":
            result = [-value for value in values[0]]
        elif expression.op == "exp":
            result = [math.exp(value) for value in values[0]]
        elif expression.op == "sqrt":
            result = [math.sqrt(value) for value in values[0]]
        elif expression.op == "cumsum":
            source = expression.operands[0]
            axis = int(expression.attr("axis"))
            reverse = builtins.bool(expression.attr("reverse"))
            inner = _numel(source.shape[axis + 1:])
            extent = source.shape[axis]
            outer = source.numel // (extent * inner)
            result = [0] * source.numel
            order = range(extent - 1, -1, -1) if reverse else range(extent)
            for outer_index in range(outer):
                for inner_index in range(inner):
                    running = 0
                    for step in order:
                        index = (
                            outer_index * extent * inner
                            + step * inner + inner_index
                        )
                        running += values[0][index]
                        result[index] = running
        elif expression.op == "matmul":
            lhs, rhs = expression.operands
            rows, inner = lhs.shape
            columns = rhs.shape[1]
            result = [0] * (rows * columns)
            for row in range(rows):
                for column in range(columns):
                    total = 0
                    for contraction in range(inner):
                        total += (
                            values[0][row * inner + contraction]
                            * values[1][contraction * columns + column]
                        )
                    result[row * columns + column] = total
        elif expression.op in {"add", "mul", "div", "not_equal"}:
            lhs, rhs = expression.operands
            lhs_indices = _broadcast_indices(lhs.shape, self.shape)
            rhs_indices = _broadcast_indices(rhs.shape, self.shape)
            if expression.op == "add":
                result = [values[0][i] + values[1][j] for i, j in zip(lhs_indices, rhs_indices)]
            elif expression.op == "mul":
                result = [values[0][i] * values[1][j] for i, j in zip(lhs_indices, rhs_indices)]
            elif expression.op == "not_equal":
                result = [values[0][i] != values[1][j] for i, j in zip(lhs_indices, rhs_indices)]
            else:
                result = [
                    _ieee_divide(values[0][i], values[1][j])
                    for i, j in zip(lhs_indices, rhs_indices)
                ]
        else:
            raise RuntimeError(f"unknown Tensor expression {expression.op!r}")
        cache[key] = result
        return result

    def _read_flat(self) -> list[object]:
        if self._buffer is None:
            raise RuntimeError("Tensor has no physical storage")
        if getattr(self._buffer, "_graphforge_torch_buffer", False):
            from ..interop.torch.tensor import read_flat

            return read_flat(self)
        if self.device.type != DeviceType.CPU:
            # Device views may be strided. Read the owned physical instance,
            # decode it once, then apply the same logical-offset traversal as
            # native CPU views.
            payload = self._buffer.read()
            storage_elements = len(payload) // self.dtype.itemsize
            if self.dtype is float16:
                physical = list(struct.unpack(f"={storage_elements}e", payload))
            elif self.dtype is float32:
                physical = list(struct.unpack(f"={storage_elements}f", payload))
            elif self.dtype is float64:
                physical = list(struct.unpack(f"={storage_elements}d", payload))
            elif self.dtype is int32:
                physical = list(struct.unpack(f"={storage_elements}i", payload))
            elif self.dtype is int64:
                physical = list(struct.unpack(f"={storage_elements}q", payload))
            elif self.dtype is bool:
                physical = list(struct.unpack(f"={storage_elements}?", payload))
            else:
                component = "f" if self.dtype is complex64 else "d"
                parts = struct.unpack(
                    f"={2 * storage_elements}{component}", payload)
                physical = [
                    complex(parts[index], parts[index + 1])
                    for index in range(0, len(parts), 2)
                ]
            return [
                physical[self.offset + logical]
                for logical in _logical_offsets(self.shape, self.strides)
            ]
        pointer = ctypes.cast(self._buffer.address, ctypes.POINTER(self.dtype.ctype))
        result = []
        for logical in _logical_offsets(self.shape, self.strides):
            value = pointer[self.offset + logical]
            if self.dtype is float16:
                value = struct.unpack("=e", struct.pack("=H", value))[0]
            elif self.dtype.kind == "complex":
                value = complex(value.real, value.imag)
            result.append(value)
        return result

    def _write_flat(self, values: Sequence[object]) -> None:
        if self._buffer is None or len(values) != self.numel:
            raise ValueError("physical write does not match Tensor size")
        if not self.is_contiguous:
            raise ValueError("writes currently require contiguous Tensor storage")
        if getattr(self._buffer, "_graphforge_torch_buffer", False):
            from ..interop.torch.tensor import write_flat

            write_flat(self, values)
            return
        if self.device.type != DeviceType.CPU:
            if self.dtype is float16:
                payload = struct.pack(f"={self.numel}e", *values)
            elif self.dtype is float32:
                payload = struct.pack(f"={self.numel}f", *values)
            elif self.dtype is float64:
                payload = struct.pack(f"={self.numel}d", *values)
            elif self.dtype is int32:
                payload = struct.pack(f"={self.numel}i", *values)
            elif self.dtype is int64:
                payload = struct.pack(f"={self.numel}q", *values)
            elif self.dtype is bool:
                payload = struct.pack(f"={self.numel}?", *values)
            else:
                components = tuple(
                    component
                    for value in values
                    for component in (complex(value).real, complex(value).imag)
                )
                spelling = "f" if self.dtype is complex64 else "d"
                payload = struct.pack(
                    f"={2 * self.numel}{spelling}", *components)
            self._buffer.write(
                payload, offset=self.offset * self.dtype.itemsize)
            return
        pointer = ctypes.cast(
            self._buffer.address + self.offset * self.dtype.itemsize,
            ctypes.POINTER(self.dtype.ctype),
        )
        for index, value in enumerate(values):
            if self.dtype is float16:
                pointer[index] = struct.unpack("=H", struct.pack("=e", value))[0]
            elif self.dtype.kind == "complex":
                converted = complex(value)
                pointer[index] = self.dtype.ctype(converted.real, converted.imag)
            else:
                pointer[index] = value

    def tolist(self):
        self.realize()
        values = self._read_flat()
        if self.shape == ():
            return values[0]

        def nest(offset: int, shape: tuple[int, ...]):
            if len(shape) == 1:
                return values[offset:offset + shape[0]]
            step = _numel(shape[1:])
            return [nest(offset + index * step, shape[1:]) for index in range(shape[0])]

        return nest(0, self.shape)

    def to_numpy(self):
        """Copy this logical Tensor into a contiguous NumPy array.

        NumPy remains optional: importing GraphForge does not import or depend
        on it.  Native CUDA storage is copied through the runtime, while a
        Torch-owned interop allocation uses its existing zero-copy view before
        the explicit host transfer.
        """
        try:
            import numpy as np
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "Tensor.to_numpy() requires the optional NumPy package"
            ) from error
        self.realize()
        if getattr(self._buffer, "_graphforge_torch_buffer", False):
            return self.to_torch().detach().cpu().numpy().copy()
        dtype = np.dtype({
            "bool": "?", "int32": "=i4", "int64": "=i8",
            "float16": "=f2", "float32": "=f4", "float64": "=f8",
            "complex64": "=c8", "complex128": "=c16",
        }[self.dtype.name])
        payload = self._buffer.read()
        byte_strides = tuple(stride * self.dtype.itemsize for stride in self.strides)
        view = np.ndarray(
            self.shape, dtype=dtype, buffer=payload,
            offset=self.offset * self.dtype.itemsize,
            strides=byte_strides if self.shape else None,
        )
        return view.copy()

    def to_torch(self, *, copy: bool = False):
        """Return a Torch tensor.

        By default this is a strict zero-copy view and only accepts
        Torch-owned storage. ``copy=True`` explicitly permits one native
        H2H/D2D copy from GraphForge-owned contiguous storage.
        """
        if copy:
            from ..interop.torch.tensor import copy_to_torch

            return copy_to_torch(self)
        from ..interop.torch.tensor import to_torch

        return to_torch(self)

    def expression(self) -> str:
        """Return a debug capture graph; this is not canonical GraphForge MLIR."""
        lines: list[str] = []
        names: dict[int, str] = {}

        def visit(value: Tensor) -> str:
            key = id(value)
            if key in names:
                return names[key]
            operands = tuple(visit(item) for item in value._expr.operands) \
                if value._expr else ()
            name = f"%{len(names)}"
            names[key] = name
            if value._expr is None:
                lines.append(
                    f"{name} = input shape={value.shape} dtype={value.dtype.name} "
                    f"device={value.device} version={value.version}"
                )
            else:
                attributes = ", ".join(
                    f"{key}={item}" for key, item in value._expr.attrs
                )
                suffix = f" {{{attributes}}}" if attributes else ""
                if value._expr.region is not None:
                    suffix += " {region=1}"
                lines.append(
                    f"{name} = {value._expr.op}({', '.join(operands)}){suffix} "
                    f"shape={value.shape} dtype={value.dtype.name}"
                )
            return name

        result = visit(self)
        return "\n".join([*lines, f"return {result}"])

    def __repr__(self) -> str:
        state = "materialized" if self._buffer is not None else "deferred"
        return (
            f"gf.Tensor(shape={self.shape}, dtype={self.dtype!r}, "
            f"device='{self.device}', {state}, requires_grad={self.requires_grad})"
        )


def empty(
    shape: int | Sequence[int],
    *,
    dtype: DType = float32,
    device: Device | str = "cpu",
    requires_grad: bool = False,
) -> Tensor:
    resolved = Device.parse(device)
    if resolved.type == DeviceType.CUDA:
        normalized = _normalize_shape(shape)
        return Tensor(
            normalized,
            dtype=dtype,
            device=resolved,
            buffer=Buffer(_numel(normalized) * dtype.itemsize, device=resolved),
            requires_grad=requires_grad,
        )
    return Tensor(shape, dtype=dtype, device=resolved, requires_grad=requires_grad)


def from_torch(value: object, *, requires_grad: bool | None = None) -> Tensor:
    """Wrap a dense Torch Tensor without copying its storage."""
    from ..interop.torch.tensor import from_torch as adapt

    return adapt(value, requires_grad=requires_grad)


def tensor(
    data: object,
    *,
    dtype: DType | None = None,
    device: Device | str = "cpu",
    requires_grad: bool = False,
) -> Tensor:
    shape, values = _flatten_data(data)
    resolved_dtype = _infer_dtype(values) if dtype is None else dtype
    result = empty(
        shape, dtype=resolved_dtype, device=device, requires_grad=requires_grad
    )
    result._write_flat(values)
    return result


def zeros_like(value: Tensor, *, requires_grad: bool = False) -> Tensor:
    result = empty(
        value.shape,
        dtype=value.dtype,
        device=value.device,
        requires_grad=requires_grad,
    )
    result._write_flat([0] * result.numel)
    return result


def ones_like(value: Tensor, *, requires_grad: bool = False) -> Tensor:
    result = empty(
        value.shape,
        dtype=value.dtype,
        device=value.device,
        requires_grad=requires_grad,
    )
    result._write_flat([1] * result.numel)
    return result


__all__ = [
    "DType",
    "Tensor",
    "bool",
    "complex64",
    "complex128",
    "empty",
    "float16",
    "float32",
    "float64",
    "from_torch",
    "int32",
    "int64",
    "ones_like",
    "tensor",
    "zeros_like",
]
