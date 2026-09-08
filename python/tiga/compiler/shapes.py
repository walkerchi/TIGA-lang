"""Symbolic Tensor shape constraints and guard-based JIT specialization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from ..runtime import Device
from ..tensor import DType, Tensor, float32


@dataclass(frozen=True)
class Dim:
    name: str
    minimum: int = 1
    maximum: int | None = None
    multiple_of: int = 1

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise ValueError("symbolic dimension name must be an identifier")
        if self.minimum < 0 or self.multiple_of <= 0:
            raise ValueError("invalid symbolic dimension bounds")
        if self.maximum is not None and self.maximum < self.minimum:
            raise ValueError("symbolic maximum must be >= minimum")

    def accepts(self, extent: int) -> bool:
        return extent >= self.minimum and (
            self.maximum is None or extent <= self.maximum
        ) and extent % self.multiple_of == 0


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int | Dim, ...]
    dtype: DType = float32
    device: Device | None = None

    def __init__(
        self, shape: Sequence[int | Dim], *, dtype: DType = float32,
        device: Device | str | None = None,
    ) -> None:
        normalized = tuple(shape)
        if any(
            not isinstance(extent, (int, Dim))
            or isinstance(extent, int) and extent < 0
            for extent in normalized
        ):
            raise ValueError("TensorSpec shape requires non-negative int or Dim")
        object.__setattr__(self, "shape", normalized)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(
            self, "device", None if device is None else Device.parse(device)
        )


@dataclass(frozen=True)
class ShapeBinding:
    dimensions: tuple[tuple[str, int], ...]
    concrete_shapes: tuple[tuple[int, ...], ...]

    def specialization_key(self) -> tuple[object, ...]:
        return self.dimensions, self.concrete_shapes


def bind_shapes(
    specs: Sequence[TensorSpec], values: Sequence[Tensor]
) -> ShapeBinding:
    if len(specs) != len(values):
        raise ValueError("symbolic signature/value arity mismatch")
    dimensions: dict[str, int] = {}
    concrete = []
    for argument, (spec, value) in enumerate(zip(specs, values, strict=True)):
        if not isinstance(value, Tensor):
            raise TypeError(f"argument {argument} must be a tiga.Tensor")
        if value.dtype is not spec.dtype:
            raise TypeError(
                f"argument {argument} dtype is {value.dtype.name}, expected {spec.dtype.name}"
            )
        if spec.device is not None and value.device != spec.device:
            raise ValueError(
                f"argument {argument} device is {value.device}, expected {spec.device}"
            )
        if value.ndim != len(spec.shape):
            raise ValueError(
                f"argument {argument} rank is {value.ndim}, expected {len(spec.shape)}"
            )
        for axis, (expected, actual) in enumerate(zip(spec.shape, value.shape, strict=True)):
            if isinstance(expected, int):
                if actual != expected:
                    raise ValueError(
                        f"argument {argument} axis {axis} is {actual}, expected {expected}"
                    )
                continue
            if not expected.accepts(actual):
                raise ValueError(
                    f"symbol {expected.name} extent {actual} violates "
                    f"[{expected.minimum}, {expected.maximum}] multiple_of={expected.multiple_of}"
                )
            previous = dimensions.setdefault(expected.name, actual)
            if previous != actual:
                raise ValueError(
                    f"symbol {expected.name} is bound to both {previous} and {actual}"
                )
        concrete.append(value.shape)
    return ShapeBinding(tuple(sorted(dimensions.items())), tuple(concrete))


class ShapeSpecializer:
    """Cache concrete compiler executables behind a symbolic signature."""

    def __init__(
        self,
        specs: Sequence[TensorSpec],
        compiler: Callable[[ShapeBinding], Callable[..., object]],
    ) -> None:
        self.specs = tuple(specs)
        self.compiler = compiler
        self._executables: dict[tuple[object, ...], Callable[..., object]] = {}
        self.hits = 0
        self.misses = 0

    def __call__(self, *values: Tensor):
        binding = bind_shapes(self.specs, values)
        key = binding.specialization_key()
        executable = self._executables.get(key)
        if executable is None:
            executable = self.compiler(binding)
            if not callable(executable):
                raise TypeError("shape specialization compiler must return a callable")
            self._executables[key] = executable
            self.misses += 1
        else:
            self.hits += 1
        return executable(*values)

    @property
    def specialization_count(self) -> int:
        return len(self._executables)


__all__ = [
    "Dim", "ShapeBinding", "ShapeSpecializer", "TensorSpec", "bind_shapes"
]
