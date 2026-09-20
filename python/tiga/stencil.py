"""Dimension-independent neighborhood macros for :meth:`Graph.stencil`.

Macros expand to ordinary integer offsets at graph construction time. This
module has no Torch dependency and does not change the CSR execution path.
"""
from dataclasses import dataclass
from itertools import product
from typing import Literal

__all__ = ["Neighborhood", "von_neumann", "moore"]


@dataclass(frozen=True)
class Neighborhood:
    """An inclusive integer-radius neighborhood, expanded in lexicographic order.

    ``kind`` selects Manhattan (von Neumann) or Chebyshev (Moore) distance.
    ``radius`` is a positive integer; ``include_center`` controls the zero offset.
    """

    kind: Literal["von_neumann", "moore"]
    radius: int = 1
    include_center: bool = True

    def __post_init__(self):
        if self.kind not in ("von_neumann", "moore"):
            raise ValueError("kind must be 'von_neumann' or 'moore'")
        if type(self.radius) is not int or self.radius < 1:
            raise ValueError("radius must be a positive integer")
        if type(self.include_center) is not bool:
            raise TypeError("include_center must be bool")

    def offsets(self, ndim: int) -> tuple[tuple[int, ...], ...]:
        """Expand for a positive integer dimension; no device allocation occurs."""
        if type(ndim) is not int or ndim < 1:
            raise ValueError("ndim must be a positive integer")

        def l1_offsets(axes, remaining):
            if axes == 0:
                yield ()
                return
            for shift in range(-remaining, remaining + 1):
                for tail in l1_offsets(axes - 1, remaining - abs(shift)):
                    yield (shift,) + tail

        candidates = (
            l1_offsets(ndim, self.radius) if self.kind == "von_neumann"
            else product(range(-self.radius, self.radius + 1), repeat=ndim)
        )
        return tuple(offset for offset in candidates if self.include_center or any(offset))


def von_neumann(radius: int = 1, *, include_center: bool = True) -> Neighborhood:
    """Offsets with Manhattan distance <= radius (2D default: five points)."""
    return Neighborhood("von_neumann", radius, include_center)


def moore(radius: int = 1, *, include_center: bool = True) -> Neighborhood:
    """Offsets with Chebyshev distance <= radius (2D default: nine points)."""
    return Neighborhood("moore", radius, include_center)


def _resolve_offsets(dims, offsets):
    if not isinstance(dims, tuple) or not dims or any(
        type(extent) is not int or extent < 1 for extent in dims
    ):
        raise ValueError("dims must be a non-empty tuple of positive integers")
    if offsets is None:
        offsets = von_neumann()
    if isinstance(offsets, Neighborhood):
        offsets = offsets.offsets(len(dims))
    if not isinstance(offsets, tuple) or not offsets or any(
        not isinstance(offset, tuple) or len(offset) != len(dims)
        or any(type(shift) is not int for shift in offset)
        for offset in offsets
    ):
        raise ValueError(
            "offsets must be a Neighborhood or a non-empty tuple of length-D offset tuples"
        )
    return offsets
