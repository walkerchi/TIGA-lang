"""Target-independent reverse rules for relation programs.

This module analyzes captured UDF algebra only. Framework storage binding and
provider execution live in optional interop packages.
"""

from __future__ import annotations

from dataclasses import dataclass

from .capture import capture_edge, recognize_weighted_sum


@dataclass(frozen=True)
class LinearSourceVJP:
    source_field: str
    edge_field: str
    transform: str = "transpose-relation-weighted-sum"


def analyze_source_vjp(kernel, params) -> LinearSourceVJP:
    """Derive the source-input VJP from a scalar additive edge region."""
    pattern = recognize_weighted_sum(capture_edge(kernel.edge, params))
    if pattern is None or getattr(kernel.reducer, "name", None) != "sum":
        raise NotImplementedError(
            "initial generated relation VJP requires an additive scalar "
            "edge expression with one source field and one edge field")
    return LinearSourceVJP(pattern.src_field, pattern.edge_field)


__all__ = ["LinearSourceVJP", "analyze_source_vjp"]
