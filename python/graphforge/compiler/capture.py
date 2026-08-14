"""Restricted expression capture shared by native lowering and recognizers.

The expression objects are typed frontend descriptors, not serialized MLIR or
generated source. Canonical Domain operations are constructed and verified by
the native C++ OpBuilder. A few structural recognizers remain here for guarded
runtime dispatch while their analyses migrate into compiler passes.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class Expr:
    op: str
    args: tuple[Any, ...]

    def _binary(self, op: str, other):
        return Expr(op, (self, as_expr(other)))

    def __add__(self, other):
        return self._binary("add", other)

    def __radd__(self, other):
        return as_expr(other)._binary("add", self)

    def __sub__(self, other):
        return self._binary("sub", other)

    def __rsub__(self, other):
        return as_expr(other)._binary("sub", self)

    def __mul__(self, other):
        return self._binary("mul", other)

    def __rmul__(self, other):
        return as_expr(other)._binary("mul", self)

    def __truediv__(self, other):
        return self._binary("div", other)

    def __rtruediv__(self, other):
        return as_expr(other)._binary("div", self)

    def __neg__(self):
        return Expr("neg", (self,))

    def sum(self, dim=None, keepdim=False):
        return Expr("sum", (self, dim, keepdim))

    def maximum(self, other):
        return Expr("maximum", (self, as_expr(other)))

    def exp(self):
        return Expr("exp", (self,))


def as_expr(value) -> Expr:
    return value if isinstance(value, Expr) else Expr("constant", (value,))


class FieldNamespace:
    def __init__(self, role: str) -> None:
        self._role = role

    def __getattr__(self, name: str) -> Expr:
        return Expr("field", (self._role, name))


@dataclass(frozen=True)
class WeightedSumPattern:
    src_field: str
    edge_field: str


@dataclass(frozen=True)
class DiffusionPattern:
    node_field: str
    edge_field: str


def _selected_params(method, params: Mapping[str, Any]):
    signature = inspect.signature(method)
    accepts_rest = any(
        item.kind is inspect.Parameter.VAR_KEYWORD
        for item in signature.parameters.values())
    return params if accepts_rest else {
        name: value for name, value in params.items()
        if name in signature.parameters
    }


def capture_edge(method, params: Mapping[str, Any]) -> Expr | None:
    try:
        result = method(
            FieldNamespace("src"),
            FieldNamespace("dst"),
            FieldNamespace("edge"),
            **_selected_params(method, params),
        )
    except Exception:
        return None
    return result if isinstance(result, Expr) else None


def recognize_weighted_sum(expr: Expr | None) -> WeightedSumPattern | None:
    if expr is None or expr.op != "mul":
        return None
    left, right = expr.args
    fields = {tuple(left.args): left, tuple(right.args): right}
    src = fields.get(("src", "x"))
    edge = fields.get(("edge", "weight"))
    if src is not None and edge is not None:
        return WeightedSumPattern(src_field="x", edge_field="weight")

    # General field names: exactly one src field and one edge field.
    roles = {}
    for item in (left, right):
        if not isinstance(item, Expr) or item.op != "field":
            return None
        role, name = item.args
        if role in roles:
            return None
        roles[role] = name
    if set(roles) == {"src", "edge"}:
        return WeightedSumPattern(
            src_field=roles["src"], edge_field=roles["edge"])
    return None


def recognize_diffusion(expr: Expr | None) -> DiffusionPattern | None:
    if expr is None or expr.op != "mul":
        return None
    left, right = expr.args
    edge = None
    difference = None
    for item in (left, right):
        if isinstance(item, Expr) and item.op == "field" and item.args[0] == "edge":
            edge = item
        elif isinstance(item, Expr) and item.op == "sub":
            difference = item
    if edge is None or difference is None:
        return None
    source, destination = difference.args
    if not (
        isinstance(source, Expr) and source.op == "field"
        and isinstance(destination, Expr) and destination.op == "field"
        and source.args[0] == "src"
        and destination.args[0] == "dst"
        and source.args[1] == destination.args[1]
    ):
        return None
    return DiffusionPattern(
        node_field=source.args[1], edge_field=edge.args[1])
