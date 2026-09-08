"""Framework-independent class frontend for procedural radius relations."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any

from .core import Graph
from ..kernel import Kernel
from ..tensor import Tensor


def _selected_params(method, params: Mapping[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(method)
    accepts_rest = any(
        item.kind is inspect.Parameter.VAR_KEYWORD
        for item in signature.parameters.values()
    )
    if accepts_rest:
        return dict(params)
    return {name: value for name, value in params.items() if name in signature.parameters}


class RadiusGraph(Kernel):
    """Reusable UDF builder captured into a generated-relation region.

    The default metric/select are implicit compiler primitives. Subclasses may
    override either method with Tiga scalar/tensor expressions; no array
    framework operation is executed by this module.
    """

    def __init__(self) -> None:
        super().__init__()
        self._last_graph: Graph | object | None = None

    def metric(self, src, dst, edge, **params):
        del src, dst, params
        return edge.distance

    def select(self, src, dst, edge, **params):
        del src, dst, edge, params
        return True

    def __call__(
        self,
        *,
        positions: Tensor | object,
        cutoff: float | Tensor | object,
        fields: Mapping[str, Tensor | object] | None = None,
        exclude_self: bool = True,
        periodic: Tensor | object | None = None,
        **params: Any,
    ):
        metric = None
        if type(self).metric is not RadiusGraph.metric:
            def metric(src, dst, edge):
                return self.metric(
                    src, dst, edge, **_selected_params(self.metric, params)
                )

        select = None
        if type(self).select is not RadiusGraph.select:
            def select(src, dst, edge):
                return self.select(
                    src, dst, edge, **_selected_params(self.select, params)
                )

        graph = Graph.radius(
            positions,
            cutoff,
            exclude_self=exclude_self,
            fields=fields,
            metric=metric,
            select=select,
            periodic=periodic,
        )
        self._last_graph = graph
        return graph

    @property
    def last_graph(self):
        if self._last_graph is None:
            raise RuntimeError("RadiusGraph builder has not been called")
        return self._last_graph

    def explain(self) -> str:
        return (
            f"builder: {type(self).__module__}.{type(self).__qualname__}\n"
            f"{self.last_graph.explain()}"
        )


__all__ = ["RadiusGraph"]
