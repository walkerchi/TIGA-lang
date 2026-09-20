"""Typed coarse MessagePassing capture consumed by the native MLIR frontend.

The objects in this file are plain semantic descriptors.  They intentionally
contain no MLIR spelling and no target schedule; C++ OpBuilder owns operation
construction and the Tiga pass pipeline owns lowering.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping

from .capture import Expr, FieldNamespace, _selected_params
from .reducer_capture import (
    ReducerDescriptor,
    capture_online_softmax,
    capture_reducer,
)
from ..reducer import OnlineSoftmaxItem, ReducerCall
from ..tensor import float32


@dataclass(frozen=True)
class DomainField:
    role: str
    name: str
    dtype: str
    shape: tuple[int, ...]
    itemsize: int = 0


@dataclass(frozen=True)
class DomainParam:
    name: str
    dtype: str


@dataclass(frozen=True)
class DomainGraph:
    realization: str
    num_src: int
    num_dst: int
    index_dtype: str
    degree_min: int | None
    degree_max: int | None
    degree_sum: int | None
    degree_histogram: tuple[int, ...] | None
    source_index_span_ratio: float | None
    mesh_shape: tuple[int, ...] | None
    mesh_axis: int | None
    halo_depth: int | None
    partition_balance: str | None
    dimensions: int | None = None
    neighbor_count: int | None = None
    cutoff: float | None = None
    periodic: bool = False
    hash_grid: bool = False
    dense_boundary: str = "full"
    k: int | None = None
    metric: str | None = None
    selection: str | None = None
    tie_break: str | None = None
    exclude_self: bool = False
    same_entity_domain: bool = False
    exact: bool = True


@dataclass(frozen=True)
class DomainDescriptor:
    symbol: str
    graph: DomainGraph
    fields: tuple[DomainField, ...]
    implicit_fields: tuple[DomainField, ...]
    params: tuple[DomainParam, ...]
    messages: tuple[Expr, ...]
    node_expression: Expr | None
    node_input_indices: tuple[int, ...]
    iteration_lanes: tuple[int, ...]
    reducer: ReducerDescriptor


def _symbol(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_$.-]", "_", value)
    return cleaned if cleaned and not cleaned[0].isdigit() else f"kernel_{cleaned}"


def _dtype_name(value) -> str:
    dtype = value.dtype
    name = getattr(dtype, "name", None)
    if name is not None:
        return str(name)
    spelling = str(dtype)
    return spelling.removeprefix("torch.")


def capture_message_passing(
    *,
    kernel,
    graph,
    src: Mapping[str, object],
    dst: Mapping[str, object],
    edge: Mapping[str, object],
    params: Mapping[str, object],
    kernel_name: str,
    directory=None,
) -> DomainDescriptor:
    """Capture the initial scalar MessagePassing contract without MLIR text."""
    staged_params = {
        name: Expr("param", (name,)) if isinstance(value, (int, float)) else value
        for name, value in params.items()
    }
    captured_message = kernel.edge(
        FieldNamespace("src"),
        FieldNamespace("dst"),
        FieldNamespace("edge"),
        **_selected_params(kernel.edge, staged_params),
    )
    if isinstance(captured_message, ReducerCall):
        if captured_message.reducer is not kernel.reducer:
            raise ValueError("edge() bound a reducer different from kernel.reducer")
        expressions = tuple(captured_message.messages)
    else:
        expressions = (captured_message,)
    if not expressions or any(not isinstance(item, Expr) for item in expressions):
        raise TypeError(
            "edge() did not capture staged scalar reducer message expressions")

    aggregate = Expr("aggregate", (0,))
    node_expression = kernel.node(
        FieldNamespace("dst"),
        aggregate,
        **_selected_params(kernel.node, staged_params),
    )
    if not isinstance(node_expression, Expr):
        raise TypeError("node() did not capture as one staged scalar expression")
    if node_expression == aggregate:
        node_expression = None

    edge_leaves: list[tuple[str, str]] = []
    node_leaves: list[tuple[str, str]] = []

    def visit(value, output: list[tuple[str, str]]) -> None:
        if not isinstance(value, Expr):
            raise TypeError("captured edge expression contains an unsupported value")
        if value.op == "field":
            binding = str(value.args[0]), str(value.args[1])
            if binding not in output:
                output.append(binding)
            return
        if value.op == "param":
            binding = "param", str(value.args[0])
            if binding not in output:
                output.append(binding)
            return
        if value.op == "constant":
            return
        if value.op == "aggregate":
            return
        for operand in value.args:
            if isinstance(operand, Expr):
                visit(operand, output)

    for expression in expressions:
        visit(expression, edge_leaves)
    if node_expression is not None:
        visit(node_expression, node_leaves)
    leaves = list(edge_leaves)
    for binding in node_leaves:
        if binding not in leaves:
            leaves.append(binding)
    order = {"src": 0, "dst": 1, "edge": 2, "param": 3}
    leaves.sort(key=lambda item: (order.get(item[0], 99), item[1]))
    ranked = graph.schema.realization == "procedural_knn"
    input_leaves = [
        binding for binding in leaves
        if not ((directory is not None or ranked) and
                binding == ("edge", "distance"))
    ]
    node_input_indices = tuple(
        input_leaves.index(binding)
        for binding in sorted(
            node_leaves, key=lambda item: (order.get(item[0], 99), item[1])
        )
    )
    namespaces = {"src": src, "dst": dst, "edge": edge}
    fields: list[DomainField] = []
    implicit_fields: list[DomainField] = []
    for role, name in leaves:
        if role == "param":
            if not isinstance(params.get(name), (int, float)):
                raise TypeError(
                    f"captured parameter {name} must be a scalar int or float"
                )
            continue
        if (directory is not None or ranked) and (
            role, name
        ) == ("edge", "distance"):
            implicit_fields.append(DomainField(role, name, "float32", ()))
            continue
        try:
            value = namespaces[role][name]
        except KeyError as error:
            raise KeyError(
                f"captured field {role}.{name} has no invocation binding"
            ) from error
        shape = tuple(value.shape)
        dtype = _dtype_name(value)
        if len(shape) not in (1, 2) or dtype != "float32":
            raise TypeError(
                "initial native Domain capture requires rank-one/two float32 fields"
            )
        if len(shape) == 2 and shape[1] <= 0:
            raise ValueError("vector field width must be positive")
        itemsize = getattr(getattr(value, "dtype", None), "itemsize", None)
        if itemsize is None:
            itemsize = value.element_size()
        fields.append(DomainField(role, name, dtype, shape, int(itemsize)))

    field_widths = {
        (field.role, field.name): (
            field.shape[1] if len(field.shape) == 2 else 1
        )
        for field in fields
    }

    def expression_width(value: Expr) -> int:
        if value.op == "field":
            return field_widths.get(
                (str(value.args[0]), str(value.args[1])), 1
            )
        if value.op in {"constant", "param", "aggregate"}:
            return 1
        widths = tuple(
            expression_width(item)
            for item in value.args
            if isinstance(item, Expr)
        )
        if value.op == "sum":
            return 1
        width = max(widths, default=1)
        if any(item not in (1, width) for item in widths):
            raise ValueError("captured expression has incompatible vector widths")
        return width

    message_widths = tuple(expression_width(item) for item in expressions)
    message_dtypes = tuple(
        "float32" if width == 1 else f"vector:{width}:float32"
        for width in message_widths
    )

    degree_min = degree_max = degree_sum = None
    degree_histogram = None
    realization = (
        "generated_radius" if directory is not None
        else graph.schema.realization
    )
    if realization == "materialized_csr":
        degree_min, degree_max, degree_sum = graph.degree_statistics()
        if degree_max <= 4096:
            degree_histogram = graph.degree_histogram()
    placement = graph.placement
    mesh_axis = None
    if placement is not None:
        mesh_axis = placement.partition.mesh_axis
        if isinstance(mesh_axis, str):
            mesh_axis = placement.mesh.names.index(mesh_axis)
        mesh_axis %= len(placement.mesh.shape)
    graph_descriptor = DomainGraph(
        realization=realization,
        num_src=graph.schema.num_src,
        num_dst=graph.schema.num_dst,
        index_dtype=(
            getattr(graph.schema.index_dtype, "name", None)
            or str(graph.schema.index_dtype).removeprefix("torch.")
        ),
        degree_min=degree_min,
        degree_max=degree_max,
        degree_sum=degree_sum,
        degree_histogram=degree_histogram,
        source_index_span_ratio=(
            graph.source_index_span_ratio()
            if realization == "materialized_csr" else None
        ),
        mesh_shape=None if placement is None else placement.mesh.shape,
        mesh_axis=mesh_axis,
        halo_depth=(
            None if placement is None
            else -1 if placement.halo_depth == "auto"
            else placement.halo_depth
        ),
        partition_balance=(
            None if placement is None else placement.partition.balance
        ),
        dimensions=(
            int(graph.ranked_positions()[0].shape[1])
            if ranked else
            None if directory is None else
            int(graph.euclidean_positions().shape[1])
        ),
        neighbor_count=(
            None if directory is None
            else int(directory.neighbor_offsets.shape[0])
        ),
        cutoff=None if directory is None else float(directory.cutoff),
        periodic=False if directory is None else bool(directory.periodic),
        hash_grid=False if directory is None else bool(directory.hash_grid),
        dense_boundary=getattr(graph, "_dense_boundary", "full"),
        k=int(graph._k) if ranked else None,
        metric="squared_euclidean" if ranked else None,
        selection="smallest" if ranked else None,
        tie_break="source_index" if ranked else None,
        exclude_self=bool(graph._exclude_self) if ranked else False,
        same_entity_domain=bool(graph._source_positions is None) if ranked else False,
        exact=True,
    )
    symbol = _symbol(kernel_name)
    return DomainDescriptor(
        symbol=symbol,
        graph=graph_descriptor,
        fields=tuple(fields),
        implicit_fields=tuple(implicit_fields),
        params=tuple(
            DomainParam(name, "float32")
            for role, name in leaves if role == "param"
        ),
        messages=expressions,
        node_expression=node_expression,
        node_input_indices=node_input_indices,
        iteration_lanes=(),
        reducer=capture_reducer(
            kernel.reducer,
            message_dtypes=message_dtypes,
            symbol=f"{symbol}_reducer",
        ),
    )


def structured_message_bindings(kernel, params: Mapping[str, object]):
    staged_params = {
        name: Expr("param", (name,)) if isinstance(value, (int, float)) else value
        for name, value in params.items()
    }
    captured = kernel.edge(
        FieldNamespace("src"),
        FieldNamespace("dst"),
        FieldNamespace("edge"),
        **_selected_params(kernel.edge, staged_params),
    )
    if not isinstance(captured, OnlineSoftmaxItem):
        raise TypeError("structured reducer edge() did not bind message operands")
    fields: list[tuple[str, str]] = []
    parameter_names: list[str] = []

    def visit(value) -> None:
        if not isinstance(value, Expr):
            return
        if value.op == "field":
            binding = str(value.args[0]), str(value.args[1])
            if binding not in fields:
                fields.append(binding)
            return
        if value.op == "param":
            name = str(value.args[0])
            if name not in parameter_names:
                parameter_names.append(name)
            return
        for operand in value.args:
            if isinstance(operand, Expr):
                visit(operand)

    visit(captured.score)
    visit(captured.value)
    original_order = {binding: index for index, binding in enumerate(fields)}
    fields.sort(
        key=lambda item: (
            {"dst": 0, "src": 1, "edge": 2}.get(item[0], 9),
            original_order[item],
        )
    )
    return captured, tuple(fields), tuple(parameter_names)


def capture_structured_message_passing(
    *, kernel, graph, src, dst, edge, params, kernel_name: str
) -> DomainDescriptor:
    """Capture vector messages and a stable online reducer as typed data."""
    if graph.schema.realization != "implicit_dense":
        raise NotImplementedError(
            "streaming reducer codegen currently requires a Cartesian relation"
        )
    captured, bindings, parameter_names = structured_message_bindings(kernel, params)
    if len(bindings) != 3 or [role for role, _ in bindings] != ["dst", "src", "src"]:
        raise NotImplementedError(
            "dense streaming capture requires one destination and two source fields"
        )
    namespaces = {"src": src, "dst": dst, "edge": edge}
    values = [namespaces[role][name] for role, name in bindings]
    shape = tuple(values[0].shape)
    source_shape = tuple(values[1].shape)
    dtype = _dtype_name(values[0])
    if (
        dtype != "float16"
        or len(shape) != 3
        or len(source_shape) != 3
        or tuple(values[2].shape) != source_shape
        or any(_dtype_name(value) != dtype for value in values[1:])
        or source_shape[0] != shape[0]
        or source_shape[2] != shape[2]
        or source_shape[1] <= 0
        or shape[1] % source_shape[1] != 0
    ):
        raise TypeError(
            "dense streaming fields require destination [entities, lanes, width] "
            "and two equal source fields whose lanes divide destination lanes"
        )
    if shape[0] != graph.schema.num_dst:
        raise ValueError("dense field entity extent does not match the relation")
    lanes, width = shape[1:]
    if len(parameter_names) != 1:
        raise NotImplementedError("dense streaming capture requires one scalar parameter")
    parameter = parameter_names[0]
    if not isinstance(params[parameter], (int, float)):
        raise TypeError("dense streaming runtime parameters must be scalar numbers")
    graph_descriptor = DomainGraph(
        realization="implicit_dense",
        num_src=graph.schema.num_src,
        num_dst=graph.schema.num_dst,
        index_dtype=(
            getattr(graph.schema.index_dtype, "name", None)
            or str(graph.schema.index_dtype).removeprefix("torch.")
        ),
        degree_min=None,
        degree_max=None,
        degree_sum=None,
        degree_histogram=None,
        source_index_span_ratio=None,
        mesh_shape=None,
        mesh_axis=None,
        halo_depth=None,
        partition_balance=None,
        dense_boundary=getattr(graph, "_dense_boundary", "full"),
    )
    symbol = _symbol(kernel_name)
    return DomainDescriptor(
        symbol=symbol,
        graph=graph_descriptor,
        fields=tuple(
            DomainField(role, name, dtype, tuple(value.shape))
            for (role, name), value in zip(bindings, values)
        ),
        implicit_fields=(),
        params=(DomainParam(parameter, "float32"),),
        messages=(captured.score, captured.value),
        node_expression=None,
        node_input_indices=(),
        iteration_lanes=(lanes,),
        reducer=capture_online_softmax(
            width=width,
            symbol=f"{symbol}_reducer",
            block_prune_threshold=kernel.reducer.block_prune_threshold,
        ),
    )


__all__ = [
    "DomainDescriptor",
    "DomainField",
    "DomainGraph",
    "DomainParam",
    "capture_message_passing",
    "capture_structured_message_passing",
    "structured_message_bindings",
]
