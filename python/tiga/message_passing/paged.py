"""Single-machine paged MessagePassing over a disk-resident CSR relation.

A ``paged_csr`` Graph (``tg.load(path)``) keeps its topology on disk and
exposes bounded destination-row reads through ``Graph.paged_page``.  This
executor streams those pages: only one page of ``row_ptr``/``col_idx`` is
resident at a time, a small thread pool keeps up to ``prefetch_depth`` page
reads in flight while pages compute, and every page runs through the
unchanged native MessagePassing path on a page-local CSR graph.

Fields may also live on disk inside the ``.gfg`` (``tg.save(graph, path,
fields={"src": ..., "dst": ..., "edge": ...})``).  ``Graph.fields(role)``
hands out full-shape shell Tensors backed by the store's row readers; the
executor materializes only each page's contiguous dst/edge row ranges
(positional ``pread``) and the source rows its ``col_idx`` touches
(read-only mmap, so the OS page cache — not the process heap — owns
residency for irregular power-law source gathers).

Differentiable calls are supported end to end: the forward result carries a
``paged_message_passing`` expression, and reverse-mode (see
``execute_paged_vjp``) rebuilds each page's forward expression with
page-local ``requires_grad`` views and differentiates against exactly those
views, so every adjoint stays page-sized.  dst/edge/param adjoints assemble
in RAM (each row belongs to exactly one page); source adjoints
scatter-accumulate into a disk-resident gradient file with read-modify-write
row updates, so the global num_src-sized gather-VJP ``segment_sum`` is never
materialized.  Prefetch workers only read; every gradient write is serial on
the main thread.

Correctness of page-local reducers: MessagePassing semantics are
``out[i] = node(dst_i, ⊕_{e=(j→i)} edge(...))`` — destination rows are
independent, and a row's aggregate depends only on that row's own edges, in
row order.  A page therefore reproduces the global per-row reduction exactly
(sum, mean, prod and custom reducers alike), including the native path's
floating-point summation order.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterator

from ..graph import Graph
from ..runtime import Buffer, DeviceType
from ..tensor import Tensor, empty, int64, tensor
from ..tensor.core import _Expr
from .core import _native_csr_forward, _validate_native_fields

_DEFAULT_PAGE_ROWS = 100_000
_PAGE_ROWS_ENV = "TIGA_PAGED_PAGE_ROWS"
_DEFAULT_PREFETCH_DEPTH = 2
_PREFETCH_DEPTH_ENV = "TIGA_PAGED_PREFETCH_DEPTH"


def _default_page_rows() -> int:
    from ..runtime.memory import current_execution
    if current_execution() is not None:
        return current_execution().page_rows
    override = os.environ.get(_PAGE_ROWS_ENV)
    if override is None:
        return _DEFAULT_PAGE_ROWS
    try:
        value = int(override)
    except ValueError:
        raise ValueError(f"{_PAGE_ROWS_ENV} must be a positive integer") from None
    if value <= 0:
        raise ValueError(f"{_PAGE_ROWS_ENV} must be a positive integer")
    return value


def _default_prefetch_depth() -> int:
    from ..runtime.memory import current_execution
    if current_execution() is not None:
        return current_execution().prefetch_depth
    override = os.environ.get(_PREFETCH_DEPTH_ENV)
    if override is None:
        return _DEFAULT_PREFETCH_DEPTH
    try:
        value = int(override)
    except ValueError:
        raise ValueError(
            f"{_PREFETCH_DEPTH_ENV} must be a positive integer") from None
    if value <= 0:
        raise ValueError(f"{_PREFETCH_DEPTH_ENV} must be a positive integer")
    return value


def _requires_grad_names(
    src: Mapping[str, Any],
    dst: Mapping[str, Any],
    edge: Mapping[str, Any],
    params: Mapping[str, Any],
) -> list[str]:
    names = [
        f"{role}.{name}"
        for role, fields in (("src", src), ("dst", dst), ("edge", edge))
        for name, value in fields.items()
        if isinstance(value, Tensor) and value.requires_grad
    ]
    names.extend(
        f"param {name!r}"
        for name, value in params.items()
        if isinstance(value, Tensor) and value.requires_grad
    )
    return names


def _page_bounds(num_dst: int, page_rows: int) -> list[tuple[int, int]]:
    return [
        (begin, min(num_dst, begin + page_rows))
        for begin in range(0, num_dst, page_rows)
    ]


def _row_width(shape: tuple[int, ...]) -> int:
    return math.prod(shape[1:])


def _field_reader(value: Tensor):
    return getattr(value, "_paged_field", None)


def _flat_leaf(
    flat: Sequence,
    shape: tuple[int, ...],
    dtype,
    device,
    *,
    requires_grad: bool = False,
) -> Tensor:
    leaf = empty(shape, dtype=dtype, device=device, requires_grad=requires_grad)
    leaf._write_flat(flat)
    return leaf


def _bytes_leaf(
    payload: bytes,
    shape: tuple[int, ...],
    dtype,
    device,
    *,
    requires_grad: bool = False,
) -> Tensor:
    if len(payload) != math.prod(shape) * dtype.itemsize:
        raise ValueError("paged leaf payload does not match its shape")
    buffer = Buffer(len(payload), device=device, _pooled=True)
    buffer.write(payload)
    return Tensor(
        shape, dtype=dtype, device=device, buffer=buffer,
        requires_grad=requires_grad)


def _drain_bytes(output: Tensor) -> bytes:
    """One realized page result as raw little-endian row bytes."""
    output.realize()
    return output._buffer.read(bytes=output.nbytes)


def _assemble_pages(
    page_payloads: list[bytes], shape: tuple[int, ...], dtype, device
) -> Tensor:
    """Concatenate per-page result bytes into one materialized Tensor.

    Byte-level assembly avoids round-tripping every result element through
    Python float objects (``tolist``), which would dominate the paged working
    set on large graphs.
    """
    buffer = Buffer(sum(map(len, page_payloads)), device=device, _pooled=True)
    offset = 0
    for payload in page_payloads:
        buffer.write(payload, offset=offset)
        offset += len(payload)
    return Tensor(shape, dtype=dtype, device=device, buffer=buffer)


def _stream_pages(
    load,
    bounds: list[tuple[int, int]],
    *,
    prefetch: bool,
    prefetch_depth: int,
) -> Iterator[tuple[int, int, tuple]]:
    """Yield ``(begin, end, page)`` serially while workers read ahead.

    The with-block joins the workers on every exit path — including generator
    teardown after an exception in the consumer — so no thread leaks.
    """
    if prefetch and len(bounds) > 1:
        with ThreadPoolExecutor(
            max_workers=prefetch_depth,
            thread_name_prefix="tiga-paged-prefetch",
        ) as pool:
            pending = [
                pool.submit(copy_context().run, load, *bounds[index])
                for index in range(min(prefetch_depth, len(bounds)))
            ]
            for index, (begin, end) in enumerate(bounds):
                page = pending.pop(0).result()
                ahead = index + prefetch_depth
                if ahead < len(bounds):
                    pending.append(pool.submit(copy_context().run, load, *bounds[ahead]))
                yield begin, end, page
    else:
        for begin, end in bounds:
            yield begin, end, load(begin, end)


class _PagedPlan:
    """Everything reverse-mode needs to replay one paged forward per page."""

    def __init__(
        self,
        *,
        kernel,
        graph: Graph,
        page_rows: int,
        prefetch: bool,
        prefetch_depth: int,
        src: Mapping[str, Tensor],
        dst: Mapping[str, Tensor],
        edge: Mapping[str, Tensor],
        params: Mapping[str, Any],
        operand_layout: tuple[tuple[str, str], ...],
    ) -> None:
        self.kernel = kernel
        self.graph = graph
        self.page_rows = page_rows
        self.prefetch = prefetch
        self.prefetch_depth = prefetch_depth
        self.src = src
        self.dst = dst
        self.edge = edge
        self.params = params
        # (role, name) for each operand of the paged_message_passing expr,
        # in operand order; only differentiable tensors become operands.
        self.operand_layout = operand_layout


class _PageArena:
    """Grow-only per-field staging slabs for one paged run.

    Page field payloads are staged into reused slabs and handed to the page
    kernel as zero-copy leaves whose runtime Buffer wraps the slab address —
    no per-page large malloc/free cycles, so the host allocator never
    accumulates page-sized fragments across a long paged run.  Slabs are
    written and consumed on the main thread only; prefetch workers just read
    topology and issue readahead hints.
    """

    def __init__(self) -> None:
        self._slabs: dict[tuple[str, str], Buffer] = {}

    def stage(self, key: tuple[str, str], nbytes: int) -> Buffer:
        slab = self._slabs.get(key)
        if slab is None or slab.nbytes < nbytes:
            if slab is not None:
                # The previous page's compute joined before this restage, so
                # no live leaf still reads the old slab.
                slab.close()
            slab = Buffer(max(nbytes, 4096), device="cpu", _pooled=False)
            self._slabs[key] = slab
        return slab

    def close(self) -> None:
        for slab in self._slabs.values():
            slab.close()
        self._slabs.clear()


def _make_page_loader(graph, src, dst, edge):
    """Read one topology page; issue field readahead hints along the way.

    The loader also runs inside prefetch worker threads: positional reads and
    ``posix_fadvise`` are read-only and safe to overlap.  Field payloads are
    staged on the main thread into the arena (the OS page cache serves them
    after the hint), and every gradient write stays on the main thread.
    """
    disk_fields = [
        (role, name, _field_reader(value))
        for role, fields in (("src", src), ("dst", dst), ("edge", edge))
        for name, value in fields.items()
        if _field_reader(value) is not None
    ]

    def load(begin: int, end: int):
        rows, columns, edge_begin = graph.paged_page(begin, end)
        for role, _name, reader in disk_fields:
            if role == "dst":
                reader.hint(begin, end)
            elif role == "edge":
                reader.hint(edge_begin, edge_begin + rows[-1])
        return rows, columns, edge_begin

    return load


def _page_view(
    role, name, value, index, columns, begin, count, arena, device, *, grad_mode
):
    """One field's page-local view: edge-domain rows or a row/edge slice.

    Disk-resident fields are staged into the arena and exposed as zero-copy
    leaves.  Forward over RAM fields keeps the same lazy gather expressions
    the materialized path builds.  Backward (``grad_mode``) materializes
    every view into a leaf proxy — autograd's program-leaf expansion rebuilds
    non-leaf expressions, so only leaf proxies keep object identity as
    differentiation targets.
    """
    shape = (count, *value.shape[1:])
    requires_grad = grad_mode and value.requires_grad
    reader = _field_reader(value)
    if reader is not None:
        if count == 0:
            return empty(shape, dtype=value.dtype, device=device,
                         requires_grad=requires_grad)
        slab = arena.stage((role, name), count * reader.row_bytes)
        if role == "src":
            reader.gather_into(columns, slab.address)
        else:
            reader.read_into(begin, begin + count, slab.address)
        nbytes = count * reader.row_bytes
        if device.type is DeviceType.CPU:
            view = Buffer.wrap_address(
                slab.address, nbytes, device=device, owner=slab)
        else:
            # A host arena address is not a CUDA address. Stage exactly this
            # page onto the destination device before the kernel consumes it.
            view = Buffer(nbytes, device=device, _pooled=True)
            view.write(slab.read(bytes=nbytes))
        return Tensor(
            shape, dtype=value.dtype, device=device, buffer=view,
            requires_grad=requires_grad)
    if not grad_mode:
        return value.gather(index)
    gathered = value.gather(index).realize()
    return Tensor(
        gathered.shape, dtype=gathered.dtype, device=device,
        buffer=gathered._buffer, requires_grad=value.requires_grad)


def _run_page(
    kernel,
    *,
    graph: Graph,
    src: Mapping[str, Tensor],
    dst: Mapping[str, Tensor],
    edge: Mapping[str, Tensor],
    params: Mapping[str, Any],
    begin: int,
    end: int,
    page: tuple,
    arena: _PageArena,
    device,
    index_dtype,
    grad_mode: bool,
) -> tuple[Tensor, dict[tuple[str, str], Tensor]]:
    """Run the native forward on one page; return ``(output, views)``.

    ``views[(role, name)]`` is the exact page-local tensor each field was
    consumed as: edge-domain src gathers, page-row dst slices, page-edge
    slices.  Reverse-mode differentiates against these views, so every
    per-page adjoint is page-sized by construction.
    """
    rows, columns, edge_begin = page
    num_rows = end - begin
    num_page_edges = rows[-1]
    # Rebased page-local CSR with GLOBAL source indexing.  Built under the
    # _IN_PAGED flag by the caller so auto-offload never re-pages a page.
    page_graph = Graph.from_csr(
        # Storage reads already return validated flat sequences. Avoid generic
        # nested-shape inference once per edge on the host before every launch.
        _flat_leaf(rows, (len(rows),), index_dtype, device),
        _flat_leaf(columns, (len(columns),), index_dtype, device),
        num_src=graph.schema.num_src,
    )
    # Row pointers are already on the host and validated by the page reader.
    # Reuse them rather than copying device indices back for degree analysis.
    degrees = [right - left for left, right in zip(rows, rows[1:])]
    page_graph._degree_bounds_cache = (
        (min(degrees), max(degrees)) if degrees else (0, 0))
    row_ptr, col_idx = page_graph.resolve_csr()
    views: dict[tuple[str, str], Tensor] = {}

    src_at_edge = {}
    for name, value in src.items():
        view = _page_view(
            "src", name, value, col_idx, columns, 0, num_page_edges,
            arena, device, grad_mode=grad_mode)
        views["src", name] = view
        src_at_edge[name] = view

    page_dst = {}
    if dst:
        row_index = tensor(list(range(begin, end)), dtype=int64, device=device)
        for name, value in dst.items():
            view = _page_view(
                "dst", name, value, row_index, columns, begin, num_rows,
                arena, device, grad_mode=grad_mode)
            views["dst", name] = view
            page_dst[name] = view

    page_edge = {}
    if edge:
        edge_index = tensor(
            list(range(edge_begin, edge_begin + num_page_edges)),
            dtype=int64, device=device)
        for name, value in edge.items():
            view = _page_view(
                "edge", name, value, edge_index, columns, edge_begin,
                num_page_edges, arena, device, grad_mode=grad_mode)
            views["edge", name] = view
            page_edge[name] = view

    output, _lowering = _native_csr_forward(
        kernel,
        graph=page_graph,
        row_ptr=row_ptr,
        col_idx=col_idx,
        src={},
        dst=page_dst,
        edge=page_edge,
        params=params,
        src_at_edge=src_at_edge,
    )
    return output, views


def execute_paged_message_passing(
    kernel,
    *,
    graph: Graph,
    src: Mapping[str, Tensor],
    dst: Mapping[str, Tensor],
    edge: Mapping[str, Tensor],
    params: Mapping[str, Any],
    page_rows: int | None = None,
    prefetch: bool = True,
    prefetch_depth: int | None = None,
):
    """Run one forward MessagePassing program over paged destination rows."""
    num_edges = graph.num_edges
    assert num_edges is not None  # paged_csr graphs always know their edge count
    _validate_native_fields("edge", edge, num_edges, graph.device)
    differentiable = _requires_grad_names(src, dst, edge, params)
    if differentiable and graph.device.type is not DeviceType.CPU:
        raise NotImplementedError(
            "paged MessagePassing backward is host-side today: "
            + ", ".join(differentiable)
            + " has requires_grad=True on a non-CPU paged graph.")
    if page_rows is None:
        page_rows = _default_page_rows()
    if not isinstance(page_rows, int) or page_rows <= 0:
        raise ValueError("page_rows must be a positive integer")
    if not isinstance(prefetch, bool):
        raise TypeError("prefetch must be bool")
    if prefetch_depth is None:
        prefetch_depth = _default_prefetch_depth()
    if not isinstance(prefetch_depth, int) or prefetch_depth <= 0:
        raise ValueError("prefetch_depth must be a positive integer")

    device = graph.device
    index_dtype = graph.schema.index_dtype
    bounds = _page_bounds(graph.schema.num_dst, page_rows)
    if not bounds:
        # Degenerate empty relation: run one empty page so the kernel's own
        # node() defines the output shape and dtype.  The empty expression
        # references the caller's fields directly and differentiates through
        # the ordinary autograd rules.
        empty = Graph.from_csr(
            tensor([0], dtype=index_dtype, device=device),
            tensor([], dtype=index_dtype, device=device),
            num_src=graph.schema.num_src,
        )
        return kernel(graph=empty, src=src, dst=dst, edge=edge, **params)

    load = _make_page_loader(graph, src, dst, edge)

    from ..graph.offload import _IN_PAGED

    # Keep only one output page on the host. The final output is resident, but
    # a second graph-sized list of host byte strings is not needed for assembly.
    result = None
    page_backends: set[str] = set()
    out_feature: tuple[int, ...] | None = None
    dtype = None
    edge_cursor = 0
    arena = _PageArena()
    token = _IN_PAGED.set(True)
    try:
        for begin, end, page in _stream_pages(
            load, bounds, prefetch=prefetch, prefetch_depth=prefetch_depth
        ):
            rows = page[0]
            output, _views = _run_page(
                kernel,
                graph=graph,
                src=src,
                dst=dst,
                edge=edge,
                params=params,
                begin=begin,
                end=end,
                page=page,
                arena=arena,
                device=device,
                index_dtype=index_dtype,
                grad_mode=False,
            )
            edge_cursor += rows[-1]
            if dtype is None:
                dtype = output.dtype
                out_feature = output.shape[1:]
                result_shape = (graph.schema.num_dst, *out_feature)
                result = Tensor(
                    result_shape, dtype=dtype, device=device,
                    buffer=Buffer(math.prod(result_shape) * dtype.itemsize,
                                  device=device, _pooled=True))
            elif output.dtype is not dtype or output.shape[1:] != out_feature:
                raise RuntimeError(
                    "paged pages produced inconsistent output dtypes/shapes")
            result._buffer.write(
                _drain_bytes(output),
                offset=begin * math.prod(out_feature) * dtype.itemsize)
            page_backends.add((output.execution or {}).get("backend", "unknown"))
            # Do not retain the previous page's bindings during the next
            # allocation (or the final full-output assembly).
            del output, _views
    finally:
        _IN_PAGED.reset(token)
        arena.close()
    assert edge_cursor == num_edges
    assert result is not None
    result._execution_info = {
        "backend": "paged-message-passing",
        "page_backends": sorted(page_backends),
        "pages": len(bounds),
        "storage_path": "host-staged" if device.type is not DeviceType.CPU else "host",
    }
    if not differentiable:
        return result

    # Attach the eager execution boundary that reverse-mode replays per page.
    operands: list[Tensor] = []
    layout: list[tuple[str, str]] = []
    for role, fields in (("src", src), ("dst", dst), ("edge", edge)):
        for name, value in fields.items():
            if isinstance(value, Tensor) and value.requires_grad:
                operands.append(value)
                layout.append((role, name))
    for name, value in params.items():
        if isinstance(value, Tensor) and value.requires_grad:
            operands.append(value)
            layout.append(("param", name))
    plan = _PagedPlan(
        kernel=kernel,
        graph=graph,
        page_rows=page_rows,
        prefetch=prefetch,
        prefetch_depth=prefetch_depth,
        src=src,
        dst=dst,
        edge=edge,
        params=params,
        operand_layout=tuple(layout),
    )
    return Tensor(
        result.shape,
        dtype=result.dtype,
        device=result.device,
        buffer=result._buffer,
        requires_grad=True,
        expression=_Expr(
            "paged_message_passing", tuple(operands), (("plan", plan),)),
        version=result.version,
        ready_event=result.ready_event,
    )


def execute_paged_vjp(
    plan: _PagedPlan, upstream: Tensor
) -> list[Tensor | None]:
    """Eager reverse-mode over one recorded ``paged_message_passing`` plan.

    Each page rebuilds its forward expression with page-local views as
    ``requires_grad`` proxies and differentiates against exactly those views,
    so no adjoint is ever larger than a page.  Source adjoints
    scatter-accumulate into a sparse zero-initialized gradient file on disk
    (read-modify-write row updates), bypassing the global gather-VJP
    ``segment_sum`` over num_src; dst/edge adjoints are page-local and
    assemble in RAM; param adjoints accumulate as values page by page.
    Returns one gradient Tensor (or ``None``) per plan operand.
    """
    from ..autograd import grad  # lazy: paged ↔ autograd import cycle
    from ..graph.offload import _IN_PAGED
    from ..graph.storage import PagedGradFile

    kernel = plan.kernel
    graph = plan.graph
    device = graph.device
    index_dtype = graph.schema.index_dtype
    upstream = upstream.realize()
    if upstream.ndim == 0 or upstream.shape[0] != graph.schema.num_dst:
        raise ValueError(
            "paged MessagePassing grad_output must match the (num_dst, ...) "
            "forward output")
    out_feature = upstream.shape[1:]
    out_row_bytes = _row_width(upstream.shape) * upstream.dtype.itemsize
    if (upstream.is_contiguous and upstream.offset == 0
            and not getattr(upstream._buffer, "_tiga_torch_buffer", False)):
        upstream_bytes = upstream._buffer.read(bytes=upstream.nbytes)
    else:
        # Rare path: strided or provider-owned grad_output — normalize once
        # through a contiguous CPU leaf.
        normalized = _flat_leaf(
            upstream._read_flat(), upstream.shape, upstream.dtype, device)
        upstream_bytes = normalized._buffer.read(bytes=normalized.nbytes)

    def differentiable_names(fields: Mapping[str, Any]) -> list[str]:
        return [
            name for name, value in fields.items()
            if isinstance(value, Tensor) and value.requires_grad
        ]

    grad_src = differentiable_names(plan.src)
    grad_dst = differentiable_names(plan.dst)
    grad_edge = differentiable_names(plan.edge)
    grad_params = [
        name for name, value in plan.params.items()
        if isinstance(value, Tensor) and value.requires_grad
    ]

    bounds = _page_bounds(graph.schema.num_dst, plan.page_rows)
    load = _make_page_loader(graph, plan.src, plan.dst, plan.edge)
    grad_dir = Path(tempfile.mkdtemp(prefix="tiga-paged-grad-"))
    arena = _PageArena()
    token = _IN_PAGED.set(True)
    src_files: dict[str, Any] = {}
    try:
        for name in grad_src:
            src_files[name] = PagedGradFile(
                grad_dir / f"src.{name}.grad.bin",
                plan.src[name].shape,
                plan.src[name].dtype,
            )
        dst_chunks: dict[str, list[bytes]] = {name: [] for name in grad_dst}
        edge_chunks: dict[str, list[bytes]] = {name: [] for name in grad_edge}
        param_values: dict[str, list | None] = {name: None for name in grad_params}
        for begin, end, page in _stream_pages(
            load, bounds, prefetch=plan.prefetch, prefetch_depth=plan.prefetch_depth
        ):
            rows, columns = page[0], page[1]
            output, views = _run_page(
                kernel,
                graph=graph,
                src=plan.src,
                dst=plan.dst,
                edge=plan.edge,
                params=plan.params,
                begin=begin,
                end=end,
                page=page,
                arena=arena,
                device=device,
                index_dtype=index_dtype,
                grad_mode=True,
            )
            grad_output = _bytes_leaf(
                upstream_bytes[begin * out_row_bytes:end * out_row_bytes],
                (end - begin, *out_feature),
                upstream.dtype,
                device,
            )
            requests: list[Tensor] = []
            keys: list[tuple[str, str]] = []
            for role, names in (
                ("src", grad_src), ("dst", grad_dst), ("edge", grad_edge)
            ):
                for name in names:
                    requests.append(views[role, name])
                    keys.append((role, name))
            for name in grad_params:
                requests.append(plan.params[name])
                keys.append(("param", name))
            adjoints = (
                grad(output, requests, grad_output=grad_output, allow_unused=True)
                if requests
                else ()
            )
            for (role, name), adjoint in zip(keys, adjoints):
                if role == "dst" or role == "edge":
                    field = (plan.dst if role == "dst" else plan.edge)[name]
                    rows_in_page = (end - begin) if role == "dst" else rows[-1]
                    zero_bytes = bytes(
                        rows_in_page * _row_width(field.shape)
                        * field.dtype.itemsize)
                if adjoint is None:
                    # A field the UDF never consumes keeps its zero gradient.
                    if role == "dst":
                        dst_chunks[name].append(zero_bytes)
                    elif role == "edge":
                        edge_chunks[name].append(zero_bytes)
                    continue
                if role == "src":
                    src_files[name].add_rows(columns, _drain_bytes(adjoint))
                elif role == "dst":
                    dst_chunks[name].append(_drain_bytes(adjoint))
                elif role == "edge":
                    edge_chunks[name].append(_drain_bytes(adjoint))
                else:
                    flat = adjoint.realize()._read_flat()
                    previous = param_values[name]
                    param_values[name] = (
                        flat
                        if previous is None
                        else [left + right for left, right in zip(previous, flat)]
                    )
        results: dict[tuple[str, str], Tensor] = {}
        for name, grad_file in src_files.items():
            grad_file.close()
            results["src", name] = _bytes_leaf(
                grad_file.path.read_bytes(),
                plan.src[name].shape,
                plan.src[name].dtype,
                device,
            )
        for name, payloads in dst_chunks.items():
            results["dst", name] = _assemble_pages(
                payloads, plan.dst[name].shape, plan.dst[name].dtype, device)
        for name, payloads in edge_chunks.items():
            results["edge", name] = _assemble_pages(
                payloads, plan.edge[name].shape, plan.edge[name].dtype, device)
        for name, flat in param_values.items():
            value = plan.params[name]
            results["param", name] = _flat_leaf(
                flat if flat is not None else [0] * value.numel,
                value.shape,
                value.dtype,
                device,
            )
        return [results.get(key) for key in plan.operand_layout]
    finally:
        _IN_PAGED.reset(token)
        arena.close()
        for grad_file in src_files.values():
            grad_file.close()
        shutil.rmtree(grad_dir, ignore_errors=True)


__all__ = ["execute_paged_message_passing", "execute_paged_vjp"]
