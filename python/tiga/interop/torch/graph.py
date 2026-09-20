from __future__ import annotations

import hashlib
import inspect
import math
from dataclasses import dataclass
from collections.abc import Callable, Mapping
from types import SimpleNamespace
from typing import Any, Literal
import warnings

import torch
from .state import tensor_version

from ...distributed import ByDestination, DeviceMesh, GraphPlacement
from ...stencil import Neighborhood, _resolve_offsets


_INDEX_DTYPES = {torch.int32, torch.int64}


def _callable_key(function: Callable[..., object] | None) -> str:
    if function is None:
        return "default"
    try:
        source = inspect.getsource(function)
    except (OSError, TypeError):
        source = repr(function)
    return hashlib.sha256(source.encode()).hexdigest()[:16]


def _pair_tensor(
    name: str,
    value: object,
    *,
    pairs: int,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(
        value, device=device, dtype=dtype)
    if tensor.device != device:
        raise ValueError(f"{name} returned a tensor on {tensor.device}, expected {device}")
    if tensor.ndim == 0:
        return tensor.expand(pairs)
    if tensor.shape != (pairs,):
        raise ValueError(f"{name} must return shape [{pairs}], got {tuple(tensor.shape)}")
    return tensor


@dataclass(frozen=True)
class GraphSchema:
    num_src: int
    num_dst: int
    origin: str
    lifecycle: str
    realization: str
    index_dtype: torch.dtype
    sorted_by_dst: bool

    def specialization_key(self) -> tuple[object, ...]:
        return (
            self.num_src,
            self.num_dst,
            self.origin,
            self.lifecycle,
            self.realization,
            str(self.index_dtype),
            self.sorted_by_dst,
        )


@dataclass(frozen=True)
class DenseCellDirectory:
    """Compact physical directory for on-kernel radius edge generation."""

    cell_ptr: torch.Tensor
    particle_order: torch.Tensor
    cell_coordinates: torch.Tensor
    extents: torch.Tensor
    strides: torch.Tensor
    neighbor_offsets: torch.Tensor
    lattice: torch.Tensor
    inverse_lattice: torch.Tensor
    cutoff: float
    periodic: bool = False
    hash_grid: bool = False


class Graph:
    """PyTorch compatibility realization of a Tiga logical relation.

    In-memory CSR and rebuildable radius relations use the same public type.
    Default Euclidean radius construction has a tensorized CPU/CUDA uniform
    cell-list baseline; arbitrary custom metrics retain a safe all-pairs
    correctness path until they provide a provable spatial bound.
    """

    _tiga_graph = True

    def __init__(
        self,
        schema: GraphSchema,
        *,
        row_ptr: torch.Tensor | None = None,
        col_idx: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        source_positions: torch.Tensor | None = None,
        cutoff: float | torch.Tensor | None = None,
        k: int | None = None,
        periodic: torch.Tensor | None = None,
        exclude_self: bool = True,
        builder_fields: Mapping[str, torch.Tensor] | None = None,
        metric: Callable[..., torch.Tensor] | None = None,
        select: Callable[..., torch.Tensor] | None = None,
        dense_boundary: str = "full",
        device: torch.device | str | None = None,
        placement: GraphPlacement | None = None,
    ) -> None:
        self._schema = schema
        self._row_ptr = row_ptr
        self._col_idx = col_idx
        self._positions = positions
        self._source_positions = source_positions
        self._cutoff = cutoff
        self._k = k
        self._periodic = periodic
        self._exclude_self = exclude_self
        self._builder_fields = dict(builder_fields or {})
        self._metric = metric
        self._select = select
        if dense_boundary not in {"full", "lower_inclusive"}:
            raise ValueError("dense_boundary must be 'full' or 'lower_inclusive'")
        self._dense_boundary = dense_boundary
        self._device = torch.device(device) if device is not None else None
        self._placement = placement
        self._builder_udf_key = (_callable_key(metric), _callable_key(select))
        self._last_builder = "unresolved"
        self._last_build_info: dict[str, object] | None = None
        self._destination_index_cache: torch.Tensor | None = None
        self._degree_bounds_cache: tuple[int, int] | None = None
        self._degree_sum_cache: int | None = None
        self._source_index_span_ratio_cache: float | None = None
        self._sparse_cache: tuple[object, ...] | None = None
        self._cell_directory_cache: tuple[object, DenseCellDirectory] | None = None
        self._radius_csr_cache: tuple[object, torch.Tensor, torch.Tensor,
                                      dict[str, object]] | None = None
        # Exact kNN always has k entries per destination.  This immutable CSR
        # metadata survives coordinate updates; neighbor columns never do.
        self._knn_row_ptr_cache: torch.Tensor | None = None

    @classmethod
    def from_csr(
        cls,
        row_ptr: torch.Tensor,
        col_idx: torch.Tensor,
        *,
        num_src: int | None = None,
        sorted_by_dst: bool = True,
        validate: Literal["basic", "full"] = "basic",
    ) -> Graph:
        cls._validate_index_tensor("row_ptr", row_ptr)
        cls._validate_index_tensor("col_idx", col_idx)
        if row_ptr.device != col_idx.device:
            raise ValueError("row_ptr and col_idx must be on the same device")
        if row_ptr.numel() == 0:
            raise ValueError("row_ptr must contain at least the initial zero")
        num_dst = row_ptr.numel() - 1
        if num_src is None:
            if col_idx.numel() == 0:
                num_src = num_dst
            else:
                # Full value validation is explicitly allowed to synchronize.
                num_src = int(col_idx.max().item()) + 1
        if num_src < 0:
            raise ValueError("num_src must be non-negative")
        if validate == "full":
            cls._validate_csr_values(row_ptr, col_idx, num_src)
        elif validate != "basic":
            raise ValueError("validate must be 'basic' or 'full'")
        schema = GraphSchema(
            num_src=num_src,
            num_dst=num_dst,
            origin="explicit",
            lifecycle="static",
            realization="materialized_csr",
            index_dtype=col_idx.dtype,
            sorted_by_dst=sorted_by_dst,
        )
        return cls(schema, row_ptr=row_ptr, col_idx=col_idx)

    @classmethod
    def from_coo(
        cls,
        src: torch.Tensor,
        dst: torch.Tensor,
        *,
        num_src: int | None = None,
        num_dst: int | None = None,
    ) -> Graph:
        cls._validate_index_tensor("src", src)
        cls._validate_index_tensor("dst", dst)
        if src.shape != dst.shape:
            raise ValueError("src and dst must have identical shapes")
        if src.device != dst.device:
            raise ValueError("src and dst must be on the same device")
        if num_src is None:
            num_src = int(src.max().item()) + 1 if src.numel() else 0
        if num_dst is None:
            num_dst = int(dst.max().item()) + 1 if dst.numel() else 0
        if not isinstance(num_src, int) or num_src < 0:
            raise ValueError("num_src must be a non-negative integer")
        if not isinstance(num_dst, int) or num_dst < 0:
            raise ValueError("num_dst must be a non-negative integer")
        if src.numel():
            source_min, source_max = int(src.min().item()), int(src.max().item())
            destination_min = int(dst.min().item())
            destination_max = int(dst.max().item())
            if source_min < 0 or source_max >= num_src:
                raise ValueError("src contains an out-of-range source index")
            if destination_min < 0 or destination_max >= num_dst:
                raise ValueError("dst contains an out-of-range destination index")
        order = torch.argsort(dst, stable=True)
        sorted_dst = dst[order]
        col_idx = src[order]
        counts = torch.bincount(sorted_dst.to(torch.int64), minlength=num_dst)
        row_ptr = torch.empty(num_dst + 1, device=dst.device, dtype=dst.dtype)
        row_ptr[0] = 0
        torch.cumsum(counts, dim=0, out=row_ptr[1:])
        return cls.from_csr(
            row_ptr, col_idx, num_src=num_src, sorted_by_dst=True, validate="basic")

    @classmethod
    def regular(
        cls,
        num_nodes: int,
        degree: int,
        *,
        device: torch.device | str = "cpu",
    ) -> Graph:
        """Create a deterministic regular relation without an RNG.

        Every destination has exactly ``degree`` in-edges; destination ``i``
        gathers from sources ``(i * degree + k) % num_nodes`` for ``k`` in
        ``[0, degree)``.  This is the fixed-degree pattern benchmarks and
        smoke tests otherwise assemble by hand with ``arange``/``%`` at every
        call site.
        """
        if not isinstance(num_nodes, int) or num_nodes < 1:
            raise ValueError("num_nodes must be a positive integer")
        if not isinstance(degree, int) or degree < 0:
            raise ValueError("degree must be a non-negative integer")
        row_ptr = torch.arange(
            num_nodes + 1, device=device, dtype=torch.int64) * degree
        col_idx = torch.arange(
            num_nodes * degree, device=device, dtype=torch.int64) % num_nodes
        return cls.from_csr(row_ptr, col_idx, num_src=num_nodes)

    @classmethod
    def cu_seqlens(cls, cu_seqlens: torch.Tensor, *, causal: bool = True) -> Graph:
        """Create a block-diagonal packed variable-length sequence relation.

        Flash-attn varlen style: ``cu_seqlens = [0, s_1, ..., N]`` packs S
        sequences into N flat positions; position ``i`` in sequence ``k``
        gathers from sources ``[s_k, i]`` (``causal=True``) or from its whole
        sequence ``[s_k, s_{k+1})`` (``causal=False``).  Fully vectorized: no
        Python loop, built directly on the input device.
        """
        cls._validate_index_tensor("cu_seqlens", cu_seqlens)
        if not isinstance(causal, bool):
            raise TypeError("causal must be bool")
        if cu_seqlens.numel() == 0 or int(cu_seqlens[0].item()) != 0:
            raise ValueError("cu_seqlens must start at zero")
        if bool(torch.any(cu_seqlens[1:] < cu_seqlens[:-1]).item()):
            raise ValueError("cu_seqlens must be monotonic")
        device, dtype = cu_seqlens.device, cu_seqlens.dtype
        lengths = cu_seqlens.diff()                                # (S,)
        total = int(cu_seqlens[-1].item())
        positions = torch.arange(total, device=device, dtype=dtype)  # (N,)
        sequence = torch.repeat_interleave(
            torch.arange(lengths.numel(), device=device, dtype=dtype),
            lengths.to(torch.int64),
        )                                                          # (N,) sequence of each position
        start = cu_seqlens[:-1][sequence]                          # (N,) its sequence's start
        degree = positions - start + 1 if causal else lengths[sequence]  # (N,)
        row_ptr = torch.cat(
            [cu_seqlens.new_zeros(1), degree.cumsum(0, dtype=dtype)])  # (N+1,)
        repeats = degree.to(torch.int64)
        offset = torch.arange(
            int(row_ptr[-1].item()), device=device, dtype=dtype)   # (E,)
        col_idx = start.repeat_interleave(repeats) + (
            offset - row_ptr[:-1].repeat_interleave(repeats)
        )                                                          # (E,) start..end within each row
        return cls.from_csr(row_ptr, col_idx, num_src=total)

    @classmethod
    def cat(cls, graphs) -> Graph:
        """Compose relations into one block-diagonal relation.

        ``cat([g_1, ..., g_S])`` packs the blocks along the diagonal: a
        destination of block ``k`` is related exactly to the sources of
        block ``k``, offset by the cumulative source count.  This is the
        varlen-attention pattern spelled as composition —
        ``cat([triangular(n) for n in lengths])`` equals
        ``cu_seqlens(cumsum(lengths), causal=True)``.  Fully vectorized:
        each block resolves to CSR (implicit blocks realize vectorized),
        then offsets and ``torch.cat`` assemble the packed relation.
        """
        blocks = list(graphs)
        if not blocks:
            raise ValueError("graphs must be a non-empty sequence")
        for block in blocks:
            if not isinstance(block, Graph):
                raise TypeError("graphs must contain only Graph instances")
        device = blocks[0].device
        if any(str(block.device) != str(device) for block in blocks):
            raise ValueError("all blocks must share a device")
        dtype = (
            torch.int64
            if any(block.schema.index_dtype == torch.int64 for block in blocks)
            else torch.int32
        )
        row_parts = [torch.zeros(1, device=device, dtype=dtype)]
        col_parts = []
        edge_offset = 0
        src_offset = 0
        for block in blocks:
            block_rows, block_cols = block.resolve_csr()
            block_rows = block_rows.to(dtype)
            col_parts.append(block_cols.to(dtype) + src_offset)
            row_parts.append(block_rows[1:] + edge_offset)
            edge_offset += int(block_rows[-1].item())
            src_offset += block.schema.num_src
        row_ptr = torch.cat(row_parts)
        col_idx = (
            torch.cat(col_parts)
            if any(part.numel() for part in col_parts)
            else torch.empty(0, device=device, dtype=dtype)
        )
        return cls.from_csr(row_ptr, col_idx, num_src=src_offset)

    @classmethod
    def stencil(
        cls,
        dims: tuple[int, ...],
        offsets: tuple[tuple[int, ...], ...] | Neighborhood | None = None,
        *,
        periodic: bool = False,
        device: torch.device | str = "cpu",
    ) -> Graph:
        """Create a regular D-dimensional grid stencil relation.

        Nodes are row-major linearized grid coordinates; each destination
        gathers from ``coord(dst) + offset`` for every offset in ``offsets``,
        in offsets order.  Non-periodic boundaries truncate out-of-grid
        sources; periodic boundaries wrap with modulo. ``offsets`` accepts
        a ``tg.stencil`` neighborhood macro; None uses ``von_neumann()``
        (radius one, including the center). Macros infer D from ``dims``.
        """
        offsets = _resolve_offsets(dims, offsets)
        if not isinstance(periodic, bool):
            raise TypeError("periodic must be bool")
        num_nodes = math.prod(dims)
        extents = torch.tensor(dims, device=device, dtype=torch.int64)
        axes = [
            torch.arange(extent, device=device, dtype=torch.int64)
            for extent in dims
        ]
        coordinates = torch.stack(  # (N, D) row-major linearization
            torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(num_nodes, -1)
        shifts = torch.tensor(offsets, device=device, dtype=torch.int64)  # (O, D)
        sources = coordinates[:, None, :] + shifts[None, :, :]         # (N, O, D)
        if periodic:
            sources = sources.remainder(extents)
            valid = torch.ones(
                sources.shape[:2], device=device, dtype=torch.bool)
        else:
            valid = ((sources >= 0) & (sources < extents)).all(dim=-1)  # (N, O)
        strides = torch.ones(len(dims), device=device, dtype=torch.int64)
        if len(dims) > 1:
            strides[:-1] = torch.cumprod(extents[1:].flip(0), dim=0).flip(0)
        linear = (sources * strides).sum(dim=-1)                       # (N, O)
        degrees = valid.sum(dim=-1)                                    # (N,)
        row_ptr = torch.empty(num_nodes + 1, device=device, dtype=torch.int64)
        row_ptr[0] = 0
        torch.cumsum(degrees, dim=0, out=row_ptr[1:])
        col_idx = linear[valid]  # row-major: destination-major, offsets order
        return cls.from_csr(row_ptr, col_idx, num_src=num_nodes)

    @classmethod
    def radius(
        cls,
        positions: torch.Tensor,
        cutoff: float | torch.Tensor,
        *,
        exclude_self: bool = True,
        fields: Mapping[str, torch.Tensor] | None = None,
        metric: Callable[..., torch.Tensor] | None = None,
        select: Callable[..., torch.Tensor] | None = None,
        periodic: torch.Tensor | None = None,
    ) -> Graph:
        """Create a rebuildable radius relation.

        ``metric(src, dst, edge)`` returns one distance per candidate pair.
        ``select(src, dst, edge)`` is an optional additional boolean filter;
        the metric distance must still be within ``cutoff``. Endpoint
        namespaces contain ``position`` plus ``fields`` and the edge namespace
        contains ``displacement``, ``distance`` and pairwise ``cutoff``.

        A custom ``select`` preserves the default Euclidean cell-list broad
        phase because it only removes candidates. An arbitrary custom metric
        uses an all-pairs correctness path until a future structured metric
        supplies a provable spatial bound.
        """
        if positions.ndim != 2:
            raise ValueError("positions must have shape [particles, dimensions]")
        if not positions.is_floating_point():
            raise TypeError("positions must use a floating-point dtype")
        if metric is not None and not callable(metric):
            raise TypeError("metric must be callable")
        if select is not None and not callable(select):
            raise TypeError("select must be callable")
        count = positions.shape[0]
        dimensions = positions.shape[1]
        if periodic is not None:
            if not isinstance(periodic, torch.Tensor):
                periodic = torch.as_tensor(
                    periodic, dtype=positions.dtype, device=positions.device
                )
            if tuple(periodic.shape) not in {
                (dimensions,), (dimensions, dimensions)
            }:
                raise ValueError(
                    "periodic must contain box lengths [D] or lattice vectors [D,D]"
                )
            if periodic.dtype != positions.dtype or periodic.device != positions.device:
                raise ValueError("periodic must share the positions dtype and device")
            if not bool(torch.isfinite(periodic).all().item()):
                raise ValueError("periodic lattice must be finite")
            lattice = torch.diag(periodic) if periodic.ndim == 1 else periodic
            determinant = torch.linalg.det(lattice)
            if not bool(torch.isfinite(determinant).item()) or determinant.abs().item() == 0:
                raise ValueError("periodic lattice must be nonsingular")
        builder_fields = dict(fields or {})
        if "position" in builder_fields:
            raise ValueError("fields cannot redefine the reserved 'position' field")
        for name, value in builder_fields.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"fields[{name!r}] must be a torch.Tensor")
            if value.ndim == 0 or value.shape[0] != count:
                raise ValueError(
                    f"fields[{name!r}] leading dimension must be {count}, "
                    f"got {tuple(value.shape)}")
            if value.device != positions.device:
                raise ValueError(
                    f"fields[{name!r}] is on {value.device}, expected {positions.device}")
        schema = GraphSchema(
            num_src=count,
            num_dst=count,
            origin="geometric_radius",
            lifecycle="dynamic",
            realization="procedural_radius",
            index_dtype=torch.int64,
            sorted_by_dst=True,
        )
        return cls(
            schema,
            positions=positions,
            cutoff=cutoff,
            periodic=periodic,
            exclude_self=exclude_self,
            builder_fields=builder_fields,
            metric=metric,
            select=select,
        )

    @classmethod
    def knn(
        cls,
        positions: torch.Tensor,
        k: int,
        *,
        candidates: torch.Tensor | None = None,
        exclude_self: bool | None = None,
    ) -> Graph:
        if not isinstance(positions, torch.Tensor):
            raise TypeError("positions must be a torch.Tensor")
        if positions.ndim != 2 or not positions.dtype.is_floating_point:
            raise TypeError("positions must be a rank-two floating Tensor")
        self_search = candidates is None
        source_positions = positions if self_search else candidates
        if not isinstance(source_positions, torch.Tensor):
            raise TypeError("candidates must be a torch.Tensor")
        if source_positions.ndim != 2 or not source_positions.dtype.is_floating_point:
            raise TypeError("candidates must be a rank-two floating Tensor")
        if source_positions.shape[1] != positions.shape[1]:
            raise ValueError("positions and candidates must share feature width")
        if source_positions.dtype != positions.dtype:
            raise TypeError("positions and candidates must share dtype")
        if source_positions.device != positions.device:
            raise ValueError("positions and candidates must share device")
        if exclude_self is None:
            exclude_self = self_search
        if not isinstance(exclude_self, bool):
            raise TypeError("exclude_self must be bool or None")
        if exclude_self and not self_search:
            raise ValueError(
                "exclude_self is only defined for self-kNN; omit candidates"
            )
        query_count = positions.shape[0]
        source_count = source_positions.shape[0]
        maximum = source_count - 1 if exclude_self else source_count
        if not isinstance(k, int) or k <= 0 or k > maximum:
            raise ValueError(
                f"k must be in [1, {maximum}] for {source_count} candidates"
            )
        schema = GraphSchema(
            num_src=source_count,
            num_dst=query_count,
            origin="geometric_knn",
            lifecycle="dynamic",
            realization="procedural_knn",
            index_dtype=torch.int64,
            sorted_by_dst=True,
        )
        return cls(
            schema,
            positions=positions,
            source_positions=None if self_search else source_positions,
            k=k,
            exclude_self=exclude_self,
        )

    @classmethod
    def dense(
        cls,
        num_src: int,
        num_dst: int | None = None,
        *,
        device: torch.device | str | None = None,
        index_dtype: torch.dtype = torch.int64,
    ) -> Graph:
        """Create an implicit Cartesian relation without storing adjacency.

        The relation contains every ``(dst, src)`` pair. Recognized consumers
        may traverse it in dense tiles; generic semantic evaluation is guarded
        against accidentally materializing a very large CSR.
        """
        if not isinstance(num_src, int) or num_src < 0:
            raise ValueError("num_src must be a non-negative integer")
        if num_dst is None:
            num_dst = num_src
        if not isinstance(num_dst, int) or num_dst < 0:
            raise ValueError("num_dst must be a non-negative integer")
        if index_dtype not in _INDEX_DTYPES:
            raise TypeError("index_dtype must be torch.int32 or torch.int64")
        resolved_device = torch.device("cpu" if device is None else device)
        if (
            resolved_device.type == "cuda"
            and resolved_device.index is None
            and torch.cuda.is_available()
        ):
            resolved_device = torch.device("cuda", torch.cuda.current_device())
        schema = GraphSchema(
            num_src=num_src,
            num_dst=num_dst,
            origin="cartesian",
            lifecycle="static",
            realization="implicit_dense",
            index_dtype=index_dtype,
            sorted_by_dst=True,
        )
        return cls(schema, device=resolved_device)

    @classmethod
    def triangular(
        cls,
        num_entities: int,
        *,
        device: torch.device | str | None = None,
        index_dtype: torch.dtype = torch.int64,
    ) -> Graph:
        graph = cls.dense(
            num_entities, num_entities, device=device, index_dtype=index_dtype)
        graph._schema = GraphSchema(
            num_src=num_entities, num_dst=num_entities,
            origin="lower_triangular", lifecycle="static",
            realization="implicit_dense", index_dtype=index_dtype,
            sorted_by_dst=True,
        )
        graph._dense_boundary = "lower_inclusive"
        return graph

    @staticmethod
    def _validate_index_tensor(name: str, value: torch.Tensor) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        if value.dtype not in _INDEX_DTYPES:
            raise TypeError(f"{name} must have int32 or int64 dtype")

    @staticmethod
    def _validate_csr_values(
        row_ptr: torch.Tensor, col_idx: torch.Tensor, num_src: int
    ) -> None:
        if int(row_ptr[0].item()) != 0:
            raise ValueError("row_ptr[0] must be zero")
        if int(row_ptr[-1].item()) != col_idx.numel():
            raise ValueError("row_ptr[-1] must equal the number of edges")
        if bool(torch.any(row_ptr[1:] < row_ptr[:-1]).item()):
            raise ValueError("row_ptr must be monotonic")
        if col_idx.numel():
            lo = int(col_idx.min().item())
            hi = int(col_idx.max().item())
            if lo < 0 or hi >= num_src:
                raise ValueError("col_idx contains an out-of-range source index")

    @property
    def schema(self) -> GraphSchema:
        return self._schema

    @property
    def placement(self) -> GraphPlacement | None:
        """Return distributed ownership metadata, if this snapshot has it."""
        return self._placement

    @property
    def is_distributed(self) -> bool:
        return self._placement is not None

    def halo(
        self,
        mesh: DeviceMesh,
        *,
        partition: ByDestination | None = None,
        depth: int | Literal["auto"] = "auto",
    ) -> Graph:
        """Return the same logical graph with declarative halo placement.

        No communication happens here.  The compiler will derive owned/ghost
        regions and lower halo exchange plus an interior/boundary dependency
        plan below the user kernel. Execution requires an active
        ``DistributedRuntime``; automatic communication/computation overlap is
        not yet implemented in the alpha runtime;
        this keeps transport selection below the Graph/MessagePassing API.
        """
        if not isinstance(mesh, DeviceMesh):
            raise TypeError("mesh must be a tiga.DeviceMesh")
        if partition is None:
            partition = ByDestination()
        if not isinstance(partition, ByDestination):
            raise TypeError("partition must be a tiga.ByDestination")
        placement = GraphPlacement(mesh, partition, depth)

        # Graph snapshots share immutable physical inputs, while planning and
        # builder caches remain local to the returned logical handle.
        clone = object.__new__(Graph)
        clone.__dict__ = self.__dict__.copy()
        clone._builder_fields = dict(self._builder_fields)
        clone._last_build_info = (
            None if self._last_build_info is None else dict(self._last_build_info)
        )
        clone._destination_index_cache = None
        clone._degree_bounds_cache = self._degree_bounds_cache
        clone._degree_sum_cache = self._degree_sum_cache
        clone._source_index_span_ratio_cache = self._source_index_span_ratio_cache
        clone._sparse_cache = None
        clone._cell_directory_cache = None
        clone._radius_csr_cache = None
        clone._knn_row_ptr_cache = self._knn_row_ptr_cache
        clone._placement = placement
        return clone

    def transpose(self) -> Graph:
        """Return one materialized snapshot with endpoint roles swapped."""
        row_ptr, col_idx = self.resolve_csr()
        destination = self.destination_index(row_ptr)
        return type(self).from_coo(
            destination,
            col_idx,
            num_src=self._schema.num_dst,
            num_dst=self._schema.num_src,
        )

    @property
    def num_edges(self) -> int | None:
        if self._schema.realization == "implicit_dense":
            if self._dense_boundary == "lower_inclusive":
                count = self._schema.num_dst
                return count * (count + 1) // 2
            return self._schema.num_src * self._schema.num_dst
        return None if self._col_idx is None else self._col_idx.numel()

    @property
    def device(self) -> torch.device:
        if self._col_idx is not None:
            return self._col_idx.device
        if self._positions is not None:
            return self._positions.device
        assert self._device is not None
        return self._device

    def euclidean_positions(self) -> torch.Tensor | None:
        """Return positions when the relation has the default Euclidean metric.

        This is a compiler-facing structural proof: generated consumers may
        recompute distance from endpoints instead of materializing an edge
        geometry tensor.  Custom metrics return ``None`` and retain the fully
        general implicit-edge path.
        """
        return (
            self._positions
            if self._positions is not None
            and self._schema.realization == "procedural_radius"
            and self._metric is None
            else None
        )

    def ranked_positions(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return query/candidate coordinates without materializing kNN CSR."""
        if self._schema.realization != "procedural_knn" or self._positions is None:
            raise TypeError("ranked_positions requires a procedural kNN graph")
        return (
            self._positions,
            self._positions if self._source_positions is None
            else self._source_positions,
        )

    def generated_cell_directory(self) -> DenseCellDirectory | None:
        """Build an O(N + cells) directory without materializing graph edges.

        This is a compiler-facing physical realization for the default scalar
        Euclidean radius relation. Contiguous CUDA FP32 non-periodic 2D/3D
        inputs use modular hash buckets and reusable buffers; other supported
        inputs retain the dense directory. Custom metric/select and excessive
        dense-grid bounds return None so the dispatcher can use CSR.
        """
        if (
            self._positions is None
            or self._metric is not None
            or self._select is not None
            or not self._exclude_self
        ):
            return None
        cutoff_version = (
            tensor_version(self._cutoff)
            if isinstance(self._cutoff, torch.Tensor) else self._cutoff
        )
        cache_key = (
            tensor_version(self._positions),
            getattr(self, "_directory_mode", "auto"),
            cutoff_version,
            self._positions.data_ptr(),
            None if self._periodic is None else (
                self._periodic.data_ptr(), tensor_version(self._periodic)),
        )
        if (
            self._cell_directory_cache is not None
            and self._cell_directory_cache[0] == cache_key
        ):
            return self._cell_directory_cache[1]
        cell_size = self._cell_list_size()
        if cell_size is None or self._schema.num_dst == 0:
            return None
        positions = self._positions
        if (getattr(self, "_directory_mode", "auto") != "dense"
                and self._periodic is None and positions.is_cuda
                and positions.dtype == torch.float32 and positions.is_contiguous()
                and positions.shape[1] in (2, 3) and math.isfinite(cell_size)):
            try:
                from .radius_grid import build_hash_directory
            except ImportError:
                pass  # Optional CUDA provider unavailable: retain dense builder.
            else:
                if not hasattr(self, "_hash_directory_pool"):
                    self._hash_directory_pool = []
                # Release only this Graph's ownership. An outstanding autograd
                # context or caller-held snapshot keeps its slot unavailable.
                self._cell_directory_cache = None
                directory = build_hash_directory(positions, cell_size, self._hash_directory_pool)
                self._last_builder = "modular_hash_grid"
                self._cell_directory_cache = (cache_key, directory)
                return directory
        dimensions = positions.shape[1]
        periodic = self._periodic is not None
        if periodic:
            lattice = self._periodic_lattice().contiguous()
            inverse_lattice = torch.linalg.inv(lattice).contiguous()
            fractional = positions @ inverse_lattice
            fractional = fractional - torch.floor(fractional)
            fractional_bound = cell_size * torch.linalg.vector_norm(
                inverse_lattice, dim=0)
            extents = torch.floor(
                1.0 / fractional_bound.clamp_min(1.0e-12)
            ).clamp(min=1).to(torch.int64)
            normalized = torch.floor(fractional * extents).to(torch.int64)
            normalized = torch.minimum(normalized, extents - 1)
        else:
            lattice = torch.eye(
                dimensions, device=positions.device, dtype=positions.dtype)
            inverse_lattice = lattice
            cells = torch.floor(positions / cell_size).to(torch.int64)
            lower = cells.amin(dim=0)
            normalized = cells - lower
            extents = normalized.amax(dim=0) + 1
        strides = torch.ones(
            dimensions, device=positions.device, dtype=torch.int64)
        if dimensions > 1:
            strides[1:] = torch.cumprod(extents[:-1], dim=0)
        cell_count = int(torch.prod(extents).item())
        # A dense directory makes lookup O(1), but must remain an auxiliary
        # O(N)-scale structure. Sparse/extreme-coordinate domains use CSR.
        if cell_count > max(4096, 8 * self._schema.num_dst):
            return None
        keys = torch.sum(normalized * strides, dim=-1)
        order = torch.argsort(keys, stable=True)
        counts = torch.bincount(keys, minlength=cell_count)
        cell_ptr = torch.empty(
            cell_count + 1, device=positions.device, dtype=torch.int64)
        cell_ptr[0] = 0
        torch.cumsum(counts, dim=0, out=cell_ptr[1:])
        axes = []
        for extent in extents.tolist():
            values = (
                (0,) if periodic and extent == 1
                else ((0, 1) if periodic and extent == 2
                      else (-1, 0, 1))
            )
            axes.append(torch.tensor(
                values, device=positions.device, dtype=torch.int64))
        mesh = torch.meshgrid(*axes, indexing="ij")
        neighbor_offsets = torch.stack(
            mesh, dim=-1).reshape(-1, dimensions)
        self._last_builder = "uniform_cell_directory"
        directory = DenseCellDirectory(
            cell_ptr=cell_ptr,
            particle_order=order.to(torch.int64),
            cell_coordinates=normalized,
            extents=extents,
            strides=strides,
            neighbor_offsets=neighbor_offsets,
            lattice=lattice,
            inverse_lattice=inverse_lattice,
            cutoff=cell_size,
            periodic=periodic,
        )
        self._cell_directory_cache = (cache_key, directory)
        return directory

    def resolve_csr(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._row_ptr is not None and self._col_idx is not None:
            return self._row_ptr, self._col_idx
        if self._schema.realization == "procedural_knn":
            assert self._positions is not None and self._k is not None
            # Use the vendor Euclidean-distance primitive itself.  Although
            # squared distance is algebraically order-equivalent, spelling it
            # as ||x||^2 + ||y||^2 - 2 x@y introduces enough cancellation in
            # fp32 to change a k-boundary on realistic large point sets.  The
            # exhaustive cdist/top-k contract remains exact rather than ANN
            # and matches the public Euclidean metric at provider precision.
            candidates = (
                self._positions
                if self._source_positions is None
                else self._source_positions
            )
            distances = torch.cdist(self._positions, candidates)
            if self._exclude_self:
                distances.fill_diagonal_(float("inf"))
            # k=1 is the assignment primitive used by clustering and vector
            # quantization.  argmin has identical exact semantics without the
            # general top-k selection machinery.
            col_idx = (
                torch.argmin(distances, dim=1, keepdim=True)
                if self._k == 1
                else torch.topk(
                    distances, self._k, dim=1, largest=False, sorted=True
                ).indices
            ).reshape(-1).to(torch.int64)
            row_ptr = self._knn_row_ptr_cache
            if row_ptr is None:
                row_ptr = torch.arange(
                    0, col_idx.numel() + 1, self._k,
                    device=self.device, dtype=torch.int64)
                self._knn_row_ptr_cache = row_ptr
            self._last_builder = (
                "torch-cdist-argmin-library-dispatch"
                if self._k == 1
                else "torch-cdist-topk-library-dispatch"
            )
            self._last_build_info = {
                "builder": self._last_builder,
                "candidate_pairs": self._schema.num_dst * self._schema.num_src,
                "accepted_edges": col_idx.numel(),
                "materialized": True,
                "cache_hit": False,
            }
            return row_ptr, col_idx
        if self._schema.realization == "implicit_dense":
            edges = int(self.num_edges or 0)
            if edges > (1 << 22):
                raise RuntimeError(
                    "implicit DenseGraph would materialize more than 2^22 edges; "
                    "use a compiler-supported tiled consumer instead")
            if self._dense_boundary == "lower_inclusive":
                degrees = torch.arange(
                    1, self._schema.num_dst + 1, device=self.device,
                    dtype=self._schema.index_dtype,
                )
                row_ptr = torch.cat((
                    torch.zeros(
                        1, device=self.device, dtype=self._schema.index_dtype),
                    degrees.cumsum(0),
                ))
                # Row r owns edges [r(r-1)/2, r(r+1)/2) with columns 0..r;
                # vectorized, no per-row Python loop.
                repeats = degrees.to(torch.int64)
                offset = torch.arange(
                    edges, device=self.device, dtype=self._schema.index_dtype)
                col_idx = offset - row_ptr[:-1].repeat_interleave(repeats)
            elif self._schema.num_src == 0:
                row_ptr = torch.zeros(
                    self._schema.num_dst + 1,
                    device=self.device, dtype=self._schema.index_dtype)
                col_idx = torch.empty(
                    0, device=self.device, dtype=self._schema.index_dtype)
            else:
                row_ptr = torch.arange(
                    0, edges + 1, self._schema.num_src,
                    device=self.device, dtype=self._schema.index_dtype)
                col_idx = torch.arange(
                    self._schema.num_src,
                    device=self.device, dtype=self._schema.index_dtype,
                ).repeat(self._schema.num_dst)
            self._last_builder = "dense_reference_materialization"
            self._last_build_info = {
                "builder": self._last_builder,
                "candidate_pairs": edges,
                "accepted_edges": edges,
                "materialized": True,
            }
            return row_ptr, col_idx
        assert self._positions is not None and self._cutoff is not None
        cutoff_key = (
            (self._cutoff.data_ptr(), tensor_version(self._cutoff))
            if isinstance(self._cutoff, torch.Tensor)
            else float(self._cutoff)
        )
        periodic_key = (
            None
            if self._periodic is None
            else (self._periodic.data_ptr(), tensor_version(self._periodic))
        )
        snapshot_key = (
            self._positions.data_ptr(), tensor_version(self._positions),
            cutoff_key, periodic_key,
            tuple(
                (name, value.data_ptr(), tensor_version(value))
                for name, value in sorted(self._builder_fields.items())
            ),
            self._builder_udf_key, self._exclude_self,
        )
        if (
            self._radius_csr_cache is not None
            and self._radius_csr_cache[0] == snapshot_key
        ):
            _, row_ptr, col_idx, info = self._radius_csr_cache
            self._last_build_info = {**info, "cache_hit": True}
            self._last_builder = str(info["builder"])
            return row_ptr, col_idx
        count = self._schema.num_dst
        indices = torch.arange(count, device=self._positions.device)
        cell_size = self._cell_list_size()
        if cell_size is None:
            dst = indices.repeat_interleave(count)
            src = indices.repeat(count)
            self._last_builder = "all_pairs"
        else:
            dst, src = self._cell_list_candidates(cell_size)
            self._last_builder = "uniform_cell_list"
        candidate_pairs = src.numel()
        adjacency = self._pair_selection(dst, src)
        dst, src = dst[adjacency], src[adjacency]
        self._last_build_info = {
            "builder": self._last_builder,
            "candidate_pairs": candidate_pairs,
            "accepted_edges": src.numel(),
            "materialized": True,
            "cache_hit": False,
        }
        counts = torch.bincount(dst, minlength=self._schema.num_dst)
        row_ptr = torch.empty(
            self._schema.num_dst + 1,
            device=self._positions.device,
            dtype=torch.int64,
        )
        row_ptr[0] = 0
        torch.cumsum(counts, dim=0, out=row_ptr[1:])
        col_idx = src.to(torch.int64)
        self._radius_csr_cache = (
            snapshot_key, row_ptr, col_idx, dict(self._last_build_info)
        )
        return row_ptr, col_idx

    def _cell_list_size(self) -> float | None:
        """Return a safe Euclidean cell size, or None for all-pairs.

        A custom metric does not imply a Euclidean bound. Scalar default-metric
        cutoff in up to three dimensions is the first performance contract.
        """
        assert self._positions is not None and self._cutoff is not None
        if self._metric is not None or self._positions.shape[1] not in (1, 2, 3):
            return None
        cutoff = self._cutoff
        if isinstance(cutoff, torch.Tensor):
            if cutoff.numel() != 1:
                return None
            value = float(cutoff.detach().item())
        else:
            value = float(cutoff)
        return value if value > 0.0 else None

    def _cell_list_candidates(
        self, cell_size: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate a directed superset from adjacent uniform-grid cells.

        The construction is tensorized: it expands only occupied neighbor-cell
        segments and never allocates an N x N matrix. Exact cutoff and select
        UDF evaluation happen in ``_pair_selection``.
        """
        assert self._positions is not None
        count, dimensions = self._positions.shape
        device = self._positions.device
        if count == 0:
            empty = torch.empty(0, device=device, dtype=torch.int64)
            return empty, empty

        periodic = self._periodic is not None
        if periodic:
            lattice = self._periodic_lattice()
            inverse = torch.linalg.inv(lattice)
            fractional = self._positions @ inverse
            fractional = fractional - torch.floor(fractional)
            # For accepted Cartesian displacement d, fractional component i
            # is bounded by cutoff*||inverse[:,i]||. A bin at least this wide
            # makes one wrapped neighbor-cell step a safe broad phase even for
            # a skew lattice.
            fractional_bound = cell_size * torch.linalg.vector_norm(
                inverse, dim=0
            )
            extents = torch.floor(
                1.0 / fractional_bound.clamp_min(1.0e-12)
            ).clamp(min=1).to(torch.int64)
            cells = torch.floor(fractional * extents).to(torch.int64)
            cells = torch.minimum(cells, extents - 1)
            lower = torch.zeros_like(extents)
            upper = extents - 1
        else:
            cells = torch.floor(self._positions / cell_size).to(torch.int64)
            lower = cells.amin(dim=0)
            upper = cells.amax(dim=0)
            extents = upper - lower + 1
        strides = torch.ones(dimensions, device=device, dtype=torch.int64)
        if dimensions > 1:
            strides[1:] = torch.cumprod(extents[:-1], dim=0)
        keys = torch.sum((cells - lower) * strides, dim=-1)
        sorted_keys, order = torch.sort(keys, stable=True)
        occupied, occupied_counts = torch.unique_consecutive(
            sorted_keys, return_counts=True)
        occupied_starts = torch.cumsum(occupied_counts, dim=0) - occupied_counts

        if periodic:
            axes = []
            for extent in extents.tolist():
                values = (
                    (0,) if extent == 1
                    else ((0, 1) if extent == 2 else (-1, 0, 1))
                )
                axes.append(
                    torch.tensor(values, device=device, dtype=torch.int64)
                )
        else:
            axis = torch.arange(-1, 2, device=device, dtype=torch.int64)
            axes = [axis] * dimensions
        mesh = torch.meshgrid(*axes, indexing="ij")
        offsets = torch.stack(mesh, dim=-1).reshape(-1, dimensions)
        neighbor_cells = cells[:, None, :] + offsets[None, :, :]
        if periodic:
            neighbor_cells = torch.remainder(neighbor_cells, extents)
            valid = torch.ones(
                neighbor_cells.shape[:-1], device=device, dtype=torch.bool
            )
        else:
            valid = torch.all(
                (neighbor_cells >= lower) & (neighbor_cells <= upper), dim=-1)
        neighbor_keys = torch.sum(
            (neighbor_cells - lower) * strides, dim=-1).reshape(-1)
        valid = valid.reshape(-1)

        # searchsorted needs an in-range gather index even for absent cells.
        locations = torch.searchsorted(occupied, neighbor_keys)
        safe_locations = locations.clamp_max(occupied.numel() - 1)
        found = (
            valid
            & (locations < occupied.numel())
            & (occupied[safe_locations] == neighbor_keys)
        )
        candidate_counts = torch.where(
            found, occupied_counts[safe_locations],
            torch.zeros_like(safe_locations))
        query = torch.repeat_interleave(
            torch.arange(candidate_counts.numel(), device=device),
            candidate_counts)
        if query.numel() == 0:
            empty = torch.empty(0, device=device, dtype=torch.int64)
            return empty, empty
        prefix = torch.cumsum(candidate_counts, dim=0) - candidate_counts
        local = (
            torch.arange(query.numel(), device=device)
            - torch.repeat_interleave(prefix, candidate_counts))
        source_in_order = occupied_starts[safe_locations[query]] + local
        src = order[source_in_order]
        dst = torch.div(query, offsets.shape[0], rounding_mode="floor")
        return dst.to(torch.int64), src.to(torch.int64)

    def _pair_selection(
        self, dst: torch.Tensor, src: torch.Tensor
    ) -> torch.Tensor:
        assert self._positions is not None
        src_fields = {
            "position": self._positions[src],
            **{name: value[src] for name, value in self._builder_fields.items()},
        }
        dst_fields = {
            "position": self._positions[dst],
            **{name: value[dst] for name, value in self._builder_fields.items()},
        }
        displacement = self._minimum_image(
            src_fields["position"] - dst_fields["position"]
        )
        edge_values: dict[str, torch.Tensor] = {"displacement": displacement}
        src_view = SimpleNamespace(**src_fields)
        dst_view = SimpleNamespace(**dst_fields)
        if self._metric is None:
            distance = torch.linalg.vector_norm(displacement, dim=-1)
        else:
            distance = _pair_tensor(
                "metric",
                self._metric(src_view, dst_view, SimpleNamespace(**edge_values)),
                pairs=src.numel(),
                device=self._positions.device,
                dtype=self._positions.dtype,
            )
            if not distance.is_floating_point():
                raise TypeError("metric must return a floating-point tensor")
        cutoff = self._pair_cutoff(dst, src)
        edge_values.update(distance=distance, cutoff=cutoff)
        adjacency = distance <= cutoff
        if self._select is not None:
            selected = _pair_tensor(
                "select",
                self._select(
                    src_view, dst_view, SimpleNamespace(**edge_values)),
                pairs=src.numel(),
                device=self._positions.device,
            )
            if selected.dtype is not torch.bool:
                raise TypeError("select must return a boolean tensor")
            adjacency &= selected
        if self._exclude_self:
            adjacency &= src != dst
        return adjacency

    def _pair_cutoff(self, dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        assert self._cutoff is not None and self._positions is not None
        cutoff = self._cutoff
        if not isinstance(cutoff, torch.Tensor):
            return torch.full(
                (src.numel(),), float(cutoff), device=self._positions.device,
                dtype=self._positions.dtype)
        if cutoff.device != self._positions.device:
            raise ValueError(
                f"cutoff is on {cutoff.device}, expected {self._positions.device}")
        if cutoff.ndim == 0 or cutoff.numel() == 1:
            return cutoff.reshape(()).expand(src.numel())
        if cutoff.shape == (self._schema.num_dst,):
            return cutoff[dst]
        if cutoff.shape == (self._schema.num_dst, self._schema.num_src):
            return cutoff[dst, src]
        raise ValueError(
            "cutoff must be scalar, per-destination [N], or pairwise [N, N]; "
            f"got {tuple(cutoff.shape)}")

    def _periodic_lattice(self) -> torch.Tensor:
        assert self._periodic is not None
        return (
            torch.diag(self._periodic)
            if self._periodic.ndim == 1
            else self._periodic
        )

    def _minimum_image(self, displacement: torch.Tensor) -> torch.Tensor:
        """Map Cartesian displacements to the centered periodic cell."""
        if self._periodic is None:
            return displacement
        lattice = self._periodic_lattice()
        fractional = displacement @ torch.linalg.inv(lattice)
        fractional = fractional - torch.round(fractional)
        return fractional @ lattice

    def destination_index(self, row_ptr: torch.Tensor | None = None) -> torch.Tensor:
        if row_ptr is None and self._destination_index_cache is not None:
            return self._destination_index_cache
        if row_ptr is None:
            row_ptr, _ = self.resolve_csr()
        degree = row_ptr[1:] - row_ptr[:-1]
        dst = torch.repeat_interleave(
            torch.arange(self._schema.num_dst, device=row_ptr.device), degree)
        if self._row_ptr is row_ptr:
            self._destination_index_cache = dst
        return dst

    def degree_bounds(self) -> tuple[int, int]:
        """Return cached physical CSR degree bounds for planning.

        The first call may synchronize an accelerator.  It is graph analysis,
        not kernel consume time, and remains valid for this immutable snapshot.
        """
        if self._degree_bounds_cache is not None:
            return self._degree_bounds_cache
        if self._schema.realization == "implicit_dense":
            if self._dense_boundary == "lower_inclusive" and self._schema.num_dst:
                return 1, self._schema.num_src
            return self._schema.num_src, self._schema.num_src
        row_ptr, _ = self.resolve_csr()
        if self._schema.num_dst == 0:
            bounds = (0, 0)
        else:
            degrees = row_ptr[1:] - row_ptr[:-1]
            extrema = torch.stack((degrees.min(), degrees.max())).cpu()
            bounds = int(extrema[0]), int(extrema[1])
        if self._row_ptr is not None:
            self._degree_bounds_cache = bounds
        return bounds

    def degree_statistics(self) -> tuple[int, int, int]:
        """Return min/max/sum without materializing destination indices."""
        minimum, maximum = self.degree_bounds()
        if self._degree_sum_cache is None:
            if self._schema.realization == "implicit_dense":
                self._degree_sum_cache = int(self.num_edges or 0)
            else:
                row_ptr, _ = self.resolve_csr()
                self._degree_sum_cache = int(row_ptr[-1].item())
        return minimum, maximum, self._degree_sum_cache

    def degree_histogram(self) -> tuple[int, ...]:
        """Return exact immutable-snapshot row counts indexed by degree."""
        minimum, maximum, _ = self.degree_statistics()
        del minimum
        if maximum > 4096:
            raise ValueError("exact degree histogram is limited to degree_max <= 4096")
        if self._schema.realization == "implicit_dense":
            counts = [0] * (maximum + 1)
            if self._dense_boundary == "lower_inclusive":
                for degree in range(1, maximum + 1):
                    counts[degree] = 1
            else:
                counts[maximum] = self._schema.num_dst
            return tuple(counts)
        row_ptr, _ = self.resolve_csr()
        import torch
        counts = torch.bincount(
            row_ptr[1:] - row_ptr[:-1], minlength=maximum + 1
        ).cpu()
        return tuple(int(value) for value in counts)

    def source_index_span_ratio(self, samples: int = 4096) -> float:
        """Estimate physical source-index distance from destination rows.

        This is a target-independent locality signal, not a semantic graph
        property.  It samples deterministic edge slots and returns the mean
        ``abs(src_id - dst_id)`` normalized by endpoint cardinality.  A value
        near zero proves index-local gathers; random numbering approaches
        one third.  Immutable snapshots cache the result for JIT planning.
        """
        if self._source_index_span_ratio_cache is not None:
            return self._source_index_span_ratio_cache
        if self._schema.realization == "implicit_dense":
            return 1.0 / 3.0
        row_ptr, col_idx = self.resolve_csr()
        edges = col_idx.numel()
        if edges == 0:
            ratio = 0.0
        else:
            count = min(samples, edges)
            slots = torch.linspace(
                0, edges - 1, count, device=col_idx.device,
                dtype=torch.float64,
            ).to(dtype=row_ptr.dtype)
            rows = torch.searchsorted(row_ptr[1:], slots, right=True)
            span = (col_idx[slots] - rows).abs().to(torch.float64)
            ratio = float(
                (span.mean() / max(1, self._schema.num_src,
                                   self._schema.num_dst)).cpu()
            )
        ratio = min(1.0, max(0.0, ratio))
        if self._row_ptr is not None:
            self._source_index_span_ratio_cache = ratio
        return ratio

    def fixed_degree(self) -> int | None:
        minimum, maximum = self.degree_bounds()
        return minimum if minimum == maximum else None

    def planning_key(self) -> tuple[object, ...]:
        physical_stats = (
            self.degree_statistics()
            if self._row_ptr is not None or self._schema.realization == "implicit_dense"
            else "dynamic")
        procedural = ()
        if self._positions is not None:
            cutoff = (
                self._k
                if self._schema.realization == "procedural_knn"
                else self._cell_list_size()
            )
            procedural = (
                str(self._positions.dtype),
                tuple(self._positions.shape[1:]),
                None if self._source_positions is None else (
                    str(self._source_positions.dtype),
                    tuple(self._source_positions.shape[1:]),
                ),
                cutoff,
                self._exclude_self,
                None if self._periodic is None else (
                    tuple(self._periodic.shape), tensor_version(self._periodic),
                    self._periodic.data_ptr(),
                ),
            )
        return (
            *self._schema.specialization_key(), physical_stats,
            self.source_index_span_ratio()
            if self._row_ptr is not None else "dynamic-locality",
            self._builder_udf_key, procedural,
            getattr(self, "_directory_mode", "auto"),
            self._dense_boundary,
            None if self._placement is None else self._placement.specialization_key(),
        )

    def sparse_csr(
        self,
        values: torch.Tensor,
        *,
        row_ptr: torch.Tensor | None = None,
        col_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a native sparse view, caching only immutable snapshots.

        Compiler paths that already resolved a dynamic graph pass that exact
        snapshot here.  This avoids silently rebuilding a procedural relation
        between edge-value generation and sparse consumption.
        """
        if (row_ptr is None) != (col_idx is None):
            raise ValueError("row_ptr and col_idx must be supplied together")
        if row_ptr is None:
            row_ptr, col_idx = self.resolve_csr()
        assert col_idx is not None
        version = tensor_version(values)
        key = (id(values), version, tuple(values.shape), values.dtype, values.device)
        # A sparse wrapper of differentiable values owns an autograd node.
        # Reusing it after backward reuses a freed tape, and a wrapper created
        # under no_grad would silently disconnect a later training call.
        can_cache = (self._row_ptr is row_ptr and self._col_idx is col_idx
                     and not values.requires_grad)
        if can_cache and self._sparse_cache is not None and self._sparse_cache[0] == key:
            return self._sparse_cache[1]
        # cuSPARSE consumes contiguous CSR values; preserve the Torch gradient
        # connection through the materialization of a strided input view.
        flat = values.reshape(-1).contiguous()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Sparse invariant checks.*")
            warnings.filterwarnings("ignore", message="Sparse CSR tensor support.*")
            sparse = torch.sparse_csr_tensor(
                row_ptr.contiguous(),
                col_idx.contiguous(),
                flat,
                size=(self._schema.num_dst, self._schema.num_src),
                check_invariants=False,
            )
        if can_cache:
            self._sparse_cache = (key, sparse)
        return sparse

    @property
    def build_info(self) -> dict[str, object]:
        if self._last_build_info is None:
            raise RuntimeError("dynamic graph has not been resolved yet")
        return dict(self._last_build_info)

    def implicit_edge_fields(
        self, row_ptr: torch.Tensor, col_idx: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if self._positions is None:
            return {}
        dst = self.destination_index(row_ptr)
        source_positions = (
            self._positions
            if self._source_positions is None
            else self._source_positions
        )
        displacement = self._minimum_image(
            source_positions[col_idx] - self._positions[dst]
        )
        if self._metric is None:
            distance = torch.linalg.vector_norm(displacement, dim=-1)
        else:
            src_fields = {
                "position": source_positions[col_idx],
                **{name: value[col_idx] for name, value in self._builder_fields.items()},
            }
            dst_fields = {
                "position": self._positions[dst],
                **{name: value[dst] for name, value in self._builder_fields.items()},
            }
            distance = _pair_tensor(
                "metric",
                self._metric(
                    SimpleNamespace(**src_fields),
                    SimpleNamespace(**dst_fields),
                    SimpleNamespace(displacement=displacement),
                ),
                pairs=col_idx.numel(),
                device=self._positions.device,
                dtype=self._positions.dtype,
            )
        return {
            "displacement": displacement,
            "distance": distance,
        }

    def explain(self) -> str:
        edges = "dynamic" if self.num_edges is None else f"{self.num_edges:,}"
        udf = ""
        if self._positions is not None:
            if self._schema.realization == "procedural_knn":
                search = "self" if self._source_positions is None else "bipartite"
                udf = (
                    f", k={self._k}, search={search}, metric=euclidean, "
                    f"builder={self._last_builder}"
                )
            else:
                broad_phase = "euclidean-cell-list-eligible" if self._metric is None else "all-pairs-only"
                udf = (
                    f", metric={'custom' if self._metric else 'euclidean'}, "
                    f"select={'custom' if self._select else 'cutoff-only'}, "
                    f"periodic={'none' if self._periodic is None else tuple(self._periodic.shape)}, "
                    f"broad_phase={broad_phase}, builder={self._last_builder}")
            if self._last_build_info is not None:
                udf += (
                    f", candidates={self._last_build_info['candidate_pairs']:,}, "
                    f"accepted={self._last_build_info['accepted_edges']:,}")
        distributed = ""
        if self._placement is not None:
            distributed = (
                f", mesh={self._placement.mesh.device_type}"
                f"{self._placement.mesh.shape}, "
                f"partition=destination:{self._placement.partition.mesh_axis}, "
                f"halo={self._placement.halo_depth}"
            )
        return (
            f"Graph(origin={self._schema.origin}, lifecycle={self._schema.lifecycle}, "
            f"realization={self._schema.realization}, "
            f"src={self._schema.num_src:,}, dst={self._schema.num_dst:,}, "
            f"edges={edges}, device={self.device}{udf}{distributed})"
        )


def from_native(graph) -> Graph:
    """Create a temporary Torch execution view of a native logical Graph."""
    if isinstance(graph, Graph):
        return graph
    cached = getattr(graph, "_torch_view_cache", None)
    if cached is not None:
        return cached
    dtype = {"int32": torch.int32, "int64": torch.int64}[graph.schema.index_dtype.name]
    schema = GraphSchema(
        num_src=graph.schema.num_src,
        num_dst=graph.schema.num_dst,
        origin=graph.schema.origin,
        lifecycle=graph.schema.lifecycle,
        realization=graph.schema.realization,
        index_dtype=dtype,
        sorted_by_dst=graph.schema.sorted_by_dst,
    )
    def topology_tensor(value):
        if value is None:
            return None
        value.realize()
        # CPU-generated topology (e.g. Graph.stencil) can be Tiga-owned.
        # Copy these integer indices once into the cached Torch view; keep
        # Torch-owned topology zero-copy. This is not a gradient bridge.
        return value.to_torch(copy=not getattr(value._buffer, "_tiga_torch_buffer", False))

    row_ptr = topology_tensor(graph._row_ptr)
    col_idx = topology_tensor(graph._col_idx)
    positions = (
        graph._positions.to_torch() if graph._positions is not None else None
    )
    source_positions = (
        graph._source_positions.to_torch()
        if graph._source_positions is not None else None
    )
    cutoff = graph._cutoff
    if hasattr(cutoff, "to_torch"):
        cutoff = cutoff.to_torch()
    periodic = (
        graph._periodic.to_torch() if graph._periodic is not None else None
    )
    fields = {
        name: value.to_torch() for name, value in graph._builder_fields.items()
    }
    device = graph.device.type.name.lower()
    if device != "cpu":
        device = f"{device}:{graph.device.ordinal}"
    result = Graph(
        schema,
        row_ptr=row_ptr,
        col_idx=col_idx,
        positions=positions,
        source_positions=source_positions,
        cutoff=cutoff,
        k=graph._k,
        periodic=periodic,
        exclude_self=graph._exclude_self,
        builder_fields=fields,
        metric=graph._metric,
        select=graph._select,
        dense_boundary=graph._dense_boundary,
        device=device,
        placement=graph.placement,
    )
    graph._torch_view_cache = result
    return result


__all__ = ["DenseCellDirectory", "Graph", "GraphSchema", "from_native"]
