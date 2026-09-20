"""Benchmark-only Triton oracles for sparse Tiga workloads.

These pre-written ``@triton.jit`` traversal skeletons establish performance
gates for compiler-generated code.  They deliberately live outside the
``tiga`` package and must never be imported by its runtime.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _multiply_reduce(left, right):
        return left * right

    @triton.jit
    def edge_weight_vjp_kernel(
        cotangent, destination, source_value, source, output,
        n_edges: tl.constexpr, BLOCK: tl.constexpr,
    ):
        edges = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = edges < n_edges
        dst = tl.load(destination + edges, mask=mask, other=0)
        src = tl.load(source + edges, mask=mask, other=0)
        dy = tl.load(cotangent + dst, mask=mask, other=0.0)
        x = tl.load(source_value + src, mask=mask, other=0.0)
        tl.store(output + edges, dy * x, mask=mask)

    @triton.jit
    def radius_distance_sum_vjp_kernel(
        positions, source_value, cotangent, destination, source,
        position_gradient, source_gradient,
        n_edges: tl.constexpr, dimensions: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Matched fixed-snapshot Euclidean radius VJP benchmark oracle."""
        edges = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = edges < n_edges
        dst = tl.load(destination + edges, mask=mask, other=0)
        src = tl.load(source + edges, mask=mask, other=0)
        dy = tl.load(cotangent + dst, mask=mask, other=0.0)
        x = tl.load(source_value + src, mask=mask, other=0.0)
        squared = tl.zeros((BLOCK,), tl.float32)
        for axis in range(dimensions):
            delta = (
                tl.load(positions + src * dimensions + axis,
                        mask=mask, other=0.0)
                - tl.load(positions + dst * dimensions + axis,
                          mask=mask, other=0.0)
            )
            squared += delta * delta
        distance = tl.sqrt(squared)
        safe_inverse = tl.where(distance > 0.0, 1.0 / distance, 0.0)
        geometry_scale = dy * x * safe_inverse
        for axis in range(dimensions):
            delta = (
                tl.load(positions + src * dimensions + axis,
                        mask=mask, other=0.0)
                - tl.load(positions + dst * dimensions + axis,
                          mask=mask, other=0.0)
            )
            contribution = geometry_scale * delta
            tl.atomic_add(
                position_gradient + src * dimensions + axis,
                contribution, mask=mask)
            tl.atomic_add(
                position_gradient + dst * dimensions + axis,
                -contribution, mask=mask)
        tl.atomic_add(source_gradient + src, dy * distance, mask=mask)

    @triton.jit
    def saved_edge_weight_vjp_kernel(
        cotangent, destination, saved_source, output,
        n_edges: tl.constexpr, BLOCK: tl.constexpr,
    ):
        """Handwritten equivalent of a backward kernel with saved x[src]."""
        edges = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = edges < n_edges
        dst = tl.load(destination + edges, mask=mask, other=0)
        dy = tl.load(cotangent + dst, mask=mask, other=0.0)
        x = tl.load(saved_source + edges, mask=mask, other=0.0)
        tl.store(output + edges, dy * x, mask=mask)

    @triton.jit
    def vector_edge_weight_vjp_kernel(
        cotangent, destination, source_value, source, output,
        n_edges: tl.constexpr, features: tl.constexpr,
        BLOCK_F: tl.constexpr,
    ):
        edge = tl.program_id(0)
        lanes = tl.arange(0, BLOCK_F)
        mask = lanes < features
        dst = tl.load(destination + edge)
        src = tl.load(source + edge)
        dy = tl.load(cotangent + dst * features + lanes, mask=mask, other=0.0)
        x = tl.load(source_value + src * features + lanes, mask=mask, other=0.0)
        tl.store(output + edge, tl.sum(dy * x, axis=0))

    @triton.jit
    def saved_vector_edge_weight_vjp_kernel(
        cotangent, destination, saved_source, output,
        n_edges: tl.constexpr, features: tl.constexpr,
        BLOCK_F: tl.constexpr,
    ):
        edge = tl.program_id(0)
        lanes = tl.arange(0, BLOCK_F)
        mask = lanes < features
        dst = tl.load(destination + edge)
        dy = tl.load(cotangent + dst * features + lanes, mask=mask, other=0.0)
        x = tl.load(saved_source + edge * features + lanes, mask=mask, other=0.0)
        tl.store(output + edge, tl.sum(dy * x, axis=0))

    @triton.jit
    def csr_product_vjp_kernel(
        message, row_ptr, destination, cotangent, output,
        n_edges: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        """Zero-safe matched oracle for a captured CSR product reducer."""
        row = tl.program_id(0)
        begin = tl.load(row_ptr + row)
        end = tl.load(row_ptr + row + 1)
        lane = tl.arange(0, BLOCK_D)
        edge = begin + lane
        active = edge < end
        values = tl.load(message + edge, mask=active, other=1.0)
        factors = tl.where(
            lane[:, None] != lane[None, :], values[None, :], 1.0)
        excluded = tl.reduce(factors, axis=1, combine_fn=_multiply_reduce)
        dy = tl.load(cotangent + row)
        tl.store(output + edge, excluded * dy, mask=active)

    @triton.jit
    def fixed_csr_spmm_kernel(
        col_idx, weight, x, out,
        n_rows: tl.constexpr,
        degree: tl.constexpr,
        n_features: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_F: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        neighbors = tl.arange(0, BLOCK_D)
        features = tl.program_id(1) * BLOCK_F + tl.arange(0, BLOCK_F)
        edge = rows[:, None] * degree + neighbors[None, :]
        edge_mask = (rows[:, None] < n_rows) & (neighbors[None, :] < degree)
        source = tl.load(col_idx + edge, mask=edge_mask, other=0)
        edge_weight = tl.load(weight + edge, mask=edge_mask, other=0.0)
        value_offset = (
            source[:, :, None] * n_features + features[None, None, :])
        value_mask = (
            edge_mask[:, :, None]
            & (features[None, None, :] < n_features))
        value = tl.load(x + value_offset, mask=value_mask, other=0.0)
        result = tl.sum(value * edge_weight[:, :, None], axis=1)
        output_offset = rows[:, None] * n_features + features[None, :]
        output_mask = (
            (rows[:, None] < n_rows)
            & (features[None, :] < n_features))
        tl.store(out + output_offset, result, mask=output_mask)

    @triton.jit
    def ragged_csr_spmm_kernel(
        row_ptr, col_idx, weight, x, out,
        n_rows: tl.constexpr,
        n_features: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_F: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rows < n_rows
        starts = tl.load(row_ptr + rows, mask=row_mask, other=0)
        ends = tl.load(row_ptr + rows + 1, mask=row_mask, other=0)
        neighbors = tl.arange(0, BLOCK_D)
        features = tl.program_id(1) * BLOCK_F + tl.arange(0, BLOCK_F)
        edge = starts[:, None] + neighbors[None, :]
        edge_mask = row_mask[:, None] & (edge < ends[:, None])
        source = tl.load(col_idx + edge, mask=edge_mask, other=0)
        edge_weight = tl.load(weight + edge, mask=edge_mask, other=0.0)
        value_offset = (
            source[:, :, None] * n_features + features[None, None, :])
        value_mask = (
            edge_mask[:, :, None]
            & (features[None, None, :] < n_features))
        value = tl.load(x + value_offset, mask=value_mask, other=0.0)
        result = tl.sum(value * edge_weight[:, :, None], axis=1)
        output_offset = rows[:, None] * n_features + features[None, :]
        output_mask = row_mask[:, None] & (features[None, :] < n_features)
        tl.store(out + output_offset, result, mask=output_mask)

    @triton.jit
    def ragged_csr_spmv_kernel(
        row_ptr, col_idx, weight, x, out,
        n_rows: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rows < n_rows
        starts = tl.load(row_ptr + rows, mask=row_mask, other=0)
        ends = tl.load(row_ptr + rows + 1, mask=row_mask, other=0)
        neighbors = tl.arange(0, BLOCK_D)
        edge = starts[:, None] + neighbors[None, :]
        edge_mask = row_mask[:, None] & (edge < ends[:, None])
        source = tl.load(col_idx + edge, mask=edge_mask, other=0)
        edge_weight = tl.load(weight + edge, mask=edge_mask, other=0.0)
        value = tl.load(x + source, mask=edge_mask, other=0.0)
        result = tl.sum(value * edge_weight, axis=1)
        tl.store(out + rows, result, mask=row_mask)

    @triton.jit
    def radius_distance_spmv_kernel(
        row_ptr, col_idx, positions, x, out,
        n_rows: tl.constexpr,
        dimensions: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Consume arbitrary-degree radius CSR without edge geometry storage."""
        row = tl.program_id(0)
        start = tl.load(row_ptr + row)
        end = tl.load(row_ptr + row + 1)
        accumulator = 0.0
        for edge_start in range(start, end, BLOCK_D):
            edges = edge_start + tl.arange(0, BLOCK_D)
            mask = edges < end
            source = tl.load(col_idx + edges, mask=mask, other=0)
            distance_squared = 0.0
            for axis in tl.static_range(0, dimensions):
                source_position = tl.load(
                    positions + source * dimensions + axis,
                    mask=mask,
                    other=0.0,
                )
                destination_position = tl.load(
                    positions + row * dimensions + axis)
                delta = source_position - destination_position
                distance_squared += delta * delta
            source_value = tl.load(x + source, mask=mask, other=0.0)
            accumulator += tl.sum(
                tl.sqrt(distance_squared) * source_value, axis=0)
        tl.store(out + row, accumulator)

    @triton.jit
    def generated_radius_distance_sum_kernel(
        cell_ptr,
        particle_order,
        cell_coordinates,
        extents,
        strides,
        neighbor_offsets,
        positions,
        x,
        out,
        cutoff_squared: tl.constexpr,
        dimensions: tl.constexpr,
        neighbor_cells: tl.constexpr,
        BLOCK_D: tl.constexpr,
        HASH_GRID: tl.constexpr = False,
    ):
        """Generate radius edges from a compact cell directory and consume."""
        row = tl.program_id(0)
        accumulator = 0.0
        for neighbor in tl.static_range(0, neighbor_cells):
            valid_cell = True
            cell_key = 0
            for axis in tl.static_range(0, dimensions):
                coordinate = tl.load(
                    cell_coordinates + row * dimensions + axis)
                offset = tl.load(
                    neighbor_offsets + neighbor * dimensions + axis)
                neighbor_coordinate = coordinate + offset
                extent = tl.load(extents + axis)
                if HASH_GRID:
                    neighbor_coordinate = neighbor_coordinate & (extent - 1)
                else:
                    valid_cell &= (
                        (neighbor_coordinate >= 0)
                        & (neighbor_coordinate < extent))
                cell_key += neighbor_coordinate * tl.load(strides + axis)
            safe_key = tl.where(valid_cell, cell_key, 0)
            start = tl.load(cell_ptr + safe_key)
            end = tl.load(cell_ptr + safe_key + 1)
            start = tl.where(valid_cell, start, 0)
            end = tl.where(valid_cell, end, 0)
            for tile_start in range(start, end, BLOCK_D):
                slots = tile_start + tl.arange(0, BLOCK_D)
                slot_mask = slots < end
                source = tl.load(
                    particle_order + slots, mask=slot_mask, other=0)
                distance_squared = 0.0
                for axis in tl.static_range(0, dimensions):
                    source_position = tl.load(
                        positions + source * dimensions + axis,
                        mask=slot_mask,
                        other=0.0,
                    )
                    destination_position = tl.load(
                        positions + row * dimensions + axis)
                    delta = source_position - destination_position
                    distance_squared += delta * delta
                edge_mask = (
                    slot_mask
                    & (source != row)
                    & (distance_squared <= cutoff_squared))
                source_value = tl.load(
                    x + source, mask=edge_mask, other=0.0)
                accumulator += tl.sum(
                    tl.sqrt(distance_squared) * source_value,
                    axis=0,
                )
        tl.store(out + row, accumulator)


@dataclass(frozen=True)
class LaunchResult:
    output: torch.Tensor
    compiled: object
    block_m: int
    block_f: int
    num_warps: int
    plan: object


@dataclass
class EdgeWeightVJPPlan:
    destination: torch.Tensor
    source: torch.Tensor
    edges: int
    block: int = 256
    output: torch.Tensor | None = None

    def run(self, cotangent: torch.Tensor, source_value: torch.Tensor):
        if self.output is None or sys.getrefcount(self.output) > 2:
            self.output = torch.empty(
                self.edges, device=source_value.device, dtype=source_value.dtype)
        edge_weight_vjp_kernel[(triton.cdiv(self.edges, self.block),)](
            cotangent, self.destination, source_value, self.source, self.output,
            n_edges=self.edges, BLOCK=self.block, num_warps=4)
        return self.output


@dataclass
class RadiusDistanceSumVJPPlan:
    destination: torch.Tensor
    source: torch.Tensor
    positions: torch.Tensor
    source_value: torch.Tensor
    dimensions: int
    block: int = 256
    position_gradient: torch.Tensor | None = None
    source_gradient: torch.Tensor | None = None

    def run(self, cotangent: torch.Tensor):
        if (
            self.position_gradient is None
            or sys.getrefcount(self.position_gradient) > 2
        ):
            self.position_gradient = torch.empty_like(self.positions)
        if (
            self.source_gradient is None
            or sys.getrefcount(self.source_gradient) > 2
        ):
            self.source_gradient = torch.empty_like(self.source_value)
        self.position_gradient.zero_()
        self.source_gradient.zero_()
        edges = self.source.numel()
        radius_distance_sum_vjp_kernel[(triton.cdiv(edges, self.block),)](
            self.positions,
            self.source_value,
            cotangent,
            self.destination,
            self.source,
            self.position_gradient,
            self.source_gradient,
            n_edges=edges,
            dimensions=self.dimensions,
            BLOCK=self.block,
            num_warps=8,
        )
        return self.position_gradient, self.source_gradient


def prepare_radius_distance_sum_vjp(
    destination, source, positions, source_value,
) -> RadiusDistanceSumVJPPlan | None:
    if (
        not available()
        or positions.device.type != "cuda"
        or positions.dtype != torch.float32
        or positions.ndim != 2
        or positions.shape[1] not in (2, 3)
        or source_value.shape != (positions.shape[0],)
        or source_value.dtype != positions.dtype
        or destination.dtype not in (torch.int32, torch.int64)
        or source.dtype != destination.dtype
        or source.shape != destination.shape
    ):
        return None
    return RadiusDistanceSumVJPPlan(
        destination, source, positions, source_value, positions.shape[1])


@dataclass
class SavedEdgeWeightVJPPlan:
    destination: torch.Tensor
    saved_source: torch.Tensor
    edges: int
    block: int = 512
    output: torch.Tensor | None = None

    def run(self, cotangent: torch.Tensor):
        if self.output is None or sys.getrefcount(self.output) > 2:
            self.output = torch.empty(
                self.edges,
                device=self.saved_source.device,
                dtype=self.saved_source.dtype,
            )
        saved_edge_weight_vjp_kernel[(triton.cdiv(self.edges, self.block),)](
            cotangent,
            self.destination,
            self.saved_source,
            self.output,
            n_edges=self.edges,
            BLOCK=self.block,
            num_warps=8,
        )
        return self.output


def prepare_edge_weight_vjp(destination, source, reference):
    if (
        not available() or reference.device.type != "cuda"
        or reference.dtype != torch.float32
        or destination.dtype not in (torch.int32, torch.int64)
        or source.dtype != destination.dtype
    ):
        return None
    return EdgeWeightVJPPlan(destination, source, source.numel())


def prepare_saved_edge_weight_vjp(destination, saved_source):
    if (
        not available() or saved_source.device.type != "cuda"
        or saved_source.dtype != torch.float32
        or destination.dtype not in (torch.int32, torch.int64)
        or destination.numel() != saved_source.numel()
    ):
        return None
    return SavedEdgeWeightVJPPlan(
        destination, saved_source, destination.numel())


@dataclass
class VectorEdgeWeightVJPPlan:
    destination: torch.Tensor
    source: torch.Tensor | None
    source_value: torch.Tensor
    edges: int
    features: int
    saved: bool
    output: torch.Tensor | None = None

    def run(self, cotangent: torch.Tensor):
        if self.output is None or sys.getrefcount(self.output) > 2:
            self.output = torch.empty(
                (self.edges, 1), device=self.source_value.device,
                dtype=self.source_value.dtype)
        block = triton.next_power_of_2(self.features)
        warps = 4 if block >= 16 else 1
        if self.saved:
            saved_vector_edge_weight_vjp_kernel[(self.edges,)](
                cotangent, self.destination, self.source_value, self.output,
                n_edges=self.edges, features=self.features, BLOCK_F=block,
                num_warps=warps)
        else:
            vector_edge_weight_vjp_kernel[(self.edges,)](
                cotangent, self.destination, self.source_value, self.source,
                self.output, n_edges=self.edges, features=self.features,
                BLOCK_F=block, num_warps=warps)
        return self.output


def prepare_vector_edge_weight_vjp(
    destination, source, source_value, features: int, *, saved: bool = False
):
    if (
        not available() or source_value.device.type != "cuda"
        or source_value.dtype != torch.float32 or features <= 1
        or destination.dtype not in (torch.int32, torch.int64)
        or (not saved and (
            source is None or source.dtype != destination.dtype
            or source.numel() != destination.numel()))
    ):
        return None
    if saved and source_value.shape != (destination.numel(), features):
        return None
    return VectorEdgeWeightVJPPlan(
        destination, source, source_value, destination.numel(), features, saved)


@dataclass
class CSRProductVJPPlan:
    row_ptr: torch.Tensor
    destination: torch.Tensor
    edges: int
    rows: int
    max_degree: int
    output: torch.Tensor | None = None
    compiled: object | None = None

    def run(self, message: torch.Tensor, cotangent: torch.Tensor):
        if self.output is None or sys.getrefcount(self.output) > 2:
            self.output = torch.empty_like(message)
        self.compiled = csr_product_vjp_kernel[(self.rows,)](
            message, self.row_ptr, self.destination, cotangent, self.output,
            n_edges=self.edges,
            BLOCK_D=triton.next_power_of_2(self.max_degree),
            num_warps=1 if self.max_degree <= 16 else 4,
        )
        return self.output


def prepare_csr_product_vjp(
    row_ptr, destination, message, *, max_degree: int,
) -> CSRProductVJPPlan | None:
    if (
        not available() or message.device.type != "cuda"
        or message.dtype != torch.float32 or message.ndim != 1
        or row_ptr.dtype not in (torch.int32, torch.int64)
        or destination.dtype != row_ptr.dtype
        or destination.numel() != message.numel()
        or max_degree <= 0 or max_degree > 64
    ):
        return None
    return CSRProductVJPPlan(
        row_ptr, destination, message.numel(), row_ptr.numel() - 1, max_degree)


@dataclass
class FixedWeightedSumPlan:
    col_idx: torch.Tensor
    num_rows: int
    degree: int
    features: int
    vector_input: bool
    block_m: int
    block_f: int
    num_warps: int
    block_d: int
    grid: tuple[int, int]
    compiled: object | None = None
    output: torch.Tensor | None = None

    def _acquire_output(self, x: torch.Tensor) -> torch.Tensor:
        # The plan owns one reference. getrefcount adds one temporary reference;
        # any larger count means a caller can still observe the old value.
        if self.output is None or sys.getrefcount(self.output) > 2:
            self.output = torch.empty(
                (self.num_rows, self.features), device=x.device, dtype=x.dtype)
        return self.output

    def run(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        x_2d = x[:, None] if self.vector_input else x
        output = self._acquire_output(x)
        self.compiled = fixed_csr_spmm_kernel[self.grid](
            self.col_idx, weight, x_2d, output,
            n_rows=self.num_rows,
            degree=self.degree,
            n_features=self.features,
            BLOCK_M=self.block_m,
            BLOCK_D=self.block_d,
            BLOCK_F=self.block_f,
            num_warps=self.num_warps,
        )
        return output[:, 0] if self.vector_input else output


@dataclass
class RaggedWeightedSumPlan:
    row_ptr: torch.Tensor
    col_idx: torch.Tensor
    num_rows: int
    features: int
    vector_input: bool
    block_m: int
    block_f: int
    num_warps: int
    block_d: int
    grid: tuple[int, ...]
    compiled: object | None = None
    output: torch.Tensor | None = None

    def _acquire_output(self, x: torch.Tensor) -> torch.Tensor:
        if self.output is None or sys.getrefcount(self.output) > 2:
            self.output = torch.empty(
                (self.num_rows, self.features), device=x.device, dtype=x.dtype)
        return self.output

    def run(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        x_2d = x[:, None] if self.vector_input else x
        output = self._acquire_output(x)
        if self.features == 1:
            self.compiled = ragged_csr_spmv_kernel[self.grid](
                self.row_ptr, self.col_idx, weight, x_2d, output,
                n_rows=self.num_rows,
                BLOCK_M=self.block_m,
                BLOCK_D=self.block_d,
                num_warps=self.num_warps,
            )
        else:
            self.compiled = ragged_csr_spmm_kernel[self.grid](
                self.row_ptr, self.col_idx, weight, x_2d, output,
                n_rows=self.num_rows,
                n_features=self.features,
                BLOCK_M=self.block_m,
                BLOCK_D=self.block_d,
                BLOCK_F=self.block_f,
                num_warps=self.num_warps,
            )
        return output[:, 0] if self.vector_input else output


@dataclass
class RadiusDistancePlan:
    row_ptr: torch.Tensor
    col_idx: torch.Tensor
    positions: torch.Tensor
    num_rows: int
    dimensions: int
    block_d: int = 32
    num_warps: int = 1
    compiled: object | None = None
    output: torch.Tensor | None = None

    def run(self, x: torch.Tensor, unused_weight: torch.Tensor | None = None) -> torch.Tensor:
        del unused_weight
        self.output = torch.empty(self.num_rows, device=x.device, dtype=x.dtype)
        self.compiled = radius_distance_spmv_kernel[(self.num_rows,)](
            self.row_ptr,
            self.col_idx,
            self.positions,
            x,
            self.output,
            n_rows=self.num_rows,
            dimensions=self.dimensions,
            BLOCK_D=self.block_d,
            num_warps=self.num_warps,
        )
        return self.output


@dataclass
class GeneratedRadiusDistancePlan:
    directory: object
    positions: torch.Tensor
    num_rows: int
    dimensions: int
    block_d: int = 32
    num_warps: int = 1
    compiled: object | None = None
    output: torch.Tensor | None = None

    def _acquire_output(self, x: torch.Tensor) -> torch.Tensor:
        if self.output is None or sys.getrefcount(self.output) > 2:
            self.output = torch.empty_like(x)
        return self.output

    def run(self, x: torch.Tensor, unused_weight=None) -> torch.Tensor:
        del unused_weight
        output = self._acquire_output(x)
        directory = self.directory
        self.compiled = generated_radius_distance_sum_kernel[(self.num_rows,)](
            directory.cell_ptr,
            directory.particle_order,
            directory.cell_coordinates,
            directory.extents,
            directory.strides,
            directory.neighbor_offsets,
            self.positions,
            x,
            output,
            cutoff_squared=directory.cutoff * directory.cutoff,
            dimensions=self.dimensions,
            neighbor_cells=directory.neighbor_offsets.shape[0],
            HASH_GRID=directory.hash_grid,
            BLOCK_D=self.block_d,
            num_warps=self.num_warps,
        )
        return output


def available() -> bool:
    return triton is not None and torch.cuda.is_available()


def _config(features: int):
    if features == 1:
        return 16, 1, 1
    if features <= 16:
        return 8, triton.next_power_of_2(features), 1
    return 8, min(64, triton.next_power_of_2(features)), 4


def _ragged_config(features: int, max_degree: int):
    # Scalar ragged rows benefit from extra warps hiding random-gather latency;
    # wide features plus a 64-neighbor mask need a narrow row tile to avoid
    # excessive register pressure and masked work on low-degree rows.
    if features == 1:
        return (32, 1, 1) if max_degree > 32 else (16, 1, 4)
    if features >= 64 and max_degree > 32:
        return 1, 64, 1
    return _config(features)


def prepare_fixed_weighted_sum(
    col_idx: torch.Tensor,
    weight: torch.Tensor,
    x: torch.Tensor,
    *,
    num_rows: int,
    degree: int,
) -> FixedWeightedSumPlan | None:
    if not available() or x.device.type != "cuda" or x.dtype != torch.float32:
        return None
    if weight.dtype != torch.float32 or weight.device != x.device:
        return None
    if x.ndim not in (1, 2) or weight.ndim not in (1, 2):
        return None
    if weight.ndim == 2 and weight.shape[1:] != (1,):
        return None
    x_2d = x[:, None] if x.ndim == 1 else x
    features = x_2d.shape[1]
    if features <= 0 or features > 128:
        return None
    block_m, block_f, num_warps = _config(features)
    return FixedWeightedSumPlan(
        col_idx=col_idx,
        num_rows=num_rows,
        degree=degree,
        features=features,
        vector_input=x.ndim == 1,
        block_m=block_m,
        block_f=block_f,
        num_warps=num_warps,
        block_d=triton.next_power_of_2(degree),
        grid=(
            triton.cdiv(num_rows, block_m),
            triton.cdiv(features, block_f),
        ),
    )


def launch_radius_distance_sum(
    row_ptr: torch.Tensor,
    col_idx: torch.Tensor,
    positions: torch.Tensor,
    x: torch.Tensor,
    *,
    num_rows: int,
) -> LaunchResult | None:
    if not available() or x.device.type != "cuda" or x.dtype != torch.float32:
        return None
    if x.ndim != 1 or positions.dtype != x.dtype or positions.device != x.device:
        return None
    if positions.ndim != 2 or positions.shape[1] not in (1, 2, 3):
        return None
    plan = RadiusDistancePlan(
        row_ptr=row_ptr,
        col_idx=col_idx,
        positions=positions,
        num_rows=num_rows,
        dimensions=positions.shape[1],
    )
    output = plan.run(x)
    return LaunchResult(
        output, plan.compiled, 1, 1, plan.num_warps, plan)


def launch_generated_radius_distance_sum(
    directory,
    positions: torch.Tensor,
    x: torch.Tensor,
) -> LaunchResult | None:
    if not available() or positions.device.type != "cuda":
        return None
    if positions.dtype != torch.float32 or x.dtype != torch.float32:
        return None
    if x.ndim != 1 or positions.ndim != 2:
        return None
    if positions.shape[0] != x.shape[0] or positions.shape[1] not in (1, 2, 3):
        return None
    plan = GeneratedRadiusDistancePlan(
        directory=directory,
        positions=positions,
        num_rows=positions.shape[0],
        dimensions=positions.shape[1],
    )
    output = plan.run(x)
    return LaunchResult(
        output, plan.compiled, 1, 1, plan.num_warps, plan)


def prepare_ragged_weighted_sum(
    row_ptr: torch.Tensor,
    col_idx: torch.Tensor,
    weight: torch.Tensor,
    x: torch.Tensor,
    *,
    num_rows: int,
    max_degree: int,
) -> RaggedWeightedSumPlan | None:
    if not available() or x.device.type != "cuda" or x.dtype != torch.float32:
        return None
    if weight.dtype != torch.float32 or weight.device != x.device:
        return None
    if row_ptr.device != x.device or col_idx.device != x.device:
        return None
    if x.ndim not in (1, 2) or weight.ndim not in (1, 2):
        return None
    if weight.ndim == 2 and weight.shape[1:] != (1,):
        return None
    x_2d = x[:, None] if x.ndim == 1 else x
    features = x_2d.shape[1]
    if features <= 0 or features > 128 or max_degree <= 0 or max_degree > 64:
        return None
    block_m, block_f, num_warps = _ragged_config(features, max_degree)
    return RaggedWeightedSumPlan(
        row_ptr=row_ptr,
        col_idx=col_idx,
        num_rows=num_rows,
        features=features,
        vector_input=x.ndim == 1,
        block_m=block_m,
        block_f=block_f,
        num_warps=num_warps,
        block_d=triton.next_power_of_2(max_degree),
        grid=((triton.cdiv(num_rows, block_m),) if features == 1 else (
            triton.cdiv(num_rows, block_m),
            triton.cdiv(features, block_f),
        )),
    )
