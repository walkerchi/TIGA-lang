from __future__ import annotations

import hashlib
import inspect
import os
import weakref
from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable
import statistics

import torch

from ...compiler.capture import (
    capture_edge,
    recognize_weighted_sum,
)
from .graph import Graph
from ...kernel import CompiledVariant, Kernel
from ...reducer import OnlineSoftmaxReducer, SumReducer
from ...runtime import (
    ArgumentBinding,
    ExecutableBundle,
    KernelInvocation,
    ResourceAccess,
)


_EXECUTABLE_MISS = object()


class _GeneratedRadiusDistanceSum(torch.autograd.Function):
    """Autograd bridge for the compiler-generated matrix-free radius consumer.

    The relation membership is a discrete, fixed forward snapshot.  Backward
    reconstructs only that snapshot's differentiable edge geometry; users do
    not write a backward kernel and the forward still never materializes CSR.
    """

    @staticmethod
    def forward(ctx, positions, source, graph, directory, runner):
        ctx.graph = graph
        ctx.snapshot_token = _dynamic_snapshot_token(graph)
        ctx.save_for_backward(positions, source)
        return runner(directory, positions, source)

    @staticmethod
    def backward(ctx, grad_output):
        positions, source = ctx.saved_tensors
        need_positions, need_source = ctx.needs_input_grad[:2]
        if not need_positions and not need_source:
            return None, None, None, None, None

        graph = ctx.graph
        if _dynamic_snapshot_token(graph) != ctx.snapshot_token:
            raise RuntimeError(
                "dynamic radius topology inputs changed between forward and backward")

        create_graph = torch.is_grad_enabled()
        if (
            not create_graph
            and positions.device.type == "cuda"
            and positions.dtype == torch.float32
            and positions.ndim == 2
            and positions.shape[1] in (2, 3)
            and source.dtype == torch.float32
        ):
            try:
                grad_positions, grad_source = \
                    _compiled_fixed_snapshot_radius_vjp(
                        graph, positions, source, grad_output)
                return (
                    grad_positions if need_positions else None,
                    grad_source if need_source else None,
                    None, None, None,
                )
            except (NotImplementedError, RuntimeError, ValueError) as error:
                # Optional provider/toolchain failures retain the semantic VJP
                # below. Correctness never depends on a particular GPU plugin.
                graph._graphforge_radius_vjp_provider_error = str(error)
        # Topology is intentionally non-differentiable. Materialize only for
        # the semantic fallback; the prepared provider path above owns its
        # already-bound fixed snapshot and must remain a true hot launch.
        with torch.no_grad():
            row_ptr, col_idx = graph.resolve_csr()
            destination = graph.destination_index(row_ptr)
        with torch.enable_grad():
            displacement = positions[col_idx] - positions[destination]
            periodic = graph._periodic
            if periodic is not None:
                lattice = (
                    torch.diag(periodic) if periodic.ndim == 1 else periodic
                )
                fractional = displacement @ torch.linalg.inv(lattice)
                displacement = (fractional - torch.round(fractional)) @ lattice
            distance = torch.linalg.vector_norm(displacement, dim=-1)
            message = distance * source[col_idx]
            output = torch.zeros(
                graph.schema.num_dst,
                dtype=message.dtype,
                device=message.device,
            ).index_add(0, destination, message)
            inputs = []
            if need_positions:
                inputs.append(positions)
            if need_source:
                inputs.append(source)
            gradients = torch.autograd.grad(
                output,
                inputs,
                grad_output,
                create_graph=create_graph,
                allow_unused=True,
            )

        cursor = iter(gradients)
        grad_positions = next(cursor) if need_positions else None
        grad_source = next(cursor) if need_source else None
        return grad_positions, grad_source, None, None, None


def _compiled_fixed_snapshot_radius_vjp(
    graph, positions, source, grad_output
):
    """Launch the first-class packed Euclidean CSR VJP Tensor operation."""
    if not positions.is_contiguous() or not source.is_contiguous():
        raise NotImplementedError("compiled radius VJP requires contiguous primals")
    upstream = grad_output.contiguous()
    fast_key = (
        _dynamic_snapshot_token(graph),
        positions.data_ptr(), positions._version,
        source.data_ptr(), source._version,
        upstream.data_ptr(), upstream._version,
    )
    cached = getattr(graph, "_graphforge_compiled_radius_vjp", None)
    if cached is not None and cached[0] == fast_key:
        return cached[1].run()
    with torch.no_grad():
        row_ptr, col_idx = graph.resolve_csr()
        destination = graph.destination_index(row_ptr)
        directory = graph.generated_cell_directory()
    if directory is None or col_idx.dtype != torch.int64:
        raise NotImplementedError("compiled radius VJP requires an i64 cell snapshot")
    if cached is None or cached[0] != fast_key:
        cached = (
            fast_key,
            _PreparedRadiusVJP(
                positions, source, destination, col_idx, upstream, directory),
        )
        graph._graphforge_compiled_radius_vjp = cached
    return cached[1].run()


class _PreparedRadiusVJP:
    """Prepared packed VJP with non-aliasing reusable output slots."""

    def __init__(
        self, positions, source, destination, col_idx, upstream, directory
    ):
        from ...tensor import from_torch
        from ...compiler.gpu_tensor import compile_tensor

        native_positions = from_torch(positions.detach())
        native_source = from_torch(source.detach())
        native_destination = from_torch(destination)
        native_col_idx = from_torch(col_idx)
        native_upstream = from_torch(upstream)
        native_lattice = from_torch(directory.lattice)
        native_inverse = from_torch(directory.inverse_lattice)
        packed = native_positions._csr_euclidean_distance_sum_vjp(
            native_source,
            native_destination,
            native_col_idx,
            native_upstream,
            native_lattice,
            native_inverse,
            periodic=directory.periodic,
        )
        self.executable = compile_tensor(packed)
        self.shape = (positions.shape[0], positions.shape[1] + 1)
        self.dimensions = positions.shape[1]
        self.dtype = positions.dtype
        self.device = positions.device
        self.slots = []

    def _new_slot(self):
        from ...tensor import from_torch

        output = torch.empty(
            self.shape, dtype=self.dtype, device=self.device)
        launch = self.executable.prepare(from_torch(output))
        slot = [output, launch, ()]
        self.slots.append(slot)
        return slot

    def run(self):
        slot = next(
            (
                candidate for candidate in self.slots
                if all(reference() is None for reference in candidate[2])
            ),
            None,
        )
        if slot is None:
            slot = self._new_slot()
        output, launch, _references = slot
        launch()
        position_gradient = output[:, :self.dimensions]
        source_gradient = output[:, self.dimensions]
        slot[2] = (
            weakref.ref(position_gradient), weakref.ref(source_gradient))
        return position_gradient, source_gradient


def _run_generated_radius_distance_sum(
    graph, runner, directory, positions, source
):
    if positions.requires_grad or source.requires_grad:
        return _GeneratedRadiusDistanceSum.apply(
            positions, source, graph, directory, runner)
    return runner(directory, positions, source)


def _cuda_median_ms(function, *, warmup: int = 4, samples: int = 15) -> float:
    """Small synchronized JIT autotune measurement on the current stream."""
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    elapsed = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        elapsed.append(start.elapsed_time(end))
    return statistics.median(elapsed)


class _StaticTaskBundleRunner:
    """Own compiler task resources and rebind fields on repeated JIT calls."""

    def __init__(
        self, plan, compiled, graph, row_ptr, col_idx, reference,
        src_resource, edge_resource,
    ):
        from .provider import (
            TorchCudaSubmissionProvider,
            reusable_vector_output,
        )

        self._provider = TorchCudaSubmissionProvider()
        resources = {
            "row_ptr": row_ptr,
            "col_idx": col_idx,
            src_resource: reference,
            # Both fields are rebound before every execute submission.  The
            # initial values exist solely to make the prepared ABI complete.
            edge_resource: reference,
            "output": torch.empty(
                graph.schema.num_dst, device=reference.device,
                dtype=reference.dtype),
        }
        for requirement in plan.resources:
            if requirement.external:
                continue
            if requirement.layout == "degree-row-worklist":
                value = torch.empty(
                    requirement.capacity_bytes // 8,
                    device=reference.device,
                    dtype=torch.int64,
                )
            elif requirement.layout == "row-partial-f32":
                value = torch.empty(
                    requirement.capacity_bytes // 4,
                    device=reference.device,
                    dtype=torch.float32,
                )
            else:
                raise ValueError(
                    "unsupported compiler task resource layout "
                    f"{requirement.layout!r}"
                )
            resources[requirement.name] = value
        resolver = lambda invocation: compiled[invocation.name]
        plan.bind_phase("materialize", resolver).submit(
            self._provider, resources).wait()
        self._execute = plan.bind_phase("execute", resolver).prepare(
            self._provider)
        self._required = self._execute.bundle.required_bindings
        slots = {name: index for index, name in enumerate(self._required)}
        self._src_slot = slots[src_resource]
        self._edge_slot = slots[edge_resource]
        self._output_slot = slots["output"]
        # A fixed slot array is the prepared runtime ABI.  Repeated launches
        # mutate only the two field slots and (when needed) the output slot;
        # graph/worklist buffers remain bound without mapping or tuple rebuilds.
        self._bound = [resources[name] for name in self._required]
        self._rows = graph.schema.num_dst
        self._reusable_output = reusable_vector_output
        bundle_invocations = self._execute.bundle.invocations
        prepared_invocations = self._execute.invocations
        dependency_free = tuple(
            prepared
            for invocation, prepared in zip(
                bundle_invocations, prepared_invocations)
            if invocation.task_kind != "join"
        )
        kernel_names = {
            invocation.name for invocation in bundle_invocations
            if invocation.task_kind != "join"
        }
        self._same_stream_invocations = dependency_free if (
            bool(dependency_free)
            and all(
                not invocation.depends_on
                for invocation in bundle_invocations
                if invocation.task_kind != "join"
            )
            and all(
                invocation.task_kind == "join"
                or hasattr(prepared, "launch_same_stream")
                for invocation, prepared in zip(
                    bundle_invocations, prepared_invocations)
            )
            and all(
                invocation.task_kind != "join"
                or set(invocation.depends_on) <= kernel_names
                for invocation in bundle_invocations
            )
        ) else ()
    def run(self, x, weight):
        self._bound[self._src_slot] = x
        self._bound[self._edge_slot] = weight
        output = self._reusable_output(
            self._bound[self._output_slot], x, rows=self._rows)
        self._bound[self._output_slot] = output
        if self._same_stream_invocations:
            for invocation in self._same_stream_invocations:
                invocation.launch_same_stream(self._bound)
        else:
            self._execute.submit_resources(self._bound)
        return output


class _GenericStaticTaskBundleRunner:
    """Prepared task bundle whose field ABI comes directly from Task IR."""

    def __init__(
        self, plan, compiled, graph, row_ptr, col_idx,
        resource_names, initial_inputs,
    ):
        from .provider import TorchCudaSubmissionProvider, reusable_vector_output

        if not initial_inputs:
            raise ValueError("a generic task bundle requires at least one field")
        reference = initial_inputs[0]
        self._provider = TorchCudaSubmissionProvider()
        resources = {
            "row_ptr": row_ptr,
            "col_idx": col_idx,
            "output": torch.empty(
                graph.schema.num_dst,
                device=reference.device,
                dtype=reference.dtype,
            ),
            **dict(zip(resource_names, initial_inputs)),
        }
        for requirement in plan.resources:
            if requirement.external:
                continue
            if requirement.layout == "degree-row-worklist":
                value = torch.empty(
                    requirement.capacity_bytes // 8,
                    device=reference.device,
                    dtype=torch.int64,
                )
            elif requirement.layout == "row-partial-f32":
                value = torch.empty(
                    requirement.capacity_bytes // 4,
                    device=reference.device,
                    dtype=torch.float32,
                )
            else:
                raise ValueError(
                    "unsupported compiler task resource layout "
                    f"{requirement.layout!r}"
                )
            resources[requirement.name] = value
        resolver = lambda invocation: compiled[invocation.name]
        plan.bind_phase("materialize", resolver).submit(
            self._provider, resources).wait()
        self._execute = plan.bind_phase("execute", resolver).prepare(
            self._provider)
        self._required = self._execute.bundle.required_bindings
        slots = {name: index for index, name in enumerate(self._required)}
        self._input_slots = tuple(slots[name] for name in resource_names)
        self._output_slot = slots["output"]
        self._bound = [resources[name] for name in self._required]
        self._rows = graph.schema.num_dst
        self._reusable_output = reusable_vector_output
        bundle_invocations = self._execute.bundle.invocations
        prepared_invocations = self._execute.invocations
        dependency_free = tuple(
            prepared
            for invocation, prepared in zip(
                bundle_invocations, prepared_invocations)
            if invocation.task_kind != "join"
        )
        kernel_names = {
            invocation.name for invocation in bundle_invocations
            if invocation.task_kind != "join"
        }
        self._same_stream_invocations = dependency_free if (
            bool(dependency_free)
            and all(
                not invocation.depends_on
                for invocation in bundle_invocations
                if invocation.task_kind != "join"
            )
            and all(
                invocation.task_kind == "join"
                or hasattr(prepared, "launch_same_stream")
                for invocation, prepared in zip(
                    bundle_invocations, prepared_invocations)
            )
            and all(
                invocation.task_kind != "join"
                or set(invocation.depends_on) <= kernel_names
                for invocation in bundle_invocations
            )
        ) else ()

    def run(self, *inputs):
        for slot, value in zip(self._input_slots, inputs):
            self._bound[slot] = value
        output = self._reusable_output(
            self._bound[self._output_slot], inputs[0], rows=self._rows)
        self._bound[self._output_slot] = output
        if self._same_stream_invocations:
            for invocation in self._same_stream_invocations:
                invocation.launch_same_stream(self._bound)
        else:
            self._execute.submit_resources(self._bound)
        return output


def _prepare_static_task_candidate(
    stages, graph, row_ptr, col_idx, reference, src_field, edge_field,
):
    """Bind a compiler-emitted chunked-tail plan, or report no legal plan."""
    if stages is None or stages.task is None:
        return None
    from ...codegen import prepare_ttir_task_primitive
    from ...compiler import translate_task_bundle

    plan = translate_task_bundle(stages.task)
    if not any(
        invocation.metadata.get("row_mapping") == "worklist-chunked"
        for invocation in plan.invocations
    ):
        return None
    compiled = {
        invocation.name: prepare_ttir_task_primitive(invocation)
        for invocation in plan.invocations
        if invocation.task_kind not in {"release", "join"}
    }
    return (
        _StaticTaskBundleRunner(
            plan, compiled, graph, row_ptr, col_idx, reference,
            f"src:{src_field}", f"edge:{edge_field}"),
        compiled,
    )


def _prepare_generic_task_candidate(
    stages, graph, row_ptr, col_idx, bindings, inputs,
):
    """Bind a degree-bucket task plan for an arbitrary scalar UDF ABI."""
    if stages is None or stages.task is None or any(
        role == "param" for role, _ in bindings
    ):
        return None
    from ...codegen import prepare_ttir_task_primitive
    from ...compiler import translate_task_bundle

    plan = translate_task_bundle(stages.task)
    if not any(
        invocation.task_kind == "degree-bucket"
        for invocation in plan.invocations
    ):
        return None
    compiled = {
        invocation.name: prepare_ttir_task_primitive(invocation)
        for invocation in plan.invocations
        if invocation.task_kind not in {"release", "join"}
    }
    resource_names = tuple(f"{role}:{name}" for role, name in bindings)
    return (
        _GenericStaticTaskBundleRunner(
            plan, compiled, graph, row_ptr, col_idx,
            resource_names, inputs,
        ),
        compiled,
    )


def _tensor_schema(fields: Mapping[str, torch.Tensor]) -> tuple[object, ...]:
    return tuple(
        (name, str(value.dtype), tuple(value.shape[1:]), str(value.device))
        for name, value in sorted(fields.items())
    )


def _field_spec(value: torch.Tensor) -> tuple[object, ...]:
    return value.dtype, tuple(value.shape), value.device, tuple(value.stride())


def _matches_field(value: object, spec: tuple[object, ...]) -> bool:
    return isinstance(value, torch.Tensor) and (
        value.dtype == spec[0]
        and value.shape == spec[1]
        and value.device == spec[2]
        and value.stride() == spec[3]
    )


def _runtime_param_schema(params: Mapping[str, Any]) -> tuple[object, ...]:
    return tuple(
        sorted(
            (name, value.dtype, tuple(value.shape), value.device)
            if isinstance(value, torch.Tensor)
            else (name, type(value).__qualname__)
            for name, value in params.items()
        )
    )


@dataclass
class _GuardedExecutable:
    graph: Graph
    src_field: str
    edge_field: str
    src_spec: tuple[object, ...]
    edge_spec: tuple[object, ...]
    require_dst_alias: bool
    variant: CompiledVariant
    runner: Callable[[torch.Tensor, torch.Tensor], object]

    def try_run(self, graph, src, dst, edge, params=None):
        del params
        # Static Graph instances are immutable snapshots. Identity therefore
        # proves topology, degree analysis and captured physical buffers still
        # match without synchronizing or recomputing a planning key.
        if graph is not self.graph:
            return _EXECUTABLE_MISS
        try:
            source = src[self.src_field]
            weight = edge[self.edge_field]
        except (KeyError, TypeError):
            return _EXECUTABLE_MISS
        if not (
            _matches_field(source, self.src_spec)
            and _matches_field(weight, self.edge_spec)
        ):
            return _EXECUTABLE_MISS
        if self.require_dst_alias:
            try:
                if dst[self.src_field] is not source:
                    return _EXECUTABLE_MISS
            except (KeyError, TypeError):
                return _EXECUTABLE_MISS
        return self.runner(source, weight)


@dataclass
class _GeneratedRadiusExecutable:
    """Identity-guarded generated relation with an adaptive frozen snapshot.

    A fresh dynamic Graph stays matrix-free. Repeated execution of the same
    immutable position snapshot may materialize CSR+distance when it fits the
    configured cache budget, because the amortized library path is then a
    legal implementation of the exact same relation.
    """

    graph: Graph
    source_field: str
    source_spec: tuple[object, ...]
    position: torch.Tensor
    position_version: int
    snapshot_token: tuple[object, ...]
    variant: CompiledVariant
    generated_runner: Callable[[object, torch.Tensor, torch.Tensor], object]
    record_variant: Callable[[CompiledVariant], None]
    reuse_threshold: int = 2
    calls: int = 0
    materialized_runner: Callable[[torch.Tensor], object] | None = None

    def try_run(self, graph, src, dst, edge, params=None):
        del dst
        if graph is not self.graph or edge or params:
            return _EXECUTABLE_MISS
        try:
            source = src[self.source_field]
        except (KeyError, TypeError):
            return _EXECUTABLE_MISS
        positions = graph.euclidean_positions()
        if (
            not _matches_field(source, self.source_spec)
            or positions is not self.position
            or positions._version != self.position_version
            or _dynamic_snapshot_token(graph) != self.snapshot_token
        ):
            return _EXECUTABLE_MISS
        self.calls += 1
        if (
            self.materialized_runner is None
            and self.calls >= self.reuse_threshold
        ):
            self._materialize_if_profitable(positions)
        if self.materialized_runner is not None:
            return self.materialized_runner(source)
        directory = graph.generated_cell_directory()
        if directory is None:
            return _EXECUTABLE_MISS
        return _run_generated_radius_distance_sum(
            graph, self.generated_runner, directory, positions, source)

    def _materialize_if_profitable(self, positions: torch.Tensor) -> None:
        # Caching differentiable distance values would retain and then reuse a
        # stale autograd tape.  Keep geometry-gradient executions matrix-free;
        # their generated forward is already the preferred implementation.
        if positions.requires_grad:
            return
        budget = int(os.environ.get(
            "GRAPHFORGE_RADIUS_CACHE_BYTES", str(512 << 20)))
        # Refuse the potentially expensive build before performing it. The
        # 64-neighbor estimate is conservative for the current default
        # generated-radius contract and accounts for col_idx+distance.
        estimated = self.graph.schema.num_dst * 64 * 12
        if budget <= 0 or estimated > budget:
            return
        row_ptr, col_idx = self.graph.resolve_csr()
        distance = self.graph.implicit_edge_fields(
            row_ptr, col_idx)["distance"]
        physical_bytes = (
            row_ptr.numel() * row_ptr.element_size()
            + col_idx.numel() * col_idx.element_size()
            + distance.numel() * distance.element_size()
        )
        if physical_bytes > budget:
            return
        sparse = self.graph.sparse_csr(
            distance, row_ptr=row_ptr, col_idx=col_idx)

        def run(source):
            return torch.mm(sparse, source[:, None])[:, 0]

        self.materialized_runner = run
        self.variant = CompiledVariant(
            key=(*self.variant.key, "reuse-materialized"),
            backend=self.variant.backend,
            provider="torch.sparse.mm",
            lowering="adaptive-materialize-radius-snapshot",
            provider_key=self.variant.provider_key,
            domain_plan=self.variant.domain_plan,
            passes=(*self.variant.passes,
                    "prove-position-snapshot-unchanged",
                    "plan-radius-reuse-materialization",
                    "dispatch-native-sparse-mm"),
            artifacts=dict(self.variant.artifacts),
            findings=self.variant.findings,
            schedules=self.variant.schedules,
            remarks=(*self.variant.remarks,
                     f"reused position snapshot materialized {physical_bytes} bytes",
                     f"materialization cache budget={budget} bytes",
                     "in-place position mutation invalidates this executable"),
        )
        self.record_variant(self.variant)


def _dynamic_snapshot_token(graph: Graph) -> tuple[object, ...]:
    positions = graph._positions
    source_positions = graph._source_positions
    cutoff = graph._cutoff
    periodic = graph._periodic
    return (
        None if positions is None else (positions.data_ptr(), positions._version),
        None if source_positions is None else (
            source_positions.data_ptr(), source_positions._version
        ),
        (cutoff.data_ptr(), cutoff._version)
        if isinstance(cutoff, torch.Tensor) else cutoff,
        None if periodic is None else (periodic.data_ptr(), periodic._version),
        tuple(
            (name, value.data_ptr(), value._version)
            for name, value in sorted(graph._builder_fields.items())
        ),
    )


@dataclass
class _DynamicSnapshotExecutable:
    """Fast path bound to one immutable procedural-relation snapshot."""

    graph: Graph
    source_field: str
    source_spec: tuple[object, ...]
    snapshot_token: tuple[object, ...]
    variant: CompiledVariant
    runner: Callable[[torch.Tensor], object]
    compatible_runner: Callable[[Graph, torch.Tensor], object]

    def try_run(self, graph, src, dst, edge, params=None):
        del dst
        if edge or params:
            return _EXECUTABLE_MISS
        try:
            source = src[self.source_field]
        except (KeyError, TypeError):
            return _EXECUTABLE_MISS
        if not _matches_field(source, self.source_spec):
            return _EXECUTABLE_MISS
        current_snapshot = _dynamic_snapshot_token(graph)
        if graph is self.graph and current_snapshot == self.snapshot_token:
            return self.runner(source)
        if (
            graph.schema.specialization_key()
            != self.graph.schema.specialization_key()
            or graph.device != self.graph.device
            or graph._metric is not None
            or graph._select is not None
            or (
                None if graph._periodic is None else tuple(graph._periodic.shape)
            ) != (
                None if self.graph._periodic is None
                else tuple(self.graph._periodic.shape)
            )
        ):
            return _EXECUTABLE_MISS
        # A newly constructed logical Graph may still name the exact same
        # immutable physical relation.  Pointer + version guards cover every
        # builder input, while the specialization checks above cover schema,
        # device and periodic representation.  Reusing the already compiled
        # runner here avoids rebuilding/wrapping CSR state in loops that create
        # short-lived Graph objects.  Any in-place input mutation changes the
        # version counter and deliberately falls through to compatible_runner.
        if current_snapshot == self.snapshot_token:
            return self.runner(source)
        return self.compatible_runner(graph, source)


@dataclass
class _StructuredExecutable:
    graph: Graph
    bindings: tuple[tuple[str, str], ...]
    specs: tuple[tuple[object, ...], ...]
    parameter_name: str
    variant: CompiledVariant
    runner: Callable[..., object]

    def try_run(self, graph, src, dst, edge, params=None):
        if graph is not self.graph:
            return _EXECUTABLE_MISS
        namespaces = {"src": src, "dst": dst, "edge": edge}
        try:
            tensors = tuple(
                namespaces[role][name] for role, name in self.bindings)
            scale = (params or {})[self.parameter_name]
        except (KeyError, TypeError):
            return _EXECUTABLE_MISS
        if not all(
            _matches_field(value, spec)
            for value, spec in zip(tensors, self.specs)
        ) or not isinstance(scale, (int, float)):
            return _EXECUTABLE_MISS
        return self.runner(*tensors, float(scale))


@dataclass
class _DenseScalarExecutable:
    graph: Graph
    bindings: tuple[tuple[str, str], ...]
    specs: tuple[tuple[object, ...] | None, ...]
    variant: CompiledVariant
    runner: Callable[..., object]

    def try_run(self, graph, src, dst, edge, params=None):
        if graph is not self.graph or edge:
            return _EXECUTABLE_MISS
        namespaces = {"src": src, "dst": dst, "param": params or {}}
        try:
            inputs = tuple(
                namespaces[role][name] for role, name in self.bindings)
        except (KeyError, TypeError):
            return _EXECUTABLE_MISS
        for (role, _), value, spec in zip(
            self.bindings, inputs, self.specs
        ):
            if role == "param":
                if not isinstance(value, (int, float)):
                    return _EXECUTABLE_MISS
            elif spec is None or not _matches_field(value, spec):
                return _EXECUTABLE_MISS
        return self.runner(*inputs)


@dataclass
class _CSRScalarExecutable(_DenseScalarExecutable):
    row_ptr_spec: tuple[object, ...] | None = None
    col_idx_spec: tuple[object, ...] | None = None

    def try_run(self, graph, src, dst, edge, params=None):
        if graph is not self.graph:
            return _EXECUTABLE_MISS
        row_ptr, col_idx = graph.resolve_csr()
        if (
            self.row_ptr_spec is None
            or self.col_idx_spec is None
            or not _matches_field(row_ptr, self.row_ptr_spec)
            or not _matches_field(col_idx, self.col_idx_spec)
        ):
            return _EXECUTABLE_MISS
        namespaces = {
            "src": src, "dst": dst, "edge": edge, "param": params or {}}
        try:
            inputs = tuple(
                namespaces[role][name] for role, name in self.bindings)
        except (KeyError, TypeError):
            return _EXECUTABLE_MISS
        for (role, _), value, spec in zip(
            self.bindings, inputs, self.specs
        ):
            if role == "param":
                if not isinstance(value, (int, float)):
                    return _EXECUTABLE_MISS
            elif spec is None or not _matches_field(value, spec):
                return _EXECUTABLE_MISS
        return self.runner(*inputs)


@dataclass
class _RankedExecutable(_DenseScalarExecutable):
    """Shape-guarded dynamic ranked relation with live coordinate rebinding."""

    position_specs: tuple[tuple[object, ...], tuple[object, ...]] | None = None

    def try_run(self, graph, src, dst, edge, params=None):
        if graph is not self.graph or graph.schema.realization != "procedural_knn":
            return _EXECUTABLE_MISS
        try:
            query, candidate = graph.ranked_positions()
        except TypeError:
            return _EXECUTABLE_MISS
        if self.position_specs is None or not all(
            _matches_field(value, spec)
            for value, spec in zip((query, candidate), self.position_specs)
        ):
            return _EXECUTABLE_MISS
        namespaces = {
            "src": src, "dst": dst, "edge": edge, "param": params or {}}
        try:
            inputs = tuple(
                namespaces[role][name] for role, name in self.bindings)
        except (KeyError, TypeError):
            return _EXECUTABLE_MISS
        for (role, _), value, spec in zip(
            self.bindings, inputs, self.specs
        ):
            if role == "param":
                if not isinstance(value, (int, float)):
                    return _EXECUTABLE_MISS
            elif spec is None or not _matches_field(value, spec):
                return _EXECUTABLE_MISS
        return self.runner(query, candidate, *inputs)


class _CSRPlanLaunch:
    """Named runtime ABI wrapper around one compiler-produced TTIR plan."""

    def __init__(self, plan):
        self.plan = plan

    def launch(self, *, output, inputs):
        self.plan.launch_into(output=output, inputs=inputs)

    def prepare_submission(self, argument_slots):
        slots = dict(argument_slots)
        output_slot = slots["output"]
        inputs_slot = slots["inputs"]
        plan = self.plan

        def submit(resources):
            plan.launch_into(
                output=resources[output_slot], inputs=resources[inputs_slot])

        return submit


def _csr_bundle_runner(plan):
    """Submit a CSR plan through the provider-neutral runtime bundle API."""
    from .provider import TorchCudaSubmissionProvider

    bundle = ExecutableBundle(
        (
            KernelInvocation(
                "csr_compute",
                _CSRPlanLaunch(plan),
                arguments=(
                    ArgumentBinding("output", "output"),
                    ArgumentBinding("inputs", "inputs"),
                ),
                accesses=(
                    ResourceAccess("row_ptr", "read"),
                    ResourceAccess("col_idx", "read"),
                    ResourceAccess("inputs", "read"),
                    ResourceAccess("output", "write"),
                ),
            ),
        ),
        required_bindings=("row_ptr", "col_idx", "inputs", "output"),
    )
    prepared = bundle.prepare(TorchCudaSubmissionProvider())

    def run(*inputs):
        output = plan.acquire_output(*inputs)
        prepared.submit_resources(
            (plan.row_ptr, plan.col_idx, tuple(inputs), output))
        return output

    return bundle, run


def _map_tree(function, value):
    if isinstance(value, torch.Tensor):
        return function(value)
    if isinstance(value, tuple):
        return tuple(_map_tree(function, item) for item in value)
    if isinstance(value, list):
        return [_map_tree(function, item) for item in value]
    if isinstance(value, dict):
        return {name: _map_tree(function, item) for name, item in value.items()}
    raise TypeError(
        "edge() must return a Tensor or a nested tuple/list/dict of Tensors"
    )


def _call_region(method, positional: tuple[object, ...], params: Mapping[str, Any]):
    signature = inspect.signature(method)
    accepts_rest = any(
        item.kind is inspect.Parameter.VAR_KEYWORD
        for item in signature.parameters.values()
    )
    selected = params if accepts_rest else {
        name: value for name, value in params.items() if name in signature.parameters
    }
    return method(*positional, **selected)


class MessagePassing(Kernel):
    reducer = SumReducer()

    def __init__(self) -> None:
        super().__init__()
        self._last_executable: (
            _GuardedExecutable
            | _StructuredExecutable
            | _DenseScalarExecutable
            | _CSRScalarExecutable
            | _RankedExecutable
            | None
        ) = None
        self._generated_radius_executables: dict[tuple[object, ...], object] = {}

    def edge(self, src, dst, edge, **params):
        raise NotImplementedError

    def node(self, dst, aggregate, **params):
        return aggregate

    def __call__(
        self,
        *,
        graph: Graph,
        src: Mapping[str, torch.Tensor],
        dst: Mapping[str, torch.Tensor],
        edge: Mapping[str, torch.Tensor] | None = None,
        **params: Any,
    ):
        if not getattr(graph, "_graphforge_graph", False):
            raise TypeError("graph must be a graphforge.Graph")
        # This module is the deprecated Torch eager/JIT compatibility facade.
        # Native gf.Tensor execution is owned by compiler/runtime modules.
        if type(graph).__module__ != "graphforge.interop.torch.graph":
            from .graph import from_native

            graph = from_native(graph)
        if graph.is_distributed:
            try:
                from .compiler_bridge import (
                    find_gf_opt,
                    lower_mlir_stages,
                    message_passing_domain_mlir,
                )

                tool = find_gf_opt()
                if tool is not None:
                    module = message_passing_domain_mlir(
                        kernel=self,
                        graph=graph,
                        src=dict(src),
                        dst=dict(dst),
                        edge={} if edge is None else dict(edge),
                        params=params,
                        kernel_name=type(self).__qualname__,
                    )
                    stages = lower_mlir_stages(module, gf_opt=tool)
                    artifacts = {
                        "iter": stages.iteration,
                        "kernel": stages.kernel,
                    }
                    if stages.task is not None:
                        artifacts["task"] = stages.task
                    self._record_variant(CompiledVariant(
                        key=("distributed-plan", graph.planning_key()),
                        backend="distributed-runtime-unavailable",
                        provider="graphforge-task",
                        domain_plan=stages.domain,
                        passes=(
                            "gf-lower-domain-to-iter",
                            "gf-lower-iter-to-kernel",
                            "gf-select-kernel-schedule",
                            "gf-plan-distributed-tasks",
                        ),
                        artifacts=artifacts,
                        remarks=(
                            "task DAG compiled; activate DistributedRuntime and "
                            "submit through the native MessagePassing entry point",
                        ),
                    ))
            except (KeyError, TypeError, NotImplementedError, RuntimeError):
                pass
            raise NotImplementedError(
                "distributed Graph execution requires an active "
                "DistributedRuntime and the native MessagePassing entry point; "
                "refusing to execute only the local shard because that would "
                "change MessagePassing semantics"
            )
        edge = {} if edge is None else edge
        executable = self._last_executable
        if executable is not None:
            result = executable.try_run(graph, src, dst, edge, params)
            if result is not _EXECUTABLE_MISS:
                self._cache_hits += 1
                self._last_variant = executable.variant
                return result
        self._cache_misses += 1
        self._validate_fields("src", src, graph.schema.num_src, graph.device)
        self._validate_fields("dst", dst, graph.schema.num_dst, graph.device)
        key = self._specialization_key(graph, src, dst, edge, params)
        if (
            graph.schema.realization == "implicit_dense"
            and isinstance(self.reducer, OnlineSoftmaxReducer)
        ):
            result = self._execute_structured_dense(
                key, graph, src, dst, edge, params)
            if result is not None:
                return result
            if self.reducer.block_prune_threshold is not None:
                raise NotImplementedError(
                    "block_prune_threshold requires the generated CUDA dense "
                    "streaming provider; refusing to silently execute exact "
                    "softmax or materialize the Cartesian relation"
                )
        if graph.schema.realization == "implicit_dense":
            result = self._execute_generic_dense(
                key, graph, src, dst, edge, params)
            if result is not None:
                return result
        if graph.schema.realization == "procedural_knn":
            result = self._execute_ranked(
                key, graph, src, dst, edge, params)
            if result is not None:
                return result
        pattern = self._weighted_sum_pattern(params)
        if pattern is not None and not edge:
            generated = self._execute_generated_radius_sum(
                key, pattern, graph, src, dst, params)
            if generated is not None:
                return generated
        row_ptr, col_idx = graph.resolve_csr()
        self._validate_fields("edge", edge, col_idx.numel(), graph.device)
        if pattern is not None:
            result = self._execute_weighted_sum(
                key, pattern, graph, row_ptr, col_idx, src, dst, edge, params)
            if result is not None:
                return result
        result = self._execute_generic_csr(
            key, graph, row_ptr, col_idx, src, dst, edge, params)
        if result is not None:
            return result
        if self._lookup_variant(key) is None:
            self._record_variant(self._make_reference_variant(key, graph, src, dst, edge))
        return self._evaluate_reference(
            graph, row_ptr, col_idx, src, dst, edge, params)

    def _execute_ranked(self, key, graph, src, dst, edge, params):
        """Compile exact ranked selection and its edge consumer as one launch."""
        k = getattr(graph, "_k", None)
        if (
            graph.device.type != "cuda"
            or not isinstance(k, int)
            or k <= 0
            or k > 64
            or k & (k - 1)
        ):
            return None
        query, candidate = graph.ranked_positions()
        if (
            query.dtype != torch.float32
            or candidate.dtype != torch.float32
            or not query.is_contiguous()
            or not candidate.is_contiguous()
        ):
            return None
        self._validate_fields(
            "edge", edge, graph.schema.num_dst * k, graph.device)
        try:
            from dataclasses import replace

            from ...codegen import prepare_ttir_ranked
            from ...compiler.domain_capture import capture_message_passing
            from .compiler_bridge import (
                find_gf_translate,
                lower_kernel_to_ttir,
                lower_mlir_stages,
                message_passing_domain_mlir,
                parse_kernel_ttir_plan,
            )

            descriptor = capture_message_passing(
                kernel=self, graph=graph, src=src, dst=dst, edge=edge,
                params=params, kernel_name=type(self).__qualname__)
            bindings = tuple(
                (field.role, field.name) for field in descriptor.fields)
            bindings += tuple(
                ("param", parameter.name) for parameter in descriptor.params)
            namespaces = {
                "src": src, "dst": dst, "edge": edge, "param": params}
            inputs = tuple(
                namespaces[role][name] for role, name in bindings)
            if any(
                value.requires_grad
                for value in (query, candidate, *(
                    value for (role, _), value in zip(bindings, inputs)
                    if role != "param"
                ))
            ):
                return None
            if any(
                not value.is_contiguous()
                for (role, _), value in zip(bindings, inputs)
                if role != "param"
            ):
                return None
            module = message_passing_domain_mlir(
                kernel=self, graph=graph, src=dict(src), dst=dict(dst),
                edge=dict(edge), params=params,
                kernel_name=type(self).__qualname__)
            stages = lower_mlir_stages(module)
            translator = find_gf_translate()
            if translator is None:
                return None
            stages = replace(
                stages,
                provider_ttir=lower_kernel_to_ttir(
                    stages.kernel, gf_translate=translator),
            )
            manifest = parse_kernel_ttir_plan(stages.provider_ttir)
            if manifest.entry != "gf_ranked_select_consume":
                return None
            plan = prepare_ttir_ranked(
                manifest.module,
                num_queries=graph.schema.num_dst,
                block_rows=manifest.block_rows,
                num_warps=manifest.num_warps,
            )
            output = plan.run(query, candidate, *inputs)
        except (KeyError, OSError, TypeError, ValueError,
                NotImplementedError, RuntimeError):
            return None

        artifacts = {
            name: value
            for name, value in plan.result.artifacts.items()
            if name in {"ttir", "ttgir", "llir", "ptx"}
            and isinstance(value, str)
        }
        domain_plan, variant_passes, artifacts = self._merge_mlir_stages(
            stages,
            "",
            (
                "capture-ranked-relation",
                "select-ranked-pairs-coordinate-hierarchy",
                "select-candidate-tile",
                "emit-local-stable-topk",
                "emit-hierarchical-topk-merge",
                "fuse-selected-edge-consumer",
                "gf-kernel-to-ttir",
                "provider-compile-serialized-ttir",
            ),
            artifacts,
        )
        variant = CompiledVariant(
            key=key,
            backend="cuda",
            provider=plan.result.provider.display_name(),
            lowering="gf-kernel-to-ttir-ranked-select-consume",
            provider_key=plan.result.provider.cache_key(),
            domain_plan=domain_plan,
            passes=variant_passes,
            artifacts=artifacts,
            remarks=(
                "exact candidate ranking and edge consumption share one launch",
                "candidate tiles retain only k stable distance/index keys",
                "no CSR row pointer or selected column tensor is materialized",
            ),
        )
        self._record_variant(variant)
        self._last_executable = _RankedExecutable(
            graph=graph,
            bindings=bindings,
            specs=tuple(
                None if role == "param" else _field_spec(value)
                for (role, _), value in zip(bindings, inputs)
            ),
            variant=variant,
            runner=plan.run,
            position_specs=(_field_spec(query), _field_spec(candidate)),
        )
        return output

    def _execute_generated_radius_sum(
        self, key, pattern, graph, src, dst, params
    ):
        del dst, params
        if (
            graph.schema.lifecycle != "dynamic"
            or pattern.edge_field != "distance"
            or graph.device.type != "cuda"
        ):
            return None
        x = src.get(pattern.src_field)
        positions = graph.euclidean_positions()
        if x is None or positions is None or x.ndim != 1:
            return None
        directory = graph.generated_cell_directory()
        if directory is None:
            return None
        direct_plan = self._generated_radius_executables.get(key)
        if direct_plan is not None:
            variant = self._lookup_variant(key)
            if variant is not None:
                output = _run_generated_radius_distance_sum(
                    graph, direct_plan.run, directory, positions, x)
                self._last_executable = _GeneratedRadiusExecutable(
                    graph=graph,
                    source_field=pattern.src_field,
                    source_spec=_field_spec(x),
                    position=positions,
                    position_version=positions._version,
                    snapshot_token=_dynamic_snapshot_token(graph),
                    variant=variant,
                    generated_runner=direct_plan.run,
                    record_variant=self._record_variant,
                    reuse_threshold=max(1, int(os.environ.get(
                        "GRAPHFORGE_RADIUS_REUSE_THRESHOLD", "2"))),
                )
                return output

        mlir_stages = None
        if self._lookup_variant(key) is None:
            mlir_stages = self._generated_radius_mlir_stages(
                graph, directory, x, pattern)
        if mlir_stages is not None and mlir_stages.provider_ttir is not None:
            from ...codegen import prepare_ttir_generated_radius
            from .compiler_bridge import parse_kernel_ttir_plan

            manifest = parse_kernel_ttir_plan(mlir_stages.provider_ttir)
            try:
                direct_plan = prepare_ttir_generated_radius(
                    manifest.module,
                    num_rows=graph.schema.num_dst,
                    block_rows=manifest.block_rows,
                    num_warps=manifest.num_warps,
                )
            except RuntimeError:
                direct_plan = None
            if direct_plan is not None:
                output = _run_generated_radius_distance_sum(
                    graph, direct_plan.run, directory, positions, x)
                artifacts = {
                    name: value
                    for name, value in direct_plan.result.artifacts.items()
                    if name in {"ttir", "ttgir", "llir", "ptx"}
                    and isinstance(value, str)
                }
                domain_plan, variant_passes, artifacts = self._merge_mlir_stages(
                    mlir_stages,
                    "",
                    (
                        "capture-edge-expression",
                        "recognize-distance-weighted-sum",
                        "prove-default-euclidean-radius",
                        "select-dense-cell-directory",
                        "elide-implicit-edge-geometry",
                        "lower-generated-cell-tile-consumer",
                        "gf-kernel-to-ttir",
                        "provider-compile-serialized-ttir",
                    ),
                    artifacts,
                )
                variant = CompiledVariant(
                    key=key,
                    backend="cuda",
                    provider=direct_plan.result.provider.display_name(),
                    lowering="gf-kernel-to-ttir-generated-radius-distance-sum",
                    provider_key=direct_plan.result.provider.cache_key(),
                    domain_plan=domain_plan,
                    passes=variant_passes,
                    artifacts=artifacts,
                    remarks=(
                        "accepted CSR, distance and message tensors are not materialized",
                        "dynamic cell-directory buffers are rebound on every invocation",
                        "autograd reconstructs the fixed-snapshot geometry VJP automatically",
                        "32-lane generated cell tiles passed the <=1.03x oracle runtime gate",
                    ),
                )
                self._record_variant(variant)
                self._generated_radius_executables[key] = direct_plan
                self._last_executable = _GeneratedRadiusExecutable(
                    graph=graph,
                    source_field=pattern.src_field,
                    source_spec=_field_spec(x),
                    position=positions,
                    position_version=positions._version,
                    snapshot_token=_dynamic_snapshot_token(graph),
                    variant=variant,
                    generated_runner=direct_plan.run,
                    record_variant=self._record_variant,
                    reuse_threshold=max(1, int(os.environ.get(
                        "GRAPHFORGE_RADIUS_REUSE_THRESHOLD", "2"))),
                )
                return output

        # There is deliberately no handwritten provider fallback here.  A
        # shape not covered by the compiler-generated path returns to the
        # semantic implementation; handwritten kernels belong in benchmarks.
        return None

    def _execute_structured_dense(
        self, key, graph, src, dst, edge, params
    ):
        if graph.device.type != "cuda" or edge:
            return None
        from dataclasses import replace

        from ...codegen import (
            prepare_ttir_dense_streaming,
            triton_provider_identity,
        )
        from .compiler_bridge import (
            find_gf_translate,
            lower_kernel_to_ttir,
            lower_mlir_stages,
            message_passing_domain_mlir,
            parse_kernel_ttir_plan,
        )
        from .compiler_bridge import (
            find_gf_opt,
            structured_message_bindings,
        )

        tool = find_gf_opt()
        if tool is None:
            return None
        try:
            captured, bindings, parameter_names = structured_message_bindings(
                self, params)
            del captured
            if len(parameter_names) != 1:
                return None
            namespaces = {"src": src, "dst": dst, "edge": edge}
            tensors = tuple(namespaces[role][name] for role, name in bindings)
            entities, lanes, width = tensors[0].shape
            required_stride = (width, entities * width, 1)
            if any(tensor.stride() != required_stride for tensor in tensors):
                return None
            module = message_passing_domain_mlir(
                kernel=self, graph=graph, src=dict(src), dst=dict(dst),
                edge=dict(edge), params=params,
                kernel_name=type(self).__qualname__)
            stages = lower_mlir_stages(module, gf_opt=tool)
            translator = find_gf_translate(next_to=tool)
            if translator is None:
                return None
            stages = replace(
                stages,
                provider_ttir=lower_kernel_to_ttir(
                    stages.kernel, gf_translate=translator),
            )
            manifest = parse_kernel_ttir_plan(stages.provider_ttir)
            plan = prepare_ttir_dense_streaming(
                manifest.module,
                num_rows=entities,
                lanes=lanes,
                width=width,
                block_rows=manifest.block_rows,
                num_warps=manifest.num_warps,
            )
            scale = float(params[parameter_names[0]])
            output = plan.run(*tensors, scale)
        except (KeyError, TypeError, ValueError, NotImplementedError, RuntimeError):
            return None

        artifacts = {
            name: artifact
            for name, artifact in plan.result.artifacts.items()
            if name in {"ttir", "ttgir", "llir", "ptx"}
            and isinstance(artifact, str)
        }
        domain_plan, variant_passes, artifacts = self._merge_mlir_stages(
            stages,
            "",
            (
                "capture-structured-message",
                "expand-reducer-regions",
                "select-cartesian-coordinate-hierarchy",
                "select-dense-query-neighbor-tile",
                "lower-vector-contraction",
                "lower-streaming-reducer-state",
                "gf-kernel-to-ttir",
                "provider-compile-serialized-ttir",
            ),
            artifacts,
        )
        variant = CompiledVariant(
            key=key,
            backend="cuda",
            provider=triton_provider_identity().display_name(),
            lowering="gf-kernel-to-ttir-dense-streaming-reduction",
            provider_key=plan.result.provider.cache_key(),
            domain_plan=domain_plan,
            passes=variant_passes,
            artifacts=artifacts,
            remarks=(
                "Cartesian relation remained implicit",
                "multi-value message and reducer state were fused into tiles",
                "no pairwise score or normalized-weight tensor was materialized",
            ),
        )
        self._record_variant(variant)
        self._last_executable = _StructuredExecutable(
            graph=graph,
            bindings=bindings,
            specs=tuple(_field_spec(value) for value in tensors),
            parameter_name=parameter_names[0],
            variant=variant,
            runner=plan.run,
        )
        return output

    def _execute_generic_dense(self, key, graph, src, dst, edge, params):
        if graph.device.type != "cuda" or edge:
            return None
        from dataclasses import replace

        from ...codegen import prepare_ttir_dense_scalar
        from ...compiler.domain_capture import capture_message_passing
        from .compiler_bridge import (
            find_gf_translate,
            lower_kernel_to_ttir,
            lower_mlir_stages,
            message_passing_domain_mlir,
            parse_kernel_ttir_plan,
        )

        try:
            descriptor = capture_message_passing(
                kernel=self, graph=graph, src=src, dst=dst, edge=edge,
                params=params, kernel_name=type(self).__qualname__)
            bindings = tuple(
                (field.role, field.name) for field in descriptor.fields)
            bindings += tuple(
                ("param", parameter.name) for parameter in descriptor.params)
            namespaces = {
                "src": src, "dst": dst, "edge": edge, "param": params}
            inputs = tuple(
                namespaces[role][name] for role, name in bindings)
            tensors = tuple(
                value for (role, _), value in zip(bindings, inputs)
                if role != "param")
            if not tensors or any(
                tensor.ndim != 1 or tensor.dtype != torch.float32
                for tensor in tensors
            ) or any(
                not isinstance(value, (int, float))
                for (role, _), value in zip(bindings, inputs)
                if role == "param"
            ):
                return None
            module = message_passing_domain_mlir(
                kernel=self, graph=graph, src=dict(src), dst=dict(dst),
                edge=dict(edge), params=params,
                kernel_name=type(self).__qualname__)
            stages = lower_mlir_stages(module)
            translator = find_gf_translate()
            if translator is None:
                return None
            stages = replace(
                stages,
                provider_ttir=lower_kernel_to_ttir(
                    stages.kernel, gf_translate=translator),
            )
            manifest = parse_kernel_ttir_plan(stages.provider_ttir)
            if manifest.entry != "gf_dense_scalar_reduce":
                return None
            plan = prepare_ttir_dense_scalar(
                manifest.module,
                num_rows=graph.schema.num_dst,
                block_rows=manifest.block_rows,
                num_warps=manifest.num_warps,
            )
            output = plan.run(*inputs)
        except (KeyError, TypeError, ValueError, NotImplementedError, RuntimeError):
            return None

        artifacts = {
            name: artifact
            for name, artifact in plan.result.artifacts.items()
            if name in {"ttir", "ttgir", "llir", "ptx"}
            and isinstance(artifact, str)
        }
        domain_plan, variant_passes, artifacts = self._merge_mlir_stages(
            stages,
            "",
            (
                "capture-scalar-message-and-node",
                "inline-reducer-regions",
                "select-cartesian-coordinate-hierarchy",
                "select-correctness-first-dense-scalar-loop",
                "gf-kernel-to-ttir",
                "provider-compile-serialized-ttir",
            ),
            artifacts,
        )
        variant = CompiledVariant(
            key=key,
            backend="cuda",
            provider=plan.result.provider.display_name(),
            lowering="gf-kernel-to-ttir-generic-dense-scalar-reducer",
            provider_key=plan.result.provider.cache_key(),
            domain_plan=domain_plan,
            passes=variant_passes,
            artifacts=artifacts,
            remarks=(
                "Cartesian relation remained implicit",
                "user reducer identity/lift/combine/finalize regions were lowered",
                "generic scalar loop is a correctness path, not a SOTA tile claim",
            ),
        )
        self._record_variant(variant)
        self._last_executable = _DenseScalarExecutable(
            graph=graph,
            bindings=bindings,
            specs=tuple(
                None if role == "param" else _field_spec(value)
                for (role, _), value in zip(bindings, inputs)
            ),
            variant=variant,
            runner=plan.run,
        )
        return output

    def _execute_generic_csr(
        self, key, graph, row_ptr, col_idx, src, dst, edge, params
    ):
        if graph.device.type != "cuda" or graph.schema.realization != "materialized_csr":
            return None
        from dataclasses import replace

        from ...codegen import prepare_ttir_csr_scalar
        from ...compiler.domain_capture import capture_message_passing
        from .compiler_bridge import (
            find_gf_translate,
            lower_kernel_to_ttir,
            lower_mlir_stages,
            message_passing_domain_mlir,
            parse_kernel_ttir_plan,
        )

        try:
            descriptor = capture_message_passing(
                kernel=self, graph=graph, src=src, dst=dst, edge=edge,
                params=params, kernel_name=type(self).__qualname__)
            bindings = tuple(
                (field.role, field.name) for field in descriptor.fields)
            bindings += tuple(
                ("param", parameter.name) for parameter in descriptor.params)
            namespaces = {
                "src": src, "dst": dst, "edge": edge, "param": params}
            inputs = tuple(
                namespaces[role][name] for role, name in bindings)
            tensors = tuple(
                value for (role, _), value in zip(bindings, inputs)
                if role != "param")
            if not tensors or any(
                tensor.ndim != 1 or tensor.dtype != torch.float32
                for tensor in tensors
            ) or any(
                not isinstance(value, (int, float))
                for (role, _), value in zip(bindings, inputs)
                if role == "param"
            ):
                return None
            module = message_passing_domain_mlir(
                kernel=self, graph=graph, src=dict(src), dst=dict(dst),
                edge=dict(edge), params=params,
                kernel_name=type(self).__qualname__)
            stages = lower_mlir_stages(module)
            translator = find_gf_translate()
            if translator is None:
                return None
            stages = replace(
                stages,
                provider_ttir=lower_kernel_to_ttir(
                    stages.kernel, gf_translate=translator),
            )
            manifest = parse_kernel_ttir_plan(stages.provider_ttir)
            if manifest.entry not in {
                "gf_csr_scalar_reduce", "gf_csr_additive_tile",
                "gf_csr_algebra_tile", "gf_csr_stable_weighted_tile",
            }:
                return None
            plan = prepare_ttir_csr_scalar(
                manifest.module,
                row_ptr=row_ptr,
                col_idx=col_idx,
                num_rows=graph.schema.num_dst,
                block_rows=manifest.block_rows,
                num_warps=manifest.num_warps,
            )
            bundle, bundle_runner = _csr_bundle_runner(plan)
            output = bundle_runner(*inputs)
        except (KeyError, TypeError, ValueError, NotImplementedError, RuntimeError):
            return None

        minimum_degree, maximum_degree, degree_sum = graph.degree_statistics()
        padded_degree = 1 << max(0, maximum_degree - 1).bit_length()
        padding_utilization = (
            1.0 if graph.schema.num_dst == 0 or padded_degree == 0
            else degree_sum / (graph.schema.num_dst * padded_degree)
        )
        selected_runner = bundle_runner
        task_runner = None
        compiled_tasks = None
        task_selected = False
        direct_ms = task_ms = None
        if padding_utilization < 0.60 and stages.task is not None:
            try:
                prepared = _prepare_generic_task_candidate(
                    stages, graph, row_ptr, col_idx, bindings, inputs)
            except (KeyError, OSError, TypeError, ValueError, RuntimeError):
                prepared = None
            if prepared is not None:
                task_runner, compiled_tasks = prepared
                direct_ms = _cuda_median_ms(lambda: bundle_runner(*inputs))
                task_ms = _cuda_median_ms(lambda: task_runner.run(*inputs))
                if task_ms <= direct_ms * 0.99:
                    selected_runner = task_runner.run
                    task_selected = True
                    output = selected_runner(*inputs)

        artifacts = {
            name: artifact
            for name, artifact in plan.result.artifacts.items()
            if name in {"ttir", "ttgir", "llir", "ptx"}
            and isinstance(artifact, str)
        }
        if compiled_tasks is not None and task_selected:
            for task_name, executable in compiled_tasks.items():
                for name, artifact in executable.result.artifacts.items():
                    if name in {"ttir", "ttgir", "llir", "ptx"} and isinstance(
                        artifact, str
                    ):
                        artifacts[f"{task_name}.{name}"] = artifact
        domain_plan, variant_passes, artifacts = self._merge_mlir_stages(
            stages,
            "",
            (
                "capture-scalar-message-and-node",
                "inline-reducer-regions",
                "select-compressed-row-coordinate-hierarchy",
                (
                    f"select-bounded-row-neighbor-tile[rows="
                    f"{manifest.block_rows}]"
                    if manifest.entry in {
                        "gf_csr_additive_tile", "gf_csr_algebra_tile",
                        "gf_csr_stable_weighted_tile",
                    }
                    else "select-correctness-first-csr-row-loop"
                ),
                "gf-kernel-to-ttir",
                "autotune-degree-bucket-task-vs-row-tile",
                "provider-compile-serialized-ttir",
                "runtime-submit-executable-bundle",
            ),
            artifacts,
        )
        load_balance_remark = (
            f"degree analysis: min={minimum_degree}, max={maximum_degree}, "
            f"padding utilization={padding_utilization:.1%}; "
            + (
                f"degree-bucket task selected by warm autotune "
                f"({task_ms:.4f} vs row-tile {direct_ms:.4f} ms)"
                if task_selected
                else (
                    f"row tile retained by warm autotune "
                    f"({direct_ms:.4f} vs degree-task {task_ms:.4f} ms)"
                    if task_runner is not None
                    else "no legal degree-bucket task was produced"
                )
            )
            if padding_utilization < 0.60
            else f"degree analysis: min={minimum_degree}, max={maximum_degree}, "
                 f"padding utilization={padding_utilization:.1%}; row-neighbor "
                 "tile is accepted"
        )
        variant = CompiledVariant(
            key=key,
            backend="cuda",
            provider=plan.result.provider.display_name(),
            lowering=(
                "gf-task-degree-bucket-csr-additive"
                if task_selected
                else (
                    "gf-kernel-to-ttir-csr-additive-tile"
                    if manifest.entry == "gf_csr_additive_tile"
                    else (
                        "gf-kernel-to-ttir-csr-algebra-tile"
                        if manifest.entry == "gf_csr_algebra_tile"
                        else (
                            "gf-kernel-to-ttir-csr-stable-weighted-tile"
                            if manifest.entry == "gf_csr_stable_weighted_tile"
                            else "gf-kernel-to-ttir-generic-csr-scalar-reducer"
                        )
                    )
                )
            ),
            provider_key=plan.result.provider.cache_key(),
            domain_plan=domain_plan,
            passes=variant_passes,
            artifacts=artifacts,
            remarks=(
                "user reducer and optional node regions were lowered",
                (
                    "zero-additive tuple state was proven and reduced in a "
                    f"bounded {manifest.block_rows}-row neighbor tile"
                    if manifest.entry == "gf_csr_additive_tile"
                    else (
                        "user tuple-state identity and combine regions were "
                        f"lowered to a variadic {manifest.block_rows}-row "
                        "tt.reduce tile"
                        if manifest.entry == "gf_csr_algebra_tile"
                        else (
                            "the proven stable weighted tuple algebra was "
                            "reassociated to max/exp/sum reduction tiles"
                            if manifest.entry == "gf_csr_stable_weighted_tile"
                            else "generic CSR row loop is a correctness path, "
                                 "not a SOTA tile claim"
                        )
                    )
                ),
                load_balance_remark,
                (
                    f"runtime executable bundle contains "
                    f"{len(task_runner._execute.bundle.invocations)} execute invocations"
                    if task_selected
                    else f"runtime executable bundle contains "
                         f"{len(bundle.invocations)} kernel invocation"
                ),
            ),
        )
        self._record_variant(variant)
        self._last_executable = _CSRScalarExecutable(
            graph=graph,
            bindings=bindings,
            specs=tuple(
                None if role == "param" else _field_spec(value)
                for (role, _), value in zip(bindings, inputs)
            ),
            variant=variant,
            runner=selected_runner,
            row_ptr_spec=_field_spec(row_ptr),
            col_idx_spec=_field_spec(col_idx),
        )
        return output

    def reference(
        self,
        *,
        graph: Graph,
        src: Mapping[str, torch.Tensor],
        dst: Mapping[str, torch.Tensor],
        edge: Mapping[str, torch.Tensor] | None = None,
        **params: Any,
    ):
        """Run the eager semantic oracle even when a compiled path exists."""
        if (
            isinstance(self.reducer, OnlineSoftmaxReducer)
            and self.reducer.block_prune_threshold is not None
        ):
            raise NotImplementedError(
                "the schedule-dependent block-pruned approximation has no "
                "eager edge-materializing reference; compare against an exact "
                "online-softmax result with an explicit accuracy budget"
            )
        if graph.is_distributed:
            raise NotImplementedError(
                "reference() cannot silently evaluate one shard of a Graph.halo() snapshot"
            )
        edge = {} if edge is None else edge
        self._validate_fields("src", src, graph.schema.num_src, graph.device)
        self._validate_fields("dst", dst, graph.schema.num_dst, graph.device)
        row_ptr, col_idx = graph.resolve_csr()
        self._validate_fields("edge", edge, col_idx.numel(), graph.device)
        return self._evaluate_reference(
            graph, row_ptr, col_idx, src, dst, edge, params)

    def _evaluate_reference(
        self, graph, row_ptr, col_idx, src, dst, edge, params
    ):
        destination = graph.destination_index(row_ptr)
        implicit_edge = graph.implicit_edge_fields(row_ptr, col_idx)
        overlap = implicit_edge.keys() & edge.keys()
        if overlap:
            names = ", ".join(sorted(overlap))
            raise ValueError(f"edge fields shadow graph-provided fields: {names}")
        all_edge = {**implicit_edge, **edge}
        src_at_edge = SimpleNamespace(**{
            name: value[col_idx] for name, value in src.items()
        })
        dst_at_edge = SimpleNamespace(**{
            name: value[destination] for name, value in dst.items()
        })
        edge_view = SimpleNamespace(**all_edge)
        message = _call_region(
            self.edge, (src_at_edge, dst_at_edge, edge_view), params)
        from .reducer_oracle import reduce as reduce_oracle

        if isinstance(self.reducer, OnlineSoftmaxReducer):
            aggregate = reduce_oracle(
                self.reducer, message, destination, graph.schema.num_dst)
        else:
            aggregate = _map_tree(
                lambda tensor: reduce_oracle(
                    self.reducer, tensor, destination, graph.schema.num_dst),
                message,
            )
        if type(self).node is MessagePassing.node:
            return aggregate
        return _call_region(self.node, (SimpleNamespace(**dst), aggregate), params)

    def _weighted_sum_pattern(self, params):
        if not isinstance(self.reducer, SumReducer):
            return None
        if type(self).node is not MessagePassing.node:
            return None
        return recognize_weighted_sum(capture_edge(self.edge, params))

    def _execute_weighted_sum(
        self, key, pattern, graph, row_ptr, col_idx, src, dst, edge, params
    ):
        x = src.get(pattern.src_field)
        weight = edge.get(pattern.edge_field)
        implicit_edge = None
        if graph.schema.lifecycle == "dynamic":
            if pattern.edge_field in {"distance", "displacement"} and weight is not None:
                raise ValueError(
                    f"edge fields shadow graph-provided fields: {pattern.edge_field}")
            implicit_edge = graph.implicit_edge_fields(row_ptr, col_idx)
            if pattern.edge_field in implicit_edge:
                weight = implicit_edge[pattern.edge_field]
        if x is None or weight is None:
            return None
        if weight.ndim not in (1, 2) or (
            weight.ndim == 2 and weight.shape[1:] != (1,)
        ):
            return None
        mlir_stages = None
        if self._lookup_variant(key) is None:
            mlir_stages = self._weighted_sum_mlir_stages(
                graph, row_ptr, col_idx, x, weight, pattern)
        if graph.schema.lifecycle == "static":
            minimum_degree, maximum_degree = graph.degree_bounds()
        elif graph.schema.num_dst == 0:
            minimum_degree = maximum_degree = 0
        else:
            degrees = row_ptr[1:] - row_ptr[:-1]
            extrema = torch.stack((degrees.min(), degrees.max())).cpu()
            minimum_degree, maximum_degree = int(extrema[0]), int(extrema[1])
        degree = minimum_degree if minimum_degree == maximum_degree else None
        direct_fixed_shape = (
            (x.ndim == 1 and weight.ndim == 1)
            or (
                x.ndim == 2
                and x.shape[1] > 1
                and (
                    weight.ndim == 1
                    or (weight.ndim == 2 and weight.shape[1:] == (1,))
                )
            )
        )
        fixed_knn = (
            graph.schema.realization == "procedural_knn"
            and degree is not None
            and degree == getattr(graph, "_k", None)
        )
        if (
            degree is not None
            and 0 < degree <= 64
            and (graph.schema.lifecycle == "static" or fixed_knn)
            and graph.device.type == "cuda"
            and direct_fixed_shape
            and mlir_stages is not None
            and mlir_stages.provider_ttir is not None
        ):
            from ...codegen import prepare_ttir_weighted_sum
            from .compiler_bridge import parse_kernel_ttir_plan

            launch_manifest = parse_kernel_ttir_plan(
                mlir_stages.provider_ttir)
            plan = prepare_ttir_weighted_sum(
                launch_manifest.module,
                row_ptr=row_ptr,
                col_idx=col_idx,
                num_rows=graph.schema.num_dst,
                block_rows=launch_manifest.block_rows,
                num_warps=launch_manifest.num_warps,
            )
            if fixed_knn:
                def selected_runner(current_x, current_weight):
                    current_row_ptr, current_col_idx = graph.resolve_csr()
                    return plan.run_csr(
                        current_row_ptr, current_col_idx,
                        current_x, current_weight)
                output = plan.run_csr(row_ptr, col_idx, x, weight)
            else:
                selected_runner = plan.run
                output = selected_runner(x, weight)
            if self._lookup_variant(key) is None:
                artifacts = {
                    name: value
                    for name, value in plan.result.artifacts.items()
                    if name in {"ttir", "ttgir", "llir", "ptx"}
                    and isinstance(value, str)
                }
                domain_plan, variant_passes, artifacts = self._merge_mlir_stages(
                    mlir_stages,
                    "\n".join((
                        "// GraphForge captured domain-plan",
                        f"apply @{type(self).__qualname__}",
                        "  recognize linear_weighted_sum reducer(sum)",
                        f"  fixed_degree {degree}",
                        (
                            "  dynamic_csr_rebind procedural_knn"
                            if fixed_knn else "  frozen_csr_binding"
                        ),
                        f"  direct_ttir rows={launch_manifest.block_rows} "
                        f"features={1 if x.ndim == 1 else x.shape[1]} "
                        f"warps={launch_manifest.num_warps}",
                    )),
                    (
                        "capture-edge-expression",
                        "recognize-linear-weighted-sum",
                        "analyze-fixed-degree",
                        "select-fixed-row-neighbor-tile",
                        "gf-kernel-to-ttir",
                        "provider-compile-serialized-ttir",
                    ),
                    artifacts,
                )
                self._record_variant(CompiledVariant(
                    key=key,
                    backend="cuda",
                    provider=plan.result.provider.display_name(),
                    lowering="gf-kernel-to-ttir-fixed-csr-weighted-sum",
                    provider_key=plan.result.provider.cache_key(),
                    domain_plan=domain_plan,
                    passes=variant_passes,
                    artifacts=artifacts,
                    remarks=(
                        "physical fixed-degree proof selected a row-neighbor-feature tile",
                        "TTIR was emitted from gf_kernel IR without a @triton.jit frontend",
                        "the direct candidate passed the SOTA runtime gate on this machine",
                        (
                            "exact kNN rebuilds column indices and rebinds the compiled CSR ABI"
                            if fixed_knn else
                            "frozen CSR indices remain bound to the compiled executable"
                        ),
                    ),
                ))
            self._cache_weighted_sum_executable(
                graph=graph,
                src=src,
                dst=dst,
                edge=edge,
                params=params,
                variant=self.last_variant,
                src_field=pattern.src_field,
                edge_field=pattern.edge_field,
                runner=selected_runner,
                resolved_weight=weight,
            )
            if isinstance(self._last_executable, _DynamicSnapshotExecutable):
                self._last_executable.compatible_runner = (
                    lambda current_graph, current_x: (
                        self._run_dynamic_distance_plan(
                            current_graph, current_x, plan)
                    )
                )
            return output
        if (
            degree is None
            and 0 < maximum_degree <= 64
            and graph.device.type == "cuda"
            and direct_fixed_shape
            and mlir_stages is not None
            and mlir_stages.provider_ttir is not None
        ):
            from ...codegen import prepare_ttir_weighted_sum
            from .compiler_bridge import parse_kernel_ttir_plan

            launch_manifest = parse_kernel_ttir_plan(
                mlir_stages.provider_ttir)
            plan = prepare_ttir_weighted_sum(
                launch_manifest.module,
                row_ptr=row_ptr,
                col_idx=col_idx,
                num_rows=graph.schema.num_dst,
                block_rows=launch_manifest.block_rows,
                num_warps=launch_manifest.num_warps,
            )
            output = plan.run(x, weight)
            if self._lookup_variant(key) is None:
                artifacts = {
                    name: value
                    for name, value in plan.result.artifacts.items()
                    if name in {"ttir", "ttgir", "llir", "ptx"}
                    and isinstance(value, str)
                }
                domain_plan, variant_passes, artifacts = self._merge_mlir_stages(
                    mlir_stages,
                    "\n".join((
                        "// GraphForge captured domain-plan",
                        f"apply @{type(self).__qualname__}",
                        "  recognize linear_weighted_sum reducer(sum)",
                        f"  degree_range [{minimum_degree}, {maximum_degree}]",
                        f"  direct_ttir masked_rows={launch_manifest.block_rows} "
                        f"features={1 if x.ndim == 1 else x.shape[1]} "
                        f"warps={launch_manifest.num_warps}",
                    )),
                    (
                        "capture-edge-expression",
                        "recognize-linear-weighted-sum",
                        "analyze-degree-bounds",
                        "guard-max-degree-64",
                        "select-bounded-ragged-row-neighbor-tile",
                        "gf-kernel-to-ttir",
                        "provider-compile-serialized-ttir",
                    ),
                    artifacts,
                )
                self._record_variant(CompiledVariant(
                    key=key,
                    backend="cuda",
                    provider=plan.result.provider.display_name(),
                    lowering="gf-kernel-to-ttir-bounded-ragged-weighted-sum",
                    provider_key=plan.result.provider.cache_key(),
                    domain_plan=domain_plan,
                    passes=variant_passes,
                    artifacts=artifacts,
                    remarks=(
                        "bounded ragged rows use compiler-emitted masked row-neighbor-feature tiles",
                        "TTIR was emitted from gf_kernel IR without a @triton.jit frontend",
                        "the direct candidate matched the Triton oracle performance gate",
                        "zero-degree rows produce the sum reducer identity",
                    ),
                ))
            self._cache_weighted_sum_executable(
                graph=graph,
                src=src,
                dst=dst,
                edge=edge,
                params=params,
                variant=self.last_variant,
                src_field=pattern.src_field,
                edge_field=pattern.edge_field,
                runner=plan.run,
                resolved_weight=weight,
            )
            if isinstance(self._last_executable, _DynamicSnapshotExecutable):
                # A dynamic ragged topology changes row_ptr/col_idx/implicit
                # distance, but not the guarded <=64 traversal skeleton.
                # Rebind those buffers to the compiled plan just as the
                # fixed-degree path does; falling back to sparse.mm here both
                # discarded generated code and added avoidable CSR wrapping.
                self._last_executable.compatible_runner = (
                    lambda current_graph, current_x: (
                        self._run_dynamic_distance_plan(
                            current_graph, current_x, plan)
                    )
                )
            return output

        if (
            degree is None
            and maximum_degree > 64
            and graph.schema.lifecycle == "static"
            and graph.device.type == "cuda"
            and x.ndim == 1
            and weight.ndim == 1
            and mlir_stages is not None
            and mlir_stages.task is not None
        ):
            try:
                prepared_candidate = _prepare_static_task_candidate(
                    mlir_stages, graph, row_ptr, col_idx, x,
                    pattern.src_field, pattern.edge_field)
            except (KeyError, OSError, TypeError, ValueError, RuntimeError):
                prepared_candidate = None
            if prepared_candidate is not None:
                task_runner, compiled_tasks = prepared_candidate
                native_sparse = graph.sparse_csr(
                    weight, row_ptr=row_ptr, col_idx=col_idx)
                native_output = torch.empty_like(x)

                def native_runner(current_x, current_weight):
                    current_sparse = (
                        native_sparse
                        if current_weight is weight
                        else graph.sparse_csr(
                            current_weight, row_ptr=row_ptr, col_idx=col_idx
                        )
                    )
                    if current_x.ndim == 1:
                        # The guarded/prepared executable owns stable output
                        # storage. Reusing it avoids an allocator round trip
                        # and matches the semantics of generated kernels that
                        # already write into persistent output buffers.
                        torch.mv(current_sparse, current_x, out=native_output)
                        return native_output
                    return torch.mm(current_sparse, current_x)
                # Compilation and immutable worklist materialization are JIT
                # costs, not consume timings.  Autotune only the two warm
                # executable choices on the actual provider and topology.
                candidate_ms = _cuda_median_ms(
                    lambda: task_runner.run(x, weight))
                native_ms = _cuda_median_ms(
                    lambda: native_runner(x, weight))
                choose_candidate = candidate_ms <= native_ms * 0.99
                selected_runner = (
                    task_runner.run if choose_candidate else native_runner)
                output = selected_runner(x, weight)
                artifacts = {}
                if choose_candidate:
                    for name, executable in compiled_tasks.items():
                        for kind, value in executable.result.artifacts.items():
                            if kind in {"ttir", "ttgir", "llir", "ptx"} and isinstance(
                                value, str
                            ):
                                artifacts[f"{name}.{kind}"] = value
                    ptx = [
                        value for name, value in artifacts.items()
                        if name.endswith(".ptx")
                    ]
                    if ptx:
                        artifacts["ptx"] = "\n\n".join(ptx)
                domain_plan, variant_passes, artifacts = self._merge_mlir_stages(
                    mlir_stages,
                    "",
                    (
                        "capture-edge-expression",
                        "recognize-linear-weighted-sum",
                        "analyze-degree-histogram",
                        "analyze-source-index-span",
                        "gf-plan-degree-buckets",
                        "gf-plan-split-rows",
                        "gf-decompose-degree-worklists",
                        "provider-compile-task-ttir",
                        "autotune-task-vs-native-sparse",
                        (
                            "select-worklist-chunked-tail"
                            if choose_candidate
                            else "select-native-sparse-mm"
                        ),
                    ),
                    artifacts,
                )
                provider = next(iter(compiled_tasks.values())).result.provider
                self._record_variant(CompiledVariant(
                    key=key,
                    backend="cuda",
                    provider=(
                        provider.display_name()
                        if choose_candidate else "torch.sparse.mm"
                    ),
                    lowering=(
                        "gf-task-worklist-chunked-tail"
                        if choose_candidate else "dispatch-native-sparse-mm"
                    ),
                    provider_key=provider.cache_key(),
                    domain_plan=domain_plan,
                    passes=variant_passes,
                    artifacts=artifacts,
                    remarks=(
                        f"source index span ratio={graph.source_index_span_ratio():.4f}",
                        f"warm JIT autotune: task={candidate_ms:.4f} ms, "
                        f"native sparse={native_ms:.4f} ms",
                        (
                            "task candidate won the conservative 1% dispatch margin"
                            if choose_candidate else
                            "native sparse retained because task candidate did not "
                            "win the conservative 1% dispatch margin"
                        ),
                        "degree worklist materialization is cached by the immutable graph executable",
                    ),
                ))
                self._cache_weighted_sum_executable(
                    graph=graph,
                    src=src,
                    dst=dst,
                    edge=edge,
                    params=params,
                    variant=self.last_variant,
                    src_field=pattern.src_field,
                    edge_field=pattern.edge_field,
                    runner=selected_runner,
                    resolved_weight=weight,
                )
                return output

        # Linear ragged/unsupported feature shapes retain their structure by
        # dispatching the native sparse provider instead of materializing
        # messages in the semantic evaluator.
        x_2d = x[:, None] if x.ndim == 1 else x
        if x_2d.ndim != 2 or weight.dtype != x.dtype:
            return None
        sparse = graph.sparse_csr(
            weight, row_ptr=row_ptr, col_idx=col_idx)
        output = torch.sparse.mm(sparse, x_2d)
        if self._lookup_variant(key) is None:
            domain_plan, variant_passes, artifacts = self._merge_mlir_stages(
                mlir_stages,
                "\n".join((
                    "// GraphForge captured domain-plan",
                    f"apply @{type(self).__qualname__}",
                    "  recognize linear_weighted_sum reducer(sum)",
                    "  dispatch native_sparse_mm",
                )),
                (
                    "capture-edge-expression",
                    "recognize-linear-weighted-sum",
                    "dispatch-native-sparse-mm",
                ),
                {},
            )
            self._record_variant(CompiledVariant(
                key=key,
                backend=str(graph.device.type),
                provider="torch.sparse.mm",
                lowering="dispatch-native-sparse-mm",
                domain_plan=domain_plan,
                passes=variant_passes,
                artifacts=artifacts,
                remarks=(
                    "no profitable generated specialization was proven; dispatched native sparse library",
                ),
            ))
        def native_sparse_runner(current_x, current_weight):
            current_sparse = (
                sparse
                if current_weight is weight
                else graph.sparse_csr(
                    current_weight, row_ptr=row_ptr, col_idx=col_idx
                )
            )
            current_2d = current_x[:, None] if current_x.ndim == 1 else current_x
            result = torch.mm(current_sparse, current_2d)
            return result[:, 0] if current_x.ndim == 1 else result

        self._cache_weighted_sum_executable(
            graph=graph,
            src=src,
            dst=dst,
            edge=edge,
            params=params,
            variant=self.last_variant,
            src_field=pattern.src_field,
            edge_field=pattern.edge_field,
            runner=native_sparse_runner,
            resolved_weight=weight,
        )
        return output[:, 0] if x.ndim == 1 else output

    def _weighted_sum_mlir_stages(
        self, graph, row_ptr, col_idx, x, weight, pattern
    ):
        from dataclasses import replace

        from .compiler_bridge import (
            find_gf_translate,
            find_gf_opt,
            lower_kernel_to_ttir,
            lower_mlir_stages,
            message_passing_domain_mlir,
        )

        tool = find_gf_opt()
        if tool is None:
            return None
        compile_graph = graph
        if graph.schema.lifecycle == "dynamic":
            compile_graph = Graph.from_csr(
                row_ptr,
                col_idx,
                num_src=graph.schema.num_src,
                validate="basic",
            )
        try:
            module = message_passing_domain_mlir(
                kernel=self,
                graph=compile_graph,
                src={pattern.src_field: x},
                dst={},
                edge={pattern.edge_field: weight},
                params={},
                kernel_name=type(self).__qualname__,
            )
        except (KeyError, TypeError, NotImplementedError):
            return None
        stages = lower_mlir_stages(module, gf_opt=tool)
        # The first direct translator is intentionally narrow.  Keep the
        # high-performance selected TTIR as the executable artifact, while
        # exposing this independently produced TTIR for correctness/perf gates.
        translator = find_gf_translate(next_to=tool)
        translatable_shape = (
            x.ndim == 1 and weight.ndim == 1
        ) or (
            x.ndim == 2 and x.shape[1] > 1 and
            (weight.ndim == 1 or
             (weight.ndim == 2 and weight.shape[1:] == (1,)))
        )
        if translatable_shape and translator is not None:
            try:
                provider_ttir = lower_kernel_to_ttir(
                    stages.kernel, gf_translate=translator)
            except RuntimeError:
                # A valid target-independent program may intentionally select
                # provider-deferred (for example a high-degree vector CSR).
                # Preserve its Task/Kernel IR and let dispatch choose a native
                # sparse library instead of turning a codegen rejection into
                # a user-visible compilation failure.
                provider_ttir = None
            stages = replace(stages, provider_ttir=provider_ttir)
        return stages

    def _generated_radius_mlir_stages(self, graph, directory, x, pattern):
        from dataclasses import replace

        from .compiler_bridge import (
            find_gf_translate,
            find_gf_opt,
            lower_kernel_to_ttir,
            lower_mlir_stages,
            message_passing_domain_mlir,
        )

        tool = find_gf_opt()
        if tool is None:
            return None
        module = message_passing_domain_mlir(
            kernel=self,
            graph=graph,
            directory=directory,
            src={pattern.src_field: x},
            dst={},
            edge={},
            params={},
            kernel_name=type(self).__qualname__,
        )
        stages = lower_mlir_stages(module, gf_opt=tool)
        translator = find_gf_translate(next_to=tool)
        if translator is not None:
            stages = replace(
                stages,
                provider_ttir=lower_kernel_to_ttir(
                    stages.kernel, gf_translate=translator),
            )
        return stages

    @staticmethod
    def _merge_mlir_stages(stages, domain_plan, passes, artifacts):
        if stages is None:
            return domain_plan, passes, artifacts
        merged = dict(artifacts)
        merged.update(iter=stages.iteration, kernel=stages.kernel)
        if stages.task is not None:
            merged["task"] = stages.task
        if stages.provider_ttir is not None:
            merged["kernel_ttir"] = stages.provider_ttir
        # Keep post-capture planning facts next to the structured Domain IR.
        # These facts can depend on runtime-proven topology properties (for
        # example fixed-k procedural kNN) and therefore cannot be reconstructed
        # from the materialized compile snapshot alone.
        merged_domain = stages.domain
        if domain_plan:
            merged_domain = f"{merged_domain.rstrip()}\n\n{domain_plan}\n"
        return (
            merged_domain,
            (
                "gf-verify-domain",
                "gf-lower-domain-to-iter",
                "gf-lower-iter-to-kernel",
                *passes,
            ),
            merged,
        )

    @staticmethod
    def _native_sparse_sum(graph, x, weight):
        x_2d = x[:, None] if x.ndim == 1 else x
        output = torch.mm(graph.sparse_csr(weight), x_2d)
        return output[:, 0] if x.ndim == 1 else output

    def _cache_weighted_sum_executable(
        self,
        *,
        graph,
        src,
        dst,
        edge,
        params,
        variant,
        src_field,
        edge_field,
        runner,
        resolved_weight=None,
        dst_alias_src=False,
    ):
        # Dynamic graphs must resolve a new snapshot before launch. Their fast
        # path will be attached to a versioned snapshot cache, not this static
        # identity guard.
        if graph.schema.lifecycle != "static":
            if graph.schema.realization == "procedural_knn":
                self._last_executable = _GuardedExecutable(
                    graph=graph,
                    src_field=src_field,
                    edge_field=edge_field,
                    src_spec=_field_spec(src[src_field]),
                    edge_spec=_field_spec(edge[edge_field]),
                    require_dst_alias=dst_alias_src,
                    variant=variant,
                    runner=runner,
                )
                return
            if (
                graph.schema.lifecycle == "dynamic"
                and not edge
                and edge_field == "distance"
                and resolved_weight is not None
            ):
                self._last_executable = _DynamicSnapshotExecutable(
                    graph=graph,
                    source_field=src_field,
                    source_spec=_field_spec(src[src_field]),
                    snapshot_token=_dynamic_snapshot_token(graph),
                    variant=variant,
                    runner=lambda current_x: runner(
                        current_x, resolved_weight
                    ),
                    compatible_runner=lambda current_graph, current_x: (
                        self._run_dynamic_distance_sparse(
                            current_graph, current_x
                        )
                    ),
                )
            return

        self._last_executable = _GuardedExecutable(
            graph=graph,
            src_field=src_field,
            edge_field=edge_field,
            src_spec=_field_spec(src[src_field]),
            edge_spec=_field_spec(edge[edge_field]),
            require_dst_alias=dst_alias_src,
            variant=variant,
            runner=runner,
        )

    @staticmethod
    def _run_dynamic_distance_sparse(graph, source):
        row_ptr, col_idx = graph.resolve_csr()
        distance = graph.implicit_edge_fields(row_ptr, col_idx)["distance"]
        sparse = graph.sparse_csr(
            distance, row_ptr=row_ptr, col_idx=col_idx
        )
        source_2d = source[:, None] if source.ndim == 1 else source
        result = torch.mm(sparse, source_2d)
        return result[:, 0] if source.ndim == 1 else result

    @staticmethod
    def _run_dynamic_distance_plan(graph, source, plan):
        row_ptr, col_idx = graph.resolve_csr()
        distance = graph.implicit_edge_fields(row_ptr, col_idx)["distance"]
        return plan.run_csr(row_ptr, col_idx, source, distance)

    @staticmethod
    def _validate_fields(
        role: str,
        fields: Mapping[str, torch.Tensor],
        entity_count: int,
        device: torch.device,
    ) -> None:
        if not isinstance(fields, Mapping):
            raise TypeError(f"{role} fields must be a mapping")
        for name, value in fields.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{role}.{name} must be a torch.Tensor")
            if value.ndim == 0 or value.shape[0] != entity_count:
                raise ValueError(
                    f"{role}.{name} leading dimension must be {entity_count}, "
                    f"got shape {tuple(value.shape)}"
                )
            if value.device != device:
                raise ValueError(
                    f"{role}.{name} is on {value.device}, but graph is on {device}"
                )

    def _specialization_key(self, graph, src, dst, edge, params):
        runtime_params = _runtime_param_schema(params)
        alias_schema = tuple(
            (name, name in dst and value is dst[name])
            for name, value in sorted(src.items())
        )
        provider_environment: tuple[object, ...] = ()
        if graph.device.type == "cuda":
            from ...codegen import triton_provider_identity
            provider_environment = triton_provider_identity().cache_key()
        return (
            type(self).__module__,
            type(self).__qualname__,
            self.reducer.specialization_key(),
            graph.planning_key(),
            _tensor_schema(src),
            _tensor_schema(dst),
            _tensor_schema(edge),
            alias_schema,
            runtime_params,
            provider_environment,
        )

    def _make_reference_variant(self, key, graph, src, dst, edge):
        try:
            source = inspect.getsource(type(self).edge)
        except (OSError, TypeError):
            source = repr(type(self).edge)
        digest = hashlib.sha256(source.encode()).hexdigest()[:16]
        node_kind = "identity" if type(self).node is MessagePassing.node else "custom"
        plan = "\n".join(
            [
                "// GraphForge domain-plan (not yet parsed MLIR)",
                f"graph @{graph.schema.origin} <{graph.schema.realization}, "
                f"{graph.schema.num_src} -> {graph.schema.num_dst}>",
                f"apply @{type(self).__qualname__} "
                f"relation(@{graph.schema.origin}) reducer({self.reducer.name}) {{",
                f"  edge_region source_hash={digest}",
                f"  node_region {node_kind}",
                f"  reads src={tuple(sorted(src))} dst={tuple(sorted(dst))} "
                f"edge={tuple(sorted(edge))}",
                "}",
            ]
        )
        return CompiledVariant(
            key=key,
            backend="reference",
            provider="torch",
            domain_plan=plan,
            passes=("validate-domain", "select-reference-csr"),
            remarks=(
                "reference evaluator selected; no performance codegen artifact exists",
                "cross-apply fusion is handled by @gf.program rather than the "
                "single-apply reference evaluator",
            ),
        )


def _executor(kernel) -> MessagePassing:
    cached = getattr(kernel, "_torch_executor", None)
    if cached is not None:
        return cached
    from ...message_passing.core import MessagePassing as NativeMessagePassing

    def edge(adapter, src, dst, edge, **params):
        del adapter
        return _call_region(kernel.edge, (src, dst, edge), params)

    attributes = {
        "reducer": kernel.reducer,
        "edge": edge,
        "__module__": __name__,
    }
    if type(kernel).node is not NativeMessagePassing.node:
        def node(adapter, dst, aggregate, **params):
            del adapter
            return _call_region(kernel.node, (dst, aggregate), params)
        attributes["node"] = node
    adapter_type = type(
        f"{type(kernel).__name__}TorchExecutor", (MessagePassing,), attributes
    )
    cached = adapter_type()
    kernel._torch_executor = cached
    return cached


def _copy_diagnostics(kernel, executor: MessagePassing) -> None:
    kernel._variants = executor._variants
    kernel._last_variant = executor._last_variant
    kernel._cache_hits = executor._cache_hits
    kernel._cache_misses = executor._cache_misses


def _install_native_fast_path(
    kernel, native_graph, torch_graph, executor, src, dst, edge, params
) -> None:
    """Attach the current guarded executable to the native lazy-JIT object.

    The executable still performs its complete topology, alias, shape, dtype,
    device and stride guard. This removes repeated adapter discovery,
    native-to-Torch graph lookup and provider dispatch from the hot call path.
    """
    executable = executor._last_executable
    if executable is None:
        kernel._torch_fast_path = None
        return

    # The dominant scalar CSR specialization receives fresh field mappings on
    # each Python call but normally the same tensors. Identity proves dtype and
    # device; retain only metadata guards that can change in place. This makes
    # the public lazy-JIT call as cheap as the compiled runner after warm-up.
    direct = None
    if isinstance(executable, _GuardedExecutable):
        try:
            captured_source = src[executable.src_field]
            captured_weight = edge[executable.edge_field]
        except (KeyError, TypeError):
            pass
        else:
            source_version = captured_source._version
            weight_version = captured_weight._version

            def direct(current_src, current_dst, current_edge, current_params):
                del current_params
                try:
                    source = current_src[executable.src_field]
                    weight = current_edge[executable.edge_field]
                except (KeyError, TypeError):
                    return _EXECUTABLE_MISS
                if (
                    source is not captured_source
                    or weight is not captured_weight
                    # Identity plus an unchanged Torch version proves that
                    # shape/stride metadata cannot have changed. Value-only
                    # mutations take one guarded miss, then rebind without
                    # recompilation; this avoids two Python metadata queries
                    # on every asynchronous kernel submission.
                    or source._version != source_version
                    or weight._version != weight_version
                ):
                    return _EXECUTABLE_MISS
                if executable.require_dst_alias:
                    try:
                        if current_dst[executable.src_field] is not source:
                            return _EXECUTABLE_MISS
                    except (KeyError, TypeError):
                        return _EXECUTABLE_MISS
                return executable.runner(source, weight)
    elif isinstance(executable, _CSRScalarExecutable):
        namespaces = {
            "src": src, "dst": dst, "edge": edge, "param": params or {}}
        try:
            captured_inputs = tuple(
                namespaces[role][name] for role, name in executable.bindings)
        except (KeyError, TypeError):
            captured_inputs = ()
        if len(captured_inputs) == len(executable.bindings):
            captured_versions = tuple(
                None if role == "param" else value._version
                for (role, _), value in zip(
                    executable.bindings, captured_inputs)
            )

            def direct(current_src, current_dst, current_edge, current_params):
                current_namespaces = {
                    "src": current_src,
                    "dst": current_dst,
                    "edge": current_edge,
                    "param": current_params or {},
                }
                try:
                    current_inputs = tuple(
                        current_namespaces[role][name]
                        for role, name in executable.bindings
                    )
                except (KeyError, TypeError):
                    return _EXECUTABLE_MISS
                for (role, _), current, captured, version in zip(
                    executable.bindings,
                    current_inputs,
                    captured_inputs,
                    captured_versions,
                ):
                    if role == "param":
                        if not isinstance(current, (int, float)):
                            return _EXECUTABLE_MISS
                    elif current is not captured or current._version != version:
                        return _EXECUTABLE_MISS
                return executable.runner(*current_inputs)

    def fast_path(graph, src, dst, edge, params):
        current_graph = torch_graph
        if graph is not native_graph:
            if not isinstance(executable, _DynamicSnapshotExecutable):
                return False, None
            from .graph import from_native

            current_graph = from_native(graph)
        result = (
            direct(src, dst, edge, params) if direct is not None
            else executable.try_run(current_graph, src, dst, edge, params)
        )
        if result is _EXECUTABLE_MISS:
            return False, None
        executor._cache_hits += 1
        executor._last_variant = executable.variant
        kernel._cache_hits += 1
        kernel._last_variant = executable.variant
        return True, result

    kernel._torch_fast_path = fast_path


def execute(kernel, *, graph, src, dst, edge, params):
    from .graph import from_native

    native_graph = graph
    graph = from_native(native_graph)
    executor = _executor(kernel)
    try:
        return executor(
            graph=graph, src=src, dst=dst, edge=edge, **dict(params)
        )
    finally:
        _copy_diagnostics(kernel, executor)
        _install_native_fast_path(
            kernel, native_graph, graph, executor, src, dst, edge, params)


def execute_reference(kernel, **kwargs):
    from .graph import from_native

    kwargs = dict(kwargs)
    kwargs["graph"] = from_native(kwargs["graph"])
    executor = _executor(kernel)
    result = executor.reference(**kwargs)
    _copy_diagnostics(kernel, executor)
    return result


def prepare(kernel, *, graph, src, dst, edge, params):
    """Prepare a guard-free launch after validating one lazy-JIT binding."""
    from .graph import from_native

    # The ordinary public call owns capture, compilation, diagnostics, and all
    # guards. Preparation only freezes the successfully validated executable.
    kernel(graph=graph, src=src, dst=dst, edge=edge, **dict(params))
    executor = _executor(kernel)
    executable = executor._last_executable
    if not isinstance(
        executable, (_GuardedExecutable, _CSRScalarExecutable,
                     _RankedExecutable)
    ):
        raise NotImplementedError(
            "prepared MessagePassing currently supports compiler-generated "
            "scalar CSR and ranked-relation executables")
    torch_graph = from_native(graph)
    if executable.graph is not torch_graph:
        raise RuntimeError("prepared executable graph identity changed")
    # Validate once using the executable's complete topology/field guards.
    if executable.try_run(torch_graph, src, dst, edge, params) is _EXECUTABLE_MISS:
        raise ValueError("prepared MessagePassing bindings failed executable guards")
    if isinstance(executable, _GuardedExecutable):
        try:
            inputs = (
                src[executable.src_field], edge[executable.edge_field])
        except KeyError as error:
            raise ValueError(
                f"missing prepared binding {error.args[0]!r}") from error
    else:
        namespaces = {
            "src": src, "dst": dst, "edge": edge, "param": params or {}}
        try:
            inputs = tuple(
                namespaces[role][name] for role, name in executable.bindings)
        except KeyError as error:
            raise ValueError(
                f"missing prepared binding {error.args[0]!r}") from error
    runner = executable.runner
    ranked_positions = (
        executable.graph.ranked_positions()
        if isinstance(executable, _RankedExecutable) else None
    )

    def launch():
        return (
            runner(*ranked_positions, *inputs)
            if ranked_positions is not None else runner(*inputs)
        )

    launch.variant = executable.variant
    launch.graph = graph
    return launch


__all__ = ["execute", "execute_reference", "prepare"]
