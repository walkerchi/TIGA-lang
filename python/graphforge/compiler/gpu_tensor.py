"""CUDA provider for canonical GraphForge Tensor IR.

The stable input is ``gf_tensor`` MLIR.  A separate GraphForge translator
emits serialized TTIR, then the active vendor Triton distribution lowers that
provider IR to target assembly. Native GraphForge buffers launch through the
CUDA Driver runtime; the optional Torch adapter supplies only external
zero-copy storage/current-stream bindings and never executes the expression.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import subprocess
import time
from typing import TYPE_CHECKING, Mapping

from ..runtime import DeviceType

if TYPE_CHECKING:
    from ..tensor import Tensor


@dataclass
class GPUExecutable:
    compiled: object
    inputs: tuple[Tensor, ...]
    source: str
    ir: str
    semantic_hash: str
    compile_ms: float
    cache_hit: bool
    fast_math: bool
    grid: tuple[int, int, int]
    artifacts: Mapping[str, str | bytes]
    provider: object
    materialize_ms: float = 0.0
    saved_bytes: int = 0
    saved_compile_ms: float = 0.0
    checkpoint_plan: Mapping[str, object] | None = None
    uses_torch_storage: bool = False
    zero_output: bool = False

    @property
    def backend(self) -> str:
        return "cuda-ttir-triton"

    @property
    def artifact(self) -> Path:
        # Triton owns a versioned persistent cache rather than one GraphForge
        # shared-object path. Keep the diagnostics shape compatible with CPU.
        return Path("triton-cache") / self.semantic_hash[:24]

    @staticmethod
    def _storage_argument(value: Tensor):
        buffer = value._buffer
        if getattr(buffer, "_graphforge_torch_buffer", False):
            return buffer.tensor
        return buffer

    def launch(self, output: Tensor):
        if self.zero_output:
            buffer = output._buffer
            if getattr(buffer, "_graphforge_torch_buffer", False):
                buffer.tensor.zero_()
            else:
                buffer.write(bytes(output.nbytes))
        if self.driver_launcher is not None:
            arguments = tuple(
                self._storage_argument(value) for value in (*self.inputs, output)
            )
            event = self.driver_launcher.launch(self.grid, arguments)
            output.ready_event = event
            return event
        arguments = [value.to_torch() for value in self.inputs]
        arguments.append(output.to_torch())
        self.compiled.compiled[self.grid](*arguments)  # type: ignore[index,operator]
        return None

    @property
    def driver_launcher(self):
        return getattr(self.compiled, "driver_launcher", None)

    def prepare(self, output: Tensor):
        """Bind stable buffers and provider launcher once for a hot executable."""
        all_external_storage = all(
            getattr(value._buffer, "_graphforge_torch_buffer", False)
            for value in (*self.inputs, output)
        )
        if self.uses_torch_storage and all_external_storage:
            # The optional Torch adapter already owns these allocations and
            # its current stream.  Reuse the vendor launcher's pre-packed ABI
            # for a stable prepared submission; this removes ctypes dispatch
            # latency without changing TTIR, binary, allocation ownership or
            # the standalone GraphForge-Buffer path below.
            arguments = tuple(value.to_torch() for value in self.inputs)
            output_buffer = output.to_torch()
            launcher = self.compiled.compiled[self.grid]  # type: ignore[index,operator]

            def launch_external():
                if self.zero_output:
                    output_buffer.zero_()
                launcher(*arguments, output_buffer)
                return output_buffer

            return launch_external
        if self.driver_launcher is not None:
            arguments = tuple(
                self._storage_argument(value) for value in (*self.inputs, output)
            )

            prepare = getattr(self.driver_launcher, "prepare", None)
            if prepare is not None:
                submit = prepare(self.grid, arguments)

                def launch_prepared():
                    if self.zero_output:
                        buffer = output._buffer
                        if getattr(buffer, "_graphforge_torch_buffer", False):
                            buffer.tensor.zero_()
                        else:
                            buffer.write(bytes(output.nbytes))
                    return submit()

                return launch_prepared

            def launch():
                if self.zero_output:
                    buffer = output._buffer
                    if getattr(buffer, "_graphforge_torch_buffer", False):
                        buffer.tensor.zero_()
                    else:
                        buffer.write(bytes(output.nbytes))
                return self.driver_launcher.launch(self.grid, arguments)

            return launch
        arguments = tuple(value.to_torch() for value in self.inputs)
        output_buffer = output.to_torch()
        launcher = self.compiled.compiled[self.grid]  # type: ignore[index,operator]

        def launch():
            if self.zero_output:
                output_buffer.zero_()
            launcher(*arguments, output_buffer)
            return output_buffer

        return launch


@dataclass
class GPULibraryExecutable:
    library: object
    inputs: tuple[Tensor, ...]
    source: str
    ir: str
    semantic_hash: str
    compile_ms: float
    cache_hit: bool
    artifacts: Mapping[str, str | bytes]
    provider: object
    uses_torch_storage: bool
    materialize_ms: float = 0.0
    saved_bytes: int = 0
    saved_compile_ms: float = 0.0
    checkpoint_plan: Mapping[str, object] | None = None
    fast_math: bool = False
    grid: tuple[int, int, int] = (1, 1, 1)

    @property
    def backend(self) -> str:
        return "cuda-cublas-library-dispatch"

    @property
    def artifact(self) -> Path:
        return Path("cuda-library-cache") / self.semantic_hash[:24]

    @staticmethod
    def _storage_argument(value: Tensor):
        buffer = value._buffer
        if getattr(buffer, "_graphforge_torch_buffer", False):
            return buffer.tensor
        return buffer

    def launch(self, output: Tensor):
        lhs, rhs = (
            self._storage_argument(value) for value in self.inputs
        )
        target = self._storage_argument(output)
        event = self.library.launch(lhs, rhs, target)
        output.ready_event = event
        return event

    @property
    def driver_launcher(self):
        return None

    def prepare(self, output: Tensor):
        lhs, rhs = tuple(
            self._storage_argument(value) for value in self.inputs
        )
        target = self._storage_argument(output)
        stream = self.library.current_stream()
        submit = self.library.prepare(lhs, rhs, target, stream=stream)

        def launch():
            submit()
            return target

        return launch


@dataclass
class GPULoopExecutable:
    """Runtime realization of one compiler-captured control repeat region."""

    initial: Tensor
    iterations: int
    first: tuple[GPUExecutable | GPULibraryExecutable, Tensor] | None
    odd: tuple[GPUExecutable | GPULibraryExecutable, Tensor] | None
    even: tuple[GPUExecutable | GPULibraryExecutable, Tensor] | None
    ir: str
    semantic_hash: str
    compile_ms: float
    cache_hit: bool
    artifacts: Mapping[str, str | bytes]
    uses_torch_storage: bool
    materialize_ms: float = 0.0
    saved_bytes: int = 0
    saved_compile_ms: float = 0.0
    checkpoint_plan: Mapping[str, object] | None = None
    fast_math: bool = False
    aliases_output: bool = True

    @property
    def backend(self) -> str:
        return "cuda-control-loop-ttir-triton"

    @property
    def source(self) -> str:
        return str(self.artifacts.get("ttir", self.ir))

    @property
    def artifact(self) -> Path:
        return Path("control-loop-cache") / self.semantic_hash[:24]

    @staticmethod
    def _launch(pair):
        executable, target = pair
        completion = executable.launch(target)
        target.ready_event = completion
        return completion, target

    def launch(self, output: Tensor):
        if self.iterations == 0:
            self.initial.realize()
            output._buffer = self.initial._buffer
            output.ready_event = self.initial.ready_event
            return output.ready_event
        assert self.first is not None
        completion, final = self._launch(self.first)
        for iteration in range(1, self.iterations):
            pair = self.even if iteration % 2 else self.odd
            assert pair is not None
            completion, final = self._launch(pair)
        output._buffer = final._buffer
        output.ready_event = final.ready_event
        return completion

    def prepare(self, output: Tensor):
        if self.iterations == 0:
            def launch_zero():
                output._buffer = self.initial._buffer
                output.ready_event = self.initial.ready_event
                return output.ready_event

            return launch_zero
        assert self.first is not None and self.odd is not None and self.even is not None
        first_executable, first_target = self.first
        odd_executable, odd_target = self.odd
        even_executable, even_target = self.even
        first_launch = first_executable.prepare(first_target)
        odd_launch = odd_executable.prepare(odd_target)
        even_launch = even_executable.prepare(even_target)

        def launch():
            completion = first_launch()
            final = first_target
            for iteration in range(1, self.iterations):
                if iteration % 2:
                    completion = even_launch()
                    final = even_target
                else:
                    completion = odd_launch()
                    final = odd_target
            output._buffer = final._buffer
            output.ready_event = getattr(final, "ready_event", None)
            return completion

        return launch


_EXECUTABLES: dict[tuple[object, ...], GPUExecutable] = {}
_LIBRARY_EXECUTABLES: dict[tuple[object, ...], GPULibraryExecutable] = {}


def _clone_loop_body(output: Tensor, state: Tensor, replacement: Tensor) -> Tensor:
    from ..tensor.core import Tensor, _Expr

    memo: dict[int, Tensor] = {id(state): replacement}

    def visit(value: Tensor) -> Tensor:
        found = memo.get(id(value))
        if found is not None:
            return found
        expression = value._expr
        if expression is None:
            memo[id(value)] = value
            return value
        if expression.region is not None:
            raise NotImplementedError("nested GPU control regions are unsupported")
        cloned = Tensor(
            value.shape,
            dtype=value.dtype,
            device=value.device,
            requires_grad=False,
            expression=_Expr(
                expression.op,
                tuple(visit(operand) for operand in expression.operands),
                expression.attrs,
            ),
            version=value.version,
        )
        memo[id(value)] = cloned
        return cloned

    return visit(output)


def _compile_repeat(output: Tensor) -> GPULoopExecutable:
    from ..runtime import Buffer
    from ..tensor.core import Tensor
    from .tensor_mlir import tensor_mlir

    expression = output._expr
    assert expression is not None and expression.op == "repeat"
    region = expression.region
    if region is None or not region.arguments:
        raise RuntimeError("gf_control.repeat is missing its captured body")
    iterations = int(expression.attr("iterations"))
    initial = expression.operands[0]
    if initial._buffer is None:
        initial.realize()
    captures = expression.operands[1:]
    uses_torch_storage = any(
        getattr(value._buffer, "_graphforge_torch_buffer", False)
        for value in (initial, *captures)
    )

    def allocate_state() -> Tensor:
        if uses_torch_storage:
            from ..interop.torch.tensor import allocate_buffer

            buffer = allocate_buffer(output.shape, output.dtype, output.device)
        else:
            buffer = Buffer(output.nbytes, device=output.device)
        return Tensor(
            output.shape, dtype=output.dtype, device=output.device,
            buffer=buffer, requires_grad=False,
        )

    canonical_ir = tensor_mlir(output)
    semantic_hash = hashlib.sha256(canonical_ir.encode()).hexdigest()
    if iterations == 0:
        return GPULoopExecutable(
            initial, iterations, None, None, None, canonical_ir,
            semantic_hash, 0.0, True, {"gf_control": canonical_ir},
            uses_torch_storage,
        )

    first_target = allocate_state()
    second_target = allocate_state()
    state = region.arguments[0]
    first_body = _clone_loop_body(region.output, state, initial)
    odd_body = _clone_loop_body(region.output, state, second_target)
    even_body = _clone_loop_body(region.output, state, first_target)
    first_executable = compile_tensor(first_body, _control_body=True)
    odd_executable = compile_tensor(odd_body, _control_body=True)
    even_executable = compile_tensor(even_body, _control_body=True)
    body_artifacts = dict(first_executable.artifacts)
    artifacts: dict[str, str | bytes] = {
        "gf_control": canonical_ir,
        "provider_ttir": first_executable.source,
        **{f"body.{name}": value for name, value in body_artifacts.items()},
    }
    ttir = body_artifacts.get("ttir")
    if isinstance(ttir, (str, bytes)):
        artifacts["ttir"] = ttir
    return GPULoopExecutable(
        initial=initial,
        iterations=iterations,
        first=(first_executable, first_target),
        odd=(odd_executable, first_target),
        even=(even_executable, second_target),
        ir=canonical_ir,
        semantic_hash=semantic_hash,
        compile_ms=sum(
            executable.compile_ms for executable in (
                first_executable, odd_executable, even_executable)
        ),
        cache_hit=all(
            executable.cache_hit for executable in (
                first_executable, odd_executable, even_executable)
        ),
        artifacts=artifacts,
        uses_torch_storage=uses_torch_storage,
    )


def _topological_sort(output: Tensor) -> list[Tensor]:
    ordered: list[Tensor] = []
    visited: set[int] = set()

    def visit(value: Tensor) -> None:
        if id(value) in visited:
            return
        visited.add(id(value))
        if value._expr is not None:
            for operand in value._expr.operands:
                visit(operand)
        ordered.append(value)

    visit(output)
    return ordered


def _physicalize_storage(
    output: Tensor,
) -> tuple[Tensor, float, int, float, Mapping[str, object]]:
    """Legalize relation indices and apply the native checkpoint plan."""
    from ..tensor.core import Tensor, _Expr, from_torch
    from .native import _load_native, checkpoint_plan

    candidates = tuple(
        value
        for value in _topological_sort(output)
        if value._expr is not None
        and value._expr.op == "checkpoint_candidate"
    )
    budget_text = os.environ.get("GRAPHFORGE_CHECKPOINT_BUDGET_BYTES")
    spill_budget_text = os.environ.get(
        "GRAPHFORGE_CHECKPOINT_SPILL_BUDGET_BYTES"
    )
    try:
        memory_budget_bytes = -1 if budget_text is None else int(budget_text)
    except ValueError as error:
        raise ValueError(
            "GRAPHFORGE_CHECKPOINT_BUDGET_BYTES must be -1 or non-negative"
        ) from error
    if memory_budget_bytes < -1:
        raise ValueError(
            "GRAPHFORGE_CHECKPOINT_BUDGET_BYTES must be -1 or non-negative"
        )
    try:
        spill_budget_bytes = (
            0 if spill_budget_text is None else int(spill_budget_text)
        )
    except ValueError as error:
        raise ValueError(
            "GRAPHFORGE_CHECKPOINT_SPILL_BUDGET_BYTES must be -1 or non-negative"
        ) from error
    if spill_budget_bytes < -1:
        raise ValueError(
            "GRAPHFORGE_CHECKPOINT_SPILL_BUDGET_BYTES must be -1 or non-negative"
        )
    if candidates:
        native_load_started = time.perf_counter_ns()
        _load_native()
        native_load_ms = (time.perf_counter_ns() - native_load_started) / 1e6
        plan_started = time.perf_counter_ns()
        native_plan = checkpoint_plan(
            output,
            memory_budget_bytes=memory_budget_bytes,
            spill_budget_bytes=spill_budget_bytes,
        )
        decisions = native_plan["decisions"]
        tiers = native_plan["tiers"]
        planned_saved_bytes = native_plan["saved_bytes"]
        planned_ir = native_plan["ir"]
        checkpoint_plan_ms = (time.perf_counter_ns() - plan_started) / 1e6
    else:
        decisions, planned_saved_bytes, planned_ir = (), 0, ""
        tiers = ()
        native_plan = {
            "spilled_bytes": 0,
            "peak_live_bytes": 0,
            "peak_spill_bytes": 0,
            "recompute_costs": (),
            "live_intervals": (),
        }
        native_load_ms = 0.0
        checkpoint_plan_ms = 0.0
    if len(decisions) != len(candidates):
        raise RuntimeError(
            "native checkpoint plan does not match the captured Tensor DAG"
        )
    selected = {
        id(candidate): (decision, tier)
        for candidate, decision, tier in zip(
            candidates, decisions, tiers, strict=True
        )
    }

    memo: dict[int, Tensor] = {}
    materialize_ms = 0.0
    saved_bytes = 0
    saved_compile_ms = 0.0
    spilled_bytes = 0
    spill_transfer_ms = 0.0
    delegated_checkpoint_candidates = 0

    # The flat pointwise translator consumes physical view descriptors at its
    # ABI, whereas structured fusions consume reshape/broadcast operations as
    # proof-carrying IR.  Cut zero-copy views only when the complete DAG is a
    # pointwise/view program; doing so for a scan/reduction/matmul would erase
    # the structure that selects its specialized lowering.
    pointwise_view_ops = {
        "reshape", "permute", "broadcast", "add", "mul", "div", "neg",
        "exp", "sqrt", "conj", "not_equal",
    }
    expression_ops = {
        item._expr.op
        for item in _topological_sort(output)
        if item._expr is not None
    }
    bind_pointwise_views = expression_ops <= pointwise_view_ops

    def candidate_count(value: Tensor) -> int:
        return sum(
            item._expr is not None
            and item._expr.op == "checkpoint_candidate"
            for item in _topological_sort(value)
        )

    def visit(value: Tensor) -> Tensor:
        nonlocal materialize_ms, saved_bytes, saved_compile_ms
        nonlocal spilled_bytes, spill_transfer_ms
        nonlocal delegated_checkpoint_candidates
        found = memo.get(id(value))
        if found is not None:
            return found
        expression = value._expr
        if expression is None:
            memo[id(value)] = value
            return value
        if (value is not output and value._buffer is not None
                and expression.op not in {"reshape", "permute", "broadcast"}):
            # Keep the semantic/autograd DAG intact, but bind any explicitly
            # realized compute intermediate as a zero-copy executable ABI
            # leaf. Views are excluded: they can alias storage eagerly while
            # still carrying structural shape proofs required by scan/matmul
            # fusion. This is the CUDA counterpart of the CPU materialization
            # barrier and is required when interior work is completed before
            # a boundary consumer is compiled.
            rewritten = Tensor(
                value.shape, dtype=value.dtype, device=value.device,
                buffer=value._buffer, offset=value.offset,
                strides=value.strides, requires_grad=False,
                version=value.version, ready_event=value.ready_event,
            )
            memo[id(value)] = rewritten
            return rewritten
        if (expression.op == "distributed_halo_snapshot"
                and value._buffer is not None):
            # Forward communication already installed owned||ghost storage.
            # Keep the semantic edge for VJP, but bind the physical snapshot
            # as an ordinary device ABI leaf for local kernel compilation.
            rewritten = Tensor(
                value.shape, dtype=value.dtype, device=value.device,
                buffer=value._buffer, offset=value.offset,
                strides=value.strides, requires_grad=False,
                version=value.version, ready_event=value.ready_event,
            )
            memo[id(value)] = rewritten
            return rewritten
        if (bind_pointwise_views and value is not output
                and value._buffer is not None
                and expression.op in {"reshape", "permute", "broadcast"}):
            rewritten = Tensor(
                value.shape, dtype=value.dtype, device=value.device,
                buffer=value._buffer, offset=value.offset,
                strides=value.strides, requires_grad=False,
                version=value.version, ready_event=value.ready_event,
            )
            memo[id(value)] = rewritten
            return rewritten
        if expression.op == "distributed_halo_reverse":
            from ..distributed.transport import (
                DeviceBufferTransport, current_distributed_runtime,
                reverse_halo_device, reverse_halo_values,
            )
            from ..tensor import tensor

            runtime = current_distributed_runtime()
            if runtime is None:
                raise RuntimeError(
                    "distributed halo VJP requires an active runtime")
            cotangent = expression.operands[0]
            if isinstance(runtime.transport, DeviceBufferTransport):
                rewritten = reverse_halo_device(
                    cotangent, expression.attr("halo"), runtime.transport)
                memo[id(value)] = rewritten
                return rewritten
            cotangent.realize()
            combined = cotangent._read_flat()
            rows = int(expression.attr("owned_rows"))
            row_width = cotangent.numel // cotangent.shape[0]
            owned = reverse_halo_values(
                expression.attr("halo"), combined, row_width=row_width,
                transport=runtime.transport,
            )
            rewritten = tensor(
                owned, dtype=value.dtype, device=value.device).reshape(
                    (rows, *value.shape[1:]))
            memo[id(value)] = rewritten
            return rewritten
        # The current TTIR translators each own one reduction skeleton. Split
        # nested reductions at physical barriers instead of rejecting a valid
        # Tensor DAG or cloning handwritten compound kernels. The root stays
        # in this executable; every nested reduction becomes a device-resident
        # ABI leaf for its consumer. This also makes the dependency boundary
        # available to bundle/pipeline planning.
        if value is not output and expression.op in {
            "sum", "segment_sum", "csr_segment_product",
            "csr_segment_product_vjp",
        }:
            was_materialized = value._buffer is not None
            if not was_materialized:
                delegated_checkpoint_candidates += candidate_count(value)
            value.realize()
            if not was_materialized:
                execution = value.execution or {}
                materialize_ms += float(execution.get("launch_ms", 0.0))
                saved_compile_ms += float(execution.get("compile_ms", 0.0))
                saved_bytes += int(execution.get("saved_bytes", 0))
            rewritten = Tensor(
                value.shape, dtype=value.dtype, device=value.device,
                buffer=value._buffer, offset=value.offset,
                strides=value.strides, requires_grad=False,
                version=value.version, ready_event=value.ready_event,
            )
            memo[id(value)] = rewritten
            return rewritten
        if value is output and expression.op in {
            "csr_segment_product", "csr_segment_product_vjp",
        }:
            # Product reductions have a dedicated row-tiled TTIR skeleton.
            # Make arbitrary message/cotangent producers explicit ABI leaves;
            # topology leaves remain untouched and no operation-specific
            # handwritten MessagePassing kernel enters the runtime.
            direct_operands = []
            for operand in expression.operands:
                if operand._expr is not None:
                    was_materialized = operand._buffer is not None
                    operand.realize()
                    if not was_materialized:
                        execution = operand.execution or {}
                        materialize_ms += float(
                            execution.get("launch_ms", 0.0))
                        saved_compile_ms += float(
                            execution.get("compile_ms", 0.0))
                        saved_bytes += int(execution.get("saved_bytes", 0))
                    operand = Tensor(
                        operand.shape, dtype=operand.dtype,
                        device=operand.device, buffer=operand._buffer,
                        offset=operand.offset, strides=operand.strides,
                        requires_grad=False, version=operand.version,
                        ready_event=operand.ready_event,
                    )
                direct_operands.append(operand)
            rewritten = Tensor(
                value.shape, dtype=value.dtype, device=value.device,
                requires_grad=value.requires_grad,
                expression=_Expr(
                    expression.op, tuple(direct_operands), expression.attrs),
                version=value.version,
            )
            memo[id(value)] = rewritten
            return rewritten
        if value is output and expression.op == "segment_sum":
            direct_operands = []
            for operand in expression.operands:
                if operand._expr is not None:
                    was_materialized = operand._buffer is not None
                    if not was_materialized:
                        delegated_checkpoint_candidates += candidate_count(
                            operand)
                    operand.realize()
                    if not was_materialized:
                        execution = operand.execution or {}
                        materialize_ms += float(
                            execution.get("launch_ms", 0.0))
                        saved_compile_ms += float(
                            execution.get("compile_ms", 0.0))
                        saved_bytes += int(
                            execution.get("saved_bytes", 0))
                    operand = Tensor(
                        operand.shape, dtype=operand.dtype,
                        device=operand.device, buffer=operand._buffer,
                        offset=operand.offset, strides=operand.strides,
                        requires_grad=False, version=operand.version,
                        ready_event=operand.ready_event,
                    )
                direct_operands.append(operand)
            rewritten = Tensor(
                value.shape, dtype=value.dtype, device=value.device,
                requires_grad=value.requires_grad,
                expression=_Expr(
                    expression.op, tuple(direct_operands), expression.attrs),
                version=value.version,
            )
            memo[id(value)] = rewritten
            return rewritten
        if value is output and expression.op == "sum" and tuple(
                expression.attr("axes")) == (0,) and len(value.shape) >= 1:
            # Axis-zero reduction traverses the leading physical dimension.
            # Materialize its producer once so the reduction emitter can bind
            # an arbitrary pointwise/gather DAG as a strided ABI leaf. This is
            # a generic reduction barrier, not a workload-specific kernel.
            source = expression.operands[0]
            if source._expr is not None:
                was_materialized = source._buffer is not None
                if not was_materialized:
                    delegated_checkpoint_candidates += candidate_count(source)
                source.realize()
                if not was_materialized:
                    execution = source.execution or {}
                    materialize_ms += float(execution.get("launch_ms", 0.0))
                    saved_compile_ms += float(execution.get("compile_ms", 0.0))
                    saved_bytes += int(execution.get("saved_bytes", 0))
                source = Tensor(
                    source.shape, dtype=source.dtype, device=source.device,
                    buffer=source._buffer, offset=source.offset,
                    strides=source.strides, requires_grad=False,
                    version=source.version, ready_event=source.ready_event,
                )
            rewritten = Tensor(
                value.shape, dtype=value.dtype, device=value.device,
                requires_grad=value.requires_grad,
                expression=_Expr(
                    expression.op, (source,), expression.attrs),
                version=value.version,
            )
            memo[id(value)] = rewritten
            return rewritten
        if expression.op in {"checkpoint", "checkpoint_candidate"}:
            source = expression.operands[0]
            if expression.op == "checkpoint":
                save, tier = True, "device"
            else:
                save, tier = selected.get(id(value), (False, "recompute"))
            if not save:
                rewritten = visit(source)
                memo[id(value)] = rewritten
                return rewritten
            was_materialized = source._buffer is not None
            if not was_materialized:
                delegated_checkpoint_candidates += candidate_count(source)
            source.realize()
            if not was_materialized:
                execution = source.execution or {}
                materialize_ms += float(execution.get("launch_ms", 0.0))
                saved_compile_ms += float(execution.get("compile_ms", 0.0))
                saved_bytes += int(execution.get("saved_bytes", 0))
            if tier == "host-pinned":
                if source.device.type != DeviceType.CUDA:
                    raise RuntimeError(
                        "host-pinned checkpoint spill currently requires CUDA"
                    )
                transfer_started = time.perf_counter_ns()
                if getattr(source._buffer, "_graphforge_torch_buffer", False):
                    from ..interop.torch.provider import cuda_pinned_roundtrip

                    rewritten = cuda_pinned_roundtrip(source)
                else:
                    from ..runtime import Buffer, Stream

                    host = Buffer.pinned_host(source.nbytes)
                    restored = Buffer(source.nbytes, device=source.device)
                    stream = Stream(source.device)
                    source._buffer.copy_to(host, stream=stream).wait()
                    host.copy_to(restored, stream=stream).wait()
                    rewritten = Tensor(
                        source.shape, dtype=source.dtype, device=source.device,
                        buffer=restored, requires_grad=False,
                        version=source.version,
                    )
                spill_transfer_ms += (
                    time.perf_counter_ns() - transfer_started
                ) / 1e6
                spilled_bytes += source.nbytes
                memo[id(value)] = rewritten
                return rewritten
            saved_bytes += source.nbytes
            # A leaf alias exposes saved storage to the backward ABI while the
            # logical DAG continues to retain the checkpoint decision.
            rewritten = Tensor(
                source.shape,
                dtype=source.dtype,
                device=source.device,
                buffer=source._buffer,
                offset=source.offset,
                strides=source.strides,
                requires_grad=False,
                version=source.version,
            )
            memo[id(value)] = rewritten
            return rewritten
        if expression.op == "gather":
            # A gather performs an arbitrary cross-lane read. Its source and
            # index therefore have to be physical ABI inputs; a pointwise
            # producer cannot be substituted into the current lane. This
            # generic barrier also covers VJPs that gather from a previously
            # partitioned/scattered distributed result.
            direct_operands = []
            for operand in expression.operands:
                if operand._expr is not None:
                    was_materialized = operand._buffer is not None
                    operand.realize()
                    if not was_materialized:
                        execution = operand.execution or {}
                        materialize_ms += float(
                            execution.get("launch_ms", 0.0))
                        saved_compile_ms += float(
                            execution.get("compile_ms", 0.0))
                        saved_bytes += int(execution.get("saved_bytes", 0))
                    operand = Tensor(
                        operand.shape, dtype=operand.dtype,
                        device=operand.device, buffer=operand._buffer,
                        offset=operand.offset, strides=operand.strides,
                        requires_grad=False, version=operand.version,
                        ready_event=operand.ready_event,
                    )
                direct_operands.append(operand)
            rewritten = Tensor(
                value.shape, dtype=value.dtype, device=value.device,
                requires_grad=value.requires_grad,
                expression=_Expr(
                    expression.op, tuple(direct_operands), expression.attrs),
                version=value.version,
            )
            memo[id(value)] = rewritten
            return rewritten
        operands = tuple(visit(item) for item in expression.operands)
        if expression.op in {"csr_expand_rows", "csr_segment_sum"}:
            source, row_ptr = operands
            # A node-domain broadcast of one scalar is invariant across the
            # relation. Canonicalize it directly to the edge domain instead of
            # materializing and reading a destination index.
            if expression.op == "csr_expand_rows":
                scalar = source
                while (
                    scalar._expr is not None
                    and scalar._expr.op in {"broadcast", "reshape"}
                    and len(scalar._expr.operands) == 1
                ):
                    scalar = scalar._expr.operands[0]
                if scalar.shape == ():
                    rewritten = scalar.reshape(1).broadcast_to(value.shape)
                    memo[id(value)] = rewritten
                    return rewritten
            destination = getattr(row_ptr, "_cuda_destination_index_cache", None)
            if destination is None:
                started = time.perf_counter_ns()
                if getattr(row_ptr._buffer, "_graphforge_torch_buffer", False):
                    from ..interop.torch.provider import cuda_destination_index

                    destination = cuda_destination_index(
                        row_ptr,
                        source.shape[0] if expression.op == "csr_expand_rows"
                        else int(expression.attr("num_rows")))
                else:
                    from ..tensor import tensor

                    rows = tuple(int(item) for item in row_ptr.tolist())
                    physical = [
                        row
                        for row, (begin, end) in enumerate(
                            zip(rows, rows[1:]))
                        for _ in range(begin, end)
                    ]
                    destination = tensor(
                        physical, dtype=row_ptr.dtype, device=row_ptr.device)
                materialize_ms += (time.perf_counter_ns() - started) / 1e6
                row_ptr._cuda_destination_index_cache = destination
            if expression.op == "csr_expand_rows":
                # Relation expansion is itself an indexed read. If its node
                # producer was already a gather (for example a partitioned
                # destination cotangent), cut between the two gathers: one
                # Triton program cannot satisfy an arbitrary second-level
                # index from values held only in the current lane.
                if source._expr is not None:
                    was_materialized = source._buffer is not None
                    source.realize()
                    if not was_materialized:
                        execution = source.execution or {}
                        materialize_ms += float(
                            execution.get("launch_ms", 0.0))
                        saved_compile_ms += float(
                            execution.get("compile_ms", 0.0))
                        saved_bytes += int(execution.get("saved_bytes", 0))
                    source = Tensor(
                        source.shape, dtype=source.dtype, device=source.device,
                        buffer=source._buffer, offset=source.offset,
                        strides=source.strides, requires_grad=False,
                        version=source.version, ready_event=source.ready_event,
                    )
                rewritten = source.gather(destination)
            else:
                if source._expr is not None:
                    was_materialized = source._buffer is not None
                    if not was_materialized:
                        delegated_checkpoint_candidates += candidate_count(
                            source)
                    source.realize()
                    if not was_materialized:
                        execution = source.execution or {}
                        materialize_ms += float(
                            execution.get("launch_ms", 0.0))
                        saved_compile_ms += float(
                            execution.get("compile_ms", 0.0))
                        saved_bytes += int(
                            execution.get("saved_bytes", 0))
                    source = Tensor(
                        source.shape, dtype=source.dtype, device=source.device,
                        buffer=source._buffer, offset=source.offset,
                        strides=source.strides, requires_grad=False,
                        version=source.version, ready_event=source.ready_event,
                    )
                rewritten = source.segment_sum(
                    destination, int(expression.attr("num_rows")))
                if value is not output:
                    rewritten.realize()
                    rewritten = Tensor(
                        rewritten.shape, dtype=rewritten.dtype,
                        device=rewritten.device, buffer=rewritten._buffer,
                        offset=rewritten.offset, strides=rewritten.strides,
                        requires_grad=False, version=rewritten.version,
                        ready_event=rewritten.ready_event,
                    )
        else:
            rewritten = Tensor(
                value.shape,
                dtype=value.dtype,
                device=value.device,
                requires_grad=value.requires_grad,
                expression=_Expr(expression.op, operands, expression.attrs),
                version=value.version,
            )
        memo[id(value)] = rewritten
        return rewritten

    physical = visit(output)
    if (saved_bytes < planned_saved_bytes and
            delegated_checkpoint_candidates == 0):
        raise RuntimeError(
            "checkpoint physicalization saved fewer bytes than the MLIR plan: "
            f"actual={saved_bytes}, planned={planned_saved_bytes}, "
            f"candidates={len(candidates)}"
        )
    plan = {
        "candidate_count": len(candidates),
        "delegated_candidate_count": delegated_checkpoint_candidates,
        "decisions": decisions,
        "memory_budget_bytes": memory_budget_bytes,
        "spill_budget_bytes": spill_budget_bytes,
        "planned_saved_bytes": planned_saved_bytes,
        "planned_spilled_bytes": int(native_plan["spilled_bytes"]),
        "actual_spilled_bytes": spilled_bytes,
        "spill_transfer_ms": spill_transfer_ms,
        "tiers": tuple(tiers),
        "recompute_costs": tuple(native_plan["recompute_costs"]),
        "live_intervals": tuple(native_plan["live_intervals"]),
        "peak_live_bytes": int(native_plan["peak_live_bytes"]),
        "peak_spill_bytes": int(native_plan["peak_spill_bytes"]),
        "planning_ms": checkpoint_plan_ms,
        "native_load_ms": native_load_ms,
        "ir": planned_ir,
    }
    return physical, materialize_ms, saved_bytes, saved_compile_ms, plan


def _find_translate() -> Path:
    from .toolchain import find_gf_translate

    executable = find_gf_translate()
    if executable is None:
        raise RuntimeError(
            "gf-translate is unavailable; set GRAPHFORGE_TRANSLATE or install "
            "a native GraphForge wheel"
        )
    return Path(executable)


def _translator_fingerprint() -> tuple[str, int, int]:
    """Invalidate in-process binaries when the provider translator changes."""
    executable = _find_translate().resolve()
    status = executable.stat()
    return str(executable), status.st_mtime_ns, status.st_size


def _translate(module: str) -> tuple[str, str, int, int, int]:
    completed = subprocess.run(
        [str(_find_translate()), "--gf-tensor-to-ttir"],
        input=module,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError("gf_tensor GPU translation failed:\n" + completed.stderr)
    match = re.match(
        r"// graphforge\.tensor entry=(\S+) block_rows=(\d+) "
        r"block_elements=(\d+) num_warps=(\d+) abi=[A-Za-z0-9_,]+\n",
        completed.stdout,
    )
    if match is None:
        raise RuntimeError("gf_tensor TTIR has no valid launch manifest")
    return (
        completed.stdout, match.group(1), int(match.group(2)),
        int(match.group(3)), int(match.group(4)),
    )


def compile_tensor(
    output: Tensor,
    *,
    _control_body: bool = False,
) -> GPUExecutable | GPULibraryExecutable | GPULoopExecutable:
    from ..runtime import DeviceType
    from ..codegen.ttir import compile_ttir
    from .tensor_mlir import tensor_mlir

    if output.device.type != DeviceType.CUDA:
        raise ValueError("GPU Tensor compiler requires a CUDA output")
    if output._expr is not None and output._expr.op == "repeat":
        return _compile_repeat(output)
    if output.dtype.name not in {
        "float16", "float32", "float64", "complex64", "complex128"
    }:
        raise NotImplementedError(
            "CUDA Tensor codegen supports FP16/FP32/FP64 and complex64/complex128"
        )
    if _control_body:
        # A loop-carried dependency may not be checkpointed outside the loop.
        # The control planner will later hoist only values proven invariant;
        # until then, keeping the complete body is the correctness-first rule.
        physical_output = output
        materialize_ms = saved_bytes = saved_compile_ms = 0
        checkpoint_plan = {
            "candidate_count": 0,
            "delegated_candidate_count": 0,
            "decisions": (),
            "tiers": (),
            "control_body": True,
        }
    else:
        physical_output, materialize_ms, saved_bytes, saved_compile_ms, checkpoint_plan = \
            _physicalize_storage(output)
    nodes = _topological_sort(physical_output)
    inputs = tuple(value for value in nodes if value._expr is None)
    if any(value.device != output.device for value in inputs):
        raise ValueError("all Tensor inputs must use the same CUDA device")

    canonical_ir = tensor_mlir(physical_output)
    semantic_hash = hashlib.sha256(canonical_ir.encode()).hexdigest()

    uses_torch_storage = any(
        getattr(value._buffer, "_graphforge_torch_buffer", False)
        for value in inputs
    )
    matmul_provider = os.environ.get(
        "GRAPHFORGE_MATMUL_PROVIDER", "auto").lower()
    if matmul_provider not in {"auto", "library", "ttir"}:
        raise ValueError(
            "GRAPHFORGE_MATMUL_PROVIDER must be auto, library, or ttir")
    from ..tensor.core import _contiguous_strides

    expression = physical_output._expr
    library_eligible = (
        expression is not None
        and expression.op == "matmul"
        and physical_output.dtype.name == "float16"
        and len(inputs) == 2
        and all(
            operand is input_value
            for operand, input_value in zip(expression.operands, inputs)
        )
        and all(
            value.offset == 0
            and value.strides == _contiguous_strides(value.shape)
            for value in inputs
        )
    )
    if matmul_provider == "library" and not library_eligible:
        raise NotImplementedError(
            "cuBLAS dispatch currently requires direct contiguous rank-2 FP16 matmul")
    if library_eligible and matmul_provider != "ttir":
        from ..runtime.cuda_blas import CUDABlasMatmul, CUDABlasProvider

        provider = CUDABlasProvider()
        key = (semantic_hash, provider.cache_key(), uses_torch_storage)
        cached = _LIBRARY_EXECUTABLES.get(key)
        if cached is not None:
            return GPULibraryExecutable(
                cached.library, inputs, cached.source, canonical_ir,
                semantic_hash, 0.0, True, cached.artifacts, provider,
                uses_torch_storage, materialize_ms, saved_bytes,
                saved_compile_ms, checkpoint_plan,
            )
        stream_provider = None
        if uses_torch_storage:
            from ..interop.torch.provider import current_cuda_stream

            stream_provider = lambda: current_cuda_stream(output.device)
        started = time.perf_counter_ns()
        rows, inner = inputs[0].shape
        columns = inputs[1].shape[1]
        library = CUDABlasMatmul(
            device=output.device, m=rows, n=columns, k=inner,
            stream_provider=stream_provider,
        )
        compile_ms = (time.perf_counter_ns() - started) / 1e6
        executable = GPULibraryExecutable(
            library, inputs, "library.call @cublasLtMatmul", canonical_ir,
            semantic_hash, compile_ms, False,
            {
                "gf_tensor": canonical_ir,
                "library_dispatch": (
                    "row-major FP16 matmul -> cublasLtMatmul; "
                    "FP32 accumulation; current GraphForge/Torch stream"
                ),
            },
            provider, uses_torch_storage, materialize_ms, saved_bytes,
            saved_compile_ms, checkpoint_plan,
        )
        _LIBRARY_EXECUTABLES[key] = executable
        return executable

    # Provider identity includes vendor backend, architecture, warp size and
    # compiler revision. It prevents reusing CUDA binaries on HIP or another
    # target even when canonical semantics are identical.
    from ..codegen.provider import triton_provider_identity
    from ..runtime import cuda_compute_capability
    from triton.backends.compiler import GPUTarget

    major, minor, warp_size = cuda_compute_capability(output.device)
    target = GPUTarget("cuda", major * 10 + minor, warp_size)
    provider = triton_provider_identity(target)
    key_prefix = (
        semantic_hash,
        provider.cache_key(),
        uses_torch_storage,
        _translator_fingerprint(),
    )
    cached = next(
        (
            executable
            for key, executable in _EXECUTABLES.items()
            if key[:4] == key_prefix
        ),
        None,
    )
    if cached is not None:
        return GPUExecutable(
            cached.compiled,
            inputs,
            cached.source,
            canonical_ir,
            semantic_hash,
            0.0,
            True,
            False,
            cached.grid,
            cached.artifacts,
            cached.provider,
            materialize_ms,
            saved_bytes,
            saved_compile_ms,
            checkpoint_plan,
            cached.uses_torch_storage,
            cached.zero_output,
        )

    start = time.perf_counter_ns()
    ttir, entry, block_rows, block_elements, num_warps = _translate(canonical_ir)
    key = (*key_prefix, num_warps)
    stream_provider = None
    if uses_torch_storage:
        from ..interop.torch.provider import current_cuda_stream

        stream_provider = lambda: current_cuda_stream(output.device)
    result = compile_ttir(
        ttir,
        target=target,
        options={
            "num_warps": num_warps,
            # The scan-contract loop carries its complete recurrent state in
            # registers. Extra software-pipeline stages only lengthen live
            # ranges on current Triton backends; two stages consistently keep
            # occupancy above the official FLA recurrent/chunk baselines.
            "num_stages": (
                4 if entry == "gf_tensor_matmul"
                else 2 if entry == "gf_tensor_scan_contract"
                else 3
            ),
        },
        launcher="driver",
        device=str(output.device),
        stream_provider=stream_provider,
    )
    compile_ms = (time.perf_counter_ns() - start) / 1e6
    if entry == "gf_tensor_pointwise":
        launch_extent = physical_output.numel
    elif entry == "gf_tensor_cumsum":
        assert physical_output._expr is not None
        axis = int(physical_output._expr.attr("axis"))
        launch_extent = physical_output.numel // physical_output.shape[axis]
    elif entry == "gf_tensor_scan_contract":
        # The structurally matched provider owns one K<=16 recurrent state
        # tile for each logical lane and 16-wide output-value tile.
        assert physical_output.ndim == 3
        value_width = physical_output.shape[2]
        launch_extent = (
            physical_output.shape[0] * (value_width + 15) // 16
        )
    elif entry == "gf_tensor_csr_product_vjp" and (
        bool(physical_output._expr.attr("uniform_degree"))
        and int(physical_output._expr.attr("max_degree")) <= 32
    ):
        feature_count = physical_output.numel // physical_output.shape[0]
        launch_extent = (
            int(physical_output._expr.attr("num_rows")) * feature_count)
    elif entry in {
        "gf_tensor_segment_sum", "gf_tensor_fused_reduce",
        "gf_tensor_csr_product", "gf_tensor_csr_product_vjp",
    }:
        launch_extent = physical_output.numel
    elif entry == "gf_tensor_matmul":
        launch_extent = (
            (physical_output.shape[0] + block_rows - 1) // block_rows
            * (physical_output.shape[1] + block_elements - 1) // block_elements
        )
        block_rows = 1
    elif entry == "gf_tensor_csr_euclidean_distance_sum_vjp":
        assert physical_output._expr is not None
        launch_extent = physical_output._expr.operands[3].numel
    else:
        launch_extent = physical_output.shape[0]
    executable = GPUExecutable(
        result,
        inputs,
        ttir,
        canonical_ir,
        semantic_hash,
        compile_ms,
        False,
        False,
        ((launch_extent + block_rows - 1) // block_rows, 1, 1),
        result.artifacts,
        result.provider,
        materialize_ms,
        saved_bytes,
        saved_compile_ms,
        checkpoint_plan,
        uses_torch_storage,
        entry == "gf_tensor_csr_euclidean_distance_sum_vjp",
    )
    _EXECUTABLES[key] = executable
    return executable


__all__ = ["GPUExecutable", "GPULoopExecutable", "compile_tensor"]
