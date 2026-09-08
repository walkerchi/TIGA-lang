"""Versioned serialized-TTIR compilation boundary.

Tiga's stable IR remains ``gf.kernel``.  A Triton provider plugin may
lower it to TTIR text and pass that text through this boundary.  Keeping the
input serialized avoids linking Tiga's MLIR libraries and a vendor
Triton's potentially incompatible LLVM/MLIR ABI into one context.

Only compiler-emitted ``gf.kernel -> TTIR`` translations enter through this
boundary. Handwritten provider kernels are benchmark oracles and live outside
the :mod:`tiga` package.
"""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import hashlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from typing import Any, Callable, Mapping

from .provider import ProviderIdentity, triton_provider_identity


@dataclass(frozen=True)
class TTIRCompileResult:
    compiled: Any
    provider: ProviderIdentity
    source_hash: str
    artifacts: Mapping[str, str | bytes]
    driver_launcher: Any | None = None
    worker_compile_ms: float | None = None
    cache_load_ms: float | None = None


@dataclass
class TTIRRankedPlan:
    """Exact ranked-relation build+consume plan with no adjacency output."""

    num_queries: int
    block_rows: int
    result: TTIRCompileResult
    output: Any | None = None

    def run(self, query_positions: Any, candidate_positions: Any, *inputs: Any):
        from ..interop.torch.provider import reusable_vector_output

        self.output = reusable_vector_output(
            self.output, query_positions, rows=self.num_queries)
        grid = (
            (self.num_queries + self.block_rows - 1) // self.block_rows,
            1,
            1,
        )
        _launch(
            self.result, grid,
            (query_positions, candidate_positions, *inputs, self.output),
        )
        return self.output


class _CUDADriverLauncher:
    """Launch a vendor-compiled image through Tiga's own runtime ABI."""

    def __init__(
        self,
        compiled: Any,
        artifacts: Mapping[str, str | bytes],
        *,
        device: str = "cuda:0",
        stream_provider: Callable[[], Any] | None = None,
    ):
        from ..runtime import Buffer, Module, Stream

        image = artifacts.get("cubin") or artifacts.get("ptx")
        if not isinstance(image, (str, bytes)):
            raise RuntimeError("CUDA provider emitted neither cubin nor PTX")
        self.device = device
        self.module = Module(image, device=self.device)
        self.kernel = self.module.kernel(compiled.name)
        self._stream_provider = stream_provider
        self._owned_stream = (
            Stream(self.device) if stream_provider is None else None
        )
        metadata = compiled.metadata
        self.global_scratch_size = int(metadata.global_scratch_size)
        self.profile_scratch_size = int(metadata.profile_scratch_size)
        if self.global_scratch_size < 0 or self.profile_scratch_size < 0:
            raise RuntimeError("provider scratch sizes must be non-negative")
        self._global_scratch = (
            Buffer(self.global_scratch_size, device=self.device)
            if self.global_scratch_size else None
        )
        self._profile_scratch = (
            Buffer(self.profile_scratch_size, device=self.device)
            if self.profile_scratch_size else None
        )
        self.block = (int(metadata.num_warps) * int(metadata.warp_size), 1, 1)
        self.shared_bytes = int(metadata.shared)

    def _bind_arguments(self, arguments: tuple[Any, ...]) -> tuple[Any, ...]:
        scalar_arguments = []
        for argument in arguments:
            if isinstance(argument, float):
                scalar_arguments.append(ctypes.c_float(argument))
            elif isinstance(argument, bool):
                scalar_arguments.append(ctypes.c_int32(argument))
            elif isinstance(argument, int):
                scalar_arguments.append(ctypes.c_int32(argument))
            else:
                scalar_arguments.append(argument)
        # NVIDIA provider ABI appends global/profile scratch pointers after
        # the semantic arguments. Buffer objects bind as device pointers;
        # absent scratch uses an explicit null pointer.
        scalar_arguments.extend((
            self._global_scratch or ctypes.c_uint64(0),
            self._profile_scratch or ctypes.c_uint64(0),
        ))
        return tuple(scalar_arguments)

    def _stream(self):
        return (
            self._owned_stream
            if self._stream_provider is None
            else self._stream_provider()
        )

    def launch(self, grid: tuple[int, int, int], arguments: tuple[Any, ...]):
        scalar_arguments = self._bind_arguments(arguments)
        return self.kernel.launch(
            self._stream(),
            grid=grid,
            block=self.block,
            arguments=scalar_arguments,
            shared_bytes=self.shared_bytes,
        )

    def prepare(self, grid: tuple[int, int, int], arguments: tuple[Any, ...]):
        """Bind a stable kernel ABI once while resolving the live stream late."""
        scalar_arguments = self._bind_arguments(arguments)
        if self._stream_provider is not None:
            return self.kernel.prepare_async(
                self._stream, grid=grid, block=self.block,
                arguments=scalar_arguments, shared_bytes=self.shared_bytes)
        kernel = self.kernel
        block = self.block
        shared_bytes = self.shared_bytes

        def launch():
            return kernel.launch(
                self._stream(), grid=grid, block=block,
                arguments=scalar_arguments, shared_bytes=shared_bytes)

        return launch


def _launch(result: TTIRCompileResult, grid, arguments):
    if result.driver_launcher is not None:
        return result.driver_launcher.launch(grid, tuple(arguments))
    return result.compiled[grid](*arguments)


@dataclass
class TTIRWeightedSumPlan:
    row_ptr: Any
    col_idx: Any
    num_rows: int
    block_rows: int
    result: TTIRCompileResult
    output: Any | None = None

    def _acquire_output(self, x: Any) -> Any:
        from ..interop.torch.provider import reusable_empty_like

        self.output = reusable_empty_like(self.output, x)
        return self.output

    def run(self, x: Any, weight: Any) -> Any:
        return self.run_csr(self.row_ptr, self.col_idx, x, weight)

    def run_csr(
        self, row_ptr: Any, col_idx: Any, x: Any, weight: Any
    ) -> Any:
        """Launch the compiled schedule against a compatible CSR snapshot."""
        output = self._acquire_output(x)
        grid = (
            (self.num_rows + self.block_rows - 1) // self.block_rows,
            1,
            1,
        )
        _launch(self.result, grid, (
            row_ptr, col_idx, x, weight, output))
        return output


@dataclass
class TTIRGeneratedRadiusPlan:
    """Compiled generated-radius code rebound to each dynamic directory."""

    num_rows: int
    block_rows: int
    result: TTIRCompileResult
    output: Any | None = None

    def _acquire_output(self, x: Any) -> Any:
        from ..interop.torch.provider import reusable_empty_like

        self.output = reusable_empty_like(self.output, x)
        return self.output

    def run(self, directory: Any, positions: Any, x: Any) -> Any:
        output = self._acquire_output(x)
        grid = (
            (self.num_rows + self.block_rows - 1) // self.block_rows,
            1,
            1,
        )
        _launch(self.result, grid, (
            directory.cell_ptr,
            directory.particle_order,
            directory.cell_coordinates,
            directory.extents,
            directory.strides,
            directory.neighbor_offsets,
            directory.lattice,
            directory.inverse_lattice,
            positions,
            x,
            output,
        ))
        return output


@dataclass
class TTIRDenseStreamingPlan:
    """Executable for a compiler-emitted Cartesian streaming reduction."""

    num_rows: int
    lanes: int
    width: int
    block_rows: int
    result: TTIRCompileResult
    output_storage: Any | None = None

    def _acquire_output(self, lhs: Any) -> Any:
        from ..interop.torch.provider import reusable_dense_output

        self.output_storage, output = reusable_dense_output(
            self.output_storage,
            lhs,
            lanes=self.lanes,
            rows=self.num_rows,
            width=self.width,
        )
        return output

    def run(self, lhs: Any, rhs: Any, payload: Any, scale: float) -> Any:
        output = self._acquire_output(lhs)
        grid = (
            (self.num_rows + self.block_rows - 1) // self.block_rows,
            self.lanes,
            1,
        )
        _launch(self.result, grid, (lhs, rhs, payload, scale, output))
        return output


@dataclass
class TTIRDenseScalarPlan:
    """Executable for generic scalar edge/reducer/optional-node regions."""

    num_rows: int
    block_rows: int
    result: TTIRCompileResult
    output: Any | None = None

    def run(self, *inputs: Any) -> Any:
        if not inputs:
            raise ValueError("dense scalar launch requires at least one field")
        from ..interop.torch.provider import reusable_vector_output

        self.output = reusable_vector_output(
            self.output, inputs[0], rows=self.num_rows)
        grid = (
            (self.num_rows + self.block_rows - 1) // self.block_rows,
            1,
            1,
        )
        _launch(self.result, grid, (*inputs, self.output))
        return self.output


@dataclass
class TTIRCSRScalarPlan:
    """Executable for generic scalar CSR edge/reducer/optional-node regions."""

    row_ptr: Any
    col_idx: Any
    num_rows: int
    block_rows: int
    result: TTIRCompileResult
    output: Any | None = None

    def acquire_output(self, *inputs: Any) -> Any:
        """Acquire output storage without submitting work."""
        from ..interop.torch.provider import reusable_vector_output

        reference = next(
            (value for value in inputs if hasattr(value, "device")), None)
        if reference is None:
            raise ValueError("CSR scalar launch requires a tensor field")
        self.output = reusable_vector_output(
            self.output, reference, rows=self.num_rows)
        return self.output

    def launch_into(self, *, output: Any, inputs: tuple[Any, ...]) -> None:
        """Submit the compiled kernel into caller-owned output storage."""
        grid = (
            (self.num_rows + self.block_rows - 1) // self.block_rows,
            1,
            1,
        )
        _launch(self.result, grid, (
            self.row_ptr, self.col_idx, *inputs, output))

    def run(self, *inputs: Any) -> Any:
        if not inputs:
            raise ValueError("CSR scalar launch requires at least one field")
        self.acquire_output(*inputs)
        self.launch_into(output=self.output, inputs=tuple(inputs))
        return self.output


@dataclass
class TTIRCSRProductPlan:
    """Executable for one compiler-fused multi-result CSR product launch."""

    row_ptr: Any
    col_idx: Any
    num_rows: int
    block_rows: int
    num_results: int
    result: TTIRCompileResult
    outputs: tuple[Any, ...] = ()

    def _acquire_outputs(self, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
        from ..interop.torch.provider import reusable_vector_output

        reference = next(
            (value for value in inputs if hasattr(value, "device")), None)
        if reference is None:
            raise ValueError("CSR product launch requires a tensor field")
        previous = self.outputs
        self.outputs = tuple(
            reusable_vector_output(
                previous[index] if index < len(previous) else None,
                reference,
                rows=self.num_rows,
            )
            for index in range(self.num_results)
        )
        return self.outputs

    def run(self, *inputs: Any) -> tuple[Any, ...]:
        outputs = self._acquire_outputs(tuple(inputs))
        grid = (
            (self.num_rows + self.block_rows - 1) // self.block_rows,
            1,
            1,
        )
        _launch(self.result, grid, (
            self.row_ptr, self.col_idx, *inputs, *outputs))
        return outputs


@dataclass(frozen=True)
class TTIRTaskPrimitivePlan:
    """Provider executable for one compiler-emitted task primitive."""

    task_kind: str
    grid_x: int
    abi: tuple[str, ...]
    rows: int | None
    index_dtype: str | None
    result: TTIRCompileResult

    def launch(self, **arguments: Any) -> None:
        missing = tuple(parameter for parameter in self.abi if parameter not in arguments)
        extra = tuple(parameter for parameter in arguments if parameter not in self.abi)
        if missing or extra:
            raise ValueError(
                f"task primitive ABI mismatch: missing={missing}, extra={extra}"
            )
        self._validate(arguments)
        _launch(
            self.result,
            (self.grid_x, 1, 1),
            tuple(arguments[parameter] for parameter in self.abi),
        )

    def _validate(self, arguments: Mapping[str, Any]) -> None:
        row_ptr = arguments.get("row_ptr")
        if row_ptr is None:
            return
        if self.rows is not None:
            numel = getattr(row_ptr, "numel", None)
            elements = numel() if callable(numel) else None
            if elements is not None and elements < self.rows + 1:
                raise ValueError(
                    f"row_ptr requires at least {self.rows + 1} elements, "
                    f"got {elements}"
                )
        if self.index_dtype is not None:
            actual = str(getattr(row_ptr, "dtype", ""))
            expected = "int64" if self.index_dtype == "i64" else "int32"
            if actual and expected not in actual:
                raise TypeError(
                    f"row_ptr dtype must match compiler plan "
                    f"{self.index_dtype}, got {actual}"
                )

    def prepare_submission(self, argument_slots):
        ordered_slots = tuple(slot for _parameter, slot in argument_slots)
        compiled = self.result.compiled
        grid = (self.grid_x, 1, 1)

        def submit(resources):
            if self.result.driver_launcher is not None:
                self.result.driver_launcher.launch(
                    grid, tuple(resources[slot] for slot in ordered_slots)
                )
            else:
                compiled[grid](*(resources[slot] for slot in ordered_slots))

        return submit

def compile_ttir(
    module: str,
    *,
    target: Any | None = None,
    options: Mapping[str, Any] | None = None,
    launcher: str | None = None,
    device: str = "cuda:0",
    stream_provider: Callable[[], Any] | None = None,
) -> TTIRCompileResult:
    """Compile one TTIR module with the active vendor Triton distribution.

    Triton 3.6 accepts IR through a filename and deliberately resumes after
    the input stage.  Therefore ``module`` must already be valid, canonical
    TTIR for exactly the provider identity returned here.
    """
    if not isinstance(module, str) or not module.strip():
        raise ValueError("TTIR module must be a non-empty string")
    if "tt.func" not in module:
        raise ValueError("TTIR module must contain a tt.func entry point")

    try:
        from triton.compiler import compile as triton_compile
    except ImportError as error:
        raise RuntimeError("the selected TTIR provider requires Triton") from error

    source_hash = hashlib.sha256(module.encode()).hexdigest()
    worker_compile_ms = None
    worker_mode = os.environ.get("TIGA_COMPILE_WORKER", "1").lower()
    if worker_mode not in {"0", "1", "false", "true", "off", "on"}:
        raise ValueError("TIGA_COMPILE_WORKER must be 0/1/false/true/off/on")
    if worker_mode in {"1", "true", "on"}:
        from .compile_worker import compile_in_worker

        response = compile_in_worker(
            module,
            source_hash=source_hash,
            target=target,
            options=options,
        )
        worker_compile_ms = float(response["compile_ms"])
    load_started = time.perf_counter_ns()
    with TemporaryDirectory(prefix="tiga-ttir-") as directory:
        path = Path(directory) / f"{source_hash[:16]}.ttir"
        path.write_text(module)
        compiled = triton_compile(
            str(path), target=target, options=dict(options or {}))
    cache_load_ms = (time.perf_counter_ns() - load_started) / 1e6
    artifacts = dict(compiled.asm)
    launcher = os.environ.get(
        "TIGA_CUDA_LAUNCHER", "triton"
    ) if launcher is None else launcher
    if launcher not in {"triton", "driver"}:
        raise ValueError("TTIR launcher must be 'triton' or 'driver'")
    driver_launcher = None
    if launcher == "driver":
        driver_launcher = _CUDADriverLauncher(
            compiled, artifacts, device=device,
            stream_provider=stream_provider,
        )
    return TTIRCompileResult(
        compiled=compiled,
        provider=triton_provider_identity(target),
        source_hash=source_hash,
        artifacts=artifacts,
        driver_launcher=driver_launcher,
        worker_compile_ms=worker_compile_ms,
        cache_load_ms=cache_load_ms,
    )


def prepare_ttir_weighted_sum(
    module: str,
    *,
    row_ptr: Any,
    col_idx: Any,
    num_rows: int,
    block_rows: int,
    num_warps: int,
) -> TTIRWeightedSumPlan:
    """Compile the direct Kernel-IR candidate and bind its physical CSR ABI."""
    result = compile_ttir(module, options={"num_warps": num_warps})
    return TTIRWeightedSumPlan(
        row_ptr=row_ptr,
        col_idx=col_idx,
        num_rows=num_rows,
        block_rows=block_rows,
        result=result,
    )


def prepare_ttir_ranked(
    module: str,
    *,
    num_queries: int,
    block_rows: int,
    num_warps: int,
) -> TTIRRankedPlan:
    """Compile a ranked selection/consume kernel emitted from Kernel IR."""
    result = compile_ttir(module, options={"num_warps": num_warps})
    return TTIRRankedPlan(
        num_queries=num_queries,
        block_rows=block_rows,
        result=result,
    )


def prepare_ttir_generated_radius(
    module: str,
    *,
    num_rows: int,
    block_rows: int,
    num_warps: int,
) -> TTIRGeneratedRadiusPlan:
    """Compile a direct generated-tile TTIR module for dynamic rebinding."""
    result = compile_ttir(module, options={"num_warps": num_warps})
    return TTIRGeneratedRadiusPlan(
        num_rows=num_rows,
        block_rows=block_rows,
        result=result,
    )


def prepare_ttir_dense_streaming(
    module: str,
    *,
    num_rows: int,
    lanes: int,
    width: int,
    block_rows: int,
    num_warps: int,
) -> TTIRDenseStreamingPlan:
    result = compile_ttir(
        module, options={"num_warps": num_warps, "num_stages": 3})
    return TTIRDenseStreamingPlan(
        num_rows=num_rows,
        lanes=lanes,
        width=width,
        block_rows=block_rows,
        result=result,
    )


def prepare_ttir_dense_scalar(
    module: str,
    *,
    num_rows: int,
    block_rows: int,
    num_warps: int,
) -> TTIRDenseScalarPlan:
    result = compile_ttir(module, options={"num_warps": num_warps})
    return TTIRDenseScalarPlan(
        num_rows=num_rows,
        block_rows=block_rows,
        result=result,
    )


def prepare_ttir_csr_scalar(
    module: str,
    *,
    row_ptr: Any,
    col_idx: Any,
    num_rows: int,
    block_rows: int,
    num_warps: int,
) -> TTIRCSRScalarPlan:
    result = compile_ttir(module, options={"num_warps": num_warps})
    return TTIRCSRScalarPlan(
        row_ptr=row_ptr,
        col_idx=col_idx,
        num_rows=num_rows,
        block_rows=block_rows,
        result=result,
    )


@dataclass
class TTIREdgeNNTilePlan:
    """Executable for one fused edge-NN tile kernel (Python emission, phase 1).

    The kernel segment-reduces with masked atomic adds, so the output buffer
    is zeroed before every launch; a compatible buffer is reused otherwise.
    Topology sizes are rebound per call because the same compiled structure
    serves any graph with matching segment/layer shapes.
    """

    out_width: int
    block_e: int
    result: TTIRCompileResult
    output: Any | None = None

    def run(self, arguments: tuple[Any, ...], *, num_edges: int, num_rows: int):
        """Launch with (dst_idx, src_idx, *segments, *weights) tensors."""
        import torch

        reference = next(
            (value for value in arguments
             if isinstance(value, torch.Tensor) and value.dtype.is_floating_point),
            None,
        )
        if reference is None:
            raise ValueError("edge nn tile launch requires a tensor field")
        if (
            self.output is None
            or self.output.shape != (num_rows, self.out_width)
            or self.output.device != reference.device
        ):
            self.output = torch.zeros(
                (num_rows, self.out_width),
                dtype=reference.dtype, device=reference.device)
        else:
            self.output.zero_()
        if num_edges:
            grid = ((num_edges + self.block_e - 1) // self.block_e, 1, 1)
            _launch(
                self.result, grid, (*arguments, self.output, num_edges))
        return self.output


def prepare_ttir_edge_nn_tile(
    module: str,
    *,
    out_width: int,
    block_e: int,
    num_warps: int,
) -> TTIREdgeNNTilePlan:
    """Compile one Python-emitted edge nn tile module and freeze its plan."""
    result = compile_ttir(module, options={"num_warps": num_warps})
    return TTIREdgeNNTilePlan(
        out_width=out_width,
        block_e=block_e,
        result=result,
    )


@dataclass
class TTIREdgeNNVjpPlan:
    """Executable for one fused edge-NN backward tile kernel.

    Gradient accumulators (field/position grads, padded weight/bias grads)
    are caller-allocated, zero-filled buffers merged by the kernel's relaxed
    atomics, so the plan only launches; topology sizes rebind per call.
    """

    block_e: int
    result: TTIRCompileResult

    def run(self, arguments: tuple[Any, ...], *, num_edges: int) -> None:
        """Launch with the full backward argument list (see edge_nn_vjp)."""
        if num_edges:
            grid = ((num_edges + self.block_e - 1) // self.block_e, 1, 1)
            _launch(self.result, grid, (*arguments, num_edges))


def prepare_ttir_edge_nn_vjp(
    module: str,
    *,
    block_e: int,
    num_warps: int,
) -> TTIREdgeNNVjpPlan:
    """Compile one Python-emitted edge nn backward module and freeze its plan."""
    result = compile_ttir(module, options={"num_warps": num_warps})
    return TTIREdgeNNVjpPlan(block_e=block_e, result=result)


@dataclass
class TTIREdgeNNAttentionPlan:
    """Executable for one row-centric fused nn-attention kernel.

    One CSR row per program; every row's ``out_width`` columns are written
    unconditionally (zero for empty rows), so the output needs no zeroing
    and is freshly allocated per call.  Topology sizes rebind per call.
    """

    out_width: int
    result: TTIRCompileResult

    def run(self, arguments: tuple[Any, ...], *, num_rows: int):
        """Launch with (row_ptr, col_idx, *segments, *weights, value).

        Returns ``(output, m, l)`` — the online-softmax row state is
        persisted for the fused backward; inference callers discard it.
        """
        import torch

        reference = next(
            (value for value in arguments
             if isinstance(value, torch.Tensor) and value.dtype.is_floating_point),
            None,
        )
        if reference is None:
            raise ValueError("edge nn attention launch requires a tensor field")
        output = torch.empty(
            (num_rows, self.out_width),
            dtype=reference.dtype, device=reference.device)
        m = torch.empty(num_rows, dtype=reference.dtype,
                        device=reference.device)
        l = torch.empty(num_rows, dtype=reference.dtype,
                        device=reference.device)
        if num_rows:
            _launch(self.result, (num_rows, 1, 1), (*arguments, output, m, l))
        return output, m, l


def prepare_ttir_edge_nn_attention(
    module: str,
    *,
    out_width: int,
    num_warps: int,
) -> TTIREdgeNNAttentionPlan:
    """Compile one Python-emitted nn-attention module and freeze its plan."""
    result = compile_ttir(module, options={"num_warps": num_warps})
    return TTIREdgeNNAttentionPlan(out_width=out_width, result=result)


@dataclass
class TTIREdgeNNAttentionVjpPlan:
    """Executable for one row-centric fused nn-attention backward kernel.

    All gradient buffers are caller-allocated and zero-filled; the kernel
    merges per-row partials with relaxed atomics.  One CSR row per program.
    """

    result: TTIRCompileResult

    def run(self, arguments: tuple[Any, ...], *, num_rows: int) -> None:
        """Launch with the full backward argument list (edge_nn_attention)."""
        if num_rows:
            _launch(self.result, (num_rows, 1, 1), arguments)


def prepare_ttir_edge_nn_attention_vjp(
    module: str,
    *,
    num_warps: int,
) -> TTIREdgeNNAttentionVjpPlan:
    """Compile one Python-emitted nn-attention backward module."""
    result = compile_ttir(module, options={"num_warps": num_warps})
    return TTIREdgeNNAttentionVjpPlan(result=result)


def prepare_ttir_csr_product(
    module: str,
    *,
    row_ptr: Any,
    col_idx: Any,
    num_rows: int,
    block_rows: int,
    num_results: int,
    num_warps: int,
) -> TTIRCSRProductPlan:
    if num_results < 2:
        raise ValueError("CSR product plans require at least two results")
    result = compile_ttir(module, options={"num_warps": num_warps})
    return TTIRCSRProductPlan(
        row_ptr=row_ptr,
        col_idx=col_idx,
        num_rows=num_rows,
        block_rows=block_rows,
        num_results=num_results,
        result=result,
    )


def prepare_ttir_task_primitive(invocation: Any) -> TTIRTaskPrimitivePlan:
    """Compile a task primitive solely from compiler-emitted plan metadata."""
    metadata = invocation.metadata
    module = metadata.get("provider_ttir")
    if not isinstance(module, str) or not module:
        raise ValueError(
            f"task {invocation.name!r} has no compiler-emitted provider_ttir"
        )
    if invocation.task_kind in {"degree-reset", "degree-prefix"}:
        grid_x = 1
    elif invocation.task_kind in {"degree-histogram", "degree-scatter"}:
        rows = metadata.get("rows")
        if not isinstance(rows, int) or rows <= 0:
            raise ValueError("row task requires a positive compiler-planned row count")
        grid_x = rows
    elif invocation.task_kind in {
        "degree-bucket", "row-split-partial", "row-split-finalize"
    }:
        grid_x = metadata.get("grid_x")
        if not isinstance(grid_x, int) or grid_x <= 0:
            raise ValueError(
                f"{invocation.task_kind} requires a positive compiler grid_x")
    else:
        raise ValueError(
            f"unsupported compiler task primitive {invocation.task_kind!r}"
        )
    return TTIRTaskPrimitivePlan(
        task_kind=invocation.task_kind,
        grid_x=grid_x,
        abi=tuple(argument.parameter for argument in invocation.arguments),
        rows=metadata.get("rows"),
        index_dtype=metadata.get("index_dtype"),
        result=compile_ttir(module, options={"num_warps": 1}),
    )
