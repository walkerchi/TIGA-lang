"""Native CPU JIT for a captured :mod:`graphforge.tensor` expression DAG.

This module deliberately contains no source emitter and does not invoke a
system C/C++ compiler.  Python owns specialization and executable caching;
the native compiler constructs ``gf_tensor`` operations with MLIR OpBuilder,
lowers them to SCF/MemRef and then LLVM, and creates an in-process
``ExecutionEngine``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
import time
from typing import TYPE_CHECKING, Mapping

from .native import compile_cpu, launch_cpu

if TYPE_CHECKING:
    from ..tensor import Tensor


_PIPELINE_VERSION = 1


@dataclass(frozen=True)
class CPUExecutable:
    capsule: object
    inputs: tuple[Tensor, ...]
    ir: str
    cpu_loop_ir: str
    llvm_ir: str
    semantic_hash: str
    compile_ms: float
    cache_hit: bool

    @property
    def backend(self) -> str:
        return "cpu-llvm-jit"

    @property
    def fast_math(self) -> bool:
        # Fast-math must eventually be represented by explicit gf fast-math
        # attributes; an environment flag must not silently change semantics.
        return False

    @property
    def source(self) -> str:
        """Compatibility alias for the final inspectable compiler stage."""
        return self.llvm_ir

    @property
    def artifacts(self) -> Mapping[str, str]:
        return {
            "gf_tensor": self.ir,
            "cpu_loop": self.cpu_loop_ir,
            "llvm": self.llvm_ir,
        }

    @property
    def artifact(self) -> Path:
        # MLIR ExecutionEngine owns an in-memory JIT object rather than a
        # temporary shared object.  This is a diagnostic identity, not a file.
        return Path("mlir-execution-engine") / self.semantic_hash[:24]

    def launch(self, output: Tensor) -> None:
        launch_cpu(
            self.capsule,
            (value._buffer.address for value in self.inputs),
            (value.numel for value in self.inputs),
            output._buffer.address,
            output.numel,
        )


_IDENTITY_EXECUTABLES: dict[tuple[object, ...], CPUExecutable] = {}
_STRUCTURE_EXECUTABLES: dict[tuple[object, ...], CPUExecutable] = {}


def _topological_inputs(output: Tensor) -> tuple[Tensor, ...]:
    ordered: list[Tensor] = []
    visited: set[int] = set()

    def visit(value: Tensor) -> None:
        if id(value) in visited:
            return
        visited.add(id(value))
        if value._expr is None:
            ordered.append(value)
            return
        for operand in value._expr.operands:
            visit(operand)

    visit(output)
    return tuple(ordered)


def compile_tensor(output: Tensor) -> CPUExecutable:
    if output.device.type.name != "CPU":
        raise NotImplementedError("the native CPU JIT only accepts CPU tensors")

    identity = (_PIPELINE_VERSION, output._jit_key)
    cached = _IDENTITY_EXECUTABLES.get(identity)
    if cached is not None:
        return replace(cached, compile_ms=0.0, cache_hit=True)

    # Native construction is also the canonical semantic cache key.  It has
    # no pointer values, so structurally identical DAGs rebind their leaves to
    # the same executable.
    from .tensor_mlir import tensor_mlir

    semantic_ir = tensor_mlir(output, function_name="graphforge_run")
    semantic_hash = hashlib.sha256(semantic_ir.encode()).hexdigest()
    structure = (_PIPELINE_VERSION, semantic_hash)
    inputs = _topological_inputs(output)
    cached = _STRUCTURE_EXECUTABLES.get(structure)
    if cached is not None:
        rebound = replace(
            cached,
            inputs=inputs,
            compile_ms=0.0,
            cache_hit=True,
            ir=semantic_ir,
        )
        _IDENTITY_EXECUTABLES[identity] = rebound
        return rebound

    start = time.perf_counter_ns()
    capsule, captured_ir, cpu_loop_ir, llvm_ir = compile_cpu(output)
    compile_ms = (time.perf_counter_ns() - start) / 1e6
    # Both paths use the same native builder.  Treat disagreement as a compiler
    # bug instead of caching an executable under the wrong semantic identity.
    if captured_ir != semantic_ir:
        raise RuntimeError("native Tensor capture was not deterministic")
    executable = CPUExecutable(
        capsule=capsule,
        inputs=inputs,
        ir=captured_ir,
        cpu_loop_ir=cpu_loop_ir,
        llvm_ir=llvm_ir,
        semantic_hash=semantic_hash,
        compile_ms=compile_ms,
        cache_hit=False,
    )
    _STRUCTURE_EXECUTABLES[structure] = executable
    _IDENTITY_EXECUTABLES[identity] = executable
    return executable


__all__ = ["CPUExecutable", "compile_tensor"]
