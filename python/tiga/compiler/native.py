"""Loader for the in-process Tiga MLIR compiler extension."""

from __future__ import annotations

from importlib import import_module, machinery, util
from pathlib import Path
import sys


def _load_native():
    name = "tiga._graphforge_compiler"
    try:
        return import_module(name)
    except ModuleNotFoundError as original:
        # Editable source trees keep generated modules under build/* rather
        # than polluting python/tiga with build artifacts.
        root = Path(__file__).resolve().parents[3]
        candidates = [
            path
            for path in root.glob(
                "build/*/python_bindings/_graphforge_compiler*.so"
            )
            if any(
                path.name.endswith(suffix)
                for suffix in machinery.EXTENSION_SUFFIXES
            )
        ]
        if not candidates:
            raise RuntimeError(
                "the native Tiga compiler extension is unavailable; "
                "build/install tiga-lang before using JIT or IR APIs"
            ) from original
        # Build directory names describe wheel tags or developer presets, not
        # freshness.  Prefer the most recently produced extension so an
        # editable tree cannot silently load a stale lexicographically-last
        # build.
        candidate = max(candidates, key=lambda path: path.stat().st_mtime_ns)
        specification = util.spec_from_file_location(name, candidate)
        if specification is None or specification.loader is None:
            raise RuntimeError("could not load the native Tiga compiler")
        module = util.module_from_spec(specification)
        sys.modules[name] = module
        specification.loader.exec_module(module)
        return module


def tensor_ir(
    output,
    *,
    function_name: str = "tensor_main",
    wrt=None,
    cotangent=None,
    run_vjp: bool = False,
) -> str:
    return _load_native().tensor_ir(
        output,
        function_name=function_name,
        wrt=wrt,
        cotangent=cotangent,
        run_vjp=run_vjp,
    )


def reducer_ir(descriptor) -> str:
    return _load_native().reducer_ir(descriptor)


def domain_ir(descriptor) -> str:
    result = _load_native().domain_ir(descriptor)
    # The native frontend computes all target-independent stages in one
    # MLIRContext.  Keep this compatibility function focused on Domain IR.
    return result[0] if isinstance(result, tuple) else result


def domain_stages(descriptor) -> tuple[str, str, str, str]:
    result = _load_native().domain_ir(descriptor)
    if not isinstance(result, tuple) or len(result) != 4:
        raise RuntimeError("native compiler did not return four IR stages")
    return result


def program_ir(
    modules: tuple[str, ...],
    bindings: tuple[tuple[int, ...], ...],
    outputs: tuple[int, ...],
) -> tuple[str, str, str, str]:
    result = _load_native().program_ir(modules, bindings, outputs)
    if not isinstance(result, tuple) or len(result) != 4:
        raise RuntimeError("native compiler did not return four GraphProgram stages")
    return result


def plan_checkpoints(
    output, *, memory_budget_bytes: int = -1, spill_budget_bytes: int = 0
) -> tuple[tuple[bool, ...], int, str]:
    """Run the native MLIR checkpoint planner for a captured Tensor DAG."""
    decisions, saved_bytes, planned_ir, _ = _load_native().plan_checkpoints(
        output,
        memory_budget_bytes=memory_budget_bytes,
        spill_budget_bytes=spill_budget_bytes,
    )
    return tuple(bool(value) for value in decisions), int(saved_bytes), planned_ir


def checkpoint_plan(
    output, *, memory_budget_bytes: int = -1, spill_budget_bytes: int = 0
) -> dict[str, object]:
    """Return the complete liveness/cost/tier checkpoint plan.

    ``plan_checkpoints`` keeps its compact compatibility tuple.  This API is
    the runtime/compiler boundary: decisions retain three states through the
    accompanying ``tiers`` field instead of collapsing spill into save.
    """
    decisions, saved_bytes, planned_ir, details = _load_native().plan_checkpoints(
        output,
        memory_budget_bytes=memory_budget_bytes,
        spill_budget_bytes=spill_budget_bytes,
    )
    result = dict(details)
    result.update(
        decisions=tuple(bool(value) for value in decisions),
        saved_bytes=int(saved_bytes),
        memory_budget_bytes=int(memory_budget_bytes),
        spill_budget_bytes=int(spill_budget_bytes),
        ir=planned_ir,
    )
    return result


def compile_cpu(output):
    """Compile a Tensor DAG in-process through Tiga MLIR and LLVM.

    Returns an opaque executable capsule together with the IR snapshots taken
    before lowering, after structured CPU loop lowering, and after conversion
    to the LLVM dialect.
    """
    return _load_native().compile_cpu(output)


def launch_cpu(capsule, addresses, sizes, output_address: int, output_size: int):
    return _load_native().launch_cpu(
        capsule, tuple(addresses), tuple(sizes), output_address, output_size
    )


__all__ = [
    "compile_cpu",
    "domain_ir",
    "domain_stages",
    "launch_cpu",
    "plan_checkpoints",
    "checkpoint_plan",
    "program_ir",
    "reducer_ir",
    "tensor_ir",
]
