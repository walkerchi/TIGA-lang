"""Framework-independent native GraphForge compiler services."""

from __future__ import annotations

from importlib import import_module


_TENSOR_MLIR_EXPORTS = {
    "lower_tensor_vjp_mlir",
    "tensor_mlir",
    "tensor_semantic_hash",
    "tensor_vjp_mlir",
    "verify_tensor_mlir",
}
_BUNDLE_EXPORTS = {"translate_task_bundle"}
_SCHEDULE_EXPORTS = {"schedules_from_mlir"}
_SHAPE_EXPORTS = {
    "Dim", "ShapeBinding", "ShapeSpecializer", "TensorSpec", "bind_shapes"
}


def __getattr__(name: str):
    if name in _TENSOR_MLIR_EXPORTS:
        value = getattr(import_module(".tensor_mlir", __name__), name)
        globals()[name] = value
        return value
    if name in _BUNDLE_EXPORTS:
        value = getattr(import_module(".bundle_plan", __name__), name)
        globals()[name] = value
        return value
    if name in _SCHEDULE_EXPORTS:
        value = getattr(import_module(".schedule", __name__), name)
        globals()[name] = value
        return value
    if name in _SHAPE_EXPORTS:
        value = getattr(import_module(".shapes", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'graphforge.compiler' has no attribute {name!r}")


__all__ = sorted(
    _TENSOR_MLIR_EXPORTS | _BUNDLE_EXPORTS | _SCHEDULE_EXPORTS | _SHAPE_EXPORTS
)
