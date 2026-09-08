"""Public Tensor IR inspection backed by the native in-process compiler.

Kept as a compatibility import path. No MLIR source is assembled or parsed in
Python; C++ OpBuilder constructs and verifies the module directly.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from .native import tensor_ir

if TYPE_CHECKING:
    from ..tensor import Tensor


def tensor_mlir(output: Tensor, *, function_name: str = "tensor_main") -> str:
    from ..tensor.core import _substitute_program_leaves

    # Inspection-only substitution: program leaves become unrealized input
    # placeholders; nothing executes.
    return tensor_ir(
        _substitute_program_leaves(output, realize=False),
        function_name=function_name,
    )


def tensor_vjp_mlir(
    output: Tensor,
    wrt: Tensor,
    cotangent: Tensor,
    *,
    function_name: str = "tensor_vjp",
) -> str:
    if output.shape != cotangent.shape or output.dtype is not cotangent.dtype:
        raise ValueError("cotangent shape and dtype must match output")
    from ..tensor.core import _substitute_program_leaves

    return tensor_ir(
        _substitute_program_leaves(output, realize=False),
        function_name=function_name,
        wrt=wrt,
        cotangent=cotangent,
    )


def tensor_semantic_hash(output: Tensor) -> str:
    return hashlib.sha256(tensor_mlir(output).encode()).hexdigest()


def verify_tensor_mlir(output: Tensor) -> str:
    # Construction and verification are inseparable in the native binding.
    return tensor_mlir(output)


def lower_tensor_vjp_mlir(
    output: Tensor, wrt: Tensor, cotangent: Tensor
) -> str:
    if output.shape != cotangent.shape or output.dtype is not cotangent.dtype:
        raise ValueError("cotangent shape and dtype must match output")
    from ..tensor.core import _substitute_program_leaves

    return tensor_ir(
        _substitute_program_leaves(output, realize=False),
        function_name="tensor_vjp",
        wrt=wrt,
        cotangent=cotangent,
        run_vjp=True,
    )


__all__ = [
    "lower_tensor_vjp_mlir",
    "tensor_mlir",
    "tensor_semantic_hash",
    "tensor_vjp_mlir",
    "verify_tensor_mlir",
]
