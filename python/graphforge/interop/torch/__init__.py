"""Optional PyTorch zero-copy and correctness adapters."""

from .graph import Graph
from .library import RegisteredMessagePassingOp, register_message_passing
from .tensor import from_torch, to_torch

__all__ = [
    "Graph", "RegisteredMessagePassingOp", "from_torch",
    "register_message_passing", "to_torch",
]
