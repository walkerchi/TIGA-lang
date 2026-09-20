"""nn-module capture for MessagePassing edge UDFs.

``tg.nn.trace(module)`` wraps a ``torch.nn`` module so it can be called
inside an edge UDF with either Torch tensors (eager/reference execution,
inputs concatenated along the feature dimension) or compiler capture
expressions (building a capturable subgraph for the fused edge-NN tile
lowering).  See ``tiga.compiler.nn_capture`` for the translation and
``tiga.compiler.edge_nn_tile`` for the tile kernel emission.
"""

from __future__ import annotations

from ..compiler.nn_capture import TracedModule, trace

__all__ = ["TracedModule", "trace"]
