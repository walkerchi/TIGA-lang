"""Torch storage binding for compiler-derived relation VJP programs."""

from __future__ import annotations

from dataclasses import dataclass
import time

import torch

from ...compiler.relation_vjp import analyze_source_vjp
from ...reducer import sum
from .graph import Graph, from_native
from .message_passing import MessagePassing


class _GeneratedTransposeApply(MessagePassing):
    reducer = sum()

    def edge(self, src, dst, edge):
        del dst
        return edge.jacobian * src.cotangent


@dataclass
class CompiledSourceVJP:
    """Executable source-input VJP and its immutable transpose snapshot."""

    program: MessagePassing
    graph: Graph
    jacobian: torch.Tensor
    compile_ms: float
    materialize_ms: float
    transform: str

    def __call__(self, cotangent: torch.Tensor):
        return self.program(
            graph=self.graph,
            src={"cotangent": cotangent},
            dst={},
            edge={"jacobian": self.jacobian},
        )


def compile_source_vjp(kernel, *, graph, src, edge, params=None) -> CompiledSourceVJP:
    """Differentiate one supported relation UDF and compile its reverse apply.

    Transpose construction is immutable-snapshot materialization and is never
    charged to warm backward execution. The returned program follows the same
    Domain -> Iter -> Kernel -> provider pipeline as a user forward program.
    """
    rule = analyze_source_vjp(kernel, params or {})
    torch_graph = from_native(graph)
    row_ptr, col_idx = torch_graph.resolve_csr()
    weight = edge[rule.edge_field]
    if src[rule.source_field].ndim != 1 or weight.ndim != 1:
        raise NotImplementedError("initial generated source VJP is scalar")

    started = time.perf_counter_ns()
    degrees = row_ptr[1:] - row_ptr[:-1]
    destinations = torch.repeat_interleave(
        torch.arange(
            torch_graph.schema.num_dst,
            device=col_idx.device,
            dtype=col_idx.dtype,
        ),
        degrees,
    )
    order = torch.argsort(col_idx, stable=True)
    transpose_columns = destinations[order]
    counts = torch.bincount(
        col_idx.to(torch.int64), minlength=torch_graph.schema.num_src)
    transpose_row_ptr = torch.empty(
        torch_graph.schema.num_src + 1,
        device=row_ptr.device,
        dtype=row_ptr.dtype,
    )
    transpose_row_ptr[0] = 0
    torch.cumsum(counts, dim=0, out=transpose_row_ptr[1:])
    jacobian = weight[order]
    transpose = Graph.from_csr(
        transpose_row_ptr,
        transpose_columns,
        num_src=torch_graph.schema.num_dst,
        validate="basic",
    )
    torch.cuda.synchronize(col_idx.device) if col_idx.is_cuda else None
    materialize_ms = (time.perf_counter_ns() - started) / 1e6

    program = _GeneratedTransposeApply()
    cotangent = torch.empty(
        torch_graph.schema.num_dst,
        device=weight.device,
        dtype=weight.dtype,
    )
    started = time.perf_counter_ns()
    program(
        graph=transpose,
        src={"cotangent": cotangent},
        dst={},
        edge={"jacobian": jacobian},
    )
    torch.cuda.synchronize(weight.device) if weight.is_cuda else None
    compile_ms = (time.perf_counter_ns() - started) / 1e6
    return CompiledSourceVJP(
        program, transpose, jacobian, compile_ms, materialize_ms, rule.transform)


__all__ = ["CompiledSourceVJP", "compile_source_vjp"]
