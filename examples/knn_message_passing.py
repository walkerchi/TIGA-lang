"""Exact dynamic kNN with one compiler-generated, snapshot-rebound consumer."""

import torch

import tiga as gf

positions = torch.rand(1024, 3, device="cuda")                          # (N, 3)
values = torch.rand(1024, device="cuda")                                # (N,)


# --8<-- [start:core]
class NeighborSum(gf.MessagePassing):
    reducer = gf.sum()

    def edge(self, src, dst, edge):
        return edge.weight * src.x


k = 16
graph = gf.Graph.knn(positions, k)
kernel = NeighborSum()
result = kernel(  # (N,)
    graph=graph, src={"x": values}, dst={},
    edge={"weight": torch.ones(1024 * k, device="cuda")})  # (N·k,)
# --8<-- [end:core]

# Inspect: kernel.explain(), kernel.ir("kernel")
