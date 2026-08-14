"""Exact dynamic kNN with one compiler-generated, snapshot-rebound consumer."""

import torch

import graphforge as gf


class NeighborSum(gf.MessagePassing):
    reducer = gf.sum()

    def edge(self, src, dst, edge):
        del dst
        return edge.weight * src.x


positions = torch.rand(1024, 3, device="cuda")
values = torch.rand(1024, device="cuda")
k = 16
graph = gf.Graph.knn(positions, k)
kernel = NeighborSum()
result = kernel(
    graph=graph, src={"x": values}, dst={},
    edge={"weight": torch.ones(1024 * k, device="cuda")})

print(result.shape)
print(kernel.explain())
print(kernel.ir("kernel"))
