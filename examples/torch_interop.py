"""The single optional PyTorch interoperability example.

GraphForge's compiler/runtime does not depend on Torch. This file demonstrates
the compatibility boundary: Torch tensors may call the same MessagePassing
UDF, and native gf.Tensor can wrap Torch storage without a copy.
"""

import torch

import graphforge as gf


class WeightedAggregation(gf.MessagePassing):
    reducer = gf.sum()

    def edge(self, src, dst, edge):
        del dst
        return edge.weight * src.x


nodes, degree = 8, 2
row_ptr = torch.arange(0, nodes * degree + 1, degree, dtype=torch.int64)
col_idx = (
    torch.arange(nodes)[:, None] + torch.tensor([-1, 1])
).remainder(nodes).flatten()
x = torch.linspace(0, 1, nodes)
weight = torch.ones(nodes * degree)

graph = gf.Graph.from_csr(row_ptr, col_idx, num_src=nodes, validate="full")
kernel = WeightedAggregation()
output = kernel(
    graph=graph, src={"x": x}, dst={}, edge={"weight": weight})

# Explicit zero-copy adaptation in both directions.
native = gf.from_torch(x)
round_trip = native.to_torch()
assert round_trip.data_ptr() == x.data_ptr()

print("Torch-compatible output:", output)
print("zero-copy pointer:", x.data_ptr())
print(kernel.explain())
for schedule in kernel.schedules:
    print("machine schedule:", schedule)
    print("resources:", schedule.resources)
    print("roles:", schedule.roles)
    print("instructions:", schedule.instructions)
