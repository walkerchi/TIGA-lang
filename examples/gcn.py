"""Torch-free feature MessagePassing and automatic gradients (SpMM shape)."""

import graphforge as gf


class WeightedFeatureAggregation(gf.MessagePassing):
    reducer = gf.sum()

    def edge(self, src, dst, edge):
        del dst
        return edge.weight * src.x


row_ptr = gf.tensor([0, 2, 3, 5], dtype=gf.int64)
col_idx = gf.tensor([0, 2, 1, 0, 1], dtype=gf.int64)
graph = gf.Graph.from_csr(row_ptr, col_idx, num_src=3, validate="full")

x = gf.tensor(
    [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True)
# [E, 1] broadcasts over the feature axis without changing graph semantics.
weight = gf.tensor(
    [[0.5], [1.0], [2.0], [1.5], [0.25]], requires_grad=True)

output = WeightedFeatureAggregation()(
    graph=graph, src={"x": x}, dst={}, edge={"weight": weight})
dx, dweight = gf.autograd.grad(output.sum(), (x, weight))

print("output:", output.tolist())
print("dx:", dx.tolist())
print("dweight:", dweight.tolist())
print(output.expression())
