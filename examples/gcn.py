"""Torch-free feature MessagePassing and automatic gradients (SpMM shape)."""

import tiga as gf


# 3 nodes, 5 edges: 0→0, 2→0, 1→1, 0→2, 1→2
row_ptr = gf.tensor([0, 2, 3, 5], dtype=gf.int64)        # (N+1,)
col_idx = gf.tensor([0, 2, 1, 0, 1], dtype=gf.int64)     # (E,)
graph = gf.Graph.from_csr(row_ptr, col_idx, num_src=3, validate="full")

x = gf.tensor(
    [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True)        # (N, F)
# [E, 1] broadcasts over the feature axis without changing graph semantics.
weight = gf.tensor(
    [[0.5], [1.0], [2.0], [1.5], [0.25]], requires_grad=True)        # (E, 1)


# --8<-- [start:core]
class WeightedFeatureAggregation(gf.MessagePassing):
    reducer = gf.sum()

    def edge(self, src, dst, edge):
        # message on edge e=(j→i): weight[e] * x[j, :] — (F,) per edge
        return edge.weight * src.x


output = WeightedFeatureAggregation()(
    graph=graph, src={"x": x}, dst={}, edge={"weight": weight})  # (N, F)
# --8<-- [end:core]

dx, dweight = gf.autograd.grad(output.sum(), (x, weight))
# d x[j, :] = Σ weight[e] over edges e out of j (broadcast over F)
# d weight[e] = Σ_f x[src(e), f]

# Inspect the captured forward relation Tensor graph: output.expression()
