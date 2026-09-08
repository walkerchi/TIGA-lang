"""Expose one Tiga UDF to torch.compile without a user backward."""

import torch

import tiga as gf


# --8<-- [start:core]
class WeightedAggregation(gf.MessagePassing):
    reducer = gf.sum()

    def edge(self, src, dst, edge):
        return edge.weight * src.x


row_ptr = torch.tensor([0, 2, 4], dtype=torch.int64)  # (M+1,)
col_idx = torch.tensor([0, 1, 1, 2], dtype=torch.int64)  # (E,)
graph = gf.Graph.from_csr(row_ptr, col_idx, num_src=3)
x = torch.tensor([1.0, 2.0, 4.0], requires_grad=True)  # (N,)
weight = torch.tensor([2.0, 3.0, 5.0, 7.0], requires_grad=True)  # (E,)

operation = gf.interop.torch.register_message_passing(
    WeightedAggregation(),
    graph=graph,
    src={"x": x},
    dst={},
    edge={"weight": weight},
)


@torch.compile(fullgraph=True)
def model(values, weights):
    return operation(src={"x": values}, dst={}, edge={"weight": weights})


compiled_output = model(x, weight)  # (M,)
# --8<-- [end:core]

dx, dweight = torch.autograd.grad(compiled_output.sum(), (x, weight))
# out = [2·1 + 3·2,  5·2 + 7·4]
# d x[j] = Σ weight[e] over edges e out of j
# d weight[e] = x[src(e)]
