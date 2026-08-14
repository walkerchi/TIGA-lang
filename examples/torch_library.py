"""Expose one GraphForge UDF to torch.compile without a user backward."""

import torch

import graphforge as gf


class WeightedAggregation(gf.MessagePassing):
    reducer = gf.sum()

    def edge(self, src, dst, edge):
        del dst
        return edge.weight * src.x


row_ptr = torch.tensor([0, 2, 4], dtype=torch.int64)
col_idx = torch.tensor([0, 1, 1, 2], dtype=torch.int64)
graph = gf.Graph.from_csr(row_ptr, col_idx, num_src=3)
x = torch.tensor([1.0, 2.0, 4.0], requires_grad=True)
weight = torch.tensor([2.0, 3.0, 5.0, 7.0], requires_grad=True)

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


compiled_output = model(x, weight)
dx, dweight = torch.autograd.grad(compiled_output.sum(), (x, weight))
print("dispatcher op:", operation.qualified_name)
print("compiled output:", compiled_output)
print("dx:", dx)
print("dweight:", dweight)
print("opcheck:", operation.opcheck(x.detach(), weight.detach()))
