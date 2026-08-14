"""A differentiable MessagePassing UDF with no user-written backward.

This example is Torch-free. The same UDF becomes a forward Tensor IR graph;
`gf-tensor-vjp` derives relation-aware backward kernels from that graph.
"""

import graphforge as gf


class ConductiveAggregation(gf.MessagePassing):
    reducer = gf.sum()

    def edge(self, src, dst, edge):
        del dst
        return edge.conductivity * src.temperature

    def node(self, dst, incoming):
        return incoming + dst.bias


row_ptr = gf.tensor([0, 2, 3, 5], dtype=gf.int64)
col_idx = gf.tensor([0, 2, 1, 0, 1], dtype=gf.int64)
graph = gf.Graph.from_csr(row_ptr, col_idx, num_src=3, validate="full")

temperature = gf.tensor([1.0, 2.0, 3.0], requires_grad=True)
conductivity = gf.tensor([2.0, 3.0, 4.0, 5.0, 6.0], requires_grad=True)
bias = gf.tensor([0.1, 0.2, 0.3], requires_grad=True)

kernel = ConductiveAggregation()
output = kernel(
    graph=graph,
    src={"temperature": temperature},
    dst={"bias": bias},
    edge={"conductivity": conductivity},
)
loss = output.sum()
d_temperature, d_conductivity, d_bias = gf.autograd.grad(
    loss, (temperature, conductivity, bias))
# The same functional API exposes the compiler's storage decision. ``save``
# inserts an explicit identity-like checkpoint in VJP IR; ``recompute`` keeps
# the primal expression fused into backward. ``auto`` emits a checkpoint
# candidate which the MLIR budget pass selects or rejects before lowering.
saved_d_conductivity = gf.autograd.grad(
    output,
    conductivity,
    grad_output=gf.tensor([1.0, 1.0, 1.0]),
    checkpoint="save",
)

print("output:", output.tolist())
print("d_temperature:", d_temperature.tolist())
print("d_conductivity:", d_conductivity.tolist())
print("d_bias:", d_bias.tolist())
print("checkpoint is explicit:",
      "checkpoint" in saved_d_conductivity.expression())
print("\n--- forward relation Tensor graph ---")
print(output.expression())
print("\n--- compiler-generated VJP IR ---")
print(gf.autograd.grad_mlir(loss, temperature))
print("\n--- MessagePassing selection ---")
print(kernel.explain())
