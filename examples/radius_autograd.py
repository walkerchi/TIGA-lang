"""Torch-free dynamic radius MessagePassing with automatic geometry VJP."""

import graphforge as gf


class DistanceWeightedSum(gf.MessagePassing):
    reducer = gf.sum()

    def edge(self, src, dst, edge):
        return edge.distance * src.x


positions = gf.tensor(
    [[0.0, 0.0], [0.3, 0.0], [0.8, 0.0]], requires_grad=True)
source = gf.tensor([2.0, 3.0, 5.0], requires_grad=True)
graph = gf.Graph.radius(positions, cutoff=0.6)

kernel = DistanceWeightedSum()
output = kernel(graph=graph, src={"x": source}, dst={"x": source})
position_grad, source_grad = gf.autograd.grad(
    output,
    (positions, source),
    grad_output=gf.tensor([1.0, 2.0, 3.0]),
)

print("output:", output.tolist())
print("d_positions:", position_grad.tolist())
print("d_source:", source_grad.tolist())
print(kernel.explain())
print(output.mlir(verify=True))
