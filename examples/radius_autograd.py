"""dynamic radius MessagePassing with automatic geometry VJP."""

import torch
import tiga as tg


# --8<-- [start:core]
class DistanceWeightedSum(tg.MessagePassing):
    reducer = tg.sum()

    def edge(self, src, dst, edge):
        return edge.distance * src.x


positions = torch.tensor(
    [[0.0, 0.0], [0.3, 0.0], [0.8, 0.0]], requires_grad=True)  # (N, 2)
source = torch.tensor([2.0, 3.0, 5.0], requires_grad=True)  # (N,)
graph = tg.Graph.radius(positions, cutoff=0.6)

kernel = DistanceWeightedSum()
output = kernel(graph=graph, src={"x": source}, dst={"x": source})  # (N,)
# Edges (dist ≤ 0.6): 1→0 (0.3), 0→1 (0.3), 2→1 (0.5), 1→2 (0.5)
# out = [0.3·3, 0.3·2 + 0.5·5, 0.5·3]

position_grad, source_grad = torch.autograd.grad(
    output,
    (positions, source),
    grad_outputs=torch.tensor([1.0, 2.0, 3.0]),
)
# --8<-- [end:core]

# Inspect the selected plan: kernel.explain()
