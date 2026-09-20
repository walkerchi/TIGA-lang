"""Edge-local nn modules in MessagePassing with a fused compiler tile kernel."""

import tiga as tg
import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

nodes, f_in, hidden, f_out = 512, 8, 16, 8
positions = torch.rand((nodes, 3), device=device)        # (N, 3)
x = torch.randn((nodes, f_in), device=device)            # (N, F_in)


# --8<-- [start:core]
class EdgeMLP(tg.MessagePassing):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = tg.nn.trace(mlp)  # capturable AND eager-callable

    def edge(self, src, dst, edge):
        # edge.displacement (E, 3), src.x (E, F_in) → message (E, F_out)
        return self.mlp(edge.displacement, src.x)


graph = tg.Graph.radius(positions, cutoff=0.15)  # avg degree ≈ 7
mlp = nn.Sequential(
    nn.Linear(3 + f_in, hidden), nn.ReLU(), nn.Linear(hidden, f_out),
).to(device)
kernel = EdgeMLP(mlp)

with torch.no_grad():
    # Inference takes the compiled fused tile kernel; grad-mode calls take
    # the fused recompute VJP — neither materializes per-edge activations.
    output = kernel(graph=graph, src={"x": x}, dst={})  # (N, F_out)
# --8<-- [end:core]

# A second architecture through the same UDF: deeper, GELU, bias-free.
mlp2 = nn.Sequential(
    nn.Linear(3 + f_in, hidden, bias=False), nn.GELU(),
    nn.Linear(hidden, hidden), nn.GELU(),
    nn.Linear(hidden, f_out),
).to(device)
kernel2 = EdgeMLP(mlp2)
with torch.no_grad():
    output2 = kernel2(graph=graph, src={"x": x}, dst={})  # (N, F_out)

# Training needs no special casing: grad-mode calls run the same fused tile
# forward, and backward replays per-edge activations inside a recompute VJP
# kernel — no [E, ·] tensor exists in either direction.
xg = x.clone().requires_grad_(True)
output_g = kernel(graph=graph, src={"x": xg}, dst={})
output_g.sum().backward()

# Inspect: kernel.explain(); kernel.last_variant.artifacts["ttir"]
