"""GAT-style edge attention: an nn scores each edge, online softmax combines.

The score network runs inside a row-centric fused tile kernel — no [E, ·]
score or attention-weight tensor is materialized in either direction.
"""

import tiga as tg
import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

nodes, f_in, hidden = 512, 8, 16
positions = torch.rand((nodes, 3), device=device)        # (N, 3)
x = torch.randn((nodes, f_in), device=device)            # (N, F_in)


# --8<-- [start:core]
class GAT(tg.MessagePassing):
    reducer = tg.online_softmax()      # attention: softmax over incoming edges

    def __init__(self, attn):
        super().__init__()
        self.attn = tg.nn.trace(attn)  # scalar score network, per edge

    def edge(self, src, dst, edge):
        # s_e = attn([pos_src − pos_dst ‖ x_src ‖ x_dst])   (E, 1)
        # out_i = Σ_e softmax_e(s) · x_src(e)               (N, F_in)
        return self.reducer(self.attn(edge.displacement, src.x, dst.x),
                            src.x)


graph = tg.Graph.radius(positions, cutoff=0.15)  # avg degree ≈ 7
attn = nn.Sequential(
    nn.Linear(3 + 2 * f_in, hidden), nn.LeakyReLU(0.2), nn.Linear(hidden, 1),
).to(device)
kernel = GAT(attn)

with torch.no_grad():
    output = kernel(graph=graph, src={"x": x}, dst={"x": x})  # (N, F_in)
# --8<-- [end:core]

# Training is fused as well: the forward persists only the per-row softmax
# state (m, l); the backward replays the score chain per row chunk and applies
# the softmax Jacobian adjoint in-tile, streaming gradients to the fields,
# the positions and the score-network weights.
positions_g = positions.clone().requires_grad_(True)
xg = x.clone().requires_grad_(True)
graph_g = tg.Graph.radius(positions_g, cutoff=0.15)
out_g = kernel(graph=graph_g, src={"x": xg}, dst={"x": xg})
out_g.sum().backward()

# Inspect: kernel.explain(); kernel.last_variant.artifacts["ttir"]
