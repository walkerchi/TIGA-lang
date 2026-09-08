"""Exact full attention (no mask) as a dense Graph + online-softmax reducer."""

import torch

import tiga as gf

torch.manual_seed(0)
nodes, heads, width = 1024, 4, 64
# the streaming kernel reads lane-major storage: build (H, N, D), view as (N, H, D)
query = torch.randn(heads, nodes, width, device="cuda", dtype=torch.float16).permute(1, 0, 2)
key = torch.randn(heads, nodes, width, device="cuda", dtype=torch.float16).permute(1, 0, 2)
value = torch.randn(heads, nodes, width, device="cuda", dtype=torch.float16).permute(1, 0, 2)


# --8<-- [start:core]
class FullAttention(gf.MessagePassing):
    reducer = gf.online_softmax()

    def edge(self, src, dst, edge, scale):
        score = (src.key * dst.query).sum(dim=-1) * scale
        return self.reducer(score, src.value)


program = FullAttention()
output = program(  # (N, H, D)
    graph=gf.Graph.dense(nodes, device="cuda"),
    src={"key": key, "value": value},
    dst={"query": query},
    scale=width**-0.5,
)
# --8<-- [end:core]

# Inspect: program.explain()         provider/lowering/cache summary
#          program.ir("kernel_ttir") TTIR (streaming online softmax)
