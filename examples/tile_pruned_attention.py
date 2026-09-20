"""Approximate tile admission as reducer semantics, not a core attention op."""

import torch

import tiga as tg

torch.manual_seed(0)
nodes, heads, width = 1024, 4, 64
# the streaming kernel reads lane-major storage: build (H, N, D), view as (N, H, D)
query = torch.randn(heads, nodes, width, device="cuda", dtype=torch.float16).permute(1, 0, 2)
key = torch.randn(heads, nodes, width, device="cuda", dtype=torch.float16).permute(1, 0, 2)
value = torch.randn(heads, nodes, width, device="cuda", dtype=torch.float16).permute(1, 0, 2)


# --8<-- [start:core]
class TilePrunedAttention(tg.MessagePassing):
    def __init__(self, threshold):
        super().__init__()
        self.reducer = tg.online_softmax(
            block_prune_threshold=threshold,
        )

    def edge(self, src, dst, edge, scale):
        score = (src.key * dst.query).sum(dim=-1) * scale
        return self.reducer(score, src.value)


program = TilePrunedAttention(width / nodes)
output = program(  # (N, H, D)
    graph=tg.Graph.dense(nodes, device="cuda"),
    src={"key": key, "value": value},
    dst={"query": query},
    scale=width**-0.5,
)
# --8<-- [end:core]

# Inspect: program.explain()           provider/lowering/cache summary
#          program.ir("domain")        Domain IR (block_prune_threshold attr)
#          program.ir("kernel_ttir")   TTIR (dynamic scf.if tile admission)
