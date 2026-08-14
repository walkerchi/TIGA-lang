"""Approximate tile admission as reducer semantics, not a core attention op."""

import torch

import graphforge as gf


class TilePrunedAttention(gf.MessagePassing):
    def __init__(self, threshold):
        super().__init__()
        self.reducer = gf.online_softmax(
            block_prune_threshold=threshold,
        )

    def edge(self, src, dst, edge, scale):
        del edge
        score = (src.key * dst.query).sum(dim=-1) * scale
        return self.reducer(score, src.value)


nodes, heads, width = 1024, 4, 64
query_storage = torch.randn(
    heads, nodes, width, device="cuda", dtype=torch.float16)
key_storage = torch.randn_like(query_storage)
value_storage = torch.randn_like(query_storage)
query, key, value = (
    item.permute(1, 0, 2)
    for item in (query_storage, key_storage, value_storage)
)

program = TilePrunedAttention(width / nodes)
output = program(
    graph=gf.Graph.dense(nodes, device="cuda"),
    src={"key": key, "value": value},
    dst={"query": query},
    scale=width**-0.5,
)

assert torch.isfinite(output).all()
print(program.explain())
print(program.ir("domain"))       # contains block_prune_threshold
print(program.ir("kernel_ttir"))  # contains dynamic scf.if tile admission
