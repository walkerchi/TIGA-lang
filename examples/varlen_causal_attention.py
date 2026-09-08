"""Varlen causal attention: packed sequences + cu_seqlens as one CSR graph."""

import torch

import tiga as gf

heads, width = 4, 64
lengths = torch.tensor([384, 1, 96, 640, 57], device="cuda")     # (S,) variable sequence lengths
cu_seqlens = torch.cat([lengths.new_zeros(1), lengths.cumsum(0)])  # (S+1,)
total = int(cu_seqlens[-1])                   # packed positions N = sum(lengths)

# --8<-- [start:core]
# Block-diagonal causal relation: position i attends j <= i inside its own
# sequence — exactly the mask flash-attn's varlen API encodes with cu_seqlens,
# written out as explicit CSR edges. Vectorized, built directly on device.
graph = gf.Graph.cu_seqlens(cu_seqlens, causal=True)

generator = torch.Generator(device="cuda").manual_seed(20260821)
query = torch.randn(
    total, heads, width, device="cuda", dtype=torch.float16,
    generator=generator)                                         # (N, H, D)
key = torch.randn_like(query)                                    # (N, H, D)
value = torch.randn_like(query)                                  # (N, H, D)


class VarlenCausalAttention(gf.MessagePassing):
    reducer = gf.online_softmax()

    def edge(self, src, dst, edge, scale):
        score = (src.key * dst.query).sum(dim=-1) * scale
        return self.reducer(score, src.value)


program = VarlenCausalAttention()
output = program(  # (N, H, D)
    graph=graph,
    src={"key": key, "value": value},
    dst={"query": query},
    scale=width**-0.5,
)
# --8<-- [end:core]

# Execution note: this edge-computed score over a general CSR runs on the
# exact eager oracle today; compiled streaming covers dense/triangular
# relations (full/causal attention) and field-score CSR segment softmax.
# Inspect: program.explain()
