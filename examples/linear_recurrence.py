"""A causal linear recurrence expressed with ordinary Tiga tensors."""

import torch

import tiga as tg


lanes, steps, key_width, value_width = 4, 128, 16, 16
q_torch = torch.randn(lanes, steps, key_width, device="cuda")      # (L, S, K)
k_torch = torch.randn_like(q_torch)                                # (L, S, K)
v_torch = torch.randn(lanes, steps, value_width, device="cuda")    # (L, S, V)
q, k, v = map(tg.from_torch, (q_torch, k_torch, v_torch))

# --8<-- [start:core]
state_shape = (lanes, steps, key_width, value_width)  # (L, S, K, V)
updates = (  # (L, S, K, V)
    k.unsqueeze(-1).broadcast_to(state_shape)
    * v.unsqueeze(-2).broadcast_to(state_shape)
)
output = (  # (L, S, V)
    q.unsqueeze(-1).broadcast_to(state_shape) * updates.cumsum(1)
).sum(axis=2)
# --8<-- [end:core]

actual = output.to_torch()

# Inspect: output.execution, output.generated_code("ttir"), output.expression()
