"""A causal linear recurrence expressed with ordinary GraphForge tensors.

There is no linear-attention operator in GraphForge. On the registered CUDA
shape family the compiler recognizes map -> scan -> contract and emits one
kernel whose recurrent state remains on chip.
"""

import torch

import graphforge as gf


lanes, steps, key_width, value_width = 4, 128, 16, 16
q_torch = torch.randn(lanes, steps, key_width, device="cuda")
k_torch = torch.randn_like(q_torch)
v_torch = torch.randn(lanes, steps, value_width, device="cuda")
q, k, v = map(gf.from_torch, (q_torch, k_torch, v_torch))

state_shape = (lanes, steps, key_width, value_width)
updates = (
    k.unsqueeze(-1).broadcast_to(state_shape)
    * v.unsqueeze(-2).broadcast_to(state_shape)
)
output = (
    q.unsqueeze(-1).broadcast_to(state_shape) * updates.cumsum(1)
).sum(axis=2)

actual = output.to_torch()
expected = (
    q_torch.unsqueeze(-1)
    * (k_torch.unsqueeze(-1) * v_torch.unsqueeze(-2)).cumsum(1)
).sum(dim=2)
torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-4)

print(output.execution)
print(output.generated_code("ttir"))
