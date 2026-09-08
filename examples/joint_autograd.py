"""One inspectable execution DAG for forward and compiler-generated VJP."""

import torch

import tiga as gf

# --8<-- [start:core]
xt = torch.tensor([2.0, 3.0], requires_grad=True)  # (2,) torch storage
x = gf.from_torch(xt, requires_grad=True)          # zero-copy into gf.Tensor
loss = (x * x).sum()                               # scalar deferred DAG
plan = gf.autograd.joint_plan(loss, x, checkpoint="auto")
value, dx = plan.run()
# loss = 2² + 3² = 13; dx = 2x = [4, 6]
dx.to_torch(copy=True)  # gf-computed results: one copy back out
# --8<-- [end:core]

# Inspect: plan.explain()  — forward/backward executable bundle summary
