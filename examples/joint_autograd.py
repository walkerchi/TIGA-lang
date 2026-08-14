"""One inspectable execution DAG for forward and compiler-generated VJP."""

import graphforge as gf


x = gf.tensor([2.0, 3.0], requires_grad=True)
loss = (x * x).sum()
plan = gf.autograd.joint_plan(loss, x, checkpoint="auto")
value, dx = plan.run()
print(value.tolist(), dx.tolist())
print(plan.explain())
