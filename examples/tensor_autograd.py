"""Minimal Torch-independent GraphForge Tensor and VJP example."""

import graphforge as gf


x = gf.tensor([1.0, 2.0, 3.0], requires_grad=True)
scale = gf.tensor(2.0, requires_grad=True)

loss = (x * x + scale * x).sum()
dx, dscale = gf.autograd.grad(loss, (x, scale))

print("loss:", loss.tolist())
print("dx:", dx.tolist())
print("dscale:", dscale.tolist())
print("\nCanonical forward MLIR:\n" + loss.mlir(verify=True))
print("\nVJP after gf-tensor-vjp:\n" + gf.autograd.grad_mlir(loss, x))
