"""Native GraphForge Tensor matmul and compiler-derived reverse mode."""

import graphforge as gf


lhs = gf.tensor(
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
rhs = gf.tensor(
    [[2.0, 1.0], [0.0, 3.0], [4.0, -1.0]], requires_grad=True)
cotangent = gf.tensor([[1.0, 2.0], [-1.0, 0.5]])

output = lhs @ rhs
dlhs, drhs = gf.autograd.grad(
    output, (lhs, rhs), grad_output=cotangent)

print("output:", output.tolist())
print("d(lhs):", dlhs.tolist())
print("d(rhs):", drhs.tolist())
print("semantic IR:\n", output.mlir())
