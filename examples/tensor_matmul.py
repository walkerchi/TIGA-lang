"""Native Tiga Tensor matmul and compiler-derived reverse mode."""

import tiga as tg


# --8<-- [start:core]
lhs = tg.tensor(
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)  # (M, K)
rhs = tg.tensor(
    [[2.0, 1.0], [0.0, 3.0], [4.0, -1.0]], requires_grad=True)  # (K, N)
cotangent = tg.tensor([[1.0, 2.0], [-1.0, 0.5]])  # (M, N)

output = lhs @ rhs  # (M, N)

dlhs, drhs = tg.autograd.grad(
    output, (lhs, rhs), grad_output=cotangent)
# d loss / d lhs = cotangent @ rhsᵀ
# d loss / d rhs = lhsᵀ @ cotangent
# --8<-- [end:core]

# Inspect the semantic Tensor IR: output.mlir()
