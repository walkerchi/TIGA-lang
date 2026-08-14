"""Complex storage, views and conjugate-Wirtinger VJP."""

import graphforge as gf


x = gf.tensor(
    [[1.0 + 2.0j, 3.0 - 4.0j], [2.0 + 0.5j, -1.0 + 3.0j]],
    requires_grad=True,
)

# Transpose is a zero-copy strided view. The following reshape materializes
# logical transpose order because that view is not contiguous.
y = x.T.reshape(-1)
energy_terms = y.conj() * y

# Complex outputs require an explicit cotangent. GraphForge uses the
# conjugate-Wirtinger VJP convention.
dx = gf.autograd.grad(
    energy_terms,
    x,
    grad_output=gf.tensor([1.0 + 0.0j] * 4),
)

print("dtype:", x.dtype)
print("transposed logical order:", y.tolist())
print("VJP:", dx.tolist())

