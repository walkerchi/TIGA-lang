"""Complex storage, views and conjugate-Wirtinger VJP."""

import tiga as gf


# --8<-- [start:core]
x = gf.tensor(
    [[1.0 + 2.0j, 3.0 - 4.0j], [2.0 + 0.5j, -1.0 + 3.0j]],
    requires_grad=True,
)  # (2, 2)

# Transpose is a zero-copy strided view. The following reshape materializes
# logical transpose order because that view is not contiguous.
y = x.T.reshape(-1)  # (4,)

energy_terms = y.conj() * y  # (4,)

# Complex outputs require an explicit cotangent. Tiga uses the
# conjugate-Wirtinger VJP convention.
dx = gf.autograd.grad(
    energy_terms,
    x,
    grad_output=gf.tensor([1.0 + 0.0j] * 4),
)
# d energy_terms / d x = 2x (conjugate-Wirtinger, unit cotangent)
# --8<-- [end:core]
