"""Minimal matrix-free P1 FEM solve; native tensors capture the CG loop."""

# --8<-- [start:core]
import tiga as tg
from solvers import linear_solve  # Helper in examples/solvers.py, not tg API.


class Poisson(tg.MessagePassing):
    reducer = tg.sum()

    def edge(self, src, dst, edge):
        return -src.u

    def node(self, dst, incoming):
        return 2.0 * dst.u + incoming


# 1. Seven interior nodes; omitted boundary values are fixed to zero.
n = 7
h = 1.0 / (n + 1)
graph = tg.Graph.stencil((n,), ((-1,), (1,)))
kernel = Poisson()


def apply(u):
    return kernel(graph=graph, src={"u": u}, dst={"u": u}, edge={}) / h


# 2. Solve A(u) = b. The driver owns the iteration loop.
b = tg.tensor([h] * n, dtype=tg.float32)
u = linear_solve(apply, b, tolerance=1e-6, max_iterations=32)

# 3. Compare the nodal solution with x(1-x)/2.
exact = [0.5 * (i * h) * (1.0 - i * h) for i in range(1, n + 1)]
error = max(abs(a - e) for a, e in zip(u.tolist(), exact))
# --8<-- [end:core]

if __name__ == "__main__":
    print(u.tolist())
    print(f"max error: {error:.2e}")
