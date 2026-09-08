"""Run and differentiate a Torch-free tuple-state reducer kernel."""

import tiga as gf


# 3 nodes, 5 edges: 0→0, 2→0, 1→1, 0→2, 1→2
row_ptr = gf.tensor([0, 2, 3, 5], dtype=gf.int64)    # (N+1,)
col_idx = gf.tensor([0, 2, 1, 0, 1], dtype=gf.int64)  # (E,)
graph = gf.Graph.from_csr(row_ptr, col_idx, num_src=3, validate="full")
values = gf.tensor([1.0, 2.0, 4.0], requires_grad=True)          # (N,)


# --8<-- [start:core]
class Mean(gf.Reducer):
    associative = True
    commutative = True

    def identity(self):
        return 0.0, 0.0  # (sum, count)

    def lift(self, value):
        return value, 1.0

    def combine(self, left, right):
        return left[0] + right[0], left[1] + right[1]

    def finalize(self, state):
        return state[0] / state[1]


class NeighborMean(gf.MessagePassing):
    reducer = Mean()

    def edge(self, src, dst, edge):
        # staged reducer item: (src.value[j], 1) per edge
        return self.reducer(src.value)


kernel = NeighborMean()
output = kernel(graph=graph, src={"value": values}, dst={})  # (N,)
# --8<-- [end:core]

gradient = gf.autograd.grad(output.sum(), values)
# d values[j] = Σ 1/degree(dst(e)) over edges e out of j

# Inspect the captured reducer algebra and the generated backward:
#   kernel.reducer.mlir(message_dtypes=(gf.float32,), symbol="user_mean")
#   gf.autograd.grad_mlir(output.sum(), values)
#   kernel.explain()


def main() -> None:
    return output, gradient


if __name__ == "__main__":
    main()
