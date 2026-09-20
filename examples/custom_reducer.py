"""Advanced custom reducer: this algebra currently needs the native runtime."""

import tiga as tg


# 3 nodes, 5 edges: 0→0, 2→0, 1→1, 0→2, 1→2
row_ptr = tg.tensor([0, 2, 3, 5], dtype=tg.int64)    # (N+1,)
col_idx = tg.tensor([0, 2, 1, 0, 1], dtype=tg.int64)  # (E,)
graph = tg.Graph.from_csr(row_ptr, col_idx, num_src=3, validate="full")
values = tg.tensor([1.0, 2.0, 4.0], requires_grad=True)          # (N,)


# --8<-- [start:core]
class Mean(tg.Reducer):
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


class NeighborMean(tg.MessagePassing):
    reducer = Mean()

    def edge(self, src, dst, edge):
        # staged reducer item: (src.value[j], 1) per edge
        return self.reducer(src.value)


kernel = NeighborMean()
output = kernel(graph=graph, src={"value": values}, dst={})  # (N,)
# --8<-- [end:core]

gradient = tg.autograd.grad(output.sum(), values)
# d values[j] = Σ 1/degree(dst(e)) over edges e out of j

# Inspect the selected plan: kernel.explain()


def main() -> None:
    return output, gradient


if __name__ == "__main__":
    main()
