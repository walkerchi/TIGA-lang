"""Run and differentiate a Torch-free tuple-state reducer kernel."""

import graphforge as gf


class Mean(gf.Reducer):
    associative = True
    commutative = True

    def identity(self):
        return 0.0, 0.0

    def lift(self, value):
        return value, 1.0

    def combine(self, left, right):
        return left[0] + right[0], left[1] + right[1]

    def finalize(self, state):
        return state[0] / state[1]


def main() -> None:
    reducer = Mean()
    row_ptr = gf.tensor([0, 2, 3, 5], dtype=gf.int64)
    col_idx = gf.tensor([0, 2, 1, 0, 1], dtype=gf.int64)
    graph = gf.Graph.from_csr(row_ptr, col_idx, num_src=3, validate="full")
    values = gf.tensor([1.0, 2.0, 4.0], requires_grad=True)

    class NeighborMean(gf.MessagePassing):
        reducer = Mean()

        def edge(self, src, dst, edge):
            del dst, edge
            return self.reducer(src.value)

    kernel = NeighborMean()
    output = kernel(graph=graph, src={"value": values}, dst={})
    gradient = gf.autograd.grad(output.sum(), values)

    print("output:", output.tolist())
    print("d(sum(output))/dvalue:", gradient.tolist())
    print("\n--- captured reducer algebra ---")
    print(reducer.mlir(message_dtypes=(gf.float32,), symbol="user_mean"))
    print("\n--- compiler-generated backward ---")
    print(gf.autograd.grad_mlir(output.sum(), values))
    print("\n--- reducer selection ---")
    print(kernel.explain())


if __name__ == "__main__":
    main()
