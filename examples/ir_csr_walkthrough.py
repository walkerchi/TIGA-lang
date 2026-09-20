"""Read destination-major CSR traversal using ordinary Torch tensors."""

import torch
import tiga as tg


class WeightedAggregation(tg.MessagePassing):
    reducer = tg.sum()

    def edge(self, src, dst, edge):
        return src.x * edge.weight


def main():
    # --8<-- [start:inputs]
    row_ptr = torch.tensor([0, 2, 3, 3], dtype=torch.int64)
    col_idx = torch.tensor([0, 1, 1], dtype=torch.int64)
    x = torch.tensor([2., 3.], requires_grad=True)
    weight = torch.tensor([4., 5., 2.])
    # --8<-- [end:inputs]
    graph = tg.Graph.from_csr(row_ptr, col_idx, num_src=2, validate="full")
    kernel = WeightedAggregation()
    output = kernel(graph=graph, src={"x": x}, dst={}, edge={"weight": weight})
    (gradient,) = torch.autograd.grad(output.sum(), x)
    torch.testing.assert_close(output, torch.tensor([23., 6., 0.]))
    torch.testing.assert_close(gradient, torch.tensor([4., 7.]))
    print("output:", output.tolist())
    print("d(sum(output))/dx:", gradient.tolist())
    return output, gradient


if __name__ == "__main__":
    main()
