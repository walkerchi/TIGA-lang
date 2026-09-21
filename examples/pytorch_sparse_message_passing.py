"""Compare weighted graph aggregation with torch.sparse.mm, including gradients."""
import argparse

import torch
import tiga as tg


class WeightedAggregation(tg.MessagePassing):
    reducer = tg.sum()

    def edge(self, src, dst, edge):
        return edge.weight * src.x


def run(device="cuda"):
    # CSR rows are destinations; column indices identify source nodes.
    row = torch.tensor([0, 2, 3, 5], dtype=torch.int64, device=device)
    col = torch.tensor([0, 2, 1, 0, 1], dtype=torch.int64, device=device)
    x = torch.tensor([1., 2., 3.], device=device, requires_grad=True)
    weight = torch.tensor([2., 3., 4., 5., 6.], device=device,
                          requires_grad=True)

    graph = tg.Graph.from_csr(row, col, num_src=3)
    result = WeightedAggregation()(
        graph=graph, src={"x": x}, dst={}, edge={"weight": weight})

    # Independent leaves make the two gradient computations independent.
    x_ref = x.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()
    adjacency = torch.sparse_csr_tensor(
        row, col, weight_ref, size=(3, 3), device=device, check_invariants=True)
    reference = torch.sparse.mm(adjacency, x_ref[:, None]).squeeze(1)

    torch.testing.assert_close(result, reference)
    gradients = torch.autograd.grad(result.sum(), (x, weight))
    reference_gradients = torch.autograd.grad(
        reference.sum(), (x_ref, weight_ref))
    for actual, expected in zip(gradients, reference_gradients):
        torch.testing.assert_close(actual, expected)
    print("Output:", result.detach().tolist())  # [11.0, 8.0, 17.0]
    print("Node gradients:", gradients[0].tolist())  # [7.0, 10.0, 3.0]
    print("Edge gradients:", gradients[1].tolist())  # [1., 3., 2., 1., 2.]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    run(parser.parse_args().device)
