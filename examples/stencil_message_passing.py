"""Five-point periodic grid averaging with Torch tensors and autograd."""
import argparse

# --8<-- [start:core]
import torch
import tiga as tg


class GridAverage(tg.MessagePassing):
    reducer = tg.mean()

    def edge(self, src, dst, edge):
        return src.x


def run(device="cuda"):
    graph = tg.Graph.stencil(
        (5, 5),
        tg.stencil.von_neumann(),  # Radius 1, including the center: five points.
        periodic=True, device=device,
    )
    x = torch.arange(25, dtype=torch.float32, device=device).requires_grad_()
    y = GridAverage()(graph=graph, src={"x": x}, dst={}, edge={})
    dx, = torch.autograd.grad(y.sum(), (x,))
    return y.reshape(5, 5), dx.reshape(5, 5)
# --8<-- [end:core]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    result, gradient = run(args.device)
    print(result.tolist())
    print(gradient.tolist())  # Every input contributes a total weight of 1.
