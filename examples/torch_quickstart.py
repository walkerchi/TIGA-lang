"""Default interface: ordinary Torch inputs, outputs and autograd."""

# --8<-- [start:quickstart]
import torch
import tiga as tg

device = torch.device("cuda")

class ConductiveAggregation(tg.MessagePassing):
    reducer = tg.sum()

    def edge(self, src, dst, edge):
        return edge.conductivity * src.temperature

    def node(self, dst, incoming):
        return incoming + dst.bias


graph = tg.Graph.from_csr(
    torch.tensor([0, 2, 3, 5], dtype=torch.int64, device=device),
    torch.tensor([0, 2, 1, 0, 1], dtype=torch.int64, device=device),
    num_src=3,
    validate="full",
)
temperature = torch.tensor([1.0, 2.0, 3.0], device=device, requires_grad=True)
conductivity = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0], device=device, requires_grad=True)
bias = torch.tensor([0.1, 0.2, 0.3], device=device, requires_grad=True)
kernel = ConductiveAggregation()
output = kernel(graph=graph, src={"temperature": temperature},
                dst={"bias": bias}, edge={"conductivity": conductivity})
d_temperature, d_conductivity, d_bias = torch.autograd.grad(
    output.sum(), (temperature, conductivity, bias))
# --8<-- [end:quickstart]

if __name__ == "__main__":
    print(output.tolist())         # approximately [11.1, 8.2, 17.3]
    print(d_temperature.tolist())  # [7.0, 10.0, 3.0]
    print(d_conductivity.tolist()) # [1.0, 3.0, 2.0, 1.0, 2.0]
    print(d_bias.tolist())         # [1.0, 1.0, 1.0]
