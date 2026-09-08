"""Automatic horizontal fusion across two MessagePassing leaf calls."""

import torch

import tiga as gf

device = "cuda" if torch.cuda.is_available() else "cpu"
nodes, degree = 4096, 16
x = torch.randn(nodes, device=device)                                        # (N,)
w0 = torch.randn(nodes * degree, device=device)                              # (E,)
w1 = torch.randn(nodes * degree, device=device)                              # (E,)


# --8<-- [start:core]
class WeightedSum(gf.MessagePassing):
    reducer = gf.sum()

    def edge(self, src, dst, edge):
        return src.x * edge.weight


# Regular degree-16 relation: dst i gathers from sources (i*degree + k) % N.
graph = gf.Graph.regular(nodes, degree, device=device)


@gf.jit
def two_observables():
    # @gf.jit captures the two sibling kernel calls automatically; there is
    # no explicit gf.compile() and no separate composition decorator.
    return (
        WeightedSum()(graph=graph, src={"x": x}, dst={},
                      edge={"weight": w0}),  # (N,)
        WeightedSum()(graph=graph, src={"x": x}, dst={},
                      edge={"weight": w1}),  # (N,)
    )


first, second = two_observables()
# The two leaves are planned as one fused kernel launch.
# --8<-- [end:core]

if device == "cuda":
    out0, out1 = first.program.run()  # observation triggers JIT

# Inspect: first.program.explain(), first.program.ir("kernel"),
#          first.program.code("ptx")
