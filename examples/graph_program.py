"""Automatic horizontal fusion across two MessagePassing leaf calls."""

import torch

import graphforge as gf


class WeightedSum(gf.MessagePassing):
    reducer = gf.sum()

    def edge(self, src, dst, edge):
        return src.x * edge.weight


device = "cuda" if torch.cuda.is_available() else "cpu"
nodes, degree = 4096, 16
row_ptr = torch.arange(nodes + 1, device=device, dtype=torch.int64) * degree
col_idx = torch.arange(
    nodes * degree, device=device, dtype=torch.int64
) % nodes
graph = gf.Graph.from_csr(row_ptr, col_idx, num_src=nodes)
x = torch.randn(nodes, device=device)
w0 = torch.randn(nodes * degree, device=device)
w1 = torch.randn(nodes * degree, device=device)


@gf.program
def two_observables():
    # These calls capture typed SSA leaves. There is no explicit gf.compile().
    return (
        WeightedSum()(graph=graph, src={"x": x}, dst={},
                      edge={"weight": w0}),
        WeightedSum()(graph=graph, src={"x": x}, dst={},
                      edge={"weight": w1}),
    )


first, second = two_observables()
print(first.program.explain())
print(first.program.ir("kernel"))

if device == "cuda":
    out0, out1 = first.program.run()  # observation triggers JIT
    print(out0[:4], out1[:4])
    print(first.program.code("ptx")[:200])
