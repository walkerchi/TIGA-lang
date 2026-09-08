"""Torch-free differentiable diffusion UDF on an explicit CSR stencil."""

import tiga as gf


# 4 nodes, 8 edges: dst i gathers from two sources each
nodes = 4
row_ptr = gf.tensor([0, 2, 4, 6, 8], dtype=gf.int64)    # (N+1,)
col_idx = gf.tensor([3, 1, 0, 2, 1, 3, 2, 0], dtype=gf.int64)  # (E,)
graph = gf.Graph.from_csr(row_ptr, col_idx, num_src=nodes, validate="full")

u = gf.tensor([0.0, 1.0, 0.5, -0.5], requires_grad=True)         # (N,)
conductivity = gf.tensor([1.0] * 8, requires_grad=True)          # (E,)


# --8<-- [start:core]
class Diffusion(gf.MessagePassing):
    reducer = gf.sum()

    def edge(self, src, dst, edge):
        # message on edge e=(j→i): conductivity[e] * (u[j] − u[i]) — scalar
        return edge.conductivity * (src.u + (-1.0) * dst.u)

    def node(self, dst, flux, dt):
        # flux: (N,) reduced messages; dt arrives from the call site
        return dst.u + dt * flux


next_u = Diffusion()(
    graph=graph,
    ndata={"u": u},  # homogeneous graph: one declaration, both roles
    edge={"conductivity": conductivity},
    dt=0.1,
)  # (N,)
# --8<-- [end:core]
# flux = [0.5, -1.5, -0.5, 1.5]

du, d_conductivity = gf.autograd.grad(next_u.sum(), (u, conductivity))
# every node has in-degree = out-degree = 2, so the flux terms cancel in du
# d conductivity[e] = dt · (u[src(e)] − u[dst(e)])

# Inspect the captured forward relation Tensor graph: next_u.expression()
