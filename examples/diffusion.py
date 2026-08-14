"""Torch-free differentiable diffusion UDF on an explicit CSR stencil."""

import graphforge as gf


class Diffusion(gf.MessagePassing):
    reducer = gf.sum()

    def edge(self, src, dst, edge):
        return edge.conductivity * (src.u + (-1.0) * dst.u)

    def node(self, dst, flux, dt):
        return dst.u + dt * flux


nodes = 4
row_ptr = gf.tensor([0, 2, 4, 6, 8], dtype=gf.int64)
col_idx = gf.tensor([3, 1, 0, 2, 1, 3, 2, 0], dtype=gf.int64)
graph = gf.Graph.from_csr(row_ptr, col_idx, num_src=nodes, validate="full")

u = gf.tensor([0.0, 1.0, 0.5, -0.5], requires_grad=True)
conductivity = gf.tensor([1.0] * 8, requires_grad=True)
next_u = Diffusion()(
    graph=graph,
    src={"u": u},
    dst={"u": u},
    edge={"conductivity": conductivity},
    dt=0.1,
)
du, d_conductivity = gf.autograd.grad(
    next_u.sum(), (u, conductivity))

print("next_u:", next_u.tolist())
print("d(sum(next_u))/du:", du.tolist())
print("d(sum(next_u))/dconductivity:", d_conductivity.tolist())
print(next_u.expression())
