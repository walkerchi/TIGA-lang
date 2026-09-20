"""Advanced native-only compiler checkpoint inspection; not the default tensor API."""

# --8<-- [start:quickstart]
import tiga as tg


# 3 nodes, 5 edges: 0→0, 2→0, 1→1, 0→2, 1→2
row_ptr = tg.tensor([0, 2, 3, 5], dtype=tg.int64)        # (N+1,)
col_idx = tg.tensor([0, 2, 1, 0, 1], dtype=tg.int64)     # (E,)
graph = tg.Graph.from_csr(row_ptr, col_idx, num_src=3, validate="full")

temperature = tg.tensor([1.0, 2.0, 3.0], requires_grad=True)             # (N,)
conductivity = tg.tensor([2.0, 3.0, 4.0, 5.0, 6.0], requires_grad=True)  # (E,)
bias = tg.tensor([0.1, 0.2, 0.3], requires_grad=True)                    # (N,)


# --8<-- [start:core]
class ConductiveAggregation(tg.MessagePassing):
    reducer = tg.sum()

    def edge(self, src, dst, edge):
        # message on edge e=(j→i): conductivity[e] * temperature[j] — scalar
        return edge.conductivity * src.temperature

    def node(self, dst, incoming):
        # incoming: (N,) reduced messages; dst.bias: (N,)
        return incoming + dst.bias


kernel = ConductiveAggregation()
output = kernel(  # (N,)
    graph=graph,
    src={"temperature": temperature},
    dst={"bias": bias},
    edge={"conductivity": conductivity},
)
# --8<-- [end:core]
# out = [2·1 + 3·3 + 0.1,  4·2 + 0.2,  5·1 + 6·2 + 0.3]

loss = output.sum()                                                      # scalar
d_temperature, d_conductivity, d_bias = tg.autograd.grad(
    loss, (temperature, conductivity, bias))
# --8<-- [end:quickstart]
# d temperature[j] = Σ conductivity[e] over edges e out of j
# d conductivity[e] = temperature[src(e)]

# The same functional API exposes the compiler's storage decision. ``save``
# inserts an explicit identity-like checkpoint in VJP IR; ``recompute`` keeps
# the primal expression fused into backward. ``auto`` emits a checkpoint
# candidate which the MLIR budget pass selects or rejects before lowering.
saved_d_conductivity = tg.autograd.grad(
    output,
    conductivity,
    grad_output=tg.tensor([1.0, 1.0, 1.0]),
    checkpoint="save",
)

# Inspect the captured graph, the generated VJP and the planner choice:
#   output.expression()               forward relation Tensor IR
#   tg.autograd.grad_mlir(loss, temperature)   compiler-generated VJP IR
#   kernel.explain()                  provider/lowering/cache summary
