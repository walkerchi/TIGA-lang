"""Default Torch interface, plus an optional native-storage round trip."""

import torch

import tiga as tg

nodes, degree = 8, 2
x = torch.linspace(0, 1, nodes)                                            # (N,)
weight = torch.ones(nodes * degree)                                        # (E,)


# --8<-- [start:core]
class WeightedAggregation(tg.MessagePassing):
    reducer = tg.sum()

    def edge(self, src, dst, edge):
        return edge.weight * src.x


# Ring adjacency with unit weights: out[i] = x[(i-1) mod N] + x[(i+1) mod N].
row_ptr = torch.arange(0, nodes * degree + 1, degree, dtype=torch.int64)  # (N+1,)
col_idx = (  # (E,)
    torch.arange(nodes)[:, None] + torch.tensor([-1, 1])
).remainder(nodes).flatten()

# Torch tensors call the UDF directly — no conversion, no copy.
graph = tg.Graph.from_csr(row_ptr, col_idx, num_src=nodes, validate="full")
kernel = WeightedAggregation()
output = kernel(  # (N,)
    graph=graph, src={"x": x}, dst={}, edge={"weight": weight})
# --8<-- [end:core]

# from_torch/to_torch are NOT required for the call above. They exist for the
# other direction: entering the native tg.Tensor system (deferred expression
# capture, compiler-generated VJP) while still sharing torch storage.
native = tg.from_torch(x)
round_trip = native.to_torch()  # same storage, zero copy

# Inspect: kernel.explain(), kernel.schedules
#          (schedule.resources, schedule.roles, schedule.instructions)
