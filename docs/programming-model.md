# Programming model

Tiga separates **what to compute** from **how to run it**. A program
consists of two things — a `Graph` describing *who talks to whom*, and a
`MessagePassing` program describing *what is said* — and one ordinary call
compiles the pair:

![Programming model: Graph supplies topology, MessagePassing supplies semantics, the call JIT-compiles the pair](assets/programming-model.svg)

## A 60-second example

This is the exact program drawn above — three nodes, five edges, one UDF:

```python
import torch
import tiga as gf

class Conductive(gf.MessagePassing):
    reducer = gf.sum()

    def edge(self, src, dst, edge):        # message on edge e=(j→i)
        return edge.c * src.T

    def node(self, dst, incoming):         # optional node update
        return incoming + dst.b

# the graph in the diagram: edges 0→0, 2→0, 1→1, 0→2, 1→2
graph = gf.Graph.from_csr(torch.tensor([0, 2, 3, 5]),               # (N+1,)
                          torch.tensor([0, 2, 1, 0, 1]),            # (E,)
                          num_src=3)
T = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)               # src field  (N,)
c = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0], requires_grad=True)     # edge field (E,)
b = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)               # dst field  (N,)

out = Conductive()(graph=graph, src={"T": T}, dst={"b": b}, edge={"c": c})
# out = [11.1, 8.2, 17.3]; e.g. out[0] = b0 + c0·T0 + c1·T2
out.sum().backward()     # torch autograd — no backward function is written
# T.grad = [7, 10, 3]   c.grad = [1, 3, 2, 1, 2]   b.grad = [1, 1, 1]
```

Torch tensors are used directly — no conversion, no copy. The same program
also runs torch-free on the native `gf.Tensor` runtime, where gradients come
from `gf.autograd.grad` and the compiler-generated VJP.

No compile step, no backward function — four ideas make this work:

## Graph is a relation, not a CSR tensor

`gf.Graph` records logical topology and provenance. CSR is one possible
materialization; dense Cartesian and generated radius/kNN relations may stay
implicit.

```python
static = gf.Graph.from_csr(row_ptr, col_idx, num_src=n)   # frozen adjacency
dense = gf.Graph.dense(n, device=x.device)                # full N×N relation
dynamic = gf.Graph.radius(position, cutoff=0.3)           # geometry-generated
nearest = gf.Graph.knn(position, k=16)                    # exact kNN search
packed = gf.Graph.cat([gf.Graph.triangular(L) for L in lengths])  # varlen blocks
```

Three internal axes remain independent:

| Axis | Meaning | Examples |
|---|---|---|
| origin | where adjacency comes from | external indices, procedural rule |
| lifecycle | when it changes | frozen, rebuildable |
| realization | how it executes | materialized, paged, generated |

This is why an SSD-backed or distributed relation uses the same `Graph` API:
storage is a physical plan, not a different algorithm API.

## MessagePassing describes semantics

`src`, `dst` and `edge` are staged field namespaces. Field names are user data,
not compiler pattern switches: a legal optimization must be proved from the
captured expression, relation properties and reducer algebra. The
[message passing guide](message-passing.md) covers the full interface —
namespaces, parameter forwarding, checkpoints, inspection.

## Reducers carry algebra

A reducer is not a string such as `"sum"` — it is explicit `identity` / `lift`
/ `combine` / `finalize` regions with declared associativity, so tuple state
can express stable streaming algorithms such as online softmax. The compiler
sees a structured algebra, never an `Attention` opcode or a class-name
dispatch. The [reducers guide](reducers.md) walks through the contract and the
lowering ladder.

## Calling is JIT compilation

```python
out = Program()(graph=graph, src=src, dst=dst, edge=edge)
```

The normal call captures, specializes, lowers and caches automatically. An
explicit compile API is reserved for prewarming, AOT export or deployment — not
required for ordinary execution.
