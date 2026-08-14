# Programming model

GraphForge separates logical semantics from physical execution. The public
surface has four core ideas.

## Graph is a relation, not a CSR tensor

`gf.Graph` records logical topology and provenance. CSR is one possible
materialization, while dense Cartesian and generated radius relations may stay
implicit.

```python
static = gf.Graph.from_csr(row_ptr, col_idx, num_src=n)
dense = gf.Graph.dense(n, device=x.device)
dynamic = gf.Graph.radius(position, cutoff=0.3)
```

Three internal axes remain independent:

| Axis | Meaning | Examples |
|---|---|---|
| origin | where adjacency comes from | external indices, procedural rule |
| lifecycle | when it changes | frozen, rebuildable |
| realization | how it executes | materialized, paged, generated |

This is why an SSD-backed or distributed relation should eventually use the
same `Graph` API: storage is a physical plan, not a different algorithm API.

## MessagePassing describes semantics

```python
class Diffusion(gf.MessagePassing):
    reducer = gf.sum()

    def edge(self, src, dst, edge):
        return edge.weight * (src.u - dst.u)
```

`src`, `dst` and `edge` are staged field namespaces. Field names are user data,
not compiler pattern switches. A legal optimization must be proved from the
captured expression, relation properties and reducer algebra.

## Reducers carry algebra

A reducer is not just a string such as `"sum"`. Its compiler form has explicit
`identity`, `lift`, `combine` and `finalize` regions plus associativity and
commutativity properties. Multi-value messages and tuple state allow stable
streaming algorithms such as online softmax.

```python
class DenseAttention(gf.MessagePassing):
    reducer = gf.online_softmax()

    def edge(self, src, dst, edge, scale):
        score = (src.key * dst.query).sum(dim=-1) * scale
        return self.reducer(score, src.value)
```

The compiler sees a contraction and a structured streaming reducer; it does not
see an `Attention` opcode or dispatch by class name.

## Calling is JIT compilation

```python
out = Program()(graph=graph, src=src, dst=dst, edge=edge)
```

The normal call captures, specializes, lowers and caches automatically. An
explicit compile API is reserved for prewarming, AOT export or deployment—not
required for ordinary execution.
