# GraphForge Compiler

GraphForge compiles relation-oriented tensor programs into efficient CPU/GPU
execution plans. You describe *what* flows across a relation and *how* messages
reduce; the compiler chooses traversal, tiling, fusion, memory placement and the
provider lowering.

!!! important "A compiler, not an operator library"

    GraphForge does not ship a `dense_attention()` or `radius_force()` kernel.
    Those are ordinary user programs. The core contains IR, analyses, passes,
    code generation, JIT/cache and external-library dispatch.

The project also contains the minimal Tensor/runtime/autograd substrate needed
to execute its compiler output without requiring a full framework. It does not
include optimizer, NN or dataset layers.

```python
import graphforge as gf

class WeightedAggregation(gf.MessagePassing):
    reducer = gf.sum()

    def edge(self, src, dst, edge):
        return edge.weight * src.x

out = WeightedAggregation()(
    graph=graph,
    src={"x": x},
    dst={"x": x},
    edge={"weight": weight},
)
```

The normal call is the JIT boundary. There is no required `gf.compile(...)`:
the first compatible call captures and compiles a variant; later calls reuse a
guarded executable. Every chosen variant is inspectable:

```python
print(program.explain())
print(program.ir("domain"))
print(program.ir("iter"))
print(program.ir("kernel"))
print(program.ir("gf.kernel.ttir"))
print(program.code("ptx"))
```

## What works today

| Compiler path | Current executable strategy |
|---|---|
| Scalar fixed/bounded-ragged CSR sum | compiler-emitted TTIR |
| Default Euclidean generated radius + distance sum | compiler-emitted TTIR |
| Dense Cartesian contraction + structured streaming reducer | compiler-emitted tensor-core TTIR |
| Vector CSR weighted sum | proven dispatch to `torch.sparse.mm` |
| Unsupported program/shape | explicit semantic evaluator |
| Torch-independent Tensor slice | native CPU buffer + symbolic add/mul/sum VJP oracle |

The current implementation is alpha software. See [performance](performance.md)
for measured evidence and [roadmap](roadmap.md) for unsupported shapes.

## Start here

- [Install and run the first program](getting-started.md)
- [Understand Graph, MessagePassing and reducers](programming-model.md)
- [Walk through the compiler IR](compiler-pipeline.md)
- [Understand the minimal Tensor runtime and autograd](runtime-and-autograd.md)
- [See how memory hierarchy and distributed execution fit](memory-and-distributed.md)
- [Run complete examples](examples.md)
- [Reproduce performance claims](performance.md)
