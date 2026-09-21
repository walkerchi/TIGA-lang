---
title: PyTorch Sparse Matrix Multiplication and Graph Message Passing
description: Match Tiga graph message passing to torch.sparse.mm on a GPU, check node and edge gradients, and understand CSR aggregation in graph neural networks.
---

# PyTorch sparse matrix multiplication and graph message passing

[PyTorch](https://en.wikipedia.org/wiki/PyTorch) exposes sparse matrix operations
through `torch.sparse`. Tiga expresses computation as
[message passing](https://en.wikipedia.org/wiki/Graph_neural_network#Message_passing)
along graph edges. For weighted neighbor summation, these are two ways to express
the same operation. General edge functions need not be matrix multiplication.

## How a sparse matrix represents a graph

In [compressed sparse row](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format))
(CSR) format, a row represents a destination node. Each stored column index
identifies a source node, and the corresponding value is an edge weight.
For an edge from source j to destination i:

$$
y_i = \sum_{e=(j\to i)} w_e x_j.
$$

With feature vectors, the same expression is sparse matrix–dense matrix
multiplication (SpMM). `torch.sparse.mm(adjacency, features)` and a Tiga
`edge.weight * src.x` message with a sum reducer describe this weighted sum.
The [graph programming model](programming-model.md#graph-is-a-relation-not-a-csr-tensor) illustrates how graph
connections map to data; the example below fixes the row/column convention explicitly.

## A GPU example with forward and gradient checks

Complete [installation](getting-started.md) first, including a CUDA-compatible
Torch installation. This example compares outputs and gradients with independent
Torch inputs; it is a correctness check, not a benchmark.

Run [`python examples/pytorch_sparse_message_passing.py --device cuda`](https://github.com/walkerchi/TIGA-lang/blob/main/examples/pytorch_sparse_message_passing.py).
The same source accepts `--device cpu` for a CPU check.

```python
--8<-- "examples/pytorch_sparse_message_passing.py"
```

Both implementations return ordinary `torch.Tensor` values. The five stored
edges remain separate edge weights. Rows without incoming edges sum to zero;
repeated source–destination pairs contribute once per stored edge in this sum.
Other reducers have their own [empty-neighborhood rules](reducers.md).
Torch sparse layout, device and gradient coverage are defined by
[`torch.sparse.mm`](https://docs.pytorch.org/docs/stable/generated/torch.sparse.mm.html);
Tiga coverage is listed in the [support matrix](roadmap.md).

## Where graph neural networks fit

A [graph neural network](https://en.wikipedia.org/wiki/Graph_neural_network)
(GNN) combines graph aggregation with feature transformations and nonlinearities.
The sum above is an aggregation building block, not a complete normalized
graph convolutional network (GCN) layer. Self-loops, degree normalization,
learned transformations and training objectives must be specified by the model.
See the [multi-feature GCN-shaped aggregation example](examples/message-passing.md#gcn-aggregation)
and [Torch model integration](examples/programs-and-interop.md).

PyTorch Geometric (PyG) provides graph learning layers and training utilities.
Tiga instead focuses on compiling graph programs, including custom edge messages
and their supported gradients. It is not a drop-in replacement for every PyG layer.
The [comparison guide](comparison.md) explains these abstraction boundaries.

## Beyond a fixed sparse adjacency matrix

Radius and k-nearest-neighbor graphs derive neighbors from coordinates or features;
stencils derive neighbors from grid rules. The
[dynamic relation examples](examples/dynamic-relations.md) explain these constructors,
distance metrics and boundaries. Whether a path materializes edges or generates
candidates during execution depends on the relation and selected backend.

For performance, compare the same graph, features, dtype, device and gradients.
Report graph construction, compilation, warmed execution and memory separately.
The [performance report](experiments.md) provides measured sparse aggregation,
radius/kNN, large-graph and distributed cases; it does not establish that Tiga
is faster for every graph neural network or `torch.sparse` operation.
