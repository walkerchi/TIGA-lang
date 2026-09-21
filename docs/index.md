---
title: Tiga — Graph Message Passing JIT Compiler for PyTorch
homepage: true
description: Write differentiable graph programs with PyTorch tensors. Tiga compiles message passing for sparse aggregation, graph neural network operators and GPU graph workloads.
hide:
  - navigation
  - toc
---

<div class="tiga-home-hero" markdown>

# ![Tiga](assets/tiga-logo.svg){ width="300" .tiga-brand-logo } { #tiga }

A differentiable [JIT compiler](https://en.wikipedia.org/wiki/Just-in-time_compilation)
for [graph message passing](https://en.wikipedia.org/wiki/Graph_neural_network#Message_passing).

</div>

Define a graph, write what each edge sends, and choose how messages combine.
Call the program to obtain node results and derive gradients without a
handwritten backward function. Ordinary use does not require compiler IR.

Directly supports [PyTorch](https://en.wikipedia.org/wiki/PyTorch) `torch.Tensor`
inputs and outputs without additional wrappers. Compute gradients with
`torch.autograd.grad` or `.backward()` and integrate Tiga into existing
PyTorch models and training workflows.

## Start here { #choose-a-path }

<div class="gf-learning-paths" markdown>

<div class="gf-learning-path" markdown>

### Use Tiga

1. [Install and run a first program](getting-started.md).
2. [Understand graphs, fields and messages](programming-model.md).
3. [Check execution and troubleshoot](execution.md).
4. [Choose a worked example](examples.md).

The starting point is Python; no MLIR knowledge is assumed.

</div>

<div class="gf-learning-path" markdown>

### Develop the compiler

1. [Understand the internal layers](compiler-pipeline.md).
2. [Follow one real program through its IR](ir-walkthrough.md).
3. [Build, test and locate the implementation](development.md).

Start after the Python programming model. The walkthrough explains
notation before showing passes and generated output.

</div>

</div>

## What a program looks like { #a-small-semantic-surface }

This example aggregates weighted neighbor values on an NVIDIA GPU and computes gradients with PyTorch:

```python
import torch
import tiga as tg

device = torch.device("cuda")

class WeightedAggregation(tg.MessagePassing):
    reducer = tg.sum()

    def edge(self, src, dst, edge):
        return edge.weight * src.x

graph = tg.Graph.from_csr(
    torch.tensor([0, 2, 3, 5], dtype=torch.int64, device=device),
    torch.tensor([0, 2, 1, 0, 1], dtype=torch.int64, device=device),
    num_src=3,
)
x = torch.tensor([1.0, 2.0, 3.0], device=device, requires_grad=True)
weight = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0], device=device, requires_grad=True)
program = WeightedAggregation()
out = program(
    graph=graph,
    src={"x": x},
    dst={},
    edge={"weight": weight},
)
print(out.tolist())  # [11.0, 8.0, 17.0]
dx, dw = torch.autograd.grad(out.sum(), (x, weight))
print(dx.tolist())   # [7.0, 10.0, 3.0]
print(dw.tolist())   # [1.0, 3.0, 2.0, 1.0, 2.0]
```

The graph supplies the edges; `edge()` supplies each message; the
[reducer](https://en.wikipedia.org/wiki/Fold_(higher-order_function)) `tg.sum()`
combines messages per destination node. `tg` is only an import alias for
`tiga`, not another package.

Graph indices, inputs, outputs and gradients reside on the GPU, with no tensor
wrapping or change of autograd interface. See the [installation guide](getting-started.md)
for environment setup.
[Execution diagnostics](execution.md) explains how to inspect the selected path.

## Current scope { #what-is-executable-today }

Supports CPU and NVIDIA GPU execution, with hierarchical storage and distributed
execution for large graphs. Standard computations accept `torch.Tensor` directly;
advanced runtime features such as hierarchical storage and distributed execution
use native `tg.Tensor`, as described in [memory and distributed execution](memory-and-distributed.md).
See the [support matrix](roadmap.md) for operation, dtype and execution-path coverage.

## Sparse graphs and neural network operators

Weighted neighbor aggregation is a [sparse matrix](https://en.wikipedia.org/wiki/Sparse_matrix)
operation; custom edge functions extend it to more general graph programs.
The [PyTorch sparse and message passing guide](pytorch-sparse-message-passing.md)
connects `torch.sparse.mm`, CSR graphs and
[graph neural network](https://en.wikipedia.org/wiki/Graph_neural_network) (GNN)
aggregation with one checked example. Tiga compiles these operators; it is not
a replacement for a complete model and training framework.

## Understand the implementation { #from-semantics-to-provider-code }

The compiler turns a graph computation into a traversal and an execution plan.
These are internal decisions, not extra APIs to write. The
[compiler introduction](compiler-pipeline.md) explains their purpose; the
[IR walkthrough](ir-walkthrough.md) shows actual `gf.apply`,
`gf_iter.traverse`, `gf_kernel.launch` and task operations.
The [technical report](https://github.com/walkerchi/tiga-lang-paper) develops the
programming model and compiler design, with evaluation and reproduction data.

## Evaluate performance { #retained-structure-changes-the-algorithm }

The [performance and scalability report](experiments.md) compares execution time,
memory use, large-graph capacity and distributed execution for specified workloads.
The [measurement protocol](performance.md) defines timing and validation;
the [benchmark archive](benchmark-results.md) contains additional measured cases.

For signatures, use the [Python API reference](api.md).
For project positioning, see [comparisons](comparison.md).
