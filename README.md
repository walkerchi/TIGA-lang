<div align="center">
  <img src="assets/tiga-logo.png" alt="Tiga — TG emblem" width="380">
  <p>English · <a href="https://github.com/walkerchi/TIGA-lang/blob/main/README.zh.md">简体中文</a></p>
  <p><strong>A differentiable JIT compiler for graph message-passing programs.</strong></p>
  <p><a href="https://github.com/walkerchi/TIGA-lang/blob/main/LICENSE">Apache-2.0</a> · Alpha · Package: <code>tiga-lang</code> · Import: <code>tiga</code></p>
  <p>
    <a href="https://github.com/walkerchi/TIGA-lang/blob/main/docs/getting-started.md">Getting started</a> ·
    <a href="https://github.com/walkerchi/TIGA-lang/blob/main/docs/getting-started.zh.md">中文指南</a> ·
    <a href="https://github.com/walkerchi/TIGA-lang/blob/main/docs/api.md">Python API</a> ·
    <a href="https://github.com/walkerchi/TIGA-lang/blob/main/docs/roadmap.md">Status and limitations</a>
  </p>
</div>

Tiga captures a graph relation, an edge function and a
[reducer](https://en.wikipedia.org/wiki/Fold_(higher-order_function)) as
[compiler IR](https://en.wikipedia.org/wiki/Intermediate_representation).
It uses that structure to select traversal, fuse message computation with
aggregation, and derive gradients. Supported paths run through
[JIT compilation](https://en.wikipedia.org/wiki/Just-in-time_compilation)
on CPU or NVIDIA GPU. The default interface uses ordinary PyTorch tensors;
the standalone native tensor runtime also runs without PyTorch.

## Installation

### From PyPI (after the first release)

The first PyPI release is still being prepared. Once published, install with:

```bash
# For the Torch example below; skip if Torch is already installed or not needed.
python -m pip install torch
python -m pip install tiga-lang
python -m tiga
```

The distribution name is `tiga-lang`; the Python import is `tiga`.
Torch is **not** a default dependency. For GPU use, install a hardware-compatible
Torch build first. Tiga's default installation does not install or upgrade Torch.
Without Torch, use [native Tensor, CPU MessagePassing and autograd](https://github.com/walkerchi/TIGA-lang/blob/main/docs/execution.md#native-execution).

For the optional Triton provider, replace the Tiga install command with
`python -m pip install "tiga-lang[cuda]"`; this installs neither Torch nor a GPU driver.
A compatible prebuilt wheel bundles Tiga's compiler tools and native libraries,
so a separate LLVM/MLIR SDK is not required. Wheels target Linux x86-64,
CPython 3.11/3.12 and glibc ≥ 2.38; see [installation details](https://github.com/walkerchi/TIGA-lang/blob/main/docs/getting-started.md#pypi-install).

### Install from source

Requirements: Python 3.11–3.12, a C++ compiler, CMake, Ninja, and a prebuilt
LLVM/MLIR **22.1.8** SDK. The supported build environment is Linux x86-64
with CPython 3.11/3.12. The optional CUDA adapter uses Torch 2.11.x and Triton 3.6.x.

```bash
git clone https://github.com/walkerchi/TIGA-lang.git tiga-lang
cd tiga-lang
python -m venv .venv
source .venv/bin/activate
# For the Torch example below, install Torch first (or use an existing Torch environment).
python -m pip install torch
export TIGA_LLVM_ROOT=/path/to/llvm-22.1.8
export CMAKE_ARGS="-DMLIR_DIR=$TIGA_LLVM_ROOT/lib/cmake/mlir -DLLVM_DIR=$TIGA_LLVM_ROOT/lib/cmake/llvm"
python -m pip install -e .
python -m tiga
```

Replace the SDK path before installing.

For installation failures, input pitfalls and bug reports, see the bilingual
[support guide](https://github.com/walkerchi/TIGA-lang/blob/main/docs/support.md) / [排错与反馈](https://github.com/walkerchi/TIGA-lang/blob/main/docs/support.zh.md).
Maintainer: **walkerchi**, Independent Developer —
[walker.chi.000@gmail.com](mailto:walker.chi.000@gmail.com).
Private vulnerabilities follow [SECURITY.md](https://github.com/walkerchi/TIGA-lang/blob/main/SECURITY.md), not public issues.

For source development, use `python -m pip install -e ".[test,docs]"` to add
test and documentation dependencies. These explicitly selected development
extras include Torch; the base installation does not.

## A complete differentiable program

A destination-row [CSR](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format))
graph supplies the topology; the edge function supplies the computation.
This NVIDIA GPU example uses Torch inputs, outputs and autograd:

```python
import torch
import tiga as tg

device = torch.device("cuda")

class WeightedSum(tg.MessagePassing):
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

kernel = WeightedSum()
out = kernel(graph=graph, src={"x": x}, dst={}, edge={"weight": weight})
dx, dw = torch.autograd.grad(out.sum(), (x, weight))

print(out.tolist())              # [11.0, 8.0, 17.0]
print(dx.tolist())               # [7.0, 10.0, 3.0]
print(dw.tolist())               # [1.0, 3.0, 2.0, 1.0, 2.0]
```

No explicit compile call, tensor conversion or handwritten backward is needed.
`out` is a `torch.Tensor` and works with ordinary Torch operations and autograd.
Use `kernel.explain()` to inspect the selected execution plan and
`kernel.cache_info` to inspect capture-variant cache statistics.
See the [programming model](https://github.com/walkerchi/TIGA-lang/blob/main/docs/programming-model.md) and
[runnable examples](https://github.com/walkerchi/TIGA-lang/blob/main/docs/examples.md) for node updates, custom reducers,
dynamic relations, and PyTorch interop.

## Current scope

| Path | Available scope | Boundary |
|---|---|---|
| Native tensors | Torch-free CPU runtime, supported tensor operations, CSR message passing and automatic gradients | Small programs may use the oracle under `auto`; unsupported native operations report errors |
| Default Torch interface | Additional dense/generated relation and neural-module paths | Coverage and fusion depend on relation, shape, dtype and selected provider |
| CPU / NVIDIA GPU | LLVM CPU JIT and serialized TTIR → Triton → PTX paths | No validated AMD or Intel GPU execution |
| Memory / distributed | Native-Tensor budgets, opt-in LRU spill/reload, paged graphs, MPI and CUDA TCP/NCCL halo/VJP | Paged CUDA is forward-only with resident output; partitioning uses fixed ownership |

The [support matrix](https://github.com/walkerchi/TIGA-lang/blob/main/docs/roadmap.md) distinguishes native and adapter
coverage. Graph construction support alone does not imply that every reducer
or gradient can execute on that graph.

Memory policy starts with `tg.execution(...)`; `.spill()` changes residency,
`.to(device)` copies, and `tg.save/load` persist value snapshots. See the
[complete memory example and limits](https://github.com/walkerchi/TIGA-lang/blob/main/docs/memory.md).
Distributed execution completes halo communication before computing local outputs.

## Performance evidence

See [performance and scalability](https://github.com/walkerchi/TIGA-lang/blob/main/docs/experiments.md)
for runtime and memory comparisons, single-GPU billion-edge capacity, and distributed overhead.

<img src="docs/assets/compiler-performance-overview.svg"
     alt="Six registered workloads compared with their matched baselines"
     width="1100">

The figure summarizes archived measurements for the specified workloads and
hardware. The [benchmark report](https://github.com/walkerchi/TIGA-lang/blob/main/docs/benchmark-results.md)
describes each baseline, timing scope and uncertainty. Raw data and plotting
commands are in the [reproduction guide](https://github.com/walkerchi/TIGA-lang/blob/main/docs/benchmark-reproducibility.md).

## Compiler and development

Relation semantics stay visible through Domain, Iter, Kernel and Task IR.
The compiler selects physical traversal, memory placement and target
[lowering](https://en.wikipedia.org/wiki/Compiler#Back_end), while gradients
are represented by an automatic
[VJP](https://en.wikipedia.org/wiki/Automatic_differentiation).
Read the [compiler pipeline](https://github.com/walkerchi/TIGA-lang/blob/main/docs/compiler-pipeline.md) for the boundaries.
The [IR walkthrough](https://github.com/walkerchi/TIGA-lang/blob/main/docs/ir-walkthrough.md) follows one checked compiler input
through real Domain, Iter, Kernel and Task output, with reproducible commands.
Internal `gf-*` tool names and `graphforge` C++ directories remain intentional;
the public distribution and import names are `tiga-lang` and `tiga`.

The [development guide](https://github.com/walkerchi/TIGA-lang/blob/main/docs/development.md) gives the complete Python,
MLIR, documentation and GPU test setup, including the test utilities omitted
from some LLVM SDKs. Contribution rules are in [CONTRIBUTING.md](https://github.com/walkerchi/TIGA-lang/blob/main/CONTRIBUTING.md);
changes are recorded in [CHANGELOG.md](https://github.com/walkerchi/TIGA-lang/blob/main/CHANGELOG.md).

Licensed under [Apache-2.0](https://github.com/walkerchi/TIGA-lang/blob/main/LICENSE).
