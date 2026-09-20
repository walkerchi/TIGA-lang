# How Tiga compares

Tiga is a **compiler** for message passing and graph computation — not a
framework, and not a general deep-learning compiler. The honest comparison
depends on what is reached for today.

## Timing one concrete operation { #aggregation-timing }

Each destination collects its neighbors' 16-feature vectors, multiplies them by
edge weights and sums the messages. The chart shows forward execution on the same
input: **milliseconds (ms), shorter is faster**, not a speedup ratio.
Purple marks the two Tiga paths; gray marks other specific implementations,
not entire frameworks.

[![Historical graph aggregation forward times in ms; shorter is faster. Prepared setup is outside timing.](assets/results/spmm-provider-compare.svg)](assets/results/spmm-provider-compare.svg)

On narrow screens, scroll the chart horizontally or open the image for the full
figure. All values are also available in the expandable table below.

For example, `Tiga (regular call)` takes 0.090 ms and `Torch sparse matmul` takes
0.263 ms. The 0.057 ms for `Tiga (prepared)*` requires fixed input bindings and
completed preparation; it is not first-call latency. These are warmed forward
executions, excluding first compilation, graph construction, transfers and backward.
Hot cache means repeated execution without deliberate cache flushing, not a
guarantee that all data fits in L2.

??? info "Complete timings and implementation meanings (including reference)"

    | Chart label | Archive identifier | Implementation | Time (ms) |
    |---|---|---|---:|
    | Tiga (prepared)* | `tiga.prepared_auto` | Repeated execution with fixed input bindings; preparation excluded | 0.057 |
    | Tiga (regular call) | `tiga.auto` | Warmed regular call | 0.090 |
    | Torch sparse matmul | `torch.sparse.mm` | Sparse matrix multiplication | 0.263 |
    | Torch compiled scatter | `torch.compile.index_add` | Compiled gather, multiply and scatter-add | 0.278 |
    | Handwritten Triton | `triton.csr` | One handwritten CSR kernel in the benchmark, not a Triton performance ceiling | 0.388 |
    | PyG gather-scatter | `pyg.message_passing` | This example's gather-scatter, not fused sparse aggregation | 1.673 |
    | Torch scatter | `torch.index_add` | Uncompiled gather, multiply and scatter-add | 1.776 |
    | Table only | `tiga.reference` | Correctness reference, not an optimized execution path | 3.061 |

    Values are medians of archived repeated measurements, rounded to three decimals.
    The benchmark checks forward outputs against `torch.sparse.mm` before timing;
    this does not validate gradients or measure full training.

Conditions: RTX 5070 Ti, 131,072 nodes, mean degree about 16, irregular CSR with
random locality, 16 features, FP32 values and 32-bit indices. These are **historical
measurements**, not new results for the current revision or current default Torch
path. [Chart reproduction](benchmark-reproducibility.md) provides raw data and
redrawing commands.

Remeasure this workload (results can change with versions and environment):

```bash
python -m benchmarks.sparse_compute.weighted_aggregation --topology irregular --locality random --device cuda --index-dtype i32 --nodes 131072 --degree 16 --features 16,64
```

A complete program is in the [CSR recipe](api-examples.md#message-passing);
this figure alone is not an API-selection criterion.

## At a glance

| Alternative | What it is | How Tiga differs |
|---|---|---|
| [PyG](https://pytorch-geometric.readthedocs.io/en/latest/notes/sparse_tensor.html) | GNN framework with gather-scatter and fused sparse aggregation paths | the figure measures PyG 2.8.0 gather-scatter, not `message_and_aggregate()`; this result does not generalize to all PyG operators |
| [DGL](https://www.dgl.ai/) | GNN framework with fused `gspmm` / `gsddmm` operators | Tiga emphasizes relation provenance, generated topology and VJP; no matched DGL ranking is established here |
| [torch.compile](https://pytorch.org/docs/stable/torch.compiler.html) | general PyTorch compilation entry point | Tiga starts from relation and reducer semantics; capture, fusion and performance cannot be generalized from one version or sparse case |
| [Taichi](https://www.taichi-lang.org/) / [Warp](https://nvidia.github.io/warp/) | Python kernel JIT | a different abstraction: Tiga starts from relations and edge/node UDFs |
| [Triton](https://triton-lang.org/) | tile / kernel DSL and compiler | Tiga's current NVIDIA path hands off serialized TTIR to Triton |
| [TileLang](https://github.com/tile-ai/tilelang) | external tile / kernel DSL | there is no integrated TileLang provider in Tiga today |
| [Graphiler](https://github.com/xiezhq-hermann/graphiler) | research GNN compiler on DGL + TorchScript | different frontend/runtime contracts; no maintenance-status or performance ranking is inferred here |
| [GALA](https://github.com/ADAPT-uiuc/GALA-GNN-Acceleration-LAnguage) | research GNN DSL and compiler | its own frontend and benchmark suite require workload alignment before comparison |

## When Tiga is the right tool, and when it is not

Evaluate Tiga for custom graph messages, sparse aggregation and research that
needs inspectable generated code. Validate each workload's outputs and gradients
before benchmarking matched inputs and timing boundaries. Supported edge-nn paths
have compiled VJPs; some Torch CSR backward paths replay Torch semantics, so
not every forward/backward is a fused compiled kernel.

Tiga is alpha, not a full training framework, model library or production-ready
cross-platform distribution. An included PageRank or attention example does not
replace those ecosystems. Ordinary applications keep Torch tensors; Torch is
installed separately, and the native CPU runtime can operate without it.
See [current status](roadmap.md#latest-validation) for validated coverage,
memory limits and multi-GPU boundaries.

The deep design survey behind these choices — Ebb, Simit, TACO, GraphIt,
Taichi, Legion, and the rest — is kept in the repository as
[`docs/RELATED_WORK.md`](https://github.com/walkerchi/TIGA-lang/blob/main/docs/RELATED_WORK.md).
