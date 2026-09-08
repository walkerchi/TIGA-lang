# How Tiga compares

Tiga is a **compiler** for message passing and graph computation — not a
framework, and not a general deep-learning compiler. The honest comparison
depends on what is reached for today.

![Measured provider comparison on the flagship sparse matrix–vector case: Tiga's fused kernel leads, PyG's gather-scatter MessagePassing and the eager index_add path trail by an order of magnitude](assets/results/spmm-provider-compare.svg)

The case above is the canonical GNN inner loop — a weighted sum over a
ragged CSR relation (N = 131 072, degree ≈ 16, feature width 16, measured on
an RTX 5070 Ti with a hot L2). Same relation, same math, every implementation
checked against `torch.sparse.mm`. Reproduce with
`python -m benchmarks.sparse_compute.weighted_aggregation --topology irregular --locality random --device cuda --index-dtype i32 --nodes 131072 --degree 16 --features 16,64`.

## At a glance

| Alternative | What it is | How Tiga differs |
|---|---|---|
| [PyG](https://pytorch-geometric.readthedocs.io/) | GNN framework; its gather-scatter MessagePassing does not fuse | measured against that path (PyG 2.8.0), Tiga's fused kernel is ~29× on the case above. PyG's separate `message_and_aggregate()` sparse path is not what this benchmark measures |
| [DGL](https://www.dgl.ai/) | GNN framework with fused `gspmm`/`gsddmm` ops | lands near `torch.sparse.mm` on regular SpMM; Tiga additionally keeps relation provenance, generated topology, and the compiler-generated VJP |
| [torch.compile](https://pytorch.org/docs/stable/torch.compiler.html) | general PyTorch compiler | cannot graph-capture `torch.sparse.mm`; over the gather-scatter formulation it reaches parity, not fusion |
| [Taichi](https://www.taichi-lang.org/) / [Warp](https://nvidia.github.io/warp/) | Python kernel JIT | per-thread kernels are written by hand; Tiga starts from the relation and the edge/node UDFs |
| [Triton](https://triton-lang.org/) / [TileLang](https://github.com/tile-ai/tilelang) | tile / kernel DSLs | backends Tiga lowers to, not competitors |
| [Graphiler](https://github.com/xiezhq-hermann/graphiler) | research GNN compiler on DGL + TorchScript, unmaintained since 2022 | same niche, but Tiga is maintained and multi-backend |
| [GALA](https://github.com/ADAPT-uiuc/GALA-GNN-Acceleration-LAnguage) | research GNN DSL + compiler (OOPSLA 2025) | the closest living academic peer; its own frontend and benchmark suite, not a drop-in comparison |

## When Tiga is the right tool, and when it is not

Reach for Tiga when the workload is message passing, sparse aggregation, or
attention over a graph — and both the forward kernel and its gradient should
come from one definition, on CPU or GPU.

Reach for something else for dense transformer training at scale (PyTorch),
for hand-tuned stencil/PDE codes (Taichi, Devito), or for pure graph
analytics like PageRank over fixed topologies (GraphIt, Gunrock).

The deep design survey behind these choices — Ebb, Simit, TACO, GraphIt,
Taichi, Legion, and the rest — is kept in the repository as
[`docs/RELATED_WORK.md`](https://github.com/walkerchi/TIGA-lang/blob/main/docs/RELATED_WORK.md).
