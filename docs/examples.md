# Examples

The default interface is `torch.Tensor`. Start with
[`python examples/torch_quickstart.py`](getting-started.md#first-differentiable-jit-program).
Expected output is approximately `[11.1, 8.2, 17.3]`, with temperature gradient `[7, 10, 3]`.

Core snippets may omit setup; run the linked script or the expanded full source.
Native `tg.Tensor` supports standalone execution without Torch, as well as
currently unsupported Torch features:
hierarchy memory, distributed execution, structured control flow, custom
reducers, and direct native Tensor IR / checkpoint inspection.
Ordinary aggregation, GCN, diffusion and radius examples use Torch.

Every example is an ordinary user program — none calls a Tiga algorithm
operator. Each page below explains the problem being solved (with a schematic
and the formula), then shows its core snippet and expandable complete source. All of them run
from a source checkout after installing Tiga and the example's dependencies.
The CPU starting point is
[`python examples/message_passing_autograd.py`](examples/message-passing.md#differentiable-messagepassing-udf):

```bash
python examples/message_passing_autograd.py
```

CUDA examples require a supported NVIDIA GPU and the `cuda` extra. Install Torch
separately before Tiga for Torch examples; native examples run without it. Distributed and visualization examples
have additional requirements documented on their pages. A successful run under
`auto` does not by itself demonstrate JIT execution; inspect execution diagnostics.

### [Message passing and graph algorithms](examples/message-passing.md)

| Example | What it computes |
|---|---|
| `message_passing_autograd.py` | custom neighbor aggregation with Torch autograd; inspect diagnostics to distinguish compiled VJP from semantic replay |
| `gcn.py` | a GCN layer: weighted sum of neighbor features (SpMM) with automatic gradients |
| `diffusion.py` | one explicit Euler diffusion step over a graph |
| `custom_reducer.py` | a neighbor mean written as custom (sum, count) reducer algebra |
| `compiler_probes/pagerank.py` | 20 PageRank iterations captured as one rolled repeat op — a compiler probe |

### [Dynamic and generated relations](examples/dynamic-relations.md)

| Example | What it computes |
|---|---|
| `stencil_message_passing.py` | von Neumann macro, five-point periodic averaging and Torch autograd |
| `distance_metrics.py` | Euclidean defaults, custom cosine radius and normalized cosine kNN |
| `radius_autograd.py` | differentiable messages on a radius relation, with selected neighbors held fixed during backward |
| `knn_message_passing.py` | an exact k-nearest-neighbor graph feeding a message-passing consumer |

### [Distributed and memory hierarchy](examples/distributed-memory.md)

| Example | What it computes |
|---|---|
| `distributed_halo.py` | one graph split across two processes with automatic halo exchange |
| `hierarchical_memory.py` | `tg.execution` budgets, spill and value snapshots |
| `paged_giant_graph.py` | a 1M-node graph saved to disk, streamed page by page with prefetch, result written back |
| `auto_offload.py` | a per-CSR size threshold that offloads topology to disk, with optional page read-ahead |

### [Tensor kernels, programs and Torch interop](examples/programs-and-interop.md)

| Example | What it computes |
|---|---|
| `tensor_matmul.py` | matrix multiply `C = A·B` with compiler-derived gradients for both operands |
| `complex_autograd.py` | differentiates through complex-number tensors (conjugate-Wirtinger gradients) |
| `linear_recurrence.py` | causal linear-attention recurrence fused into one CUDA kernel from generic map/cumsum/contract |
| `graph_program.py` | automatic cross-kernel capture with `@tg.jit`: horizontal kernel fusion |
| `joint_autograd.py` | forward and backward compiled into one executable |
| `torch_interop.py` | torch tensors calling a Tiga UDF with zero-copy storage sharing |
| `torch_library.py` | a UDF registered as a `torch.library` op, Inductor-compatible |
| `edge_nn_message_passing.py` | a `torch.nn` MLP on every edge, compiled into one fused tile kernel |

### [Visualization](examples/visualization.md)

| Example | What it computes |
|---|---|
| `gpu_heatmap.py` | turns a scalar field into an RGB heatmap with tensor broadcast and arithmetic |
| `visualize_fields.py` | relaxes steady heat conduction on a triangulated disc (Jacobi via mean reducer), rendered as particles, Delaunay mesh and GIF video |
| `visualize_mesh.py` | loads an OBJ icosphere, diffuses a bump over its mesh edges, renders flat/wireframe/particles plus an orbit GIF, and exports PLY/OBJ for Blender |
| `visualize_gaussians.py` | renders a colored torus of anisotropic 3-D Gaussians as splats and as a ray-marched density volume, plus an orbit GIF |

### [Linear solvers and control flow](examples/solvers.md)

| Example | What it computes |
|---|---|
| `fem_poisson_minimal.py` | Poisson in three steps: define a neighbor operator, solve, check the answer |
| `fem_poisson.py` | advanced Poisson solve: fixed iterations and load gradients |
| `meshfree_linear_solve.py` | the same solver style on a radius graph generated from point positions |
| `solvers.py` | one `linear_solve` entry — `method="cg" / "bicgstab" / "richardson"` — as plain Python loops under `@tg.jit` |
| `nonlinear_solve.py` | nonlinear diffusion `−∇·((1+u²)∇u) = f` solved by Picard fixed-point iteration, VJP through the loop |

### [Attention and dense relations](examples/attention.md)

| Example | What it computes |
|---|---|
| `full_attention.py` | exact attention with no mask — every query attends every key |
| `causal_dense_relation.py` | causal attention expressed as a triangular graph — no masking code |
| `varlen_causal_attention.py` | independent causal sequences composed with `Graph.cat` and `Graph.triangular` |
| `tile_pruned_attention.py` | approximate attention that skips value work for selected tiles after computing scores; forward only |

## Current native example boundary

The examples above are executable claims, not API sketches. Coverage
boundaries live in one canonical place:
[native and Torch-adapter support matrix](roadmap.md).
