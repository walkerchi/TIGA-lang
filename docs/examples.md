# Examples

Every example is an ordinary user program — none calls a Tiga algorithm
operator. Each page below explains the problem being solved (with a schematic
and the formula), then shows the complete runnable program. All of them run
from a source checkout:

```bash
python examples/message_passing_autograd.py   # any example runs the same way
```

### [Message passing and graph algorithms](examples/message-passing.md)

| Example | What it computes |
|---|---|
| `message_passing_autograd.py` | a custom neighbor aggregation with a compiler-generated backward pass |
| `gcn.py` | a GCN layer: weighted sum of neighbor features (SpMM) with automatic gradients |
| `diffusion.py` | one explicit Euler diffusion step over a graph |
| `custom_reducer.py` | a neighbor mean written as custom (sum, count) reducer algebra |
| `compiler_probes/pagerank.py` | 20 PageRank iterations captured as one rolled repeat op — a compiler probe |

### [Dynamic and generated relations](examples/dynamic-relations.md)

| Example | What it computes |
|---|---|
| `radius_autograd.py` | a differentiable radius graph: points within a cutoff distance become edges |
| `knn_message_passing.py` | an exact k-nearest-neighbor graph feeding a message-passing consumer |

### [Distributed and memory hierarchy](examples/distributed-memory.md)

| Example | What it computes |
|---|---|
| `distributed_halo.py` | one graph split across two processes with automatic halo exchange |
| `hierarchical_memory.py` | chainable `.disk()` / `.cpu()` tensor spill — gf manages the files, named spills cross processes |
| `paged_giant_graph.py` | a 1M-node graph saved to disk, streamed page by page with prefetch, result written back |
| `auto_offload.py` | a RAM budget that pages an oversized graph to disk automatically, prefetch pipeline included |

### [Tensor kernels, programs and Torch interop](examples/programs-and-interop.md)

| Example | What it computes |
|---|---|
| `tensor_matmul.py` | matrix multiply `C = A·B` with compiler-derived gradients for both operands |
| `complex_autograd.py` | differentiates through complex-number tensors (conjugate-Wirtinger gradients) |
| `linear_recurrence.py` | causal linear-attention recurrence fused into one CUDA kernel from generic map/cumsum/contract |
| `graph_program.py` | automatic cross-kernel capture with `@gf.jit`: horizontal kernel fusion |
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
| `fem_poisson.py` | the Poisson equation solved by matrix-free conjugate gradient on a graph stencil |
| `meshfree_linear_solve.py` | the same solver style on a radius graph generated from point positions |
| `solvers.py` | one `linear_solve` entry — `method="cg" / "bicgstab" / "richardson"` — as plain Python loops under `@gf.jit` |
| `nonlinear_solve.py` | nonlinear diffusion `−∇·((1+u²)∇u) = f` solved by Picard fixed-point iteration, VJP through the loop |

### [Attention and dense relations](examples/attention.md)

| Example | What it computes |
|---|---|
| `full_attention.py` | exact attention with no mask — every query attends every key |
| `causal_dense_relation.py` | causal attention expressed as a triangular graph — no masking code |
| `varlen_causal_attention.py` | packed variable-length sequences with per-sequence causal masks (`cu_seqlens`) |
| `tile_pruned_attention.py` | attention that skips key tiles which cannot affect the softmax result |

## Current native example boundary

The examples above are executable claims, not API sketches. Coverage
boundaries live in one canonical place:
[What is executable today](index.md#what-is-executable-today).
