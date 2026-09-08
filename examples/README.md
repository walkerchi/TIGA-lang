# Examples

Runnable user programs, grouped by category — none calls a Tiga
algorithm operator. The rendered documentation with full embedded source lives
under the **Examples** tab of the docs site (`mkdocs serve`, or
https://tiga-docs.app.walkerchi.com/examples/).

Run any example from the repository root:

```bash
export PYTHONPATH="$PWD/python"
python examples/message_passing_autograd.py   # any example runs the same way
```

## Tensor and GPU kernels

| Example | What it demonstrates |
|---|---|
| `tensor_matmul.py` | first-class `gf_tensor.matmul`, CPU LLVM / FP16 GPU `tt.dot`, compiler-derived gradients for both operands |
| `complex_autograd.py` | complex64 storage, strided views, explicit conjugate-Wirtinger cotangents |
| `linear_recurrence.py` | ordinary map/cumsum/contract fused into one recurrent CUDA kernel — no core linear-attention operator |
| `gpu_heatmap.py` | scalar-to-RGB kernel from Tensor broadcast/arithmetic; generated MLIR artifact exposed for inspection |

## Attention and dense relations

| Example | What it demonstrates |
|---|---|
| `full_attention.py` | exact mask-free attention as a dense relation + online-softmax reducer, checked against SDPA |
| `causal_dense_relation.py` | `Graph.triangular()` topology through Domain → Iter → Kernel, causal streaming TTIR |
| `varlen_causal_attention.py` | packed sequences with per-sequence causal masks: `cu_seqlens` as a block-diagonal CSR graph |
| `tile_pruned_attention.py` | online-softmax reducer carrying a semantic block threshold into dynamic TTIR tile admission |

## Message passing and graph algorithms

| Example | What it demonstrates |
|---|---|
| `message_passing_autograd.py` | static CSR edge/node UDF, sum reducer VJP, explicit save/recompute checkpoint IR |
| `gcn.py` | multi-feature aggregation with edge broadcasting and automatic gradients (SpMM/GCN shape) |
| `diffusion.py` | directional neighbor difference, node update, automatic field/edge gradients |
| `custom_reducer.py` | user-defined tuple-state mean algebra with compiler-generated VJP |
| `compiler_probes/pagerank.py` | fixed-iteration PageRank in one `gf_control.repeat`; a compiler probe, not a performance claim |

## Dynamic and generated relations

| Example | What it demonstrates |
|---|---|
| `radius_autograd.py` | Torch-free dynamic Euclidean radius relation, differentiable `edge.distance`, position/source VJP |
| `knn_message_passing.py` | exact kNN kept procedural; fixed-k CSR ABI rebound to a generated MessagePassing consumer |

## Linear solvers and control flow

| Example | What it demonstrates |
|---|---|
| `fem_poisson.py` | matrix-free P1 stiffness as a MessagePassing operator inside fixed `repeat` and residual-driven `while` CG |
| `meshfree_linear_solve.py` | generated radius graph feeding a shifted-Laplacian operator, tolerance CG in one bounded device region |
| `solvers.py` | shared solver sugar: `linear_solve(operator, rhs, method="cg"|"bicgstab"|"richardson")` as natural Python loops under `@gf.jit`; not part of the core package |

## Programs, autograd and Torch interop

| Example | What it demonstrates |
|---|---|
| `graph_program.py` | optional `@gf.program` SSA capture, horizontal fusion, observation-triggered JIT |
| `joint_autograd.py` | compiler-generated VJP in one executable, inspectable forward/backward dependency DAG |
| `torch_interop.py` | optional Torch adapter: zero-copy storage sharing and provider-neutral schedule inspection |
| `torch_library.py` | `torch.library` registration with dispatcher schema, FakeTensor kernel, `opcheck`, Inductor forward/backward |
| `edge_nn_message_passing.py` | `gf.nn.trace` captures an edge-local `torch.nn` MLP; compiled fused tile kernel vs exact eager oracle, grad-mode fallback |

## Distributed and memory hierarchy

| Example | What it demonstrates |
|---|---|
| `distributed_halo.py` | one versioned `.gfg` reopened on two processes, rank-local halo MessagePassing plus automatic VJP |
| `hierarchical_memory.py` | chainable `.disk()` / `.cpu()` tensor spill with gf-managed lifecycle; named spills via `gf.from_disk` |

## Current native example boundary

The examples above are executable claims, not API sketches. Native
differentiable MessagePassing supports static/paged CSR with `gf.sum()`,
structurally proven tuple/product reducers and stable online-softmax; its
edge/node/reducer UDF can use supported Tensor broadcasting, views and
arithmetic. Default Euclidean generated radius/periodic-radius works in both
the native runtime and optional CUDA adapter; exact kNN has an executable
compiler path. Published performance claims remain limited to the registered
benchmark matrix.

Tiga is a compiler, not the source of these algorithms. Recognized
programs lower through Domain → Iter → Kernel → provider IR; a proven library
dispatch or semantic evaluator handles shapes that do not yet have generated
code. Handwritten performance oracles exist only under `benchmarks/kernels/`.
