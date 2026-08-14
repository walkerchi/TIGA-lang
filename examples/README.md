# Examples

Run the Python examples from the repository root:

```bash
export PYTHONPATH="$PWD/python"
python3 examples/tensor_autograd.py
python3 examples/tensor_matmul.py
python3 examples/linear_recurrence.py
python3 examples/tile_pruned_attention.py
python3 examples/causal_dense_relation.py
python3 examples/complex_autograd.py
python3 examples/message_passing_autograd.py
python3 examples/radius_autograd.py
python3 examples/graph_program.py
python3 examples/diffusion.py
python3 examples/gcn.py
python3 examples/custom_reducer.py
python3 examples/torch_interop.py
python3 examples/torch_library.py
python3 examples/hierarchical_memory.py
python3 examples/joint_autograd.py
python3 examples/distributed_halo.py
python3 examples/knn_message_passing.py
python3 examples/gpu_heatmap.py --device cuda
```

- `tensor_autograd.py`: Torch-independent native CPU storage, canonical
  `gf_tensor` IR and the native reverse-mode VJP pass;
- `tensor_matmul.py`: first-class `gf_tensor.matmul`, native CPU/GPU lowering
  and compiler-derived gradients for both matrix operands;
- `linear_recurrence.py`: an ordinary Tensor map/cumsum/contract expression
  structurally fused into one recurrent CUDA kernel, without a core
  linear-attention operator;
- `tile_pruned_attention.py`: a benchmark-style DenseGraph UDF whose
  online-softmax reducer carries a semantic block threshold into dynamic TTIR
  tile admission, without a core sparse-attention operator;
- `causal_dense_relation.py`: `Graph.triangular()` carries lower-inclusive
  topology through Domain/Iter/Kernel and an ordinary online reducer lowers to
  causal streaming TTIR; Torch is only external CUDA storage and the oracle;
- `complex_autograd.py`: complex64 storage, strided views and explicit
  conjugate-Wirtinger cotangents;
- `message_passing_autograd.py`: native static-CSR edge/node UDF, sum reducer,
  LLVM forward execution and compiler-generated relation VJP; the user writes
  no backward function. It also shows explicit save/recompute checkpoint IR;
- `radius_autograd.py`: Torch-free dynamic Euclidean radius relation,
  differentiable `edge.distance`, periodic-capable minimum-image geometry and
  automatically generated position/source VJP on one fixed topology snapshot;
- `graph_program.py`: optional `@gf.program` straight-line SSA capture,
  automatic horizontal fusion, observation-triggered JIT, and Kernel IR/PTX
  inspection without an explicit compile call; it uses the optional Torch
  adapter for CUDA storage in this first executable provider example;
- `diffusion.py`: native directional neighbor difference with both source and
  destination fields, node update and automatic field/edge gradients;
- `gcn.py`: native multi-feature aggregation with edge broadcasting and
  automatic gradients, matching the SpMM/GCN tensor shape;
- `custom_reducer.py`: Torch-free execution of a user-defined tuple-state mean
  algebra plus compiler-generated VJP; selection is based on a structural
  component-wise additive proof rather than the reducer class name;
- `torch_interop.py`: the isolated optional adapter example: Torch tensors call
  a GraphForge UDF, a `gf.Tensor` shares Torch storage zero-copy, and the lazy
  JIT object exposes the verified provider-neutral machine schedule selected by
  `gf.kernel`.
- `torch_library.py`: dynamically registers a UDF specialization with an exact
  functional dispatcher schema, FakeTensor/meta kernel, automatic autograd,
  `opcheck` and a full-graph Inductor-compiled forward/backward. CSR topology
  is part of the dispatcher ABI even though the wrapper binds it automatically.
- `hierarchical_memory.py`: capacity/version-accounted RAM↔NVMe physical
  instances and transfer completions consumed by compiler bundle plans;
- `joint_autograd.py`: compiler-generated VJP in one executable and inspectable
  forward/backward dependency DAG;
- `distributed_halo.py`: saves one versioned `.gfg`, reopens the same ordinary
  `Graph` on two processes, reads only each rank's destination/edge pages, and
  runs rank-local `Graph.halo()` MessagePassing plus automatic VJP;
  compiler/runtime derives forward owner→ghost exchange and reverse
  ghost-cotangent→owner accumulation below the unchanged user kernel.
- `gpu_heatmap.py`: composes ordinary Tensor broadcast/arithmetic into one
  scalar-to-RGB kernel, exposes the generated MLIR/provider artifact, and keeps
  optional PNG encoding outside compiler core.
- `knn_message_passing.py`: keeps exact kNN procedural, rebuilds the current
  position snapshot lazily, and rebinds its fixed-k CSR ABI to one generated
  MessagePassing TTIR consumer.

## Current native example boundary

The examples above are executable claims, not API sketches. Native
differentiable MessagePassing supports static/paged CSR with `gf.sum()`,
structurally proven tuple/product reducers and stable online-softmax; its
edge/node/reducer UDF can use supported Tensor broadcasting, views and
arithmetic. Default Euclidean generated radius/periodic-radius works in both
the native runtime and optional CUDA adapter; exact kNN has an executable
compiler path. Published performance claims remain limited to the registered
benchmark matrix.

GraphForge is a compiler, not the source of these algorithms. Recognized
programs lower through Domain → Iter → Kernel → provider IR; a proven library
dispatch or semantic evaluator handles shapes that do not yet have generated
code. Handwritten performance oracles exist only under `benchmarks/kernels/`.
