# Python API

This page documents the intentionally small alpha surface. API stability is not
promised before the first release candidate.

## `Tensor` and `autograd`

- `gf.tensor(data, *, dtype=None, device="cpu", requires_grad=False)` creates a
  Tensor backed by the native GraphForge runtime rather than a Torch Tensor.
- `gf.empty(shape, ...)` allocates an uninitialized contiguous Tensor.
- `float32/float64`, `complex64/complex128` and integer/bool storage are native
  GraphForge dtypes. Only floating and complex values may require gradients.
- `reshape`, `permute`, `transpose`, `swapaxes`, `squeeze`, `unsqueeze`,
  `broadcast_to`/`expand`, `+`, `*`, `conj` and axis-aware `sum` form the current
  expression slice. Broadcasting follows right-aligned NumPy semantics.
- `gf.autograd.grad(output, inputs, checkpoint="auto")` builds functional
  reverse-mode VJP expressions. `save`/`recompute` override the compiler's
  primal-storage choice; `auto` emits `gf_tensor.checkpoint_candidate`, while
  selected and explicit saves remain inspectable as `gf_tensor.checkpoint`.
- `gf.autograd.value_and_grad(function, argnums=...)` transforms a function
  without `.grad` mutation.
- `tensor.mlir(verify=False)` returns canonical target-independent `gf_tensor`
  IR; `verify=True` parses it with the native compiler.
- `gf.autograd.grad_mlir(output, input, grad_output=..., lower=True)` exposes
  the explicit VJP request or its `gf-tensor-vjp` result for inspection.

The CPU has a Python correctness oracle and an in-process MLIR JIT with
structural executable caching. Its implemented path is
`gf_tensor → SCF/MemRef → LLVM dialect → ExecutionEngine`.
`Tensor.execution` exposes semantic hash, compile/launch/materialization timing,
the MLIR checkpoint plan, saved bytes and the selected backend;
`generated_code("gf_tensor"|"cpu_loop"|"llvm"|"ptx")` exposes real compiler
stages. `Tensor.expression()` remains informal debug capture, while
`Tensor.mlir()` is the compiler contract.

`Tensor.to_numpy()` performs an explicit host copy from native or optional
Torch-owned storage. `Tensor.prepare()` returns a hot callable with stable
input/output buffers after the first JIT; it is intended for simulation loops
and interactive rendering, while cold compilation remains visible in
`Tensor.execution`.

## `visualize`

`gf.visualize.heatmap(values, vmin=0, vmax=1, low=..., high=...)` returns a lazy
RGB `Raster`. The color transform is ordinary Tensor broadcast/arithmetic and
there is no visualization operation in compiler core. `Raster.mlir()`,
`generated_code()` and `prepare()` expose the same compiler/runtime path;
`to_numpy()`, `save()` and `show()` are optional host interop/encoding helpers.

`gf.compiler.Dim`, `TensorSpec` and `ShapeSpecializer` express bounded symbolic
signatures. Repeated symbols must bind to one extent across arguments;
minimum/maximum/divisibility guards are checked before compilation. Each legal
binding forms a concrete MLIR specialization/cache key, which keeps provider
loops and launch bounds static while avoiding one user kernel per input size.

## `Graph`

- `Graph.from_csr(row_ptr, col_idx, *, num_src=None)` creates an external frozen
  relation.
- `gf.save(graph, path)` persists a static CSR in the versioned `.gfg` format;
  `gf.load(path)` returns the same ordinary `Graph` type backed by paged storage.
  It reads only the manifest until a planner requests a bounded row range.
- `Graph.dense(num_nodes, *, device=None)` creates an implicit Cartesian
  relation; it does not allocate an `N×N` index tensor.
- `Graph.triangular(num_nodes, *, device=None)` creates an implicit
  lower-inclusive relation (`src <= dst`). It remains a graph topology in IR;
  any MessagePassing reducer may consume it, and dense lowering uses bounded
  source tiles plus a pair mask without materializing CSR.
- `Graph.radius(positions, cutoff, *, fields=None, metric=None, select=None)`
  creates a rebuildable procedural relation.
- `Graph.knn(positions, k, *, exclude_self=True)` creates an exact procedural
  kNN relation; adjacency is not allocated until an explicit materialization
  or compiler-selected consumer requires it.
- `graph.halo(mesh, *, partition=gf.ByDestination(), depth="auto")` returns the
  same logical `Graph` type with owned/ghost requirements. It is declarative:
  communication is inserted below the user kernel as typed pack, exchange,
  unpack, interior and boundary tasks.

??? info "Current distributed execution boundary"

    CPU rank-local execution, reverse halo VJP, paged CSR shards and real
    two-process MPI transport are executable. CUDA native buffers use separate
    communication/compiler streams and verified event ordering. The one-rank
    NCCL fixture proves provider binding only; real multi-GPU NCCL/RCCL
    correctness and overlap performance remain fail-closed gates. See
    [Memory hierarchy and distributed execution](memory-and-distributed.md).

`gf.DeviceMesh(device_type, shape, names=...)` describes logical devices without
initializing a process group. Deployment binds it to Torch `DeviceMesh`, MPI,
NCCL/RCCL or a vendor communicator.

## `MessagePassing`

Subclass `MessagePassing`, set `reducer`, and implement `edge(src, dst, edge,
**params)`. `node(dst, aggregate, **params)` is optional.

- `program(...)` performs lazy JIT selection and execution.
- `program.reference(...)` executes the semantic implementation explicitly.
- `program.explain()` reports planner/lowering/cache information.
- `program.diagnostics` returns typed `AnalysisFinding` records; `explain()` is
  their human-readable rendering. Each record identifies its compiler stage,
  disposition and optional resource/target contract/estimate/action.
- `program.schedules` returns typed `MachineSchedule` records extracted by the
  native compiler from verified `gf.kernel` IR. Each record exposes the chosen
  schedule family, row/neighbor tile, subgroup count, pipeline depth, named
  resources, execution roles, producer/consumer handoffs and admitted
  instruction classes. This API parses dialect attributes in process; it does
  not reconstruct decisions from pass names, logs or benchmark metadata.
- `program.ir(stage)` returns `domain`, `iter`, `kernel`, `task`, provider input or
  provider IR when available.
- `program.code(kind)` returns code artifacts such as `ttgir`, `llir` or `ptx`.
- `program.cache_info` returns hit/miss/variant counts.
- `program.prepare(...)` optionally freezes one validated scalar-CSR CUDA
  binding into a zero-argument hot submission; ordinary execution remains
  lazy-JIT and never requires this call.

`pipeline_stages == 1` is an explicit negative result: the compiler has not
admitted a compiler-controlled asynchronous producer/consumer pipeline. The
corresponding diagnostic is `unknown`, so provider instruction reordering must
not be presented as proven overlap. GraphForge does not expose a public manual
schedule DSL in this alpha; ordinary UDFs remain semantic code and the compiler
owns placement.

## Reducers

- `gf.sum()` is an associative, commutative additive reducer.
- `gf.online_softmax(accumulation_dtype=...)` exposes a stable multi-value
  streaming reducer. The edge region returns `program.reducer(score, value)`.

Reducer regions and type/state metadata are first-class compiler IR. Users may
subclass `gf.Reducer` and implement `identity`, `lift`, `combine`, and
`finalize`. Native capture accepts scalar FP32 messages and scalar or tuple
state. Implicit dense and CSR scalar reducers have generic CUDA TTIR paths;
structurally proven online-softmax uses a tiled path. Vector tuple reducers are
not yet available in the generic CSR provider.

??? info "Automatic reducer VJP coverage"

    Additive/stable algebras and captured associative scalar reducers lower to
    balanced or deterministic trees through CPU LLVM and CUDA TTIR. Product is
    promoted to zero-safe CSR product/VJP IR. Empty/ragged rows and ordered
    non-commutative tuple state are covered; unregistered reducer shapes remain
    correctness-only claims.
