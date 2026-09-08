# Python API

Flat reference for the intentionally small public surface. Conceptual
background lives in the guide pages linked from each section; this page
lists signatures, constraints and return values only.

## Tensor and autograd { #tensor-and-autograd }

`float16/float32/float64`, `complex64/complex128`, `int32/int64` and `bool`
are native Tiga dtypes. Only floating and complex values may require
gradients. Broadcasting follows right-aligned NumPy semantics.

### gf.tensor(data, *, dtype=None, device="cpu", requires_grad=False) { #gftensor }

Creates a Tensor backed by the native Tiga runtime from nested Python
sequences or a scalar. The dtype is inferred from the data when omitted.

### gf.empty(shape, *, dtype=gf.float32, device="cpu", requires_grad=False) { #gfempty }

Allocates an uninitialized contiguous Tensor.

### gf.zeros_like(value, *, requires_grad=False) { #gfzeros_like }

Allocates a zero-filled Tensor with the shape, dtype and device of `value`.

### gf.ones_like(value, *, requires_grad=False) { #gfones_like }

Allocates a one-filled Tensor with the shape, dtype and device of `value`.

### gf.from_torch(value, *, requires_grad=None) { #gffrom_torch }

Wraps a dense Torch tensor without copying its storage. Example:
[Optional Torch interoperability](examples/programs-and-interop.md#optional-torch-interoperability).

### Tensor.reshape(*shape) { #tensorreshape }

Returns a lazy view with a new shape; one `-1` extent is inferred.

### Tensor.permute(*axes) { #tensorpermute }

Returns a lazy view with axes reordered.

### Tensor.transpose(dim0, dim1) { #tensortranspose }

Returns a lazy view with two axes swapped. `Tensor.T` reverses all axes.

### Tensor.squeeze(dim=None) { #tensorsqueeze }

Removes size-one extents, at `dim` only when given.

### Tensor.unsqueeze(dim) { #tensorunsqueeze }

Inserts a size-one extent at `dim`.

### Tensor.broadcast_to(shape) { #tensorbroadcast_to }

Returns a lazy broadcast view. `Tensor.expand(*shape)` is the variadic
spelling of the same view.

### Arithmetic, comparison and matmul operators { #tensor-operators }

`+`, `-`, `*`, `/`, unary `-`, comparisons and `@` build lazy expression
nodes. A `Tensor` used as a Python boolean (`if tensor:`) raises `TypeError`.

### Tensor.matmul(other) { #tensormatmul }

Rank-two matrix multiplication with a conjugate-Wirtinger VJP for complex
inputs. Example:
[Matrix multiplication with derived gradients](examples/programs-and-interop.md#matrix-multiplication-with-derived-gradients).

### Tensor.exp() { #tensorexp }

Elementwise exponential.

### Tensor.sqrt() { #tensorsqrt }

Elementwise square root.

### Tensor.conj() { #tensorconj }

Elementwise complex conjugate; identity for real dtypes. Example:
[Complex VJP](examples/programs-and-interop.md#complex-vjp).

### Tensor.cumsum(dim, *, reverse=False) { #tensorcumsum }

Inclusive scan along one axis. Example:
[Linear recurrence from map/cumsum/contract](examples/programs-and-interop.md#linear-recurrence-from-mapcumsumcontract).

### Tensor.sum(axis=None, *, keepdims=False) { #tensorsum }

Reduction over one axis, several axes, or all axes.

### Tensor.gather(index) { #tensorgather }

Selects rows by an index Tensor; the VJP is a segment sum.

### Tensor.segment_sum(index, num_segments) { #tensorsegment_sum }

Sums rows into `num_segments` buckets given a per-row segment index.

### Tensor.checkpoint() { #tensorcheckpoint }

Marks a non-leaf value as an explicit backward save
(`gf_tensor.checkpoint` in IR), overriding the checkpoint planner for this
value.

### Tensor.realize() { #tensorrealize }

Forces JIT compilation and execution of the deferred expression and returns
the materialized Tensor.

### Tensor.prepare() { #tensorprepare }

Returns a zero-argument hot callable that resubmits the compiled DAG with
stable input/output buffers, after a first `realize()`. Cold compilation
stays visible in `Tensor.execution`.

### Tensor.execution { #tensorexecution }

Property returning codegen/launch diagnostics after realization: semantic
hash, compile/launch/materialization timing, the MLIR checkpoint plan, saved
bytes and the selected backend. `None` before realization.

### Tensor.generated_code(kind=None) { #tensorgenerated_code }

Returns a real compiler stage artifact such as `"gf_tensor"`, `"cpu_loop"`,
`"llvm"` or `"ptx"`; the default picks the most lowered available stage.
Guide: [Compiler pipeline](compiler-pipeline.md#inspecting-a-compiled-program).

### Tensor.mlir(*, verify=False) { #tensormlir }

Returns canonical target-independent `gf_tensor` IR; `verify=True`
round-trips it through the native parser and C++ verifiers. This is the
compiler contract; `Tensor.expression()` is informal debug capture.

### Tensor.tolist() / Tensor.to_numpy() / Tensor.to_torch(*, copy=False) { #tensor-host-interop }

Explicit host or Torch copies. `to_numpy()` performs a host copy from native
or optional Torch-owned storage; `to_torch()` shares storage unless
`copy=True`.

### gf.autograd.grad(output, inputs, *, grad_output=None, allow_unused=False, checkpoint="auto") { #gfautogradgrad }

Builds functional reverse-mode VJP expressions without mutating inputs.
`checkpoint` selects backward primal storage: `save` marks non-leaf values
as saved, `recompute` retains the full primal expression, and `auto` emits
`gf_tensor.checkpoint_candidate` for the target-independent checkpoint
planner. Non-scalar or complex outputs require an explicit `grad_output`.
Example:
[Matrix multiplication with derived gradients](examples/programs-and-interop.md#matrix-multiplication-with-derived-gradients).

### gf.autograd.value_and_grad(function, *, argnums=0) { #gfautogradvalue_and_grad }

Returns a transform of `function` that produces the primal output and the
gradients of the positional arguments selected by `argnums`, without `.grad`
mutation.

### gf.autograd.joint_plan(output, inputs, *, checkpoint="auto") { #gfautogradjoint_plan }

Builds an executable, inspectable joint forward/backward task DAG with one
snapshot version. `plan.run()` returns `(output, gradients)`;
`plan.explain()` renders the bundle. Example:
[Joint forward/backward DAG](examples/programs-and-interop.md#joint-forwardbackward-dag).

### gf.autograd.grad_mlir(output, input, *, grad_output=None, lower=True) { #gfautogradgrad_mlir }

Returns the explicit VJP request, or with `lower=True` the `gf-tensor-vjp`
result, as MLIR text for inspection.

## Graph { #graph }

An immutable logical relation over Tiga tensors. Materialized CSR owns
two index tensors; dense, triangular, radius and kNN relations stay implicit
or procedural and are never silently materialized into O(N²) adjacency.

### Graph.from_csr(row_ptr, col_idx, *, num_src=None, sorted_by_dst=True, validate="basic") { #graphfrom_csr }

Creates an external frozen relation from CSR index arrays (Tiga
Tensors, or Torch tensors routed through the interop adapter). `num_src`
defaults to `max(col_idx) + 1`; `validate="full"` additionally checks
endpoints, monotonicity and index ranges. Example:
[GCN aggregation](examples/message-passing.md#gcn-aggregation).

### Graph.from_coo(src, dst, *, num_src=None, num_dst=None) { #graphfrom_coo }

Creates the same frozen relation from coordinate lists, performing a stable
destination sort into canonical CSR.

### Graph.regular(num_nodes, degree, *, device="cpu") { #graphregular }

Creates a deterministic fixed-degree relation — destination `i` gathers from
sources `(i * degree + k) % num_nodes` — so benchmarks and smoke tests do not
hand-assemble `arange`/`%` CSR arrays.

### Graph.dense(num_src, num_dst=None, *, device=None, index_dtype=gf.int64) { #graphdense }

Creates an implicit Cartesian relation; no `N×N` index tensor is allocated.
`num_dst` defaults to `num_src`. Example:
[Full attention, no mask](examples/attention.md#full-attention-no-mask).

### Graph.triangular(num_entities, *, device=None, index_dtype=gf.int64) { #graphtriangular }

Creates an implicit lower-inclusive relation (`src <= dst`). It remains a
graph topology in IR; any MessagePassing reducer may consume it, and dense
lowering uses bounded source tiles plus a pair mask without materializing
CSR. Example:
[Causal dense relation](examples/attention.md#causal-dense-relation).

### Graph.cu_seqlens(cu_seqlens, *, causal=True) { #graphcu_seqlens }

Creates a materialized block-diagonal relation for flash-attn style packed
variable-length sequences: position `i` in sequence `k` gathers from sources
`[s_k, i]` (`causal=True`) or from its whole sequence (`causal=False`).
`cu_seqlens` must start at zero and be monotonic. Example:
[Varlen causal attention with cu_seqlens](examples/attention.md#varlen-causal-attention-with-cu_seqlens).

### Graph.cat(graphs) { #graphcat }

Composes relations into one block-diagonal relation: a destination of block
`k` is related exactly to the sources of block `k`, offset by the cumulative
source count. `cat([Graph.triangular(n) for n in lengths])` equals
`Graph.cu_seqlens(cumsum(lengths), causal=True)`; `Graph.dense` blocks give
the `causal=False` variant. The composition realizes one materialized CSR
snapshot. Example:
[Varlen causal attention with cu_seqlens](examples/attention.md#varlen-causal-attention-with-cu_seqlens).

### Graph.stencil(dims, offsets, *, periodic=False, device="cpu") { #graphstencil }

Creates a materialized regular-grid stencil relation over row-major
linearized nodes; each destination gathers from `coord(dst) + offset` in
offsets order. Non-periodic boundaries truncate out-of-grid sources, periodic
boundaries wrap with modulo. Examples:
[Matrix-free FEM operator and solver loop](examples/solvers.md#matrix-free-fem-operator-and-solver-loop),
[A disk-resident giant graph](examples/distributed-memory.md#a-disk-resident-giant-graph).

### Graph.radius(positions, cutoff, *, exclude_self=True, fields=None, metric=None, select=None, periodic=None) { #graphradius }

Creates a rebuildable procedural relation from rank-two floating positions;
adjacency is recomputed from the current snapshot. `metric`/`select` are
custom builder UDFs, `periodic` supplies box lengths `[D]` or lattice vectors
`[D,D]`. Example:
[Differentiable radius relation](examples/dynamic-relations.md#differentiable-radius-relation).

### Graph.knn(positions, k, *, candidates=None, exclude_self=None) { #graphknn }

Creates an exact procedural kNN relation; adjacency is not allocated until an
explicit materialization or a compiler-selected consumer requires it.
Omitting `candidates` creates self-kNN and defaults `exclude_self=True`;
passing candidates creates a bipartite query→candidate relation. Example:
[Exact kNN feeding MessagePassing](examples/dynamic-relations.md#exact-knn-feeding-messagepassing).

### Graph.open(path, *, device="cpu") { #graphopen }

Opens a persistent graph without eagerly loading its CSR arrays; the result
is an ordinary `Graph` backed by paged storage that reads only the manifest
until a bounded row range is requested. `gf.load(path, *, device="cpu")` is
the module-level alias. Example:
[A disk-resident giant graph](examples/distributed-memory.md#a-disk-resident-giant-graph).

### gf.save(graph, path, *, fields=None) { #gfsave }

Persists a static CSR graph in the versioned `.gfg` format (manifest plus
`row_ptr.bin`/`col_idx.bin`); `gf.load` reopens it. `fields={"src": {...},
"dst": {...}, "edge": {...}}` additionally writes node/edge fields as
fixed-width rows so they page from disk alongside the topology.

### Graph.fields(role, *, requires_grad=False) { #graphfields }

On a `paged_csr` graph saved with `fields=`, returns the per-role mapping of
lazily-read field shells (full shape, payload on disk) to pass straight into
a kernel call. `requires_grad=True` makes the shells differentiable; paged
execution then supports both forward and `gf.autograd.grad`.

### graph.halo(mesh, *, partition=None, depth="auto") { #graphhalo }

Returns the same logical `Graph` type with owned/ghost requirements;
`partition` defaults to `gf.ByDestination()`. Declarative only: communication
is inserted below the user kernel as typed pack, exchange, unpack, interior
and boundary tasks. Example:
[Two-process halo exchange](examples/distributed-memory.md#two-process-halo-exchange).

### graph.paged_rows(begin, end) { #graphpaged_rows }

Compiler/runtime hook returning one bounded CSR destination tile; valid only
on a `paged_csr` graph opened with `gf.load`.

### graph.resolve_csr() { #graphresolve_csr }

Explicitly materializes and returns `(row_ptr, col_idx)` — an inspection or
debug request, not part of normal compiler consumption.

### graph.transpose() { #graphtranspose }

Returns one materialized snapshot with endpoint roles swapped.

### graph.explain() { #graphexplain }

Renders a one-line summary: origin, lifecycle, realization, entity and edge
counts, device, and halo/paged backing when present. Example:
[A disk-resident giant graph](examples/distributed-memory.md#a-disk-resident-giant-graph).

### Graph introspection properties { #graph-properties }

`graph.schema` (the `GraphSchema` dataclass), `graph.device`,
`graph.num_edges` (`None` for procedural relations until realized),
`graph.placement` and `graph.is_distributed`.

### Planner statistics { #graph-planner-statistics }

`graph.degree_bounds()`, `graph.degree_statistics()`,
`graph.degree_histogram()`, `graph.fixed_degree()` and
`graph.source_index_span_ratio(samples=4096)` expose the physical statistics
the planner consumes; results are cached per snapshot.

## MessagePassing { #messagepassing }

Subclass `MessagePassing`, set the class attribute `reducer`, and implement
`edge(src, dst, edge, **params)`; `node(dst, aggregate, **params)` is
optional and defaults to the identity. Node fields bind through
`src={...}`/`dst={...}` endpoint-role mappings, or through one `ndata={...}`
mapping on homogeneous graphs. Guide:
[Message passing](message-passing.md).

### program(graph=..., src=None, dst=None, edge=None, ndata=None, **params) { #messagepassing-call }

Calling a kernel instance performs lazy JIT selection and execution and
returns the aggregated node Tensor. On a `paged_csr` graph the call streams
bounded destination-row pages and accepts three extra call-time parameters:
`page_rows` (page height, default 100 000 or `TIGA_PAGED_PAGE_ROWS`),
`prefetch=True` (overlap page reads with compute) and `prefetch_depth`
(pages read ahead concurrently, default 2 or
`TIGA_PAGED_PREFETCH_DEPTH`).
Examples:
[GCN aggregation](examples/message-passing.md#gcn-aggregation),
[A disk-resident giant graph](examples/distributed-memory.md#a-disk-resident-giant-graph).

### gf.runtime.auto_offload(ram) { #gfruntimeautoffload }

Context manager: while active, every explicit CSR builder
(`Graph.from_csr`, `Graph.stencil`, `Graph.cat`, ...) persists the topology
as `.gfg` and returns a `paged_csr` graph when the CSR exceeds `ram` bytes,
so kernel calls page it with prefetch automatically. The env var
`TIGA_GRAPH_RAM_BUDGET` sets a process-wide default; the context
manager wins. Offloads land in a per-process directory cleaned up at exit,
or under `TIGA_SPILL_DIR` when set. CPU graphs only; fields can join the
topology on disk (`gf.save(..., fields=...)`), and both forward and backward
run paged. Example:
[Automatic offload under a RAM budget](examples/distributed-memory.md#automatic-graph-offload).

### program.reference(**kwargs) { #messagepassingreference }

Executes the semantic implementation explicitly through the optional Torch
interop oracle; it is a correctness reference, not a native backend.

### program.prepare(*, graph, src=None, dst=None, edge=None, ndata=None, **params) { #messagepassingprepare }

Optionally freezes one validated scalar-CSR CUDA binding into a
zero-argument hot submission. Ordinary execution remains lazy JIT and never
requires this call.

### program.explain() { #messagepassingexplain }

Renders planner/lowering/cache information for the last variant as text.

### program.diagnostics { #messagepassingdiagnostics }

Property returning typed `AnalysisFinding` records — stage, disposition
(`accepted`/`rejected`/`warning`/`unknown`/`remark`) and optional resource,
target contract, estimate and suggested action. `explain()` is their
human-readable rendering.

### program.schedules { #messagepassingschedules }

Property returning typed `MachineSchedule` records extracted by the native
compiler from verified `gf.kernel` IR: schedule family, row/neighbor tile,
subgroup count, pipeline depth, named resources, execution roles,
producer/consumer handoffs and admitted instruction classes.
`pipeline_stages == 1` is an explicit negative result — no
compiler-controlled asynchronous pipeline was admitted — so provider
instruction reordering must not be presented as proven overlap.

### program.ir(stage="domain") { #messagepassingir }

Returns `domain`, `iter`, `kernel`, `task`, provider input or provider IR
text when available. Guide:
[Inspecting the compilation](message-passing.md#inspecting-the-compilation).

### program.code(kind="ptx") { #messagepassingcode }

Returns code artifacts such as `ttgir`, `llir` or `ptx` for the last
variant.

### program.cache_info { #messagepassingcache_info }

Property returning executable-cache hit/miss/variant counts.

### program.last_variant and program.variants { #messagepassinglast_variant }

The `CompiledVariant` record of the most recent call, or of every cached
specialization: backend, provider, lowering, passes, artifacts, diagnostics
and schedules. `last_variant` raises before the first call.

## Reducers { #reducers }

A reducer is executable compiler input, not an eager tensor operation. The
[reducers guide](reducers.md) explains the algebra contract, attributes and
lowering selection in prose.

### gf.sum(*, identity=0, deterministic=False) { #gfsum }

Associative, commutative additive reducer. Example:
[GCN aggregation](examples/message-passing.md#gcn-aggregation).

### gf.mean(*, deterministic=False) { #gfmean }

Neighbor-mean reducer over built-in tuple state `(sum, count)`; a degree-0
row yields NaN. Guide: [gf.mean()](reducers.md#gfmean).

### gf.prod(*, deterministic=False) { #gfprod }

Multiplicative reducer, promoted to a zero-safe product IR; a degree-0 row
yields the identity 1. Example:
[Product of edge gates](reducers.md#product-of-edge-gates).

### gf.online_softmax(*, accumulation_dtype=None, deterministic=False, block_prune_threshold=None) { #gfonline_softmax }

Stable multi-value streaming reducer; the edge region returns
`reducer(score, value)` (an `OnlineSoftmaxItem`). `block_prune_threshold` in
`(0, 1]` is a semantic approximation policy for dense streaming tiles, not a
scheduling hint. Examples:
[Full attention, no mask](examples/attention.md#full-attention-no-mask),
[Tile-pruned sparse attention](examples/attention.md#tile-pruned-sparse-attention).

### class gf.Reducer { #gfreducer }

User-defined reducer base class: subclass and implement `identity()`,
`lift(*messages)`, `combine(left, right)` and `finalize(state)`; state may
be a scalar or a tuple. Class attributes `associative`, `commutative` and
`deterministic` declare the algebra — parallel lowering requires an explicit
associativity declaration, which Tiga never proves from Python source.
Example:
[User-defined reducer](examples/message-passing.md#user-defined-reducer);
guide: [Inventing a reducer](reducers.md#inventing-your-own).

### reducer(*messages) { #reducer-call }

Calling a reducer inside `edge()` binds it to one or more edge-local
messages as a staged `ReducerCall`.

### reducer.mlir(*, message_dtypes=None, symbol=None) { #reducermlir }

Captures the four regions as a verified native `gf.reducer` op and returns
the MLIR text; `message_dtypes` defaults to a single `float32` message.
Guide: [Inspecting a reducer](reducers.md#inspecting-a-reducer).

??? info "Automatic reducer VJP coverage"

    Additive/stable algebras and captured associative scalar reducers lower to
    balanced or deterministic trees through CPU LLVM and CUDA TTIR. Product is
    promoted to zero-safe CSR product/VJP IR. Empty/ragged rows and ordered
    non-commutative tuple state are covered; unregistered reducer shapes remain
    correctness-only claims.

## nn { #nn }

### gf.nn.trace(module, *, block_e=None, num_warps=None) { #gfnntrace }

Wraps a `torch.nn` module so it can be called inside an edge UDF: eager with
Torch tensors (inputs concatenated along the feature dimension) or captured
into a compilable subgraph for the fused edge-NN tile lowering. `block_e`
(edge-tile size, a power of two in [16, 1024]) and `num_warps` are launch
geometry only and never change numerics; the defaults are per-lowering,
128/4 for edge-centric sum tiles and 16/1 for row-centric attention tiles.
Examples:
[GAT edge attention with an nn score](examples/attention.md#gat-edge-attention-with-an-nn-score),
[Edge nn modules with a fused tile kernel](examples/programs-and-interop.md#edge-nn-modules-with-a-fused-tile-kernel);
guide: [Edge nn modules (CUDA, torch interop)](message-passing.md#edge-nn-modules-cuda-torch-interop).

## Control and jit { #control }

Structured compiler control flow over Tiga semantic values: the body
is captured once and lowered to `gf_control.repeat` / `gf_control.while`
ops, not executed as a Python loop.

### @gf.jit or @gf.jit(max_iterations=k) { #gfjit }

AST route that captures `for i in range(k)` as `gf_control.repeat` and a
bare `while cond:` as `gf_control.while` bounded by the decorator argument.
A leading `if cond: break` in a `for` body is an early exit. Loop-carried
variables are Tensors assigned before the loop; `continue`, mid-body
`break`, `while True`, non-range iteration and loop `else` fail closed.
MessagePassing UDF regions are never AST-transformed. `@gf.jit` also
activates the composition context: sibling MessagePassing calls in the
straight-line body capture as `GraphProgram` SSA leaves and fuse where
legal, leaves participate in tensor arithmetic, and kernel calls inside a
staged loop body keep per-iteration semantics (inlined, never registered
as top-level leaves). Leaves are differentiable when a captured field
requires gradients: reverse mode re-expands each leaf inline (per leaf,
unfused) and accumulates contributions onto shared inputs; unsupported
leaf forms fail closed. Examples:
[Fixed-iteration PageRank probe](examples/message-passing.md#fixed-iteration-pagerank-probe),
[Matrix-free FEM operator and solver loop](examples/solvers.md#matrix-free-fem-operator-and-solver-loop).

### gf.repeat(initial, body, *, iterations) { #gfrepeat }

Anonymous shorthand capturing a fixed-count loop with one or more
Tensor-carried states; `body` is traced once and must return one Tensor per
carried state with preserved shape, dtype and device. CPU lowering allocates
two reusable buffers per carried value.

### gf.while_loop(initial, condition, body, *, max_iterations) { #gfwhile_loop }

Anonymous shorthand capturing bounded data-dependent control without a host
scalar check; `condition` must return a rank-zero boolean Tensor and
`max_iterations` remains in IR as a finite resource guard. CPU lowers to
`scf.while`; CUDA fails closed until a provider loop plan exists.

### class gf.control.Repeat { #gfcontrolrepeat }

Builder form of the same `gf_control.repeat` op: subclass and override
`body`; captured constants are ordinary instance attributes. Calling the
instance as `kernel(initial, iterations=k)` traces `body` exactly once.

### class gf.control.While { #gfcontrolwhile }

Builder form of `gf_control.while`: subclass and override `condition` and
`body`; calling the instance as `kernel(initial, max_iterations=k)` applies
the same bounds as `gf.while_loop`.

Stationary solvers are grammar sugar over these primitives, not core API:
[examples/solvers.py](https://github.com/walkerchi/TIGA-lang/blob/main/examples/solvers.py)
provides `dot`, `vector_norm`, `richardson` and `cg`; see
[Linear solvers and control flow](examples/solvers.md).

## Distributed and halo { #distributed-and-halo }

### gf.DeviceMesh(device_type, shape, *, names=()) { #gfdevicemesh }

Describes logical devices (an int or tuple `shape`) without initializing a
process group. Deployment binds it to Torch `DeviceMesh`, MPI, NCCL/RCCL or
a vendor communicator. Example:
[Two-process halo exchange](examples/distributed-memory.md#two-process-halo-exchange).

### gf.ByDestination(mesh_axis=0, balance="auto") { #gfbydestination }

Partition policy assigning destination entities and final reducer state to
mesh shards; `balance` is `"edges"`, `"entities"` or `"auto"`.

### gf.GraphPlacement(mesh, partition, halo_depth) { #gfgraphplacement }

Logical ownership and ghost requirements attached to a Graph snapshot by
`graph.halo(...)`; `halo_depth` is a non-negative integer or `"auto"`.

### gf.HaloMap { #gfhalomap }

Concrete per-rank owner/ghost map: `owned_begin`/`owned_end`, `ghost_ids`,
`receive_from` and `send_to`, with `owned_entities`, `ghost_entities` and
`bytes_for(itemsize, trailing_elements=1)` helpers.

### gf.derive_halo_map(row_ptr, col_idx, *, num_entities, world_size, rank, peer_requests=None) { #gfderive_halo_map }

Derives exact ghosts from owned CSR rows without a framework dependency;
`peer_requests[p]` lists the locally owned entity IDs requested by peer `p`
and populates `send_to`. Guide:
[Ownership and ghosts, exactly](memory-and-distributed.md#ownership-and-ghosts-exactly).

### gf.collective_halo_maps(row_ptr, col_idx, *, num_entities, world_size) { #gfcollective_halo_maps }

Creates mutually consistent receive/send maps for all ranks.

### gf.exchange_halo(halo, owned_data, *, element_bytes, transport) { #gfexchange_halo }

Packs, exchanges and unpacks one compiler-derived neighbor halo over a
transport and returns a `HaloBuffer`. `owned_data` is destination-owner
local byte storage ordered from `halo.owned_begin`; no framework tensor type
is involved.

### DistributedRuntime(transport, *, progress_threads=1) { #distributedruntime }

Owns asynchronous communication progress below graph kernels. Used as a
context manager it becomes the active runtime; calling a distributed graph
without one fails closed. `runtime.exchange_halo(halo, owned_data, *,
element_bytes)` starts halo progress and returns a provider-neutral
completion; `runtime.last_execution_trace` reports timings from the latest
automatic sharded execution.

### DistributedRuntime.from_provider(name, *, progress_threads=1, **options) { #distributedruntimefrom_provider }

Binds a deployment transport plugin below the unchanged graph API:
`"mpi"` (mpi4py wrapper), `"tcp"` (dependency-free cross-machine sockets)
and `"nccl"` (grouped send/recv over `libnccl`) are registered by default;
third parties register through the `tiga.transport` entry-point group.
Examples:
[Two-process halo exchange](examples/distributed-memory.md#two-process-halo-exchange),
[Running across processes and machines](examples/distributed-memory.md#running-across-processes-and-machines);
guide: [Runtimes and transports](memory-and-distributed.md#runtimes-and-transports).

??? info "Current distributed execution boundary"

    CPU rank-local execution, reverse halo VJP, paged CSR shards and real
    two-process MPI transport are executable. CUDA native buffers use separate
    communication/compiler streams and verified event ordering. The one-rank
    NCCL fixture proves provider binding only; real multi-GPU NCCL/RCCL
    correctness and overlap performance remain fail-closed gates. See
    [Memory hierarchy and distributed execution](memory-and-distributed.md).

## Spill and disk { #spill-and-disk }

### Tensor.disk(*, name=None) { #tensordisk }

Spills the payload to disk and frees the in-memory buffer; any later read
lazily reloads it. Anonymous spills are deleted when the tensor is collected
or the process exits; a named spill persists in `TIGA_SPILL_DIR`
(default `~/.cache/tiga/spill`). Example:
[Spilling tensors to disk](examples/distributed-memory.md#spilling-tensors-to-disk).

### Tensor.cpu() { #tensorcpu }

Ensures the payload is resident in CPU RAM, explicitly reloading a spill;
reads also reload automatically.

### gf.from_disk(name) { #gffrom_disk }

Attaches a named on-disk spill from any process as a lazily-loaded Tensor;
raises `FileNotFoundError` for unknown names. Example:
[A disk-resident giant graph](examples/distributed-memory.md#a-disk-resident-giant-graph).

## Program and visualize { #program-and-visualize }

### @gf.program { #gfprogram }

Compatibility capture boundary kept for straight-line code that cannot
offer source access: MessagePassing calls inside the function become
applies of one `GraphProgram`, a typed acyclic SSA composition.
`@gf.jit` activates the same context automatically (and additionally
captures loops), so new code should reach for `@gf.jit`; stacking the two
is legal and idempotent. Ordinary leaf calls outside either decorator
remain automatic JIT. Example:
[cross-kernel SSA capture](examples/programs-and-interop.md#gfprogram-ssa-capture).

### GraphProgram.apply(kernel, *, graph, src, dst, edge=None, **params) { #graphprogramapply }

Adds one MessagePassing leaf to the composition and returns its typed
`ProgramValue`.

### GraphProgram.outputs(*values) { #graphprogramoutputs }

Selects the values observed at the composition boundary. Bare
`ProgramValue` leaves select their producers directly; a Tensor expression
contributes every program leaf it transitively references, so epilogue
arithmetic composed on top of leaves is a legal output spelling without
claiming fusion into the leaf kernels.

### GraphProgram.run() { #graphprogramrun }

JITs and executes the selected outputs; independent leaves are fused
horizontally where legal, dependent applies run through a typed runtime
dependency DAG.

### GraphProgram.ir(stage="domain") { #graphprogramir }

Returns `domain`, `fused`, `iteration`/`iter` or `kernel` IR text. Calling
`ir` is the observation boundary that triggers composition, verification and
lowering.

### GraphProgram.code(kind="ptx") { #graphprogramcode }

Returns provider artifacts such as `ttir` or `ptx`; for dependent applies a
mapping of per-apply artifacts.

### GraphProgram.explain() and GraphProgram.semantic_hash { #graphprogramexplain }

Render apply/fusion counts plus the content hash of the composed domain IR.

### ProgramValue.materialize() { #programvaluematerialize }

JITs and executes the connected program at this observation boundary and
returns the concrete value. `value.ir(stage)` inspects the owning program.

### gf.visualize.heatmap(values, *, vmin=0.0, vmax=1.0, low=None, high=None, cmap=None) { #gfvisualizeheatmap }

Compiles a colormapped heatmap of a contiguous rank-two floating Tensor into
one fused kernel and returns a lazy RGB `Raster`. The default colormap is
`viridis`; `cmap` selects another multi-stop colormap — a name from
`gf.visualize.colormaps()`, a list of evenly spaced RGB colors, or a list of
`(position, RGB)` stops with strictly increasing positions in [0, 1] — and
`low`/`high` select a plain two-color ramp instead (a missing end falls back
to the matching viridis endpoint). Two-color ramps leave values outside
`[vmin,vmax]` unclipped in the lazy Tensor, keeping the expression
differentiable; multi-stop colormaps clamp to the end colors like
Matplotlib. The color transform is ordinary Tensor broadcast/arithmetic
(`sqrt(x²)` tent functions for multi-stop ramps); there is no visualization
operation in compiler core. Example:
[GPU heatmap visualization prep](examples/visualization.md#gpu-heatmap-visualization-prep).

### gf.visualize.colormaps() { #gfvisualizecolormaps }

Returns the names of the built-in heatmap colormaps: `viridis`, `magma`,
`plasma`, `inferno`, `jet`, `coolwarm`, `gray` — eight-stop samples of the
Matplotlib maps of the same name, interpolated piecewise-linearly inside the
fused kernel.

### Raster { #raster }

Lazy RGB raster whose pixels remain a Tiga Tensor: `realize()`,
`prepare()`, `execution`, `mlir(*, verify=False)` and `generated_code(kind)`
expose the same compiler/runtime path as `Tensor`; `to_numpy(*, clip=True)`,
`save(path)` (Pillow) and `show(**imshow_options)` (Matplotlib) are optional
host interop/encoding helpers.

### gf.visualize.Camera { #gfvisualizecamera }

Host-side perspective camera for point-cloud rendering:
`Camera(position, target, fov=45.0, up=(0.0, 0.0, 1.0))`, with
`Camera.auto(positions, *, fov=45.0, margin=1.2)` fitting target and
distance from a bounding sphere — planar (N, 2) input is viewed top-down,
3-D input follows the smallest-variance principal axis and falls back to an
isometric direction for isotropic clouds — and
`Camera.from_angles(elevation=30.0, azimuth=-60.0, *, positions=None, distance=None, fov=45.0)`
placing the camera on a sphere around the target.
`camera.world_to_ndc(points_xyz)` projects (N, 3) points to
`(ndc, depth, visible)`. Example:
[particles/video](examples/visualization.md#particles-and-video).

### gf.visualize.particles(positions, values=None, *, camera="auto", width=512, height=512, point_radius=2.0, vmin=None, vmax=None, low=None, high=None, cmap=None) { #gfvisualizeparticles }

Splats every point as a disk into a host-side scalar field — the point
value, or a unit density when `values=None` — then colors that field through
the `heatmap` Tensor expression and returns a lazy `Raster`. `camera`
accepts `"auto"`, a `Camera`, or an `(elevation, azimuth)` tuple;
`vmin`/`vmax` default to the field's data range; `cmap` picks a multi-stop
colormap (see [`heatmap`](#gfvisualizeheatmap)), overriding `low`/`high`.
Example:
[Delaunay mesh from points](examples/visualization.md#simulation-meshes-load-shade-orbit).

### gf.visualize.delaunay(positions, values=None, *, camera="auto", width=512, height=512, vmin=None, vmax=None, low=None, high=None, cmap=None, wireframe=False) { #gfvisualizedelaunay }

Triangulates positions — planar (N, 2) input directly, 3-D input after
camera projection — fills every triangle with the mean of its vertex values
(`values=None` uses a uniform value; `wireframe=True` draws the triangle
edges instead) and colors the resulting scalar field through `heatmap`
(`cmap` overrides `low`/`high`), returning a lazy `Raster`. Example:
[mesh rendering](examples/visualization.md#simulation-meshes-load-shade-orbit).

### gf.visualize.mesh(positions, faces, values=None, *, camera="auto", width=512, height=512, wireframe=False, vmin=None, vmax=None, low=None, high=None, cmap=None) { #gfvisualizemesh }

Renders a given triangle mesh — where `delaunay` computes the triangulation
from points, `mesh` takes explicit `faces` (an (M, 3) index array, e.g. a
simulation mesh or a model from `load_obj`). Triangles are painted
far-to-near by mean vertex depth (painter's algorithm) so nearer triangles
occlude farther ones; each is filled with the mean of its finite vertex
values (`values=None` uses a uniform value; all-non-finite triangles are
skipped), and `wireframe=True` draws edges instead. Camera handling, planar
input, and behind-camera clipping match `delaunay`. Example:
[simulation meshes: load, shade, orbit](examples/visualization.md#simulation-meshes-load-shade-orbit).

### gf.visualize.splats(positions, colors, scales, *, rotations=None, opacities=None, camera="auto", width=512, height=512, cutoff=3.0, background=(0,0,0)) { #gfvisualizesplats }

Renders anisotropic 3-D Gaussians into a lazy `Raster` (EWA splatting, as in
3-D Gaussian Splatting): the covariance `R·diag(scales²)·Rᵀ` projects
through the perspective Jacobian to a 2-D ellipse, evaluated within
`cutoff` sigma, and colors alpha-composite near-to-far with front-to-back
transmittance. `colors` (N, 3) RGB in [0, 1], `scales` (N, 3) or (N,)
positive standard deviations, `rotations` (N, 4) `(w, x, y, z)`
quaternions, `opacities` (N,) in [0, 1]. Gaussians behind the camera,
sub-pixel, or fainter than 1/255 are skipped. Example:
[Gaussian splats and volumes](examples/visualization.md#gaussian-splats-and-volumes).

### gf.visualize.volume(density, *, camera="auto", width=512, height=512, steps=128, cmap="viridis", vmin=None, vmax=None, scale=8.0, background=(0,0,0)) { #gfvisualizevolume }

Ray-marches a density grid (X, Y, Z) — mapped onto the unit cube — into a
lazy `Raster` with the emission-absorption model: trilinear samples along
each pixel ray emit the `cmap` color of the normalized density and absorb
with `alpha = 1 - exp(-density·scale·Δt)`. `vmin`/`vmax` default to the
data range; `camera` frames the cube when `"auto"`. Example:
[Gaussian splats and volumes](examples/visualization.md#gaussian-splats-and-volumes).

### gf.visualize.load_ply(path) { #gfvisualizeloadply }

Parses a PLY file (ASCII or binary_little_endian) into
`(positions, faces, extras)`: an (N, 3) float64 array from the vertex
`x y z` properties, an (M, K) int64 array from the face list property (or
`None`), and a dict of every remaining vertex property — the entry point
for 3-D Gaussian Splatting payloads (`f_dc_*`, `opacity`, `scale_*`,
`rot_*`). Example:
[Gaussian splats and volumes](examples/visualization.md#gaussian-splats-and-volumes).

### gf.visualize.load_obj(path) { #gfvisualizeloadobj }

Parses a minimal Wavefront OBJ file into `(positions, faces)`: `v x y z`
vertex rows become an (N, 3) float64 array, `f` face rows (including
`f a/b/c` token forms; only the vertex index is used) become an (M, 3)
integer array with 1-based indices resolved and polygons fan-triangulated.
Empty files, missing vertices/faces, and out-of-range indices raise
`ValueError`. Example:
[simulation meshes: load, shade, orbit](examples/visualization.md#simulation-meshes-load-shade-orbit).

### gf.visualize.export_ply(path, positions, *, faces=None, values=None, low=None, high=None, cmap=None) { #gfvisualizeexportply }

Dumps geometry as binary_little_endian [PLY](https://en.wikipedia.org/wiki/PLY_(file_format))
for [Blender](https://en.wikipedia.org/wiki/Blender_(software)) import:
vertices carry `x y z` float32 (planar (N, 2) input gets `z = 0`), optional
`faces` (M, 3) become a `vertex_indices` list element, and `values` adds
`red green blue` uint8 (the same colormap as `heatmap` — `viridis` by
default, or `cmap`/`low`/`high` — scaled to the data range) plus
`scalar_value` float32 with the raw field. Positions and
values arrive through one host copy (GPU tensor included) and the payload is
assembled in a single structured buffer, so a million points export in well
under a second. Example:
[simulation meshes: load, shade, orbit](examples/visualization.md#simulation-meshes-load-shade-orbit).

### gf.visualize.export_obj(path, positions, faces) { #gfvisualizeexportobj }

Writes a text Wavefront OBJ — the exact inverse of `load_obj`: `v` rows at
full float64 round-trip precision and 1-based `f` rows, so
`load_obj(export_obj(...))` restores coordinates and faces identically.
Example:
[simulation meshes: load, shade, orbit](examples/visualization.md#simulation-meshes-load-shade-orbit).

### gf.visualize.export_vdb(path, positions, values, *, voxel_size=0.05) { #gfvisualizeexportvdb }

Rasterizes points plus a scalar field into an [OpenVDB](https://en.wikipedia.org/wiki/OpenVDB)
dense grid. Requires the optional `pyopenvdb` dependency — the
dependency-free PLY export already covers the Blender interchange path, so
OpenVDB is only pulled in when a volumetric grid is explicitly wanted;
without it the function raises `ModuleNotFoundError`.

### gf.visualize.save_video(frames, path, *, fps=30) { #gfvisualizesavevideo }

Encodes any iterable of frames — `Raster` or float (H, W, 3) arrays in
[0, 1] — as `.gif` (Pillow) or `.mp4` (raw RGB piped into an ffmpeg
subprocess); generators are consumed frame by frame without buffering.
Example:
[particles/video](examples/visualization.md#particles-and-video).

### gf.compiler.Dim / TensorSpec / ShapeSpecializer { #gfcompiler-symbolic-shapes }

Bounded symbolic signatures: `Dim(name, minimum=1, maximum=None,
multiple_of=1)`, `TensorSpec(shape, *, dtype=gf.float32, device=None)` and
`ShapeSpecializer(specs, compiler)`. Repeated symbols must bind to one
extent across arguments; minimum/maximum/divisibility guards are checked
before compilation, and each legal binding forms a concrete MLIR
specialization/cache key. Guide:
[Tensor runtime and autograd](runtime-and-autograd.md#current-alpha-slice).
