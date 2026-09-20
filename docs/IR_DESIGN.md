# Tiga IR implementation notes

!!! note "Non-normative implementation notes"
    This historical research note is not a support contract. The API reference,
    support matrix and regression tests define current behavior.

Status: architecture draft  
Date: 2026-08-14

This document defines the IR boundary of Tiga from a lazy Python/Torch program to a
CPU/GPU/distributed executable. The goal is not to implement every op immediately, but to keep
the M0 minimal dialect from locking out generated relations, auto-fusion, hierarchical storage,
and distributed MessagePassing.

---

## 1. Design principles

1. **Graph schema and graph data are separate.** The IR stores Entity/Relation schema,
   provenance, lifecycle, and properties; the actual CSR indices, positions, and Field values
   are runtime operands.
2. **GraphProgram uses SSA value semantics.** `edge()` reads the invocation-input snapshot and
   `node()` produces a new Field version; physical in-place behavior is a result of later
   bufferization.
3. **Assembly is generalized MessagePassing.** Local contributions of a HyperRelation are
   reduced through a typed destination map; there is no public `gf.assemble()` that bypasses
   relation semantics.
4. **Distribution does not change mathematical semantics.** A single-device reduction can be
   split into rank-local partial reducers, with the destination owner performing the final
   combine; `node()` executes exactly once.
5. **Materialized/Paged/Generated are plans, not Graph types.** The same logical relation can
   have multiple ResolvedRelations.
6. **No IR op is created per edge.** The IR describes relation schema and iteration domains;
   runtime graph size does not linearly inflate compiler IR.
7. **Fast path first.** Canonical semantic IR, template traversal, layered caches, and bounded
   specialization prevent the JIT from degenerating into whole-program autotuning.

---

## 2. Overall hierarchy

```text
Python lazy FieldValue DAG / torch.compile region
  ↓ capture
gf.domain
  GraphProgram SSA, Entity/Field/Relation, edge/node, Reducer, Effect
  ↓ whole-program legality/canonicalization/fusion candidates
gf.task
  partition, ownership, halo, page/copy/collective, Event DAG
  ↓ per compute task
gf.iter
  entity/port/neighbor/feature axes, generated/paged/materialized iteration, reduce/route
  ↓ target planning
gf.kernel
  work tile, workgroup/subgroup/lane mapping, memory promotion, local pipeline
  ↓ dialect conversion
upstream MLIR
  arith/math/scf/affine/vector/tensor/memref/gpu/linalg/async/LLVM
  ↓ provider/backend
PTX/cubin, HIP/hsaco, PPU binary, CPU object
```

Not every target must physically pass through all dialects. The M0 reference evaluator can
interpret `gf.domain` directly; a simple CPU stencil can go straight from `gf.domain` into
`affine/scf/vector`; but the semantic boundaries of each layer must be consistent, and
inspection uses these stable stage names.

---

## 3. `gf.domain`: the semantic IR

### 3.1 Core types

The following syntax is schematic MLIR, not the final parser spelling:

```mlir
!gf.entity_set<"node", id = i64, coord = [index, index]>
!gf.entity_ref<@nodes>

!gf.field<@nodes, f32>
!gf.field<@nodes, tensor<3xf32>>
!gf.field_value<@nodes, tensor<3xf32>, version = ?>

!gf.relation<
  sources = [@nodes],
  destinations = [@nodes],
  arity = 2,
  origin = #gf.origin<external>,
  lifecycle = #gf.lifecycle<frozen>
>

!gf.relation_item<@edges>
!gf.snapshot<@edges>
!gf.reducer_state<@sum_f32>
!gf.event
```

`Field` is the schema; `FieldValue` is the SSA value of a specific invocation/version. The
high-level IR does not permit changing the visible value in place through the same FieldValue
handle.

### 3.2 Entity and Field schema

```mlir
gf.entity_set @nodes {
  id_type = i64,
  coordinate_rank = 2,
  global_cardinality = #gf.dynamic
}

gf.field @u on @nodes : f32 {
  mutability = "versioned",
  semantic_role = "state"
}
```

Runtime binding binds `%u : !gf.field_value<@nodes, f32>` to a Torch Tensor, a Tiga
Region, or a distributed local shard. Global cardinality and runtime Tensor addresses do not
enter symbol identity.

### 3.3 Relation origin, lifecycle, and realization

The three dimensions must be independent:

```text
origin:      External | Procedural
lifecycle:   Frozen | Rebuildable
realization: Materialized | Paged | Generated
```

External CSR:

```mlir
gf.relation @adj (%row_ptr, %col_idx) : ... {
  origin = #gf.external<format = "csr", index = i32>,
  lifecycle = #gf.frozen,
  properties = {sorted = true, symmetric = false}
}
```

Affine stencil:

```mlir
gf.relation @dx : ... {
  origin = #gf.procedural<"affine_stencil">,
  lifecycle = #gf.frozen,
  ports = [
    #gf.port<id = 0, offset = [-1, 0], side = "minus">,
    #gf.port<id = 1, offset = [ 1, 0], side = "plus">
  ],
  lattice_basis = #gf.identity_basis<2>,
  boundary = #gf.boundary<periodic, periodic>
}
```

Ports hold topology/orientation facts; the numerical coefficient of a particular derivative
belongs to the edge/node operator. Non-Cartesian lattices generate derived edge values through
`physical_displacement = basis * integer_offset`, so axial hex ports are still a fixed affine
domain.

Regular tilings cannot be represented as a Cartesian/hex-style case enum. The general schema
in `gf.domain` is:

```mlir
gf.index_domain @cells {
  rank = 2,
  set = #gf.presburger_set<(i, j) : ...>,
  boundary_action = #gf.periodic_quotient<...>
}

gf.indexed_complex @complex on @cells {
  generators = dense<...> : tensor<2x2xf64>,
  sites = [
    #gf.site<"A", offset = [...]>,
    #gf.site<"B", offset = [...]>
  ]
}

gf.relation @nearest on @complex {
  links = [
    #gf.endpoint_map<dst_site = "A", src_site = "B",
                     map = (i, j) -> (i, j)>,
    #gf.endpoint_map<dst_site = "A", src_site = "B",
                     map = (i, j) -> (i - 1, j + 1)>
  ]
}
```

Formal core:

```text
IndexDomain D ⊂ Z^d
Entity (i, site), i ∈ D
EndpointMap: (i, dst_site) → (A i + b, src_site)
Embedding: x(i, site) = B i + c_site, or runtime coordinate Field
BoundaryAction: quotient/map/ghost/mask
```

Cartesian, triangular, honeycomb, Kagome, FCC/BCC, and staggered/MAC grids are just different
domain/sites/maps/embeddings. A finite polygonal patch can use Presburger pieces; a general
aperiodic generator uses `GeneratedTileProducer`; an unstructured mesh uses explicit
cell-complex incidence. Passes query through interfaces whether an endpoint map is affine,
whether a port domain is finite, and whether an embedding is constant, instead of
`switch (tiling_kind)`.

An unstructured `Mesh` is not another kind of kernel input; it is a Python handle over several
EntitySets/Relations/Fields. The Domain IR stores typed oriented incidence:

```mlir
gf.entity_set @vertices {dimension = 0}
gf.entity_set @faces    {dimension = 2}
gf.entity_set @cells    {dimension = 3}

gf.relation @cell_faces (%offsets, %indices, %orientation)
  : !gf.relation<sources = [@cells], destinations = [@faces],
                   origin = external, lifecycle = frozen>

gf.field @position on @vertices : tensor<3xf64>
gf.field @normal   on @faces    : tensor<3xf64>
gf.field @volume   on @cells    : f64
```

Incidence can be fixed-arity, type-bucketed, or ragged. `cell adjacency through faces` is
relation composition:

```text
cell_dst ← incidence → face ← incidence → cell_src
```

The planner may materialize CSR, or traverse the incidence composition without writing out the
cell-cell adjacency. Orientation is a relation item property, ensuring that the same face
produces opposite normal/sign for the cells on its left and right. A curvilinear mesh only
turns the embedding/Jacobian/metric into runtime Fields; AMR adds level, parent/child, and
coarse-fine interface Relations. All of them lower to the same `gf.apply`.

Radius relation:

```mlir
gf.relation @radius (%positions, %cutoff, %box) : ... {
  origin = #gf.procedural<"radius">,
  lifecycle = #gf.rebuildable<dependencies = [0, 1, 2],
                              predicate = "field_version_or_skin">,
  properties = {directed = true}
}
```

The compiled function receives a graph/relation handle or the underlying buffers; the relation's
actual contents and logical version do not enter the code cache key. A version change
invalidates materialization/statistics/plans.

### 3.4 Snapshot resolve

```mlir
%snapshot, %ready = gf.resolve %relation
  budget(%memory_budget)
  : !gf.relation<...> -> (!gf.snapshot<...>, !gf.event)
```

The result of `gf.resolve` is concretized after whole-program planning into one of:

```text
MaterializedInstance
PagedTileStream
GeneratedTileProducer
```

In the Domain IR, resolve represents a snapshot/version correctness barrier; it does not
promise an independent builder kernel or a complete adjacency allocation.

### 3.5 Reducer definition

A Reducer is a symbol op whose regions can be inlined, specialized, or mapped to backend
primitives:

```mlir
gf.reducer @sum_f32 : f32 -> f32 -> f32 {
  properties = [associative, commutative, atomic_add],
  determinism = #gf.determinism<order_independent_not_bitwise>
  identity { gf.return 0.0 : f32 }
  lift(%x: f32) { gf.return %x : f32 }
  combine(%a: f32, %b: f32) {
    %c = arith.addf %a, %b : f32
    gf.return %c : f32
  }
  finalize(%x: f32) { gf.return %x : f32 }
}
```

The state of online softmax is `(m, l, o)`; `combine` uses the stable rescale formula. The
state of PortGather is a compile-time port-indexed tuple/vector plus a validity mask, and it
declares `unique_by_port` or the behavior for duplicates. The verifier does not attempt to
prove the associativity of an arbitrary user reducer; builtin reducers are guaranteed by the
system, and the algebra properties of a custom reducer are an explicit contract that enters
determinism diagnostics.

The public `NeighborhoodKernel.compute(center, nbr, ...)` is an ergonomic frontend for a fixed
finite port domain; at capture time it is normalized into `PortGather + local region`, sharing
`gf.apply` and the iteration/storage/distributed lowering with MessagePassing. It is not a
random-access container usable with arbitrary ragged/dynamic neighbor streams.

### 3.6 Unified MessagePassing op

The public `edge()/node()` captures into a single `gf.apply`:

```mlir
%u_next = gf.apply %snapshot
    fields(%u_src, %u_dst, %weight)
    params(%dt)
    reducer(@sum_f32)
    effects [read(%u_src), read(%u_dst), read(%weight)]
    : ... -> !gf.field_value<@nodes, f32> {

  ^edge(%src: !gf.entity_ref<@nodes>,
        %dst: !gf.entity_ref<@nodes>,
        %item: !gf.relation_item<@adj>):
    %su = gf.read %u_src[%src] : f32
    %du = gf.read %u_dst[%dst] : f32
    %w  = gf.read %weight[%item] : f32
    %d  = arith.subf %su, %du : f32
    %m  = arith.mulf %w, %d : f32
    gf.edge_yield %m : f32

  ^node(%dst: !gf.entity_ref<@nodes>, %total: f32):
    %old = gf.read %u_dst[%dst] : f32
    %du  = arith.mulf %dt, %total : f32
    %new = arith.addf %old, %du : f32
    gf.node_yield %new : f32
}
```

Semantics:

```text
contribution[e] = edge(input snapshot)
aggregate[d]    = reducer(contribution[e] where destination(e)=d)
output[d]       = node(destination input snapshot, aggregate[d])
```

The `node` region can be omitted, in which case the output is the reducer's finalized result.
There is no separate `old` parameter; the node reads the old value through the destination
snapshot. Bufferization may alias output storage with input storage only after proving the
absence of read-after-overwrite.

### 3.7 HyperRelation and assembly

Binary MessagePassing is a special case of `gf.apply`. A HyperRelation item can have fixed or
ragged endpoints:

```text
item = element
endpoints(item) = [node_0, ..., node_{P-1}]
```

The element region produces port-indexed local contributions; the IR uses a typed route
terminator:

```mlir
^edge(%element, %endpoints, %item):
  %re = gf.call_local @element_residual(%endpoints, %geometry)
        : ... -> tensor<Pxf32>
  gf.route_yield %re to %endpoints by #gf.endpoint_identity
                 reducer @sum_f32
```

`route_yield` is a terminator inside `gf.apply`, not a public `gf.assemble()`. It declares how
the local result axis maps to the destination EntitySet. Lowering can choose among:

- matrix-free element tile: compute once → scatter residual;
- coloring/atomics/segmented reduction;
- assemble a sparse matrix: the destination becomes a typed `(row_dof, col_dof)` EntitySet;
- distributed partial route: combine by destination owner.

### 3.8 GraphProgram SSA and auto-fusion

Consecutive leaf calls in an ordinary Python function form one `func.func`/GraphProgram:

```mlir
func.func @euler_rhs(%mesh, %q, %geometry) -> !gf.field_value<...>
    attributes {gf.program} {
  %q_face = gf.apply @reconstruct ...
  %flux = gf.apply @riemann ...
  %residual = gf.apply @divergence ...
  %out = gf.local_map @add_source(%q, %residual)
  return %out
}
```

A Tiga-native `FieldValue` is lazy by default: a leaf call returns a deferred SSA handle;
the first external observation, unsupported escape, mutation/barrier, memory-pressure flush, or
explicit materialization triggers planning/JIT. `@tg.jit` is the single user entry point — loop
capture plus automatic composition; `@tg.program` remains only as a compatibility
capture/AOT/export/debug boundary.

The auto-fusion pass derives candidates from SSA def-use chains and Effects; it does not use
decorators as fusion hints:

```text
legal if:
  producer result has no observable escape
  snapshots/versions are compatible
  iteration domains can be composed
  effects and ownership preserve ordering
  reducer route/finalize can be transformed

profitable if:
  saved bytes/launch/communication > duplicate work
  register/shared footprint respects capability
  occupancy and parallelism stay acceptable
```

Typical rewrites:

- local map → edge/node inline;
- same-relation multi-consumer traversal fusion;
- generated relation builder → consumer fusion;
- reconstruction → face flux tile chaining;
- face flux → cell divergence partial routing;
- pointwise time update → node region fusion.

Communication, global reductions, unsupported ops, external Tensor observation, alias barriers,
or excessive resource cost form fusion boundaries. `explain()` must give the reason for every
unfused SSA edge.

---

## 4. `gf.task`: the storage/distributed execution graph

### 4.1 Why it comes before `gf.iter`

Distributed/out-of-core planning first splits one logical apply into local compute, transfer,
halo, and combine tasks; each compute task then lowers its own relation iteration. A
single-device in-memory program may use a trivial task graph and does not have to print this
layer explicitly.

### 4.2 Region, instance, and event

```mlir
%owned = gf.region.subset %u [#gf.partition_owned]
%ghost = gf.region.subset %u [#gf.partition_ghost]
%tile = gf.storage.acquire %stream[%tile_id] budget(%bytes)
%copy_done = gf.task.copy %tile_src to %tile_hbm
%event = gf.task.launch @kernel(...) depends_on(%copy_done)
gf.storage.release %tile after(%event)
```

A `Region` is a logical Field subset + version; a `PhysicalInstance` is a layout/compression
copy in some memory space. Register/shared placement does not use these long-lived ops; it is
expressed in `gf.kernel`.

### 4.3 Invariant semantics of distributed MessagePassing

The global definition is unchanged:

```text
aggregate[d] = reduce(contribution[e] for e where destination(e)=d)
```

After partitioning, every destination must have a unique logical owner. The planner can choose
among:

#### Owner-compute / pull

```text
destination owner stores/iterates incoming relation items
remote source Field → ghost/halo fetch
owner computes all contributions
owner runs node() once
```

Suitable for destination-partitioned CSR, stencils, and spatial radius.

#### Edge-compute / push

```text
edge/element owner computes contributions
local combine by destination ID
send (destination, partial reducer state) to destination owner
owner combines received states
owner runs node() once
```

Suitable when the edge/element geometry is large, when the graph is already partitioned by
element, or when moving source features is expensive.

#### Hybrid/2D

The source feature, edge, and destination axes are distributed across a device mesh, with
different dimensions broadcast/reduced separately. Dense all-pairs, very long-range
interactions, and extremely high-degree graphs may select this plan.

The reducer must provide a `combine` that is legal across ranks. If a reducer cannot be
combined, semantics can only be preserved by delivering all messages to the owner in the
prescribed order; the planner must reject an unaffordable distributed plan rather than silently
change the result.

### 4.4 Overlap task graph

A typical task IR for owner-compute stencil/radius:

```text
halo_pack(local boundary fields)
  → halo_exchange ───────────────→ boundary_apply
interior_apply ──────────────────→ merge/finalize → node/output version
```

Interior and exchange have no dependency and can run in parallel; the boundary waits on the
halo event. For a push plan:

```text
local edge compute
  → partial combine
  → pack by destination owner
  → exchange
  → owner combine
  → node
```

Snapshot version, halo version, and output version must match; the verifier forbids mixing
ghost data from different timesteps.

### 4.5 Public distributed binding

Users do not write communication loops, but they must supply the semantic metadata that cannot
be inferred: global entity IDs, a mapping between local shards and the global EntitySet, or a
partition manifest. Schematic:

```python
mesh = tg.DeviceMesh("cuda", (2, 4), names=("rack", "gpu"))
graph = tg.load("mesh.gfg").halo(
    mesh,
    partition=tg.ByDestination(mesh_axis="gpu", balance="edges"),
    depth="auto",
)
u = tg.Field.from_local(local_u, entities=graph.nodes)

u_next = Diffusion()(graph=graph, src={"u": u}, dst={"u": u})
```

`Graph.halo()` is a declarative logical transformation that still returns an ordinary
`tg.Graph` and does not communicate immediately. The process group/device mesh can bind a
deployment config, a Torch DeviceMesh, or a `torchrun`/MPI/vendor launcher. Partition
algorithms, halo packing, transport, overlap, and kernels are planner/runtime decisions;
ownership and global IDs cannot be inferred out of thin air.

---

## 5. `gf.iter`: target-independent iteration IR

The standalone dialect spelling of the current minimal implementation is
`gf_iter.traverse/yield`. It can already express `compressed-row` and
`generated-neighborhood`, `destination-major` ordering, and fully preserves the
edge/optional-node regions and reducer/effect metadata. The Axis ops below are a later,
incremental expansion and should not all be built in advance before any optimization need
arises.

### 5.1 Axis model

```text
EntityAxis:   destination/source/entity set
SegmentAxis:  CSR row, bucket, page segment
NeighborAxis: extent depends on destination/segment
PortAxis:     small compile-time stencil/element endpoint domain
FeatureAxis:  dense payload axis
CellAxis:     geometric bin/neighborhood
```

Axes can be dependent/ragged; do not pad neighbor extents to the global maximum degree just to
use an ordinary tensor loop.

### 5.2 CSR lowering

```text
forall dst in D:
  state = reducer.identity
  for p in row_ptr[dst] .. row_ptr[dst+1]:
    src = col_idx[p]
    state = reducer.combine(state, edge(src, dst, p))
  out[dst] = node(dst, reducer.finalize(state))
```

`gf.iter` preserves the `dst → segment → neighbor` dependency, so that later stages can choose
row-per-thread, warp-per-row, split-row, edge-atomic, or degree buckets.

### 5.3 Affine stencil lowering

```text
forall (i, j) in owned grid:
  for port in static Ports:
    (si, sj) = affine_map(i, j, port.offset)
    state = combine(state, edge((si, sj), (i, j), port))
```

The affine map and boundary region should lower to upstream `affine`/`vector` wherever
possible, without generating CSR.

### 5.4 Generated radius lowering

```text
forall owned destination_cell:
  for neighboring_cell_port:
    for destination_particle_tile:
      for source_particle_tile:
        if distance(src, dst) <= cutoff:
          reduce edge(src, dst, derived_geometry)
```

The materialized plan replaces the intermediate layers with COO/CSR segments; the
generated-fused plan consumes the predicate result directly; the paged plan adds tile
acquire/release outside the SegmentAxis.

### 5.5 HyperRelation route lowering

```text
forall element:
  endpoints = relation.endpoints(element)
  local = element_kernel(endpoints)
  for port in endpoint_ports:
    route local[port] to endpoints[port] using reducer
```

The route can lower to atomics, coloring, sort/segment, owner partial state, or a
destination-tile accumulator.

---

## 6. `gf.kernel`: target-aware physical IR

The standalone dialect spelling of the current minimal implementation is
`gf_kernel.launch/yield`, containing the `csr-row/generated-tile` skeleton, degree
worklist/split-row planning, and typed storage/task boundaries. Domain→Iter→Kernel→TTIR
already runs and executes in the CUDA Driver runtime; the more general
WorkTile/Promotion/Pipeline below is still a design — do not mistake the schematic for
implemented ops.

This is the layer that introduces hardware mapping and kernel-local memory:

```text
WorkTile
  logical axes/subset
  → program/workgroup/subgroup/lane/vector/sequential mapping

Promotion
  Field/indices/partial state slice
  → register/shared/LDS/local memory

Pipeline
  async copy → barrier → compute → writeback
  stages/buffers/events
```

Schematic:

```mlir
gf.kernel.launch @radius_tile mapping(#gf.workgroup) {
  %src_smem = gf.kernel.promote %src_tile to #gf.memory<shared>
  %dst_smem = gf.kernel.promote %dst_tile to #gf.memory<shared>
  gf.kernel.pipeline stages(3) {
    gf.kernel.async_copy ...
    gf.kernel.compute ...
  }
}
```

These ops are generated by the compiler/autotuner and do not enter the coarse public API. CPU
lowering may drop the GPU mapping and turn the WorkTile into cache-blocked/vectorized loops.

### 6.1 `gf_tensor` and `gf_control`: differentiable data flow and bounded iteration

`gf_tensor` is the typed SSA layer shared by the runtime Tensor DAG, automatic VJP, and
provider codegen; it is not a second IR stitched together from Python strings. The relation
lowering of static CSR preserves
`gather → edge algebra → csr_segment_sum → node algebra`, where
`csr_segment_sum` carries `degree_min/degree_max` proven by Graph analysis. These attributes
only influence the later schedule; the numerical semantics remain a CSR row reduction. Unknown
degrees use `(0, 0)` and fall back to a tiled loop over arbitrarily long rows; the compiler
never guesses a uniform relation from `E/N`.

`gf_control.repeat` expresses a fixed-count loop-carried Tensor state:

```mlir
%rank1 = "gf_control.repeat"(%rank0, %row_ptr, %col_idx, %weight) <{
  iterations = 20 : i64, num_carried = 1 : i64
}> ({
^bb0(%rank: tensor<Nxf32>, %row: tensor<?xi64>,
     %col: tensor<Exi64>, %w: tensor<Exf32>):
  %next = ... : tensor<Nxf32>
  "gf_control.yield"(%next) : (tensor<Nxf32>) -> ()
}) : (...) -> tensor<Nxf32>
```

`num_carried` allows multiple Tensors of different shapes/types in the operand prefix to
simultaneously become SSA results; the remaining operands are explicit immutable captures. The
loop body is captured only once, and the iteration count does not add compiler IR nodes. CPU
lowering produces an `scf.for` and reuses two ping-pong MemRefs per carried value; the
single-state CUDA runtime reuses two device buffers and a prepared executable, while the
multi-state CUDA loop plan is not yet implemented. The CPU pass also hoists rank-0 reductions
and their scalar algebra inside the body into once-per-iteration temporaries, so that the CG
dot product does not degenerate into `O(N^2)` from being referenced by multiple vector updates;
multi-use vector SSAs (for example `A(p)` and the next residual) are likewise materialized only
once per iteration in block SSA order, while single-use vectors remain fused into their
consumers. Currently, fixed-degree weighted CSR plus a per-node epilogue structurally matches a
two-dimensional row×neighbor TTIR tile with one launch per iteration; unknown/long-tail CSR
uses one program per row with an arbitrarily long blocked loop. There is no PageRank-named op
or codegen case here. `gf_control.while` uses a rank-0
`gf_tensor.compare` condition and a mandatory `max_iterations`; CPU lowers to `scf.while`, the
body is no longer executed after convergence, and the scalar is never converted back to Python.
The GPU command-graph/cooperative persistent plan and the distributed collective condition are
still unimplemented, so PageRank remains only partial evidence for `G0`.

Reverse mode already has an automatic correctness fallback for fixed repeat: the autograd
transform specializes the body per iteration and directly reuses the existing pointwise, CSR
relation, and reducer VJPs, so users do not write a backward pass. The reverse IR/compile work
on this path is O(iterations); it is not yet a reversed `gf_control.repeat`, so it counts only
as semantic coverage, not as registered performance. A later control-autodiff pass must
generate the reverse loop and state tape according to a save/recompute budget, then merge with
the checkpoint/hierarchy planner.

---

## 7. Pass pipeline

### 7.1 Semantic normalization

```text
gf-verify-domain
gf-infer-field-access-and-effects
gf-canonicalize-origin-lifecycle
gf-normalize-edge-node-apply
gf-canonicalize-reducers
gf-build-program-ssa
gf-identify-fusion-candidates
```

### 7.2 Whole-program planning

```text
gf-resolve-shape-and-statistics-buckets
gf-select-materialized-paged-generated
gf-select-ownership-and-partition-plan
gf-form-profitable-fusion-groups
gf-plan-storage-and-events
gf-domain-to-task
```

### 7.3 Iteration lowering

```text
gf-lower-relation-to-axes
gf-lower-route-and-reducer
gf-select-traversal-skeleton
gf-bucket-or-split-ragged-work
gf-task-compute-to-iter
```

### 7.4 Target planning/codegen

```text
gf-map-work-tiles
gf-plan-promotion
gf-plan-local-pipeline
gf-bufferize-field-versions
gf-iter-to-kernel
gf-kernel-to-triton / gpu / vector / LLVM / source provider
```

Every complex pass must have a naive fallback. M0/M1 implement only a minimal subset of these,
but any unimplemented op must produce an explicit diagnostic — no silent miscompiles.

---

## 8. Verification

### 8.1 Domain verifier

- The Field's EntitySet matches the relation endpoint types;
- edge/node access only the declared/inferred Fields and parameters;
- the output Field version is not read as an input snapshot within the same apply;
- relation port, boundary, sortedness, and uniqueness properties are used legally;
- reducer message/state/result types match;
- the destination map and local tensor axis of a HyperRelation route match;
- persistent edge state must not bind to unstable generated edge IDs;
- runtime parameters and specialization constants are clearly distinguished.

### 8.2 Task/distributed verifier

- Every destination has a unique logical owner;
- the node region runs after the final combine, exactly once;
- halo/partial state versions match the snapshot version;
- transfer/launch/release Events have no use-before-ready;
- cross-device reducers have a legal combine;
- the ordering/algorithm in deterministic mode satisfies the declaration.

### 8.3 Kernel verifier

- register/shared/LDS allocation does not exceed target capability;
- barriers are reachable by all participating threads with the correct memory scope;
- a promoted tile's lifetime covers all uses;
- masked/padded work stays in bounds;
- atomic dtype/op is supported by the backend.

---

## 9. Caching and fast compilation

### 9.1 Not in the code cache key

```text
actual CSR/COO contents
Graph logical version
positions/cutoff runtime values
runtime pointer/address
exact rank-local partition contents
```

### 9.2 In the semantic/planning key

```text
canonical GraphProgram/gf.domain hash
Entity/Field/Relation schema
origin/lifecycle/enumerator kind
edge/node/reducer IR
dtype/index width/rank
shape/degree/statistics bucket
target capabilities and compiler/provider versions
determinism/effect/compile options
```

### 9.3 Layered caches

```text
semantic cache: canonical gf.domain and per-region hashes
fusion cache:   GraphProgram fusion groups and guards
planning cache: traversal/materialization/partition/storage plan
provider cache: Triton/LLVM/source artifact
runtime cache:  loaded module, launch metadata, communication plan
```

### 9.4 JIT latency policy

- small builtin reducer/edge/node regions are canonicalized and hashed independently;
- traversal uses a small number of pre-verified skeletons instead of duplicating the planner
  for every formula;
- the first call uses a heuristic quick plan and does not block on full autotuning;
- autotune winners are persisted in the background/offline, and guarded replacement does not
  change the semantic key;
- graph sizes use bounded buckets to avoid recompiling for every N/degree;
- runtime entities/edges are never unrolled; only very small static PortAxes are unrolled;
- distributed ranks share binaries; rank IDs/partition metadata are runtime operands;
- inspection artifacts are read from caches and do not trigger unrelated target compilation.

---

## 10. Inspection contract

```python
kernel = Diffusion()
u_next = kernel(graph=graph, src={"u": u}, dst={"u": u}, dt=dt)

kernel.ir("domain")       # -> IR text at each stage
kernel.ir("task")
kernel.ir("iteration")
kernel.ir("kernel")

variant = kernel.inspect()
variant.explain()             # -> schedule summary text
variant.fusion_groups()
variant.relation_plan()
variant.distribution_plan()
variant.memory_plan()
variant.artifacts()
```

`explain()` reports at least:

```text
selected snapshot/realization
fusion groups and barriers
traversal and reducer strategy
ownership/halo/partial-combine plan
working-set and memory-space estimates
compile/cache/build/transfer/consume timings
fallbacks, guards and rejected candidates
```

---

## 11. M0 minimal implementation slice

M0 does not implement the full task/iter/kernel dialects, but the Domain IR must round-trip the
following:

1. EntitySets, Field schema/value;
2. External+Frozen CSR Relations;
3. Procedural+Rebuildable Radius schema;
4. the origin/lifecycle/realization interface;
5. edge/optional node regions of binary `gf.apply`;
6. the builtin sum Reducer;
7. read/write/reduce Effects and snapshot versions;
8. the ResolvedRelation interface;
9. straight-line GraphProgram SSA;
10. target-independent canonical hashes and source locations.

M1 adds CSR/radius baseline lowering; M4 adds IndexDomain/IndexedComplex/EndpointMap,
AffineRelation, PortGather, HyperRelation routes, and auto-fusion; M6/M7 implement
PagedTileStream/storage tasks and distributed task lowering.

---

## 12. Historical prototype issues and current resolutions

1. whether `gf.apply` should use a multi-region op, or whether reducer/node as symbol calls
   would be better for region hashing;
2. whether HyperRelation fixed/ragged endpoints should be unified into one type or kept
   separate to simplify the verifier;
3. the maintainability and graph-break behavior of the Tiga lazy Field and Torch
   Tensor-subclass bridge;
4. how the cost models of Domain-level fusion and target-level tile fusion divide their work;
5. which distributed ownership/partition attributes enter the IR and which belong only to the
   deployment plan;
6. the trust/verification/debug contract for custom reducer algebra properties;
7. how much of the upstream `sparse_tensor` coordinate hierarchy can be reused, and when to
   keep dedicated relation ops;
8. the minimal linking boundary of codegen skeleton + generated region in Triton/LLVM
   providers.

These issues have been settled by the M0 round-trip, the vertical slice, and the composition
prototype: apply/reducer keep regions to support structural analysis and hashing; fixed/ragged
endpoints share the relation contract while their physical instances are separate; Torch is
only an optional adapter; Domain fusion and tile fusion belong to the semantic/physical layers
respectively; ownership enters the IR while transport placement stays in the deployment plan;
reducer properties must be structurally proven or fail-closed; the generic sparse hierarchy can
be reused without flattening generated relations; TTIR is the vendor provider boundary. The
Python API does not expose target-specific schedules.
