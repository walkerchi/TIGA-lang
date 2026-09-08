# Tiga Scheduling Abstractions Survey: From Tensor Axes to Multi-Space Mappings

> Non-normative research note. Adopted scheduling contracts take effect only in
> `PROJECT.md` at the repository root.

Date: 2026-08-06

## 1. Conclusion

Using the tensor/iteration axis as the scheduling entry point works, and most
local optimizations for regular tensors, stencils, FFT stages, and contractions
should be expressed that way. But "place every axis on some hardware or storage
level" is not a complete scheduling abstraction for Tiga.

The reason is that a high-performance implementation contains six kinds of
decisions that are interrelated but semantically distinct:

```text
Schedule
  = IterationTransform   how the iteration domain is split/fused/reordered/segmented
  + AccessLayout         how logical coordinates access and lay out each tensor/field
  + ExecutionMapping     which device/block/warp/lane/vector executes each iteration point
  + StoragePlan          which version of which region/tile resides in which memory space
  + PipelinePlan         when producer/copy/compute/store run, and how they synchronize and reuse buffers
  + TaskPlacement        how partitions, halos, collectives, and cross-device tasks are placed
```

These are not six sets of annotations users must hand-write; they are the
compiler's internal decision spaces and the structure of `explain()` output. In
M0–M2 the user input is coarse MessagePassing; the compiler lowers it into an
internal IR that still preserves iteration/relation/reduction structure, and
then constructs these mappings.

If all six are collapsed into a single `axis -> level` table, simple regular
GEMM examples look beautiful, but CSR, online softmax, asynchronous copies,
multiple tensor layouts over one iteration domain, or distributed halos produce
ambiguity. A better unification is to keep several typed spaces and describe
the mappings between them explicitly.

## 2. Why a Single Axis Hierarchy Is Not Enough

### 2.1 Iteration axes are not tensor axes

The logical iteration domain of matrix multiplication is `(m, n, k)`, but the
accesses of A, B, and C are `(m, k)`, `(k, n)`, and `(m, n)` respectively. An
iteration axis may not exist in a given tensor, and it may reach multiple
tensor axes through affine, gather, or relation maps. This is why MLIR Linalg
separates `iterator_types` from each operand's `indexing_maps` instead of
binding threads directly to tensor dimensions.

Tiga should likewise distinguish:

- `IterAxis`: the logical coordinates of a compute instance;
- `AccessMap`: from iteration coordinates to Field/Relation coordinates;
- `Layout`: from Field coordinates to physical addresses, lanes, or fragment
  coordinates.

### 2.2 Sparse/ragged is not a rectangular axis

The neighbor axis of CSR depends on the parent coordinate:

```text
dst in [0, N)
neighbor in [rowptr[dst], rowptr[dst + 1])
```

Its extent changes with `dst`. A radius relation may even need to generate this
set at runtime. A plain `tensor<N, max_degree>` axis can only express this
through padding/masking, which hides the effective workload and degree skew.

Hence the need for `segmented/dependent/generated` axes and for a mapping from
logical dimensions to physical coordinate levels. TACO and MLIR SparseTensor's
dense/compressed/singleton levels, Finch's looplets, and CoRa's ragged
dimensions all show that sparse structure is a coordinate hierarchy, not just a
dense tensor with different strides.

### 2.3 Execution mapping is not memory placement

`dst_outer -> block` describes which program instance executes;
`Q_tile -> shared` describes where data resides; `feature_inner -> lane` may
simultaneously determine cooperative loading and the register fragment layout.
These constraints are related, but they are not the same relation.

In particular, a tile can start in HBM, be asynchronously copied to shared
memory, and then be read into registers by each lane under a different fragment
layout. It is not uniquely "placed on the shared axis".

### 2.4 A pipeline is not a spatial axis mapping

Double buffering additionally has to describe:

- which producer/copy forms a stage with which consumer/compute;
- whether the copy of `k+1` can overlap the compute of `k`;
- buffer counts, phases, barriers/tokens, and reuse conditions;
- fill/drain, tail predication, and capacity constraints.

TileLang's `Pipelined` and MLIR NVGPU's async-copy token/group/wait all treat
these as timing and dependence, not layout. FlashAttention's performance
likewise comes from the combination of IO-aware tiling, online reducers, and
data-movement timing, not merely from choosing a thread tile.

### 2.5 Storage and machine topology are not always a tree

Register/shared/HBM approximate a hierarchy inside one kernel; HBM, CPU RAM,
NVMe, peer HBM, and remote RAM form a graph with multiple transfer paths,
different engines, concurrency capabilities, and coherence scopes. A read-only
Region may also have several replicas at once. Long-term storage placement
should therefore map onto a `MemoryTopology`, not just an integer level.

### 2.6 Distributed tensor sharding is powerful, but still not a complete distributed schedule

GSPMD and MLIR Shard prove that `tensor axis -> device mesh axis` is an
excellent sharding annotation: it unifies data/model/spatial parallelism and
lets the compiler propagate shards and insert collectives. Tiga should
adopt this layer directly.

But irregular relations additionally need graph/mesh partitioning, owned/ghost
regions, halo packing, load balancing, and migration; communication/computation
overlap needs a task/event DAG. Sharding is therefore an important subset of
`TaskPlacement`, not the complete runtime schedule.

## 3. Concrete Lessons from Related Systems

| System | Core abstraction | Lesson for Tiga | What it does not cover |
|---|---|---|---|
| Halide | algorithm/schedule separation; split/reorder/compute_at/store_at | beyond axis transforms, producer-consumer placement and storage lifetime are needed | irregular coordinates, distributed |
| TVM TensorIR | blocks, loops, buffer regions, cache, tensorize | schedules act on iteration/blocks; accessed regions are analyzed separately | relation provenance, dynamic graph runtime |
| MLIR Linalg/Transform | iterator types + operand indexing maps; standalone transform IR | separate iteration from access; expert and automatic scheduling share one transform mechanism | sparse/dynamic relations need extensions |
| CuTe | hierarchical shape/stride and layout composition/division | map data and thread fragments with composable layouts instead of enumerating layout names | mainly dense, single-kernel, NVIDIA |
| TileLang | tile instructions, fragment layouts, explicit memory scopes, software pipelines | `gf.kernel` needs tile, copy, pipeline, and reducer visible together | Domain/Relation/Task are not its goal |
| TACO / MLIR SparseTensor | per-level formats, coordinate hierarchies, coiteration | separate logical dimensions from storage levels; reuse mature sparse theory | geometric builders, task runtime |
| Finch | looplets and stepwise-lowered structured iteration | generated/run/sequence/control-flow structure must not be flattened too early | GPU mapping/pipelines still need a lower IR |
| CoRa | ragged dimensions and dimension graphs, minimal padding | dependent extents are first-class information; schedules must support ragged fusion | general sparse formats and distributed |
| GraphIt | graph iteration space; direction, segment, parallel, layout schedules | traversal direction and degree segmentation are first-class decisions beyond axis transforms | continuous tensor tiles, memory pipelines |
| DaCe | dataflow, memlets, storage locations, and stateful graphs | cross-kernel data movement and lifetimes need data/task flow | not a replacement for relation/domain IR |
| GSPMD / MLIR Shard | sharding tensor dimensions onto device mesh axes | regular distributed tensors use axis sharding and propagation | irregular partitions, halo overlap |

Primary references:

- [MLIR Linalg dialect](https://mlir.llvm.org/docs/Dialects/Linalg/)
- [MLIR Transform tutorial](https://mlir.llvm.org/docs/Tutorials/transform/)
- [TVM TensorIR](https://tvm.apache.org/docs/deep_dive/tensor_ir/index.html)
- [CuTe Layout Algebra](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html)
- [TileLang instructions](https://tilelang.com/programming_guides/instructions.html) and
  [software pipeline](https://www.tilelang.com/programming_guides/software_pipeline.html)
- [MLIR SparseTensor dialect](https://mlir.llvm.org/docs/Dialects/SparseTensorOps/)
- [Finch paper](https://arxiv.org/abs/2404.16730)
- [CoRa paper](https://proceedings.mlsys.org/paper_files/paper/2022/file/afe8a4577080504b8bec07bbe4b2b9cc-Paper.pdf)
- [GraphIt paper](https://arxiv.org/abs/1805.00923)
- [DaCe SDFG paper](https://arxiv.org/abs/1902.10345)
- [GSPMD paper](https://arxiv.org/abs/2105.04663) and
  [MLIR Shard dialect](https://mlir.llvm.org/docs/Dialects/Shard/)
- [MLIR GPU dialect](https://mlir.llvm.org/docs/Dialects/GPU/) and
  [NVGPU dialect](https://mlir.llvm.org/docs/Dialects/NVGPU/)
- [FlashAttention paper](https://arxiv.org/abs/2205.14135)

## 4. Proposed Unified Model

### 4.0 Coarse MessagePassing first, fine-grained language later

In M0–M2 users write only coarse `MessagePassing`, where the edge/optional-node
regions allow tensor and scalar expressions, but node/edge/neighbor traversal is
not exposed through public loop syntax:

```python
class Diffusion(gf.MessagePassing):
    reducer = gf.sum()

    def edge(self, src, dst, edge):
        return edge.weight * (src.u - dst.u)
```

The compiler lowers Graph/Relation, Message, Reducer, and Effect into an
internal fine-grained SSA/iteration IR, and the same internal IR is then
lowered per target:

```text
                         ┌→ CPU: scf/affine loops + vector + parallel runtime
gf.domain → gf.iter ─────┤
                         └→ GPU: tile/map + working-set promotion + pipeline
                                  → gf.kernel → CUDA/HIP/PPU provider
```

The GPU path derives register/shared/LDS usage from reuse/liveness, cooperative
access, tile footprints, layout conversion, memory capacity, and occupancy; it
then inserts cooperative copies, barriers, double buffers, and async pipelines.
The CPU path does not emulate shared memory; the same logical temporary lowers
to SSA/vector values, stack allocations, or cache-blocked loops.

"GPU block-tensor/dense lowering" means turning the feature/message/reducer
computation over valid relation items into regular tiles, vectors, dot/MMA, and
reductions; it does not require densifying a sparse graph into an `N×N`
adjacency. The CSR coordinate stream can stay sparse, while inside a degree
bucket the cost model may choose a small amount of padding/masking, trading
regularity for higher hardware utilization.

Here "fine-grained" is first a property of the compiler IR, not a public
language commitment. There is no public `gf.Schedule` in M0–M2: only after the
naive Static/Dynamic Graph backends and at least one hardware optimization are
complete is user syntax reverse-engineered from validated needs such as
dependent axes, work tiles, partial reducers, and materialize/reuse.

### 4.1 IterationDomain: rectangular, dependent, and generated axes

Axes must have stable names and types; they cannot be referenced by loop number
alone:

```text
axis kind:
  parallel       independent spatial axis
  reduction      reduction axis with reducer algebra
  sequential     has a loop-carried dependency
  scan           prefix axis with ordering/associativity constraints
  segmented      extent depends on the parent coordinate, e.g. a CSR row
  generated      produced dynamically by a relation builder/iterator
  stage          algorithmic stage such as FFT/time integration; not an ordinary data dimension
```

An `IterationDomain` is a DAG/coordinate hierarchy of axes and their
dependencies, not an always-rectangular shape tuple. Regular subdomains lower
directly to `linalg/affine/scf`; structured sparse subdomains lower to
`sparse_tensor` where possible; Tiga keeps only what upstream struggles
to express: relation provenance, generation, and traversal policy.

### 4.2 AccessMap and Layout: composable mappings

Treat layout as a function, not an enumeration:

```text
iteration coordinate --AccessMap--> logical field coordinate
logical field coordinate --DataLayout--> physical address
logical tile coordinate --ThreadLayout--> (lane, lane-local coordinate)
```

The minimal combinators are `compose`, `product`, `divide/tile`, `permute`,
`broadcast`, `pad`, `swizzle`, and `vectorize`. M0/M1 need not implement the
full CuTe type algebra, but the IR must not be locked into the two strings
`row_major/col_major`.

Sparse layouts cannot be forced into stride algebra; they are expressed by
dimension-to-level maps, level properties, and position/coordinate buffers.
Dense layouts and sparse coordinate layouts meet at the `AccessMap` rather than
being forced into a single underlying representation.

### 4.3 ExecutionMapping: mapping onto logical machine axes

Use a target-neutral machine vocabulary:

```text
cluster / device / program / workgroup / subgroup / lane / vector / sequential
```

Target capabilities then interpret `subgroup` as an NVIDIA warp, an AMD wave,
or the corresponding PPU entity. A mapping may co-map several axes, split one
axis hierarchically and map it onto multiple machine-axis levels, and carry
predication, replication, and ownership constraints.

Do not hard-code `warp_size=32` in a high-level schedule; schedule parameters
can be target-dependent symbols that become constants only after
specialization.

### 4.4 StoragePlan: Regions/Instances, not tensor-axis labels

StoragePlan chooses:

- materialize, cache, recompute, or stream;
- in which memory space a Region tile establishes a PhysicalInstance;
- the instance's layout, capacity, alignment, version, lifetime, and ownership;
- where the producer is `compute_at` and where values are `store_at`;
- spill/evict/writeback/replicate constraints.

A user may write `.cache(q, "workgroup", at=k_outer)`, but its semantics should
expand into an instance, copies, and a lifetime — not permanently attach a
scope attribute to some axis of q.

### 4.5 PipelinePlan: explicit stages, buffers, and events

PipelinePlan acts on executable statements/dataflow edges:

```text
copy(K_tile): HBM -> shared       stage 0, async
copy(V_tile): HBM -> shared       stage 0, async
score + online_reduce             stage 1
output_accumulate                 stage 2
store                             epilogue
```

It records `num_stages`, buffer rotation, prefetch distance, async tokens,
barrier scopes, and resource budgets. The generic layer expresses only
dependence; the backend chooses `cp.async/TMA`, the AMD/PPU equivalents, or a
synchronous copy fallback.

### 4.6 TaskPlacement: mesh sharding and relation partition coexist

Regular tensors use:

```text
tensor axis -> device mesh axis -> inferred collective
```

Relations use:

```text
entity/relation partition -> owned/ghost regions -> halo/update/migration tasks
```

Both ultimately generate `gf.task` region privileges, copies/collectives, and
an Event DAG, so the pipeline planner can overlap interior compute with halo
transfers.

## 5. Where This Lands in MLIR

Creating one permanent `gf.schedule` mega-dialect that carries everything is
not recommended. A two-layer representation is better:

1. **Transform program**: mostly generated by the compiler/autotuner, in the
   style of the MLIR Transform dialect, matching payload IR through stable
   handles/names; Tiga adds only the extension ops needed for
   relation/segment/storage/pipeline. Experts can supply external overrides,
   but ordinary users never touch it.
2. **Scheduled payload IR**: after transforms are applied, the results live
   explicitly in `gf.iter`, `linalg`, `gf.kernel`, `gf.storage`, and `gf.task`;
   codegen does not depend on hidden Python schedule objects.

Conceptual IR:

```mlir
gf.iter.domain @attn {
  %dst  = gf.iter.axis parallel [0, %num_nodes)
  %nbr  = gf.iter.axis segmented parent(%dst) offsets(%rowptr)
  %feat = gf.iter.axis parallel [0, %feature_width)
  gf.iter.reduce %nbr
      reducer = #gf.reducer<online_softmax>
}

// Separate transform program, schematic syntax.
transform.sequence failures(propagate) {
  %op = transform.gf.match_relation @attn
  %dst_o, %dst_i = transform.gf.tile_axis %op["dst"] by [128]
  transform.gf.bind %dst_o to "program"
  transform.gf.bucket_axis %op["nbr"] by #gf.degree_classes
  transform.gf.cache %op["K", "V"] in "workgroup" at %dst_i
  transform.gf.pipeline %op along "nbr" stages 3
}
```

Proposed passes:

```text
gf-normalize-iteration-domain
gf-lower-relations-to-coordinate-hierarchy
gf-apply-schedule-transforms
gf-infer-access-and-layout
gf-map-execution
gf-plan-physical-instances
gf-form-software-pipeline
gf-lower-sharding-and-halo
gf-verify-schedule
gf-report-schedule
```

`gf-verify-schedule` checks at least reduction/scan algebra, races, ragged
bounds, layout injectivity, barrier convergence, instance capacity, async
buffer lifetimes, and distributed region versions.

## 6. User API Recommendation for M0–M2

As stated in §4.0, ordinary users write only coarse MessagePassing semantics
and there is no public `gf.Schedule`. The first kernel call triggers automatic
JIT, and the compiler picks the schedule from the internal fine-grained IR. The
information available to the compiler includes named/dependent IterAxes,
AccessMaps, Effects, Relation provenance/statistics, reducer algebra,
dependencies, reuse distances, value lifetimes, and target capabilities plus
runtime profiles. Layout, placement, copies, and pipelines are compilation
results, not part of the Field definition.

Memory budgets, available devices, and permitted storage tiers cannot be
derived from compute code; they are separate deployment inputs:

```python
deployment = gf.DeploymentPolicy(
    memory_budget={"hbm": "12GiB"},
    allowed_tiers=("hbm", "pinned", "ram", "nvme"),
)
# Provided through a separate runtime/deployment context, not written into Field or physics source.
```

The compiler may internally emit Transform artifacts to dump, test, and
reproduce autotune results, but it commits to no Python syntax or stable ABI
for them. Which overrides deserve to become expert interfaces is decided after
the naive Static/Dynamic Graph backends and one significant optimization are
done.

The API must distinguish three kinds of information:

- **Semantic declaration**: Effects, Reducers, persistence, external ownership,
  determinism — anything affecting correctness or observable behavior;
- **Deployment constraint**: devices, capacities, permitted
  storage/transport — coming from the runtime environment;
- **Optimization decision**: tiles, layouts, placement, prefetch, pipelines,
  sharding — produced by the compiler/runtime by default, overridable by an
  external Transform artifact.

The API must not require every op to have identically named axes. For example,
an FFT can expose `batch/stage/butterfly/element`, a stencil can expose
`time/x/y/z/offset`, and CSR can expose `dst/neighbor/feature`. What is shared
is the axis kinds, the transforms, and the mapping protocol — not a fixed axis
list.

## 7. Online Softmax Example: Why Six Kinds of Decisions Are Needed

Attention over the neighbors of each destination:

```text
iteration:
  dst: parallel
  neighbor: segmented reduction
  feature: parallel/reduction

reducer state in registers:
  (m, l, o)

execution:
  dst tile -> program/workgroup
  neighbor chunk -> subgroup/lane + sequential stream
  feature tile -> lane-local vector/fragment

storage:
  Q accumulator -> register
  K/V tile -> workgroup/shared
  input/output -> HBM

pipeline:
  prefetch K/V chunk k+1 while computing chunk k
  rotate 2/3 shared buffers with async token/barrier

dynamic variants:
  small degree row kernel
  medium degree subgroup kernel
  large degree split-row + tuple-state combine
```

`dst/neighbor/feature -> block/warp/lane` alone cannot express the lifetime of
`(m,l,o)`, the shared instances of K/V, the phases of async copies, or the
combine of the tuple reducer after split-row. This is the minimal counterexample
showing why the mappings must be kept separate.

## 8. Phased Implementation Recommendation

### M0/M1: implement only the internal structures needed for naive lowering

- named `IterAxis` supporting `parallel/reduction/segmented/generated`;
- separation of iteration and access mappings;
- StaticGraph edge-atomic/CSR-row and CPU nested loops;
- Dynamic RadiusGraph cell-list materialize + consume;
- no public Schedule, no general working-set promotion, no pipeline language.

### M2: implement only the one optimization selected by profiling

- Static neighbor×feature/subgraph tiling, or Dynamic cell-tile/fusion/reuse;
- add only the tile, working-set, layout, and copy/pipeline passes that
  optimization needs;
- profitability guards, naive fallback, and end-to-end benchmarks.

### Post-M3 extensions

- distill reusable internal transforms from at least two validated workloads;
- then decide whether public fine-grained traversal and an expert Schedule are
  needed;
- affine stencil axes, halo tiles, time skewing;
- layout composition, swizzle, fragment mapping;
- backend async-copy lowering.

### M6/M7 extensions

- HBM/RAM/NVMe instances and pipelines;
- device mesh sharding propagation;
- relation partitioning, owned/ghost/halo;
- task/event overlap and runtime feedback.

## 9. Questions That Require Experiments

1. Can named axes + dependent extents lower CSR, stencils, and block-sparse
   attention naturally at the same time, without introducing three schedule
   APIs?
2. How far can layouts go with simple affine maps before a CuTe-style
   hierarchical algebra becomes mandatory?
3. Is degree bucketing an iteration transform, variant dispatch, or a
   combination — and which IR is most stable?
4. Which producer/consumer patterns can automatic pipeline inference cover, and
   when must expert annotation be required?
5. What capability/legality constraints does a target-neutral `subgroup` need
   on warp32, wave64, and PPU?
6. Can a canonical form plus payload hash for Transform programs keep
   Tiga optimization at millisecond scale and isolate slow vendor
   compilation behind a persistent cache?
7. Where are the crossovers for online softmax row/split-row and 1/2/3-stage
   pipelines across degree, feature width, and shared/register budgets?

The first prototype should not chase a complete schedule language. It should
first prove that one set of named/dependent axis, layout, mapping, and pipeline
IR can generate three structurally different kernels — static CSR sum, online
softmax, and an affine stencil. That validates the abstraction better than
polishing a single GEMM tile.
