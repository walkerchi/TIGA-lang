# Tiga Related Work

Date: 2026-08-06

## 1. Purpose

This survey does not attempt to catalog every HPC, graph-computing, or compiler
project. It answers the engineering questions that matter most to Tiga
right now:

1. Is a relation sufficient as the core data model for scientific computing?
2. How should physical semantics, data structures, schedules, and the runtime be
   separated?
3. How should sparse, structured, dynamic, and hierarchical relations be
   described?
4. How should traversal, layout, and reduction be chosen on different hardware?
5. Which layer should own distribution, the memory hierarchy, and overlap?
6. What is Tiga's genuinely new contribution relative to prior work?

First-pass conclusion: the relational model as the core of a simulation DSL has
strong precedent, especially Ebb and Simit. Tiga cannot treat "simulation
as relations/message passing" itself as its sole innovation; it must build on
this prior work and solve the modern engineering problems those systems did not
solve jointly.

## 2. Projects Closest to Tiga

### 2.1 Ebb

References:

- [Ebb paper](https://arxiv.org/abs/1506.07577)
- [Stanford project page](https://graphics.stanford.edu/~mdfisher/ebb.html)

Ebb is the existing work closest to Tiga's core idea. It splits the system
into three layers:

```text
simulation code
domain/data-structure libraries
CPU/GPU runtime
```

Different geometric domains are realized through a unified relational data
model. Compute code is separated from geometric data structures, and the runtime
handles the CPU/GPU mapping. The paper reports performance close to hand-written
GPU code on several simulations.

What to learn from it:

- the precise semantics of relation, key, field, group, and projection;
- how domain libraries define grids, triangle meshes, and tet meshes on top of
  the relation model;
- how the compiler derives races, locality, and reductions from field accesses;
- how relation storage is decoupled from simulation kernels;
- why the three-layer architecture can add new domains without growing the
  runtime API.

Direct consequences for Tiga:

- `EntitySet + Relation + Field` should be the M0 core, not just a
  `CSRRelation` wrapper;
- the relation schema must express key uniqueness, arity, cardinality, inverse,
  grouping, and mutability;
- domain libraries should be built on a common Relation API — stencil, mesh,
  and radius are not hard-coded frontends inside the compiler;
- field access/effect is the central input to scheduling and parallel
  correctness.

What not to copy wholesale: Ebb targeted the CPU/GPU simulation DSLs of its
time. Tiga still has to validate dynamic graph generation, modern GPU
tile/scheduling, fast Python JIT, multi-backend capability, a native
distributed task IR, and tiered storage.

### 2.2 Simit

References:

- [Simit paper and project](https://simit-lang.org/tog16)
- [Simit language](https://simit-lang.org/language)
- [Getting started](https://simit-lang.org/getting-started)

Simit represents physical systems as hypergraphs while letting users program in
a global vector/matrix/tensor language. Its assembly construct establishes the
mapping between local graph elements and global linear algebra, and the
compiler can turn some global operations back into in-place computation on the
graph, avoiding the actual construction of intermediate sparse matrices.

What to learn from it:

- vertex/edge sets and hyperedges of arbitrary arity;
- the type relation between graph fields and system vectors/matrices;
- how the assembly construct avoids indexing bugs while preserving provenance
  for the compiler;
- index expression fusion and in-place lowering;
- how local graph computation composes with global algebra.

Direct consequences for Tiga:

- plain binary message passing is not enough; the IR must reserve multi-input
  messages for `HyperRelation`;
- an explicit assembly/view is needed between local relation programs and
  global solver/library calls;
- a matrix/tensor must not become a plain tensor that loses its provenance; it
  should know which relation it was assembled from;
- calls to external libraries such as PETSc, FFT, and BLAS should still
  preserve the mapping between results and relations/entities.

### 2.3 Liszt and OP2

References:

- [Liszt paper](https://graphics.stanford.edu/hackliszt/liszt_sc2011.pdf)
- [OP2 papers](https://op-dsl.github.io/papers.html)
- [OP2 documentation](https://op-dsl.readthedocs.io/en/latest/)

Liszt targets mesh PDEs, exposing parallelism, locality, and synchronization
through restricted mesh statements, and generates cluster, SMP, and GPU code.
OP2 expresses unstructured-mesh computation with sets, maps, data, and access
descriptors, then maps it to multiple backends via source
transformation/codegen.

What to learn from them:

- the minimal set/map/dat API;
- access descriptors such as `READ/WRITE/RW/INC/MIN/MAX`;
- coloring, atomics, or partitioning strategies forced by indirect increments;
- how halos and distributed mesh partitions are generated;
- how a restricted DSL buys analyzability.

Direct consequence for Tiga: the M0 effect system needs at least `read`,
`write`, and `reduce(op)`; it cannot rely on scanning expressions to guess every
alias and cross-field effect.

## 3. Graph Computing and Message Passing Systems

### 3.1 GraphIt

Reference: [GraphIt project and paper](https://graphit-lang.org/).

The most valuable idea in GraphIt is the separation of the algorithm language
from the schedule language. The size and structure of the input graph change
the trade-offs among locality, work efficiency, and parallelism, so traversal
and layout optimizations must compose rather than hide inside a fixed executor.

Tiga should adopt:

- the schedule as an independent, serializable object;
- push/pull, edge/vertex traversal, frontiers, and direction switching;
- how graph structure statistics drive the schedule;
- legality checking of schedule transformations.

GraphIt targets graph analytics; Tiga additionally needs typed fields,
geometric relations, continuous numerical kernels, hyperrelations, time
evolution, and global solvers.

### 3.2 Gunrock

Reference: [Gunrock paper](https://arxiv.org/abs/1701.01170).

Gunrock uses a data-centric GPU abstraction over vertex/edge frontiers, showing
that irregular graph workloads cannot be reduced to static SpMV; frontier
primitives such as active set, filter, advance, and compute matter for
non-uniform iteration.

Tiga should eventually reserve a representation for active relation
subsets/frontiers, but M0 does not need a full graph analytics runtime.

### 3.3 GraphBLAS

Reference: [GraphBLAS specification](https://graphblas.org/graphblas-api-cpp/).

GraphBLAS defines generalized sparse matrix/vector operations over semirings.
It is a mature reference for the algebraic contract of a Reducer: identity,
monoid, semiring, mask, accumulator, and descriptor must be clearly
distinguished.

Tiga should not replicate the entire GraphBLAS API, but its reducer/type
verifier should borrow its rigor.

### 3.4 PyG and DGL

References:

- [PyG MessagePassing API](https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.nn.conv.MessagePassing.html)
- [DGL gspmm](https://www.dgl.ai/dgl_docs/en/1.1.x/generated/dgl.ops.gspmm.html)

PyG's `message/aggregate/update` is a good user mental model; DGL reduces
message/reduce to fused GSpMM/GSDDMM kernels, proving that common algebraic
combinations are well suited to template fusion.

The limitation is that the adjacency is usually already an explicit sparse
assignment matrix; relation provenance, geometric builders, hierarchy, and
solver/task semantics are not central. Tiga can offer PyTorch/PyG
interop, but it cannot treat the PyG execution model as its Domain IR.

## 4. Sparse and Structured Compilers

### 4.1 TACO

Reference: [TACO publications](https://tensor-compiler.org/publications.html).

TACO's key contribution is describing sparse formats as compositions of
per-dimension levels and using iteration graphs/merge lattices to generate
multi-format sparse tensor algebra. Tiga's `ExplicitRelation` physical
formats should not be hard-coded as a CSR/COO enumeration; they should borrow
levelized format capabilities: ordered, unique, compressed, singleton, dense,
locate, and so on.

### 4.2 MLIR SparseTensor dialect

Reference: [MLIR SparseTensor dialect](https://mlir.llvm.org/docs/Dialects/SparseTensorOps/).

This dialect already implements level encodings, iteration graphs, iteration
lattices, iteration spaces, and the sparse runtime bridge. Tiga should
not reimplement general sparse tensor iteration theory.

Proposed boundary:

- `gf.domain` holds Entity/Relation/geometry/effect/provenance;
- anything expressible as tensor algebra lowers to `linalg + sparse_tensor`;
- dynamic geometric builders, frontiers, and distributed relations stay in the
  Tiga dialect/runtime;
- decide the reuse scope only after experimentally validating upstream
  SparseTensor's performance and scalability on GPU irregular reductions.

### 4.3 Finch

References:

- [Finch repository](https://github.com/finch-tensor/Finch.jl)
- [Finch paper](https://arxiv.org/abs/2404.16730)

Finch handles both sparse and structured arrays, including run-length, banded,
triangular, blocks, different background values, and control flow. It uses the
high-level FinchLogic for fusion/scheduling, then lowers to FinchNotation with
more explicit control flow.

What to learn from it:

- looplets/structured coiteration;
- the format language;
- the boundary between the high-level logic IR and the low-level control IR;
- how algebraic zeros/background values eliminate work;
- compiler code dumps and debuggability.

Finch is highly valuable for the design of Tiga's `gf.iter`, especially
for avoiding the regression of structured relations back into sparse coordinate
lists.

## 5. Scientific Computing DSLs

### 5.1 Taichi

References:

- [Taichi paper](https://yuanming.taichi.graphics/publication/2019-taichi/)
- [Sparse data structures](https://docs.taichi-lang.org/docs/sparse)

Taichi separates computation from hierarchical data structures, expressing
dense, pointer, bitmasked, and other structures through composable SNodes. This
proves that high-level structural information can drive automatic sparse
traversal and memory maintenance.

What to learn from it:

- the staged Python frontend and kernel specialization;
- the hierarchical IR and offload/task splitting;
- SNode trees, activation, and sparse iteration;
- JIT caching, diagnostics, and framework interoperability.

At the same time, heed the lesson of backend feature erosion: if a high-level
sparse data structure demands heavy special runtime support from every backend,
maintenance cost is high. Tiga's core Relation capabilities must be
layered, and backends must be allowed to explicitly decline or fall back — do
not claim all targets are equivalent.

### 5.2 Devito

References: [Devito paper](https://arxiv.org/abs/1707.03776) and
[project](https://www.devitoproject.org/index.html).

Devito generates optimized stencil code from symbolic PDE/finite-difference
expressions, showing that preserving the equation, time dimension, spatial
offsets, and boundary information is more valuable than converting early into
edge lists.

Tiga does not have to take on PDE discretization, but `AffineRelation`
should be usable as a low-level target for Devito/FEniCS-style frontends while
preserving stencil/time-step metadata.

### 5.3 NVIDIA Warp

Reference: [Warp documentation](https://nvidia.github.io/warp/latest/index.html).

Warp demonstrates the experience a modern Python simulation JIT should provide:
typed kernels, CPU/GPU JIT, geometry/physics primitives, PyTorch/JAX interop,
and differentiability.

Tiga should borrow its error messages, typing experience, caching, and
interop; the difference is that Tiga tries to make the compiler
understand relations and schedules, rather than primarily having users
hand-write per-thread kernels.

## 6. Kernel DSLs and Hardware Portability

### 6.1 Triton

Reference: [Triton programming model](https://triton-lang.org/main/programming-guide/chapter-1/introduction.html).

Triton's blocked programs are efficient for regular tile computation and are a
good fit as the first NVIDIA backend and for parts of `gf.kernel` lowering. But
it is not a Tiga Domain IR, and it does not natively solve dynamic graph
construction, distributed tasks, the storage hierarchy, or all irregular sparse
schedules.

### 6.2 TileLang

Reference: [TileLang paper](https://arxiv.org/abs/2504.17577).

TileLang emphasizes composable control over tile primitives, layouts, and
pipelines. Tiga's later expert `Schedule` and `gf.kernel` should follow
this explicit-but-composable design instead of accumulating backend-specific
decorators.

### 6.3 CAKE

Reference: [CAKE: Compiler–Agent Co-Design for Frontier Kernel Evolution](https://arxiv.org/abs/2608.12629).

CAKE's core idea is not another layer of high-level operators, but letting an
agent edit a typed, hardware-explicit physical schedule: named resources, warp
roles, barriers, pipelines, and instruction forms are all visible, while
mechanical details such as addresses, phase bits, and descriptor encodings are
derived by lowering. Its verifier/cost model returns structured findings
localized to a specific resource/role/stage, and distills repeated failures
into IR primitives, verifier rules, cost calibration, and corpus regressions.
It also explicitly separates single-shape search from dispatcher portfolio
generalization; the latter must validate guard overlap/gap, tails, held-out
shapes, and fallbacks.

This directly reinforces Tiga's existing bottom-up fine-language gate:
ordinary users still write only coarse `MessagePassing`, and the compiler first
produces the machine schedule automatically. Only when multiple implemented
optimizations repeatedly need the same resource/role/barrier/pipeline decisions
are they distilled from `gf.kernel` into a schedule vocabulary that experts and
agents can transform. The debugging interface should not return just PTX or a
single latency; it should return findings carrying source locations, legality
dispositions, resource estimates, bottleneck attribution, and suggested fix
locations.

What cannot be copied matters just as much. CAKE is currently a single-device
schedule IR for NVIDIA Ampere–Blackwell, and the paper explicitly does not
validate transfer cost to non-NVIDIA targets; it does not cover Tiga's
relation provenance, generated topology, hierarchical remote storage, or
distributed task/event semantics. Tiga therefore borrows only its
"explicit but verifiable machine schedule" and its harness methodology: it does
not promote vendor instructions like `mma/tmem/TMA` into a cross-vendor
semantic IR, and it does not ask simulation users to hand-write memory
placement.

### 6.4 Kokkos

Reference: [Kokkos programming model](https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/ProgrammingModel.html).

Kokkos explicitly separates Execution Space, Execution Pattern, Execution
Policy, Memory Space, Memory Layout, and Memory Trait. It is an important
reference for Tiga's capability/machine model.

Caveat: performance portability does not mean the same launch parameters or
layout are optimal on every target. Tiga should unify the semantic
schedule vocabulary and let each target choose concrete parameters
independently.

## 7. Dynamic Neighborhoods and Particle Systems

### 7.1 LAMMPS

References:

- [Neighbor-list internals](https://docs.lammps.org/Developer_par_neigh.html)
- [Accelerator package options](https://docs.lammps.org/latest/package.html)

LAMMPS engineering experience directly supports making the topology builder and
reuse policy first-class IR:

- Verlet lists use a force cutoff plus a skin and are rebuilt after reuse for
  several timesteps;
- the neighbor list is usually one of the most memory-hungry data structures;
- CPUs often prefer half lists/Newton on; GPUs may use full lists/Newton off,
  trading redundant computation for thread safety and fewer atomics or less
  communication;
- GPUs can choose device build, CPU build, or hybrid build;
- a transposed layout is sometimes faster, but the conversion costs extra
  memory.

This means `symmetric=True` cannot by itself decide to store only half the
edges. A Tiga schedule must be able to choose between half+atomic/coloring
and full+duplicated-compute.

### 7.2 HOOMD-blue

Reference: [HOOMD neighbor lists](https://hoomd-blue.readthedocs.io/en/latest/hoomd/md/module-nlist.html).

HOOMD offers different neighbor builders such as Cell, Tree, and Stencil,
showing that the physical implementation of a radius relation must be
replaceable and scheduled according to density, cutoff ratio, box, and particle
distribution.

## 8. Distributed and Heterogeneous Runtimes

### 8.1 Legion / Realm

References:

- [Legion publications](https://legion.stanford.edu/publications/index.html)
- [Realm overview](https://legion.stanford.edu/realm/)

Legion expresses locality, independence, partitioning, and data usage through
logical regions and privileges; Realm uses a distributed event-based runtime in
which tasks, copies, and synchronization all compose asynchronously.

Tiga should adopt the logical/physical instance separation, region
privileges, mappers, and events, but the first phase should not reimplement a
full Legion. A viable path is to keep `gf.task` semantics clean enough that
reusing Realm/Legion, or implementing a smaller relation-specific runtime, can
be evaluated later.

### 8.2 StarPU

Reference: [StarPU project](https://starpu.gitlabpages.inria.fr/).

StarPU lets applications provide CPU/GPU implementations and constraints for
tasks, while the runtime manages dependencies, heterogeneous scheduling, data
replication, cluster communication, and asynchronous execution. It shows that
backend executable variants and the task/data runtime can be orthogonal.

Consequence for Tiga: an `Executable` should not be just a synchronous
callable; it should have input/output regions, target requirements, estimated
costs, and asynchronous Events.

## 9. Comparative Summary

| Project | Strongest design | Ideas Tiga should reuse | Tiga goals not yet covered |
|---|---|---|---|
| Ebb | relational simulation model | Entity/Relation/Field, domain library layering | modern multi-level IR, dynamic graphs, full distributed/storage stack |
| Simit | hypergraph + global algebra | HyperRelation, assembly provenance | dynamic topology, general kernel schedule/runtime |
| GraphIt | algorithm/schedule separation | composable schedules, structure-aware choices | continuous numerics, geometry, solvers |
| TACO/MLIR Sparse | format/iteration theory | level formats, iteration space/lattice | geometric builders, task/runtime |
| Finch | structured coiteration | structured formats, logic/control IR layering | simulation entity/effect/distributed |
| Taichi | data structure/computation separation | staged frontend, hierarchical storage | relation/global solver/distributed semantics |
| Devito | symbolic stencil compiler | affine/time structure preservation | unstructured and dynamic graphs |
| Warp | Python simulation kernel JIT | typed UX, interop, autodiff path | relation-aware automatic scheduling |
| Triton/TileLang | GPU tile kernel DSL | final kernel lowering, expert schedules | Domain/Task/Relation runtime |
| CAKE | agent-facing typed machine schedule, localized diagnostics, portfolio stage | distilling resource/role/barrier/pipeline bottom-up from production kernels; co-evolving compiler harness | relation/dynamic graphs, distributed/storage, non-NVIDIA portability |
| LAMMPS/HOOMD | dynamic neighbor engineering | builder/reuse/half-full policy | general compiler IR |
| Legion/Realm | distributed region/event runtime | privileges, instances, mappers, overlap | relation-specific kernel generation |

## 10. Concrete Revisions to PROJECT.md

After the first survey pass, the following designs should be promoted to
explicit engineering requirements:

1. M0 Relation must not have only CSR parameters; a logical schema comes first;
2. the schema adds arity, key/cardinality, functional/multi, inverse,
   mutability, and lifetime;
3. field access gains explicit `read/write/reduce` effects;
4. the IR reserves extension points for hyperedges and relation-derived
   tensor/assembly;
5. the Schedule IR follows GraphIt and is serialized separately from the Domain
   IR;
6. `ExplicitRelation` formats later connect to MLIR SparseTensor levels instead
   of building a complete sparse iteration theory in-house;
7. the radius builder must support materialize, generate-consume, and
   reuse/rebuild simultaneously;
8. a symmetric relation does not imply half storage; the schedule decides
   full/half;
9. `Executable.launch()` returns an Event from day one, so the synchronous API
   cannot lock out distributed overlap;
10. Tiga's differentiation should be stated against Ebb/Simit as the
    baseline, not as a claim of being the first relation-based simulation
    system.

## 11. Scheduling Abstractions Survey

For the comparison of Halide/TVM/MLIR Linalg/CuTe on dense axes and layouts,
TACO/Finch/CoRa on sparse/ragged coordinate hierarchies, TileLang/NVGPU on
pipelines, and GSPMD/MLIR Shard on distributed sharding, see
[`SCHEDULING_ABSTRACTIONS.md`](SCHEDULING_ABSTRACTIONS.md). Its conclusion: the
axis is the unified scheduling entry point, but iteration, access/layout,
execution, storage, pipeline, and task placement must be composable independent
mappings, not collapsed into a single `axis -> level` attribute.

For GPU graph hardware mapping, subgraph/work-tiles, neighbor×feature 2D
partitioning, cache blocking, persistent CTAs, and the implementation order for
Dynamic RadiusGraph, see
[`GPU_GRAPH_OPTIMIZATION.md`](GPU_GRAPH_OPTIMIZATION.md). That route explicitly
implements naive Static/Dynamic backends for coarse MessagePassing first, then
lets profiling select one significant optimization, and only then decides on a
fine-grained public language.
