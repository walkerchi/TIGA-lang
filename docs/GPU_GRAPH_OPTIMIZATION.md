# Tiga GPU Graph Hardware Optimization Survey and Experiment Plan

> Non-normative research and experiment candidates. Implementation and release
> gates are defined by `PROJECT.md` at the repository root.

Date: 2026-08-06

## 1. Conclusion

Tiga should not design a fine-grained user language in its first version.
Phase one provides only coarse-grained `MessagePassing` covering `StaticGraph`
and `DynamicGraph`, with semantically correct, complexity-sound naive CPU/GPU
lowering. Then, driven by profiling, implement one hardware optimization with
clear payoff. Only after that should the stable concepts a fine-grained
language must expose be distilled from real optimizations.

Recommended route:

```text
coarse MessagePassing semantics
  → naive StaticGraph / DynamicGraph
  → profile traversal, feature, atomic, build and memory traffic
  → subgraph/work-tile hardware optimization
  → prove end-to-end speedup and identify reusable compiler primitives
  → design fine-grained graph language from evidence
```

## 2. How to State "One Subgraph on One SM" Precisely

An ordinary CUDA/HIP kernel cannot permanently pin a thread block to a physical
SM. Blocks are placed dynamically by the hardware scheduler, and blocks cannot
depend on a fixed execution order. The cooperative boundary that can be relied
on is the CTA/thread block/workgroup: threads in the same block share shared
memory/LDS and barriers.

A portable compiler IR should therefore express:

```text
Graph/Relation in HBM
    ↓ partition/bucket/tile
WorkTile = subgraph fragment / row range / edge range / cell neighborhood
    ↓ map
CTA/workgroup
    ↓ stage selected working set
shared/LDS + registers
```

The hardware scheduler then places CTAs onto SMs/CUs. Persistent CTA workers
are worth considering only when a device-side dynamic work queue, cross-round
locality, or extreme load imbalance demands them; they are not the naive
baseline.

"Putting a subgraph into shared memory" usually does not mean copying the whole
subgraph either. Shared/LDS capacity is limited; the objects actually worth
staging are things like:

- compact local vertex IDs, row offsets, or edge tiles;
- source/destination feature tiles reused by many edges;
- positions, types, and short features of particle cells;
- per-row/per-node partial reducer state;
- layout conversion or cooperative-copy buffers.

The full graph topology and the home instances of fields usually stay in HBM.

## 3. Direct Lessons from Related Work

### 3.1 CUDA execution model

The [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)
states explicitly that block scheduling order is not guaranteed, and that
threads within a block share shared memory/L1/register-file resources.
Tiga therefore uses target-neutral `workgroup/subgroup` at the high level
and offers no `subgraph.bind(sm_id)` semantics.

### 3.2 Gunrock: hierarchical load balancing

[Gunrock](https://arxiv.org/abs/1701.01170) (surveyed in
[`RELATED_WORK.md`](RELATED_WORK.md) §3.2) shows that a graph workload must
switch between thread, warp, block, and global granularity according to size
and degree, rather than fixing "one warp per row" or "one thread per edge". The
first schedule families should include at least edge, row, split-row, and
bucket.

### 3.3 GNNAdvisor: the neighbor × feature 2D workload

[GNNAdvisor](https://www.usenix.org/system/files/osdi21-wang-yuke.pdf) drives
two-dimensional workload management from graph and model characteristics, using
coarse-grained neighbor partitioning. This shows that the performance space of
MessagePassing is not only the graph axis; feature width determines the second
parallel axis, the reduction organization, and the shared/register footprint.

### 3.4 FeatGraph: coarse sparse template + fine-grained payload

[FeatGraph](https://arxiv.org/abs/2008.11359) jointly optimizes graph traversal
and feature-dimension computation. The lesson for Tiga: first-version
coarse MessagePassing is enough to capture the sparse traversal template while
the message region supplies the tensor payload; there is no need to expose an
arbitrary fine-grained graph language first.

### 3.5 GraphCage: cache-aware subgraph blocking

[GraphCage](https://arxiv.org/abs/1904.02241) shows that ordinary dense cache
blocking cannot be ported to graphs directly; blocking overhead, subgraph
sparsity, and load balancing must be considered together. The paper reports
clear gains from cache-centric blocking over prior optimized implementations,
but applicability depends on the graph and the iteration phase. Tiga's
cost model must be allowed to reject subgraph tiling.

### 3.6 TC-GNN: sparse blocks to regular Tensor Core tiles

[TC-GNN](https://arxiv.org/abs/2112.02052) uses graph translation to turn parts
of sparse GNN computation into regular blocks that Tensor Cores can process. It
validates the "sparse topology + dense tile payload" route, but conversion,
padding, and sparsity patterns all cost something; it should not be the
first-version general lowering.

### 3.7 Dynamic radius graph: cell lists and reuse

[LAMMPS neighbor-list internals](https://docs.lammps.org/Developer_par_neigh.html)
(surveyed in [`RELATED_WORK.md`](RELATED_WORK.md) §7.1) use spatial bins,
neighbor stencils, and a Verlet skin, reusing a built neighbor list across
several timesteps. The local conclusion: the first correct optimization for
DynamicGraph is not an arbitrary dynamic-graph incremental algorithm, but
putting the builder, the reuse condition, and the MessagePassing consumer into
the same cost model.

## 4. First-Version Naive Implementation

"Naive" must mean simple optimization, not wrong complexity.

### 4.1 StaticGraph

Input: COO/CSR, node/edge Fields, coarse-grained Message/Reducer/Update.

CPU baseline:

```text
for dst:
    state = init
    for edge in CSR row(dst):
        state = combine(state, message(edge))
    out[dst] = finalize(state)
```

GPU baseline:

1. edge-centric: a batch of edges per thread, atomic reduce;
2. row-centric: one program/warp handles one or several CSR rows with a local
   reduce;
3. only simple, safe dispatch — no graph reordering, subgraph packing, or
   persistent kernels.

### 4.2 DynamicGraph

The first DynamicGraph is defined as a geometric radius graph, not arbitrary
edge mutation:

```python
graph = gf.RadiusGraph(
    positions=position,
    cutoff=cutoff,
    periodic_box=box,
)
```

The reference may use all-pairs on small inputs to define the semantics; the
performance baseline must use a uniform cell list with reasonable complexity:

```text
bin particles
  → build cell offsets/members
  → enumerate neighbor cells
  → materialize COO/CSR neighbor list
  → reuse StaticGraph MessagePassing kernels
```

The first version does no generate-consume fusion, Verlet reuse, cell-tile
shared staging, or dynamic reordering, so that build and consume costs can be
measured independently.

## 5. First-Round Hardware Optimization Candidates

### 5.1 Degree-aware hierarchical mapping

```text
small row   → thread/lane
medium row  → subgroup/warp
large row   → CTA or split-row + second reduction
```

Pros: low implementation risk; almost every non-uniform graph needs it. Cons:
it mainly fixes load balance and does not necessarily improve random feature
traffic.

Current degree-bucketing benchmark numbers are tracked in
[`benchmark-results.md`](benchmark-results.md).

### 5.2 Static subgraph/work-tile blocking

Form a tile from a range of destination rows and their source frontier:

```text
row range
  → collect/compact source IDs
  → stage reused source features into shared/LDS or rely on L2 tile
  → process local edge stream
  → reduce/store destination tile
```

The gains come from source feature reuse and more regular feature-axis
computation; the costs include partitioning, compact metadata, boundary
duplication, shared capacity, barriers, and occupancy. StaticGraph can amortize
preprocessing across many MessagePassing runs, so it is the first choice for
subgraph optimization.

### 5.3 Neighbor × feature 2D tiling

For workloads with feature widths of 16–256, tile neighbor and feature
simultaneously:

```text
CTA.x → destination/neighbor partition
CTA.y or subgroup/lane → feature tile
```

This is usually easier to implement first than "put the whole subgraph in
shared", and it directly controls coalescing, reducer register pressure, and
occupancy.

### 5.4 Dynamic cell-tile staging/fusion

One CTA handles one or more spatial cells, staging the position/short-feature
tiles of the local and neighbor cells into shared memory in batches, then
computes pairwise messages. Later options:

- materialize the neighbor list;
- generate-consume fused, without writing an edge list;
- reuse across timesteps via a skin/rebuild condition.

This scheme suits particle simulations with very short features; its cost
structure differs from StaticGraph subgraph packing.

### 5.5 Persistent CTA queue

A fixed number of CTAs pull degree buckets, row chunks, or cell tiles from a
device queue. This can improve load balancing under extreme skew and dynamic
graphs, and can reuse some worker state. The costs are queue atomics, exit
protocols, occupancy limits, and more complex cross-backend lowering. Implement
it only when ordinary overdecomposition still shows a significant tail.

### 5.6 Sparse-to-MMA translation

Only for tiles with high feature width, semiring/linear messages, and
sufficient local density. Translation, padding, and preprocessing amortization
must all be counted in end-to-end time. This is a second-round optimization,
not a general guarantee of MessagePassing.

## 6. The "Clearly Effective" Optimization Experiment to Pick

Keep two candidates and let the naive profile decide which single one gets a
complete vertical slice:

### Candidate A: StaticGraph feature aggregation

Workload: `sum(weight * x[src])`, feature width `16/32/64/128`, same topology
reused many times.

Compare:

```text
edge atomic
CSR row
degree bucket + split-row
neighbor × feature 2D tile
subgraph tile + compact source-feature cache
```

This candidate most directly validates subgraph/CTA mapping, dense payload
tiling, and preprocessing amortization.

### Candidate B: Dynamic radius interaction

Workload: 3D positions, short features, cutoff interaction, multiple timesteps.

Compare:

```text
cell-list build + materialized CSR + consume
cell tile shared staging
generate-consume fusion
Verlet skin + rebuild/reuse
```

This candidate most directly validates DynamicGraph and the joint optimization
of builder and consumer.

Selection criteria are not single-kernel peaks, but:

- the bottleneck accounts for at least 30% of end-to-end time in the naive
  profile;
- the optimization reaches `>=1.5x` end-to-end speedup on at least two input
  distributions;
- preprocessing/build/copy/padding are all timed;
- inapplicable inputs fall back automatically, with no more than 10%
  regression;
- CPU and GPU reference correctness and empty/extreme-degree cases all pass.

`1.5x` is an initial engineering gate and can be adjusted once local data
exists.

## 7. Deriving the Fine-Grained Language from Optimizations

Once the optimizations above are implemented and validated, the validated needs
determine whether the fine-grained language must expose:

```text
neighbor/feature dependent axes
work tile / degree bucket
reducer partial state
materialize vs generate-consume
reuse/rebuild condition
cooperative staging intent
```

Even then, prefer exposing semantics and legality information over `SM id`s,
concrete shared-memory byte layouts, or CUDA-only primitives. A concept not
jointly needed by two or more real optimizations stays in the internal IR and
does not enter the public API.
