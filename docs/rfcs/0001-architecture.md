# RFC 0001: Tiga architecture

Status: discussion draft

## Thesis

Broad message passing can be a unifying semantic interface when its relations
retain affine, butterfly, hypergraph, hierarchical and dynamic structure.  A
flat sparse `src/dst` message-passing IR must not be the universal compiler IR.
Tiga should expose relations and reductions at the user level while
preserving structured stencils, tensor contractions, collectives, scans/sorts,
external solvers and task dependencies until a profitable lowering is known.

The compiler therefore separates four concerns:

1. **Meaning**: entity sets, fields, relations, messages, reductions and state
   transitions.
2. **Structure**: affine neighborhoods, explicit sparse formats, geometric
   neighborhood builders, symmetry, conservation and topology lifetime.
3. **Schedule**: edge/node/hybrid traversal, tiling, degree bucketing, fusion,
   vectorization, precision and deterministic reduction.
4. **Placement**: partitions, physical instances, memory tiers, transfers,
   streams and events.

## User model

The friendly API can support subclassing, but it is staged and restricted
rather than ordinary dynamic Python.  `reduce` is a declared monoid; arbitrary
Python in `aggregate` is not safely parallelizable.

```python
class Diffusion(gf.MessagePassing):
    topology = gf.relation(
        src="particles", dst="particles",
        build=gf.radius(r=0.08, skin=0.01),
        hints={"symmetric": True, "dynamic": True},
    )

    def edge(self, src, dst, edge):
        return edge.weight * (src.u - dst.u)

    reducer = gf.sum(identity=0.0, deterministic="optional")

    def node(self, dst, total, dt: gf.const):
        return {"u": dst.u + dt * total}
```

Equivalent functional syntax should exist for composition.  Subclassing is
only surface syntax; methods are captured into typed expression IR at class
creation or first specialization.  The API must keep topology construction,
message algebra and scheduling separate.

Reducers declare `identity`, `combine`, associativity, commutativity, atomic
support and numerical/determinism requirements.  The node function runs after
reduction.  This distinction is required for parallel and distributed lowering.

## Relations

`Relation<Src, Dst>` is the common abstraction, with multiple representations:

- static: COO, CSR/CSC, BSR, ELL/SELL, bitmap and vendor-specific formats;
- affine: stencil/window/strided maps, retained symbolically;
- geometric: radius, kNN, cell list, Verlet list, BVH or user builder;
- implicit: generated and consumed without materializing all edges;
- distributed: owned entities, ghosts, remote references and migration rules.

Topology builders are first-class programs, not preprocessing utilities.  A
radius relation records its metric, cutoff, periodic domain, maximum occupancy,
skin distance and rebuild condition.  This allows cell-list construction to be
fused or overlapped and lets a Verlet list be reused safely.

## Multi-level IR

Use MLIR infrastructure, but define and own the domain dialects.  Owning an IR
does not require owning parsing, SSA verification, pass management and every
backend utility.

### 1. Domain IR (`gf.domain`)

Preserves entity/field/relation types, topology provenance and lifetime,
access privileges, units/shapes, geometry, symmetry, conservation laws,
reducer algebra, precision, differentiability and determinism.

### 2. Iteration IR (`gf.iter`)

Represents coordinate hierarchies and traversal: locate, merge, segment,
gather, scatter, reduce and neighborhood generation.  Sparse formats are
compositions of level properties instead of hard-coded CSR-only operations.

### 3. Task/region IR (`gf.task`)

Represents partitions, logical regions, read/write/reduce privileges, halo
plans, asynchronous copies, collectives, events and time loops.  Dependency
analysis here exposes communication/computation overlap.

### 4. Kernel/tile IR (`gf.kernel`)

Represents program instances, lane/warp/block layouts, vector packs, local
tiles, memory scopes, pipelines and barriers.  It selects edge-centric,
node-centric or hybrid schedules and degree buckets.

### 5. Target lowering

- CPU: MLIR/LLVM vector + OpenMP/thread-pool runtime.
- NVIDIA: initially Triton or generated CUDA C++; later NVVM/PTX when needed.
- Hygon DCU: generated portable HIP C++ compiled by DTK/hipcc.
- Zhenwu PPU: generated CUDA-compatible device C++ compiled by `ppu-clang`;
  never require PTX as the interchange format.
- Other accelerators: capability-described plugin with source, LLVM, SPIR-V or
  vendor compiler boundary.

Target capabilities, not vendor names in optimization passes, describe warp
width, address spaces, atomics, async copies, collectives, tensor operations,
barriers and launch constraints.  Vendor intrinsics remain optional escape
hatches below the portable subset.

## Scheduling

The compiler chooses among:

- one program per edge + atomic reduction (balanced, extra traffic/atomics);
- one program/warp/block per destination CSR row (no atomics, degree skew);
- sorted/segmented reduction;
- degree buckets with different mappings;
- symmetric-edge processing with coloring, duplicated work or atomics;
- format conversion when amortized by topology reuse;
- generate-and-consume for dynamic geometric neighbors.

Scheduling is a separate, serializable object.  Users can leave it automatic,
apply constraints, or provide an expert schedule without rewriting the physics.

## Memory and distributed execution

There is one logical data model but two physical scheduling layers:

- kernel-local: registers, local memory, shared/LDS, device global/HBM;
- runtime tiers: HBM, pinned host memory, RAM, mmap/NVMe, object storage and
  remote/distributed instances.

Registers and SSD must not pretend to have identical allocation, consistency
or lifetime semantics.  A logical `Region` can have versioned physical
`Instance`s.  Placement policies request cache, prefetch, eviction, replication
and persistence; asynchronous `copy` operations return events.

Distributed execution lowers a relation partition into owned/ghost entities and
a task DAG:

```text
exchange halo ─┬─> boundary kernel ─┐
               └─> interior kernel ─┼─> reduction/update ─> migrate/rebuild
collective/reduce ------------------┘
```

Backends may use NCCL/RCCL/vendor collectives, UCX or MPI.  The semantic IR
contains access and reduction facts; the runtime owns streams, progress and
events.  “Native distributed” therefore lives above a single device kernel.

## Fast compilation

1. Capture a restricted typed Python expression tree; avoid tracing arbitrary
   Python and avoid specializing on actual edge-index values.
2. Cache by semantic IR hash, target capability fingerprint, dtype/shape class,
   topology schema and schedule—not by process-local object identity.
3. Ship precompiled traversal/reduction templates; JIT only the small message
   and update algebra when possible.
4. Use shape/degree buckets and guarded variants rather than recompiling every
   dynamic size.
5. Persist binaries and tuning results; support AOT bundles for clusters where
   compute nodes cannot compile.
6. Keep autotuning asynchronous and optional: run a reasonable analytical
   schedule immediately, then replace it at a safe boundary.

## Repository shape

```text
python/tiga/       Python API, capture and framework adapters
include/tiga/      public C++ runtime/compiler API
lib/Dialect/             domain, iter, task and kernel MLIR dialects
lib/Transforms/          domain/sparse/distributed/schedule passes
lib/Target/              cpu, cuda, hip, ppu and plugin targets
runtime/                 regions, events, memory, communication, cache
templates/               precompiled traversal/reduction kernels
tests/                   semantics, IR, codegen and differential tests
benchmarks/              motifs and end-to-end mini-apps
```

## Milestones

1. **Semantic MVP**: static COO/CSR, sum/max reducers, CPU and NVIDIA, eager
   reference interpreter, differential tests and IR dump.
2. **Schedule proof**: edge/node/hybrid GPU schedules, format conversion and
   degree-distribution benchmark suite.
3. **Dynamic geometry**: cell-list radius graph, Verlet reuse and implicit
   generate-consume for SPH/MD.
4. **Distributed runtime**: region privileges, static partition/halo, interior
   versus boundary overlap, MPI + NCCL/RCCL abstraction.
5. **Domestic targets**: HIP/DTK CI on DCU, CUDA-source/`ppu-clang` CI on PPU,
   capability conformance tests and vendor profiler integration.
6. **Tiered storage**: out-of-core fields, asynchronous prefetch/eviction and
   checkpointing; only after the in-memory runtime is stable.

## Non-goals for the first releases

- representing every scientific algorithm as an explicit graph;
- inventing a new general-purpose Python implementation;
- promising bitwise determinism for unordered floating-point reductions;
- transparent SSD/distributed performance without explicit policy and cost;
- matching hand-tuned kernels on every degree distribution without profiling.

## Local experiment (2026-08-06)

Environment: AMD Ryzen 7 255 (8C/16T), NVIDIA RTX 5070 Ti 16 GB,
PyTorch 2.11.0+cu128 and Triton 3.6.0.  The operation is
`out[dst] += weight * (x[src] - x[dst])` on a regular directed graph.

| nodes | degree | schedule | CPU Gedge/s | GPU Gedge/s |
|---:|---:|---|---:|---:|
| 131,072 | 16 | PyTorch scatter/index_add | 0.368 | 16.926 |
| 131,072 | 16 | Triton edge atomic | — | 36.069 |
| 131,072 | 16 | CSR row reduction | 1.218 | 22.771 |
| 131,072 | 64 | PyTorch scatter/index_add | 0.204 | 10.555 |
| 131,072 | 64 | Triton edge atomic | — | 15.368 |
| 131,072 | 64 | CSR row reduction | 0.483 | 53.466 |

The implementations pass differential checks against the PyTorch reference.
This is a microbenchmark, not a general performance claim: its value is the
schedule crossover.  Low degree favors many cheap edge programs here; at higher
degree, row-local reduction amortizes program overhead and avoids atomics.

A previously unseen Triton specialization took about 295 ms from launch through
completion; the immediately repeated cached launch took about 0.09 ms.  This
motivates template reuse, persistent caching and guarded/bucketed specialization.
