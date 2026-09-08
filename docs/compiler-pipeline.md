# Compiler pipeline

Tiga compiles one Python program into different physical algorithms,
chosen from the structure of the relation rather than from a workload name.
To keep those choices correct and inspectable, lowering proceeds through four
IR stages — Domain, Iter, Kernel, and Task IR — each fixing exactly one kind
of decision before the program leaves the compiler through a stable provider
boundary.

<figure class="gf-figure gf-figure--architecture">
  <object type="image/svg+xml" data="/assets/compiler-pipeline-overview.svg" aria-label="Tiga compiler pipeline: Domain, Iter, Kernel and Task IR, then the provider handoff">
    <img src="/assets/compiler-pipeline-overview.svg" alt="Tiga compiler pipeline: Domain, Iter, Kernel and Task IR, then the provider handoff">
  </object>
  <figcaption><a href="/assets/compiler-pipeline-overview.svg">Open the full-size SVG</a>. Each stage fixes one kind of decision; the provider handoff is the only vendor-specific step.</figcaption>
</figure>

## The pipeline at a glance

| Stage | What it fixes | Who consumes it |
|---|---|---|
| Domain IR — `gf.domain` | What the program means: relations, field roles, UDF regions, reducers | Semantic optimization: canonicalization, fusion legality, automatic VJP |
| Iter IR — `gf.iter` | How to traverse: coordinate hierarchy; dense, sparse, ragged, or generated order | Iteration lowering: builder–consumer fusion, row and tile bounds |
| Kernel IR — `gf.kernel` | How to launch: launch geometry, tiles, masks, local working set | Provider translation: serialized TTIR on GPU, MLIR → LLVM on CPU |
| Task IR — `gf.task` + `gf.storage` | Where data lives and when it moves: physical instances, owned/ghost/halo sets, event DAG | Runtime planning: async transfer, halo exchange, communication overlap |

The GPU handoff is serialized TTIR, and the CPU path is a sibling lowering
through upstream MLIR to LLVM. Neither boundary leaks vendor-specific layout
encodings back into the stable IR.

Storage and task information is not a Python-side scheduling hint: cost and
dependency facts feed back into iteration and kernel choices before local
tasks are emitted.

## Inspecting a compiled program

Every compiled kernel or program doubles as an inspection handle:

```python
program(...)
program.explain()        # backend, provider, lowering, pass sequence, cache state

for stage in ("domain", "iter", "kernel", "task", "gf.kernel.ttir"):
    program.ir(stage)    # IR text at each pipeline stage

program.code("ttgir")    # generated code; also "llir" and "ptx"
```

`explain()` reports the backend, provider identity, selected lowering, pass
sequence, cache state, and optimization remarks. A stage that produced no
artifact raises a clear error — a semantic or external-library path never
pretends to have generated PTX.

A worked example with real `explain()` output lives in
[Inspecting the compilation](message-passing.md#inspecting-the-compilation).

## Details

??? info "What each dialect carries"

    - `gf.domain` preserves mathematical meaning: Entity/Field/Relation
      schema, `edge()`/`node()` UDF regions, typed reducer regions, and
      effects. `gf.tensor` keeps shape, dtype, and broadcasting explicit
      alongside map/reduce/scan/contract and VJP requests.
    - `gf.iter` makes sparse, ragged, dense, and generated iteration explicit
      — coordinate hierarchy and traversal order — without committing to one
      GPU vendor.
    - `gf.kernel` fixes launch geometry, tiles, masks, and the local working
      set while keeping vendor-specific layout encodings out of the stable
      compiler boundary.
    - `gf.storage` preserves logical Region, PhysicalInstance, memory space,
      snapshot version, async transfer, and lifetime events across register,
      shared, HBM, RAM, NVMe, and distributed tiers.
    - `gf.task` preserves partition, owned/ghost/halo semantics, and an Event
      DAG. Each local compute task lowers through the same
      `gf.iter → gf.kernel` path, so communication stays below the Python
      kernel without being forced inside a single device kernel.

??? info "Pass inventory by stage"

    | Stage | Information introduced | Representative transformations | Inspect with |
    |---|---|---|---|
    | Capture | field roles, relation provenance, UDF regions, shape/dtype guards | region verification, effect discovery, semantic hashing | `ir("domain")`, `Tensor.mlir()` |
    | Semantic optimization | reducer algebra, Tensor DAG and VJP requests | canonicalization, fusion legality, automatic VJP, checkpoint candidates | `ir("domain")`, VJP IR |
    | Iteration lowering | coordinate hierarchy, traversal order and generated/materialized choice | builder–consumer fusion, row bounds, dense/triangular tile bounds | `ir("iter")` |
    | Kernel scheduling | launch geometry, tiles, masks, local working set and reduction | degree buckets, split rows, feature tiling, pipeline legality | `ir("kernel")`, `schedules` |
    | Storage/task planning | physical instances, capacity, versions, owned/ghost sets and events | spill/recompute, async transfer, halo tasks, communication overlap | `ir("task")`, `explain()` |
    | Provider translation | provider ABI and legal target operations | Tiga kernel IR → serialized TTIR, or CPU MLIR → LLVM | `ir("gf.kernel.ttir")`, `code(...)` |

    The native passes live under `lib/Transforms/` — e.g. `LowerDomainToIter`,
    `LowerIterToKernel`, `PlanDegreeBuckets`, `PlanSplitRows`,
    `SelectKernelSchedule`, `FusionPasses`, `TensorVJP`,
    `PlanTensorCheckpoints`, and `PlanDistributedTasks`.

??? info "Why the GPU boundary is serialized TTIR"

    Tiga and a vendor Triton fork each carry their own MLIR revision.
    Serializing TTIR at the boundary prevents the two revisions from being
    linked into one process, and keeps the handoff inspectable and cacheable.
    A provider may be NVIDIA, ROCm, or a vendor Triton fork; Tiga never
    links their MLIR into its core. Downstream of the boundary, vendor Triton
    continues through TTGIR and LLVM IR to PTX, cubin, or a vendor ISA, while
    the runtime launches compiled tasks and honors Event DAG dependencies.
