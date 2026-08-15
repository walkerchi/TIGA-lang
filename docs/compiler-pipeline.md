# Compiler pipeline

GraphForge keeps semantic information long enough to choose different physical
algorithms for the same user program.

<figure class="gf-figure gf-figure--architecture">
  <object type="image/svg+xml" data="../assets/compiler-pipeline-overview.svg" aria-label="GraphForge compiler architecture from Python to TTIR">
    <img src="../assets/compiler-pipeline-overview.svg" alt="GraphForge compiler architecture from Python to TTIR">
  </object>
  <figcaption><a href="../assets/compiler-pipeline-overview.svg">Open the full-size SVG</a>. TTIR is the stable GPU provider handoff; the CPU path lowers through MLIR to LLVM.</figcaption>
</figure>

## Why three GraphForge IR levels?

- `gf.domain` preserves mathematical meaning and reducer regions.
- `gf.iter` makes sparse/ragged/dense/generated iteration explicit without
  committing to one GPU vendor.
- `gf.kernel` fixes launch/tile decisions while keeping vendor-specific layout
  encodings out of the stable compiler boundary.
- `gf.storage` preserves logical Region, PhysicalInstance, memory space,
  snapshot version, async transfer and lifetime events.
- `gf.task` preserves partition, owned/ghost/halo semantics and an Event DAG;
  each local compute task is lowered through the same `gf.iter → gf.kernel`
  path. Communication remains below the Python kernel but is not incorrectly
  forced inside a single device kernel.

The serialized TTIR boundary prevents GraphForge's MLIR revision and a vendor
Triton fork's MLIR revision from being linked into one process.

## Pass ownership

| Stage | Information introduced | Representative transformations | Inspect with |
|---|---|---|---|
| Capture | field roles, relation provenance, UDF regions, shape/dtype guards | region verification, effect discovery, semantic hashing | `ir("domain")`, `Tensor.mlir()` |
| Semantic optimization | reducer algebra, Tensor DAG and VJP requests | canonicalization, fusion legality, automatic VJP, checkpoint candidates | `ir("domain")`, VJP IR |
| Iteration lowering | coordinate hierarchy, traversal order and generated/materialized choice | builder–consumer fusion, row bounds, dense/triangular tile bounds | `ir("iter")` |
| Kernel scheduling | launch geometry, tiles, masks, local working set and reduction | degree buckets, split rows, feature tiling, pipeline legality | `ir("kernel")`, `schedules` |
| Storage/task planning | physical instances, capacity, versions, owned/ghost sets and events | spill/recompute, async transfer, halo tasks, communication overlap | `ir("task")`, `explain()` |
| Provider translation | provider ABI and legal target operations | GraphForge kernel IR → serialized TTIR, or CPU MLIR → LLVM | `ir("gf.kernel.ttir")`, `code(...)` |

`gf.storage` and `gf.task` are not Python scheduling hints. Their cost and
dependency information feeds back into iteration/kernel choices before local
tasks are emitted.

## Debugging a variant

```python
program(...)
print(program.explain())

for stage in ("domain", "iter", "kernel", "gf.kernel.ttir"):
    print(stage, program.ir(stage), sep="\n")

print(program.code("ttgir"))
print(program.code("llir"))
print(program.code("ptx"))
```

`explain()` reports the backend, provider identity, selected lowering, pass
sequence, cache state and optimization remarks. Missing artifacts raise a clear
error; a semantic or external-library path never pretends to have generated
PTX.

## Compiler-only enforcement

`python/graphforge/` contains no `@triton.jit` workload kernel. Handwritten
templates are isolated in `benchmarks/kernels/` and are used only as performance
oracles. CI should retain a source scan enforcing this boundary.
