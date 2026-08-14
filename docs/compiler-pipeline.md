# Compiler pipeline

GraphForge keeps semantic information long enough to choose different physical
algorithms for the same user program.

```text
Python AST/tensor bindings
          │ capture
          ▼
gf.domain: relation + field roles + message regions + reducer algebra
          │ legality, fusion, provenance/effect analysis
          ├──────── ownership/residency planning ────────┐
          │                                               ▼
          │                                  gf.storage + gf.task
          │                                  region/instance/event DAG
          │                                               │ local compute task
          ▼                                               ▼
gf.iter: coordinate hierarchy + ordering + generated/materialized traversal
          │ schedule selection, bounds, tiling, masks
          ▼
gf.kernel: target-independent launch, working set and local reduction
          │ provider-local translation
          ▼
serialized TTIR ── vendor Triton ── TTGIR/LLVM ── PTX/cubin
```

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
