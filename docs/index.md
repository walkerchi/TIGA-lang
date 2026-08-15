---
hide:
  - navigation
  - toc
---

<section class="gf-hero">
  <div class="gf-eyebrow">Relation-oriented compiler · alpha</div>
  <h1>Compile structure,<br>not operator names.</h1>
  <p class="gf-hero__lead">
    GraphForge captures graph relations, Tensor programs, message UDFs and
    reducer algebra, then chooses traversal, fusion, memory placement,
    communication overlap and provider lowering.
  </p>
  <div class="gf-actions">
    <a class="gf-button gf-button--primary" href="getting-started/">Get started →</a>
    <a class="gf-button" href="compiler-pipeline/">Inspect the compiler</a>
    <a class="gf-button" href="benchmark-results/">See measured results</a>
  </div>
  <div class="gf-install">pip install graphforge-compiler <span>· first public release pending</span></div>
</section>

<div class="gf-status-row">
  <div class="gf-stat"><strong>Domain → Iter → Kernel</strong><span>Semantic relation IR stays visible until physical scheduling.</span></div>
  <div class="gf-stat"><strong>Serialized TTIR</strong><span>Inspectable GPU handoff to the selected vendor toolchain.</span></div>
  <div class="gf-stat"><strong>Torch optional</strong><span>Native Tensor, autograd and CUDA Driver runtime in the core.</span></div>
  <div class="gf-stat"><strong>Hierarchy + halo</strong><span>Placement and distributed overlap are compiler decisions.</span></div>
</div>

!!! warning "Release status"

    GraphForge is alpha software and is not published to PyPI yet. The command
    above is the intended release interface; install from source today. CUDA
    and CPU are executable locally. ROCm/DCU, Metal and PPU require provider
    plugins and real-hardware conformance before they are called supported.

## One compiler, several relation realizations

<p class="gf-section-lead">
The core is not an attention, radius or SpMV operator library. A relation may
be materialized CSR, implicit dense/triangular structure, generated spatially,
paged from storage or partitioned across ranks. User code states the message
and reduction; retained structure lets passes choose a physical algorithm.
</p>

<div class="gf-feature-grid">
  <div class="gf-card"><div class="gf-card__label">Semantics</div><h3>Graph + Tensor UDFs</h3><p>Message regions, reducer algebra, shapes, effects and topology versions remain first-class IR.</p></div>
  <div class="gf-card"><div class="gf-card__label">Scheduling</div><h3>Dense, sparse and generated</h3><p>Tiles, masks, row splitting, degree buckets and builder–consumer fusion are selected from structure.</p></div>
  <div class="gf-card"><div class="gf-card__label">Runtime</div><h3>Memory and communication</h3><p>Physical instances, transfer lifetimes, halo tasks and events form one capacity-aware execution plan.</p></div>
</div>

## From a Python call to TTIR

<p class="gf-section-lead">
The normal program call is the lazy JIT boundary. The figure distinguishes
semantic IR, physical lowering, storage/task planning, and the serialized TTIR
handoff. CPU lowering is a sibling MLIR-to-LLVM route—TTIR is the GPU provider
boundary, not a fictional universal IR.
</p>

<figure class="gf-figure gf-figure--architecture">
  <object type="image/svg+xml" data="assets/compiler-pipeline-overview.svg" aria-label="GraphForge compiler pipeline from Python to TTIR">
    <img src="assets/compiler-pipeline-overview.svg" alt="GraphForge compiler pipeline from Python to TTIR">
  </object>
  <figcaption><a href="assets/compiler-pipeline-overview.svg">Open the full-size SVG</a> · Dialect and pass details are in the <a href="compiler-pipeline/">compiler pipeline</a>.</figcaption>
</figure>

## Measured compiler impact

<p class="gf-section-lead">
Structural wins and mature-primitive comparisons are deliberately separated. Hover
for the exact case, peer and confidence floor; switch views instead of mixing
incompatible workloads into one score.
</p>

<figure class="gf-figure gf-figure--chart">
  <iframe src="assets/charts/compiler-performance-report.html" title="Interactive GraphForge registered compiler performance" loading="lazy"></iframe>
  <noscript><img src="assets/compiler-performance-report.svg" alt="Static GraphForge registered compiler performance report"></noscript>
  <figcaption>Generated from the evidence manifest and registered JSON · <a href="assets/compiler-performance-report.svg">SVG fallback</a> · See <a href="benchmark-results/">all registered cases and exclusions</a>.</figcaption>
</figure>

## The programming surface stays small

<div class="gf-split" markdown>

```python
import graphforge as gf

class WeightedAggregation(gf.MessagePassing):
    reducer = gf.sum()

    def edge(self, src, dst, edge):
        return edge.weight * src.x

out = WeightedAggregation()(
    graph=graph,
    src={"x": x},
    edge={"weight": weight},
)
```

<div class="gf-card">
  <div class="gf-card__label">Lazy JIT + observability</div>
  <h3>The ordinary call compiles</h3>
  <p>The first compatible call captures and compiles a guarded variant. Later calls reuse the executable. No explicit <code>gf.compile(...)</code> is required.</p>

```python
print(program.explain())
print(program.ir("domain"))
print(program.ir("iter"))
print(program.ir("kernel"))
print(program.ir("gf.kernel.ttir"))
print(program.code("ptx"))
```
</div>

</div>

## Current executable coverage

| Path | Current implementation | Boundary |
|---|---|---|
| Static relations | scalar/vector fixed and bounded-ragged CSR; power-law degree scheduling | compiler-emitted TTIR or explicit library dispatch outside proven shapes |
| Dynamic/ranked relations | generated radius-cell traversal plus exact kNN ranked tile-select–consume TTIR | kNN M0 is CUDA/FP32 squared Euclidean with power-of-two k≤64; custom metrics, larger k and generated backward remain partial |
| Dense relations | Cartesian and lower-triangular traversal, grouped lanes, online reducers | generated tensor-core TTIR for registered shapes |
| Tensor + VJP | broadcast/view/reduce/scan/matmul, complex dtype, relation-aware automatic VJP | native CPU LLVM and CUDA TTIR subsets; unsupported programs fail or use the labeled oracle |
| Hierarchical memory | capacity/version planning, pinned↔HBM DMA, RAM↔NVMe spill | single-node executable |
| Distributed | owned/ghost/halo Task IR, MPI two-process path, CPU overlap, CUDA submission ordering | real 2+ GPU NCCL/RCCL performance remains unclaimed |

<div class="gf-callout">
The authoritative completion ledger is <code>PROJECT.md §15.3</code>. Public
documentation summarizes that ledger; it must not promote PARTIAL work to a
supported feature.
</div>

## Choose a path

<div class="gf-feature-grid">
  <div class="gf-card"><div class="gf-card__label">Use it</div><h3><a href="getting-started/">Install and run</a></h3><p>Build the native compiler, execute the first differentiable program and inspect a compiled variant.</p></div>
  <div class="gf-card"><div class="gf-card__label">Understand it</div><h3><a href="programming-model/">Programming model</a></h3><p>Learn Graph, MessagePassing, reducer UDFs and dynamic relation lifecycles.</p></div>
  <div class="gf-card"><div class="gf-card__label">Verify it</div><h3><a href="performance/">Reproduce performance</a></h3><p>Read the measurement contract, registered boundaries and generated artifacts.</p></div>
</div>
