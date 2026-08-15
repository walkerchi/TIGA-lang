# Design notes

This section contains the long-form reasoning behind GraphForge: architecture
decisions, IR sketches, scheduling research, related systems, benchmark rules,
and compiler bootstrap details.

!!! warning "Normative versus exploratory"

    These pages preserve design context and may discuss future work. The
    [Python API](api.md), [current status](roadmap.md), and the completion ledger
    in `PROJECT.md §15.3` define what is executable today. If a research note
    conflicts with those sources, it is not a supported feature.

<div class="gf-link-grid">
  <a class="gf-link-card" href="../rfcs/0001-architecture/"><span>Architecture</span><strong>RFC 0001</strong><p>The product boundary, relation model, IR levels, runtime, and initial non-goals.</p></a>
  <a class="gf-link-card" href="../IR_DESIGN/"><span>Compiler</span><strong>IR implementation notes</strong><p>Operation contracts, effects, lowering boundaries, task/storage IR, and observability.</p></a>
  <a class="gf-link-card" href="../SCHEDULING_ABSTRACTIONS/"><span>Scheduling</span><strong>From axes to machine plans</strong><p>Why iteration, execution mapping, storage placement, and pipelines remain distinct.</p></a>
  <a class="gf-link-card" href="../GPU_GRAPH_OPTIMIZATION/"><span>GPU</span><strong>Irregular graph optimization</strong><p>Degree-aware mapping, chunked tails, subgraph locality, and dynamic relation construction.</p></a>
  <a class="gf-link-card" href="../RELATED_WORK/"><span>Research</span><strong>Related systems</strong><p>What GraphForge learns from graph DSLs, sparse compilers, simulation languages, and tensor systems.</p></a>
  <a class="gf-link-card" href="../BENCHMARKS/"><span>Evidence</span><strong>Benchmark protocol</strong><p>Matched semantics, cache states, confidence gates, roofline definitions, and artifact rules.</p></a>
</div>

## Reading order

Start with the [architecture RFC](rfcs/0001-architecture.md), then use the
[IR notes](IR_DESIGN.md) as a detailed reference. Read the scheduling and GPU
optimization notes when working on passes. The benchmark protocol is required
reading before adding or changing a performance claim.
