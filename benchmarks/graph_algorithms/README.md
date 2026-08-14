# Graph-algorithm compiler probes

This directory will contain measured versions of exactly three representative
probes:

| Probe | Compiler question | Matched performance peer |
|---|---|---|
| PageRank | Can repeated MessagePassing remain a bounded loop with fused CSR/node work and reused buffers? | `torch.sparse.mm` for the fixed-iteration executable; cuGraph/GraphBLAS remains required for convergence mode |
| BFS | Can a dynamic frontier select push/pull traversal and balance a power-law graph? | cuGraph/GraphBLAS or a matched direction-optimizing BFS |
| Triangle counting | Can sorted CSR intersection choose merge/galloping/binary-search schedules without materializing a second-order graph? | cuGraph/GraphBLAS or a matched intersection kernel |

`pagerank.py` is the first measured entry point. It reports canonical control
IR, TTIR entry/schedule, two loop buffers, one launch per iteration, compilation
latency, matched numerical semantics, latency and a provider-independent
roofline. Python NetworkX is an optional correctness/API oracle, never the SOTA
performance denominator.
