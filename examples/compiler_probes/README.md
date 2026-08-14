# Compiler probes

These are deliberately a small set of algorithms, not a GraphForge algorithm
library. Each program is first expressed with general GraphForge semantics and
kept outside compiler core. Its job is to reveal a missing reusable IR or
schedule primitive before that primitive is designed.

Current probes:

- `pagerank.py`: executable fixed-iteration PageRank with dangling-node mass.
  Its body is captured once as `gf_control.repeat`; CPU lowering uses two
  ping-pong buffers. Device-side convergence remains the next PageRank gap.

BFS and triangle counting are the next two probes. They are intentionally not
added as API sketches: BFS waits for a typed frontier/worklist contract, and
triangle counting waits for a sorted relation-intersection contract.
