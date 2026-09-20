# Performance, capacity, and distributed experiments

The paper and [documentation](../docs/experiments.md) organize current evidence
around performance/memory, single-GPU capacity through 1B edges, and heterogeneous
distributed spatial-mesh partitioning. Results include negative outcomes.

- `graph_operations/run_paper_comparison.py`: bounded 36-process matrix, four
  warmups and ten samples, stored CSR plus genuinely changing radius/kNN inputs.
  kNN extends through 131,072 points. Peers include pure Torch chunked graph
  construction/aggregation and PyG, with complete dynamic edge-set and output
  checks. Every process reports measured peak Torch allocation and provenance.
- `memory_hierarchy/billion_edges.py`: existing fully checked 10M/100M/1B sweep.
  `memory_hierarchy/profile_paging.py --edges 1000000000` adds a fully checked
  1B host/CUDA profile. Its 414.81-second window is measured, not extrapolated;
  the earlier 10M diagnostics remain historical evidence.
- `distributed/paper_scaling.py --transport nccl --topology mesh`: paired two-host
  results on 3D hexahedral meshes split along a spatial plane, through 22.51M
  directed edges. Halo traffic grows with interface area, not volume.
  The public method completes communication before computation. Critical path
  is the per-sample maximum across ranks; old overlap probes are archive-only.

The exact commands and environment are in the downloadable reproduction artifact
linked from the docs and the sibling paper's `data/three-questions/REPRODUCE.md`.
Current revisions have independent protocols/source archives in
`data/comparison-v3/`, `data/distributed-spatial/` and `data/paging-1b/`; earlier ring and artificial
large-halo runs remain historical evidence, not the main distributed curve.
The figure generator is `tiga-lang-paper/tools/build_three_questions.py`.
Exploratory failed comparisons, smoke runs and allocation audits are not pooled
into formal timing curves. No new run exceeds 1B edges; the original long sweep
is preserved without remeasurement.
