# Billion-edge and out-of-core benchmark suite

This directory defines large graphs independently of any provider. The
compiler, vendor libraries, graph frameworks and CPU/distributed baselines must
consume the same normalized relation and validation target.

`suite.json` is the source of truth. It records both source edges and physical
CSR adjacency entries because an undirected edge normally becomes two message
visits. `plan.py` estimates canonical CSR plus algorithm state and chooses the
first viable storage tier; it never downloads data.

```bash
export PYTHONPATH="$PWD/python"
python -m benchmarks.large_graphs.plan
python -m benchmarks.large_graphs.plan \
  --hbm-gib 16 --ram-gib 128 --ssd-gib 2048 \
  --json output/large_graphs/capacity_plan.json
```

The implementation sequence is deliberate:

1. Graph500 scale 22, generated locally, proves import/build, correctness and
   GPU hub scheduling without a large download.
2. Scale 26 and Friendster force RAM-resident/HBM-streamed execution on common
   workstations and expose partition imbalance.
3. ogbn-papers100M adds feature traffic and compares full-graph aggregation
   against cuSPARSE, PyG/DGL and cuGraph-compatible baselines.
4. Scale 29 forces NVMe or distributed execution and gates halo/paging overlap.

A profile is not marked runnable until its required runtime path exists.
This dataset suite is a capacity plan, not a measurement of every listed dataset.
Measured Tiga paging and distributed workloads are documented separately in
[the performance report](../../docs/experiments.md); support for those workloads
does not imply that every dataset profile here has been executed.

Required result groups are `import`, `kernel-only`, `iteration`, and
`end-to-end`. Every result records partition edge counts, hub spill counts,
bytes moved at each storage boundary, overlap percentage, raw timing samples,
and the exact dataset checksum. Graph500 reports construction throughput and
TEPS; Graphalytics cases also run the official reference-output validator.
