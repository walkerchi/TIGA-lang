# Performance methodology

GraphForge accepts a performance claim only when the program semantics, dtype,
shape, index convention, cache state and timed boundary match the declared
peer. Raw samples and device metadata—not a screenshot—are the source of truth.

## Human view and machine evidence

--8<-- "docs/includes/performance-report.html"

The interactive report is generated from `benchmarks/evidence_manifest.json` and
the same registered case JSON as the static report. Generation fails if an
exact case, filter or provider is missing, preventing an old chart from being
silently reused after benchmark names change.

## Artifact contract

```text
output/roofline/<operation>/<case>/
  roofline.json          # raw samples, statistics, model and metadata
  roofline.svg           # scalable static roofline
  provider_latency.svg   # human comparison; stable method colors
  roofline.png           # raster fallback only
  provider_latency.png   # raster fallback only
  REPORT.md              # boundary, peer, accuracy and exclusions

output/roofline/<operation>/
  summary.svg            # matching cases connected across input size
  summary.png            # fallback
  SUMMARY.md

docs/assets/charts/
  compiler-performance-report.html  # interactive registered-case view
```

`output/` is reproducible and Git-ignored. Only selected publication assets are
copied to `docs/assets/`, where SVG/HTML is preferred and PNG is a fallback.

## Measurement rules

- Report cold capture/provider compilation separately from warm execution.
- Time a result-ready boundary, including required synchronization.
- Interleave providers within the sample loop to reduce thermal/order bias.
- Store medians, raw samples and bootstrap confidence intervals.
- Use one stable color per exact provider throughout the corpus; numeric
  markers distinguish coincident points without moving measurements.
- Connect points only when semantics and conditions match and input size is the
  varying dimension.
- Label external-library dispatch and correctness oracles; neither may be
  presented as compiler-emitted TTIR.

!!! warning "Do not combine unlike operations"

    Exact attention, linear attention and tile-pruned sparse attention have
    different work and accuracy contracts. Radius build, consume and
    build+consume are also distinct boundaries. They use separate cases and
    are never averaged into a cross-workload score.

## Reproduce

```bash
export PYTHONPATH="$PWD/python"
export GRAPHFORGE_OPT="$PWD/build/bin/gf-opt"
export GRAPHFORGE_TRANSLATE="$PWD/build/bin/gf-translate"

python -m benchmarks.compiler.provider_gate --fail-on-gate
python -m benchmarks.sparse_compute.weighted_aggregation
python -m benchmarks.graph_operations.radius_roofline
python -m benchmarks.neural_networks.dense_attention
python -m benchmarks.distributed.automatic_overlap

python -m benchmarks.common.check_outputs
python -m benchmarks.common.plot_cases
python -m benchmarks.common.plot_collections
```

The final command regenerates operation summaries, the SVG/PNG publication
report and the interactive HTML report. See the [benchmark protocol](BENCHMARKS.md)
for semantic matching, cache definitions, roof models and acceptance gates;
see [benchmark results](benchmark-results.md) for current claims and exclusions.
