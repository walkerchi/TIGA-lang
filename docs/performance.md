# Performance methodology

Tiga accepts a performance claim only when the program semantics, dtype,
shape, index convention, cache state and timed boundary match the declared
peer. Raw samples and device metadata — not a screenshot — are the source of
truth. The interactive evidence report lives on
[benchmark results](benchmark-results.md); it is generated from
`benchmarks/evidence_manifest.json` and fails if a registered case, filter or
provider is missing. Each report identifies its measurement snapshot.

## Semantic matching

Both sides of a comparison must execute the same mathematical program:

- identical math expression, dtype, index width and determinism contract;
- provider-native CSR/COO layouts are allowed, but conversion and preprocessing
  costs are reported separately from the warm number — it is a provider-native
  execution comparison, not a same-kernel-layout one;
- differential accuracy against the declared oracle runs before any timing;
- the Tiga reference implementation is a semantic and overhead baseline
  only and never enters a fastest-backend conclusion;
- a missing or failing optional provider reports `SKIP + reason`, never an
  invented number.

## Cache states and coverage axes

Hot cases reuse a steady working set; cold cases flush caches with an explicit
large-buffer pass before execution. The two states are measured and gated
separately — a hot graph whose working set fits in L2 must not be judged
against a DRAM roof. The sparse benchmarks expose these axes:

| Axis | Values | Purpose |
|---|---|---|
| topology | regular, irregular | fixed degree vs zero/high-degree ragged rows |
| source locality | local, random | cache-friendly stencil vs random gather |
| cache regime | hot, cold | steady reuse vs explicit flush |
| degree | CLI integer | traversal crossover and reducer work |
| feature width | 1, 16, 64 by default | scalar, vectorized SpMM, working-set growth |
| device | CPU, CUDA | reference portability vs GPU provider |

Recommended nightly coverage:

```text
(regular, local,  hot) × degree {4,16,64} × feature {1,16,64,128}
(regular, random, cold) × degree {16,64} × feature {1,16,64}
(irregular, random, hot/cold) × mean-degree {16,64} × feature {1,16,64}
```

Result JSON stores the full schema, software versions and device name, never a
bare throughput number.

## Measurement rules

- Report cold capture/provider compilation separately from warm execution.
- Time a result-ready boundary, including required synchronization.
- Interleave providers within the sample loop to reduce thermal/order bias.
- Store medians, raw samples and bootstrap confidence intervals.
- Use one stable color per exact provider throughout the corpus; numeric
  markers distinguish coincident points without moving measurements.
- Connect points only when semantics and conditions match and input size is
  the varying dimension.
- Label external-library dispatch and correctness oracles; neither may be
  presented as compiler-emitted TTIR.

## Roofline definition

Each run measures hierarchical roofs on the machine under test:

- DRAM sustained bandwidth: 256 MiB tensor device-to-device copy;
- L2-sized bandwidth: source and destination working sets kept inside L2;
- FP32 compute ceiling: dense matmul with TF32 disabled;
- kernel FLOPs counted from the workload's semantic formula;
- no-reuse algorithmic bytes: every edge's source/destination fields charged
  to DRAM;
- ideal-cache bytes: index/edge data streamed once, node fields read once,
  output written once.

Hot cases use the L2 roof, cold/flush cases the DRAM roof. `roof%` is measured
against the optimistic roof for the matching level:

```text
optimistic_roof = min(measured_FP32_peak,
                      measured_L2_or_DRAM_bandwidth × FLOPs / ideal_cache_bytes)
```

`algorithmic_gbs_no_reuse` is an algorithmic traffic metric: cache reuse
reduces real DRAM traffic, so it may exceed physical bandwidth and must never
be presented as profiler-measured DRAM bytes.

The following is the contract for new measurements, not a claim that every historical file satisfies it. See [chart reproduction](benchmark-reproducibility.md) for archived inputs and provenance gaps.

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

Plots read the JSON only and never re-run benchmarks, so reports build on a
GPU-less machine. `output/` is reproducible and Git-ignored; only selected
publication assets are copied to `docs/assets/`, SVG/HTML first, PNG as
fallback.

## Fairness rules

- Build/preprocess, cold JIT, warm kernel and peak memory are reported
  separately.
- Dynamic-graph end-to-end numbers include the builder; cache reuse is never
  reported as a rebuild.
- A baseline may amortize fixed preprocessing such as CSR construction, but
  must label it.
- Every claimed case keeps raw samples, median, bootstrap confidence interval,
  provider/compiler commit, artifact hash, full device information and tuning
  budget.
- Microsecond kernels repeat in independent processes to exclude one-off clock
  or cache states.

## Acceptance gate

Generating a binary is not a performance result. Every advertised
workload × target × dtype/index × shape bucket must beat the fastest measured
equivalent implementation in the same environment:

- candidate baselines, input buckets and tuning budgets are registered before
  measurement; hard cases may not be dropped after seeing results;
- linear sparse patterns must include a vendor sparse library or
  `torch.sparse` peer — dispatching to them without extra materialization is a
  legal and preferred lowering;
- custom fusion and generated relations must include the best runnable Triton,
  TileLang or target-native handwritten kernel;
- a bucket passes only when `best_baseline_median / tiga_median >= 1.00×`
  and the bootstrap 95% confidence-interval lower bound also reaches 1.00×; a
  geomean cannot offset a failing bucket, and statistical uncertainty counts as
  not passed;
- warm consume, cold-cache consume, cold JIT, disk-cache hit and amortized
  end-to-end are judged separately and never substituted for one another;
- a provider that passes correctness but not the performance gate stays
  opt-in; the default planner must dispatch to the faster path.

## Reproduce

```bash
# Optional overrides for the development compiler test build:
export TIGA_OPT="$PWD/build/compiler/bin/gf-opt"
export TIGA_TRANSLATE="$PWD/build/compiler/bin/gf-translate"

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
report and the interactive HTML report. See
[benchmark results](benchmark-results.md) for current claims and exclusions.
