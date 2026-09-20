# Reproduce the charts

**Redraw stored measurements** without a GPU, or **run new benchmarks** with the matching device, dependencies and timing scope. The steps below cover both workflows.

## Redraw without a GPU { #redraw}

```bash
python -m pip install matplotlib==3.11.1 numpy==2.4.4
python tools/verify_benchmark_evidence.py
python -m benchmarks.common.plot_docs_results --root benchmarks/evidence_snapshot/output
python tools/verify_benchmark_evidence.py --zip docs/assets/results/evidence.zip
```

## Regenerate every registered case { #all-cases}

```bash
mkdir -p output/chart-reproduction
cp -R benchmarks/evidence_snapshot/output/roofline output/chart-reproduction/
python -m benchmarks.common.plot_cases --root output/chart-reproduction/roofline
python -m benchmarks.common.plot_collections --root output/chart-reproduction/roofline --report-output output/chart-reproduction/report.png --showcase-output output/chart-reproduction/showcase.png --interactive-report-output output/chart-reproduction/report.html
```

The rendering environment is Python 3.12.3, Matplotlib 3.11.1 and NumPy 2.4.4. SVG timestamps are omitted and element IDs are fixed. Repeated renders in the same environment have stable bytes; different platform fonts can still alter layout. Inputs are verified with [SHA-256](https://en.wikipedia.org/wiki/SHA-2).

Use a fresh `output/chart-reproduction` directory so stale figures cannot be mistaken for new output. These commands write derived images, not measurements; a non-zero exit means regeneration failed.

## Figure inventory { #inventory }

| Figure | What it compares | Scope to retain |
|---|---|---|
| Sparse relations | Prepared Tiga / Torch sparse | Exact topology, F, index dtype, hot cache |
| Transformations | Four fixed compiler workloads | Different named baselines; not an aggregate |
| Primitive comparisons | Fixed manifest panels | Matching semantics and provider for each row |
| CPU relations | Tiga / SciPy and Torch | N=16k and 131k; degree 16; 16 threads |
| Edge-NN forward | Tiga / handwritten and eager peers | Forward only; N=262144 |
| Edge-NN memory | Eager intermediate sizes / fused representation | Estimated per-edge storage, not measured peak |
| Edge-NN backward | Compiled / eager autograd | Forward+backward; peak allocation is a separate metric |
| GAT | Compiled / eager autograd | Supporting GAT case; not dense SDPA |
| Attention | Dense, linear, sparse attention | Distinct workloads and peers; no average |
| Dynamic boundaries | Build, consume, reuse, fresh pipeline | Phase timing must not be relabeled end-to-end |
| Distributed overlap | Communication and interior trace | Two CPU processes, controlled delay, sample vs median |
| Provider comparison (Reference) | All providers in one CSR slice | Gather-scatter PyG path only, not all PyG operators |

Each of these 12 SVG/PNG pairs has an entry in the [input index](assets/results/chart-inputs.json), containing exact JSON paths and SHA-256 checksums. The [downloadable archive](assets/results/evidence.zip) also includes all 44 registered cases and its own `index.json`. No measurements are encoded as hand-copied constants in these chart generators.

## Run a new measurement { #measure }

Install the [benchmark dependencies](benchmark-suite.md) and use the recorded case configuration, dtype, device, provider versions, warmup, repetitions, seed and timing boundary. Read a module's `--help` before selecting parameters. The recipe below records a fresh run and does not replace the archived evidence:

```bash
PYTHONPATH=python python tools/run_benchmark.py --record output/fresh-smoke/provenance.json --module benchmarks.sparse_compute.weighted_aggregation -- --quick
```

The wrapper records the exact command, UTC times, exit status, Git revision and dirty-worktree fingerprints, package versions, CPU/platform and available GPU/driver identifiers. It never labels missing seeds or absent old revisions as known. The benchmark module still owns its output directory and random-seed behavior; record its effective arguments and preserve its output alongside the provenance file.

Keep the environment and source metadata with the samples collected in that run. A new version's performance requires a new measurement with matched peers and repeated samples. See [timing protocol](performance.md).

## Chart and evidence coverage { #audit }

The documentation generator produces the 12 figure pairs and the full matrix. The generic renderers cover every registered case (44 latency panels plus 43 roofline figures), five multi-case operation summaries, a dashboard and static/interactive reports. A case without roof calibration deliberately has no roofline chart.

Operation summaries are paginated at three conditions per image; `SUMMARY.md` links every page. Diagnostic renderers under `benchmarks/common/diagnostic_plotting.py` consume additional local JSON separately from the registered cases. Kernel-only tensor-fusion timings are separate from end-to-end timings. The evidence manifest identifies the inputs included in the report.
