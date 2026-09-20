# Tiga benchmarks

Benchmarks are grouped by **workload semantics**, never by provider. Triton,
Torch, vendor libraries and Tiga-generated code are peers inside a case;
they are not top-level benchmark categories.

| Directory | Scope | Current entry points |
|---|---|---|
| `sparse_compute/` | SpMV/SpMM, sparse reductions, diffusion, fusion | `weighted_aggregation`, `cpu_relation`, `diffusion_roofline`, `fusion` |
| `graph_operations/` | topology build/rebuild and generated relations | `radius_build`, `radius_pipeline`, `radius_roofline`, `knn_build` |
| `graph_algorithms/` | representative compiler probes, not NetworkX coverage | PageRank (loop/convergence), BFS (frontier), triangle counting (intersection) |
| `neural_networks/` | NN workloads expressed with generic Tiga semantics | `dense_attention`, `linear_attention`, `sparse_attention`, `online_softmax`, `dense_matmul`, `radius_edge_mlp` |
| `compiler/` | compile/JIT/cache/provider translation—not an algorithm result | `provider_gate`, `jit_latency`, `tensor_fusion`, `cpu_pointwise` |
| `memory_hierarchy/` | register/shared/HBM/RAM/NVMe placement and pipeline | pinned↔HBM DMA and RAM↔NVMe spill |
| `distributed/` | partition/halo/collective/overlap | stdlib and real MPI two-process exact halo exchange |
| `large_graphs/` | billion-edge datasets, validation contracts and memory-tier planning | `plan`, declarative `suite.json` |
| `autograd/` | generated reverse kernels vs framework and handwritten backward | `message_passing_backward`, `reducer_product_backward`, `dynamic_radius_backward` |
| `common/` | roof calibration, fairness gates, output schema, plotting | support modules |
| `kernels/` | handwritten performance oracles | never imported by `tiga` |

Run entry points as modules from the repository root so imports are independent
of the current working directory:

```bash
export PYTHONPATH="$PWD/python"

python -m benchmarks.sparse_compute.weighted_aggregation --quick
python -m benchmarks.sparse_compute.weighted_aggregation --topology lognormal --locality random --features 1
python -m benchmarks.sparse_compute.weighted_aggregation --topology exponential --locality random --features 1
python -m benchmarks.sparse_compute.cpu_relation --quick --fail-on-gate
python -m benchmarks.graph_operations.radius_roofline --quick
python -m benchmarks.graph_operations.knn_build --quick
python -m benchmarks.neural_networks.dense_attention --quick
python -m benchmarks.neural_networks.dense_attention --quick --causal --fail-on-gate
python -m benchmarks.neural_networks.linear_attention --quick --fail-on-gate
python -m benchmarks.neural_networks.sparse_attention --quick --fail-on-gate
python -m benchmarks.neural_networks.radius_edge_mlp --quick
python -m benchmarks.compiler.provider_gate --quick
python -m benchmarks.compiler.tensor_fusion --quick --torch-compile
python -m benchmarks.compiler.cpu_pointwise --quick --fail-on-gate
python -m benchmarks.autograd.message_passing_backward --quick
python -m benchmarks.autograd.dynamic_radius_backward --quick
python -m benchmarks.memory_hierarchy.transfer --quick
python -m benchmarks.distributed.halo_exchange --quick
python -m benchmarks.distributed.automatic_overlap --quick
python -m benchmarks.providers.conformance
python -m benchmarks.large_graphs.plan
python -m benchmarks.common.check_outputs
python -m benchmarks.common.plot_collections
python -m benchmarks.common.plot_diagnostics
python -m benchmarks.common.plot_cases
```

`benchmarks.compiler.provider_gate` additionally needs the built `gf-opt` and
`gf-translate`; they are auto-discovered from `build/*/bin/` or can be set
explicitly with `TIGA_OPT` / `TIGA_TRANSLATE`.

`benchmarks.neural_networks.radius_edge_mlp` additionally needs the
`warp-lang` package (`pip install warp-lang`); NVIDIA Warp is the handwritten
fused-kernel peer, never a Tiga dependency.

Every performance claim must compare identical semantics and keep its artifacts
under `output/roofline/<operation>/<case>/`. Handwritten oracle code may set a
performance target, but it is never reported as Tiga compiler output.

Machine JSON is the reproducible source of truth, but it is not the human
interface. Every measured JSON directory has an SVG visualization and a PNG
fallback. Formal roofline cases additionally receive an operation-level `summary.svg`:
matching conditions are connected across input sizes, condition changes are
faceted, provider colors are corpus-stable, and numeric markers expose
coincident points without moving their measured coordinates.

`plot_collections` also regenerates the portrait all-in-one compiler report at
`docs/assets/compiler-performance-report.{svg,png}` and its interactive Plotly
view at `docs/assets/charts/compiler-performance-report.html`. Panel filters, provider
order, and baseline are declared in `evidence_manifest.json`; rendering fails
if an exact matched bucket or provider is absent, so the README cannot silently
substitute a favorable measurement.

Static sparse generators cover fixed, bounded-uniform, discrete power-law,
continuous log-normal, and exponential degree families. Continuous cases are
seeded, rescaled to the requested mean, explicitly capped, and record
min/mean/p50/p95/p99/max, zero-degree fraction, and coefficient of variation in
every result.
