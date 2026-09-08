# Benchmark suite

The `benchmarks/` tree is the executable evidence base behind every row in
[measured results](benchmark-results.md). It is grouped by **workload
semantics**, never by provider: Triton, Torch, vendor libraries and
Tiga-generated code are peers inside a case, not top-level categories.

## Layout

| Directory | Scope | Entry points |
|---|---|---|
| `sparse_compute/` | SpMV/SpMM, sparse reductions, diffusion, fusion | `weighted_aggregation`, `cpu_relation`, `diffusion_roofline`, `fusion` |
| `graph_operations/` | topology build/rebuild and generated relations | `radius_build`, `radius_pipeline`, `radius_roofline`, `knn_build` |
| `graph_algorithms/` | representative compiler probes, not NetworkX coverage | PageRank (loop/convergence), BFS (frontier), triangle counting (intersection) |
| `neural_networks/` | NN workloads with generic Tiga semantics | `dense_attention`, `linear_attention`, `sparse_attention`, `online_softmax`, `dense_matmul` |
| `compiler/` | compile/JIT/cache/provider translation — not an algorithm result | `provider_gate`, `jit_latency`, `tensor_fusion`, `cpu_pointwise` |
| `memory_hierarchy/` | register/shared/HBM/RAM/NVMe placement and pipeline | pinned↔HBM DMA and RAM↔NVMe spill |
| `distributed/` | partition/halo/collective/overlap | stdlib and real-MPI two-process exact halo exchange |
| `large_graphs/` | billion-edge datasets, validation contracts, memory-tier planning | `plan`, declarative `suite.json` |
| `autograd/` | generated reverse kernels vs framework and handwritten backward | `message_passing_backward`, `reducer_product_backward`, `dynamic_radius_backward` |
| `visualization/` | rendering-adjacent prep kernels | `heatmap` |
| `providers/` | provider conformance, not performance | `conformance` |
| `common/` | roof calibration, fairness gates, output schema, plotting | support modules |
| `kernels/` | handwritten performance oracles | never imported by `tiga` |

## Running cases

Entry points run as modules from the repository root so imports are
independent of the current working directory. Most accept `--quick` for a
reduced sample count and `--fail-on-gate` to turn the acceptance gate into a
non-zero exit:

```bash
python -m benchmarks.sparse_compute.weighted_aggregation --quick
python -m benchmarks.sparse_compute.weighted_aggregation --topology lognormal --locality random --features 1
python -m benchmarks.graph_operations.radius_pipeline --quick --fail-on-gate
python -m benchmarks.graph_operations.knn_build --quick --fail-on-gate
python -m benchmarks.neural_networks.dense_attention --quick --fail-on-gate
python -m benchmarks.neural_networks.dense_attention --quick --causal --fail-on-gate
python -m benchmarks.neural_networks.linear_attention --quick --fail-on-gate
python -m benchmarks.neural_networks.sparse_attention --quick --fail-on-gate
python -m benchmarks.neural_networks.radius_edge_mlp --quick
python -m benchmarks.compiler.provider_gate --quick
python -m benchmarks.autograd.message_passing_backward --quick
python -m benchmarks.autograd.dynamic_radius_backward --quick
python -m benchmarks.memory_hierarchy.transfer --quick
python -m benchmarks.distributed.halo_exchange --quick
python -m benchmarks.distributed.automatic_overlap --quick
python -m benchmarks.providers.conformance
python -m benchmarks.large_graphs.plan
python -m benchmarks.common.check_outputs
```

`benchmarks.compiler.provider_gate` additionally needs the built `gf-opt` and
`gf-translate`; they are auto-discovered from `build/*/bin/` or can be set
explicitly with `TIGA_OPT` / `TIGA_TRANSLATE`.

`benchmarks.neural_networks.radius_edge_mlp` additionally needs the
`warp-lang` package (`pip install warp-lang`); NVIDIA Warp is a handwritten
peer kernel, never a Tiga dependency.

## Artifact and fairness rules

- Every performance claim compares identical semantics and keeps its
  artifacts under `output/roofline/<operation>/<case>/` — raw samples,
  median, bootstrap confidence interval, device/software metadata, and the
  arithmetic-intensity model.
- Machine JSON is the source of truth but not the human interface: every
  measured directory gets an SVG visualization with a PNG fallback, and
  formal roofline cases get an operation-level `summary.svg`.
- Handwritten oracle code under `kernels/` may set a performance target but
  is never reported as Tiga compiler output.
- `evidence_manifest.json` declares the panel filters, provider order and
  baseline of the aggregate compiler report; rendering fails if an exact
  matched bucket or provider is absent, so a favorable measurement cannot be
  substituted silently.
- Static sparse generators cover fixed, bounded-uniform, discrete power-law,
  continuous log-normal and exponential degree families. Continuous cases
  are seeded, rescaled to the requested mean, explicitly capped, and record
  min/mean/p50/p95/p99/max, zero-degree fraction and coefficient of
  variation in every result.

The [methodology page](performance.md) defines semantic matching, cache states,
cold-JIT accounting, the acceptance gate, the common artifact layout and
reproduction commands.
