# Performance

Performance claims are accepted only for a matched workload, dtype, shape,
index convention and end-to-end boundary. Every operation has an isolated
artifact directory:

```text
output/roofline/<operation>/<case>/
  roofline.json
  roofline.png
  provider_latency.png
  REPORT.md
```

## Current compiler-generated evidence

Measurements below were collected on the repository's RTX 5070 Ti environment.
Raw samples and device metadata are stored with each artifact.

![Dense attention throughput](assets/dense-attention-performance.svg)

| Workload | Shape | GraphForge | Matched peer | Result |
|---|---:|---:|---:|---:|
| Dense exact SDPA | B=1, H=16, N=4096, D=64, FP16 | 0.7720 ms | PyTorch Flash SDPA 0.8288 ms | 1.074× peer/GraphForge |
| Dense grouped-query | B=1, Hq=16, Hkv=4, N=4096, D=64, FP16 | 0.7769 ms | PyTorch Flash SDPA 0.8424 ms | 1.084× peer/GraphForge |
| GPU heatmap | 2048×2048 scalar→RGB, FP32 | 0.0996 ms | torch.compile/Inductor 0.1148 ms | 1.153× peer/GraphForge |
| Exact kNN build + weighted consume | N=8192, D=3, k=32, FP32 | 3.9072 ms | cdist/top-k/gather-multiply-sum 3.9137 ms | 1.0017× peer/GraphForge |
| Scalar CSR weighted sum, random source | 131072 rows, degree=4, FP32 | 0.0183 ms | Triton template 0.0222 ms | 0.824× GraphForge/peer latency |
| Scalar CSR weighted sum, random source | 131072 rows, degree=16, FP32 | 0.0513 ms | Triton template 0.0571 ms | 0.899× |
| Scalar CSR weighted sum, random source | 131072 rows, degree=64, FP32 | 0.1869 ms | Triton template 0.1887 ms | 0.990× |
| Scalar CSR weighted sum, regular local/hot | 131072 rows, degree=16, FP32 | 0.0186 ms | `torch.sparse.mm` 0.0310 ms | 1.67× peer/GraphForge |
| Generated radius distance sum | 3D, target degree=32 | 0.3343 ms | benchmark Triton oracle 0.3383 ms | 1.012× peer/GraphForge |

Dense attention is generated from a user-defined Cartesian message and reducer;
the core contains no attention kernel. Cold capture + MLIR + provider JIT is
reported separately from warm execution in the registered artifact.

For weighted aggregation, F=1 is a compiler-generated TTIR path. F=16/64 in
the registered mixed-width case are explicit `torch.sparse.mm` dispatches; they
include GraphForge guards/runtime overhead but are not claimed as generated
SpMM kernels.

!!! warning "Do not compare unlike operations on one roofline"

    Dense exact attention, linear attention (FLA) and sparse attention (FSA)
    have different mathematical work and byte models. They require separate
    operation directories and matched baselines. Radius build and radius
    consume are also reported separately and together.

## Reproduce

```bash
export PYTHONPATH="$PWD/python"
export GRAPHFORGE_OPT="$PWD/build/bin/gf-opt"
export GRAPHFORGE_TRANSLATE="$PWD/build/bin/gf-translate"

python -m benchmarks.compiler.provider_gate --fail-on-gate
python -m benchmarks.graph_operations.radius_roofline
python -m benchmarks.neural_networks.dense_attention
python -m benchmarks.visualization.heatmap --fail-on-gate
python -m benchmarks.common.check_outputs
```

The deeper fairness, cache, roof and confidence-interval rules are in the
[benchmark protocol](BENCHMARKS.md).
