# Current status and roadmap

## Supported environment { #latest-validation }

Tiga is alpha software. The supported package environment is Linux x86-64,
CPython 3.11/3.12 and glibc ≥ 2.38. The Torch adapter uses Torch 2.11.x;
CUDA compilation additionally uses Triton 3.6.x. Torch is installed separately.
See [installation](getting-started.md) and [release availability](support.md#publication).

## Support matrix { #support-matrix }

| Target | Execution path | Availability |
|---|---|---|
| NVIDIA CUDA | Domain/Tensor → Iter → Kernel → TTIR → Triton → PTX/cubin | Implemented for supported operators, shapes and dtypes |
| CPU | Tensor → Vector/SCF/MemRef → LLVM → ExecutionEngine | Native JIT, including Torch-free execution |
| AMD ROCm / Hygon DCU | Provider extension interface | No supported device implementation |
| Apple Metal | Provider extension interface | No supported device implementation |
| PPU | Provider extension interface | No supported device implementation |

GPU detection by Torch or Triton is independent of Tiga provider support.
Use [execution diagnostics](execution.md) to inspect the selected provider.

## Feature support { #feature-support }

Graph construction and graph execution have separate coverage. The following
table describes the consumer path, including its gradient behavior.

| Path | Supported behavior | Constraints |
|---|---|---|
| Native Tensor MessagePassing | Materialized CSR and default Euclidean radius; forward and VJP | Other relation realizations are rejected by this entry point |
| Native paged CSR | CPU forward/VJP and CUDA FP32 forward; [1B-edge measurement](memory.md#billion-edge-capacity) | CUDA backward unsupported; output remains resident; staging uses host memory |
| Torch interface | Ordinary Torch input/output and autograd; dense, triangular and generated relations | Fusion depends on relation, dtype and shape; some CSR backward paths replay Torch operations |
| Generated exact kNN | FP32 selection and fused consumption, k ≤ 64 | Target/shape guards; larger k, general metric UDF and generated backward use other paths or remain unsupported |
| Graph composition | `Graph.cat([Graph.triangular(n), ...])`, stencil and CSR construction | `cat` materializes CSR; discrete neighbor selection is not differentiated |
| Reducers | sum, mean, product, online softmax and supported custom algebras | Lowering depends on declared algebraic properties |
| Tensor dtypes | f16/f32/f64, i32/i64, complex64/128 and bool storage | Operator/backend coverage is narrower than storage coverage |
| EdgeNN | Captured Torch modules; guarded CUDA tile forward and VJP for inputs, positions and parameters | Supported trace operators only; gradients hold selected neighbors fixed |
| Distributed | CPU/MPI and CUDA TCP/NCCL halo exchange with reverse VJP | Fixed ownership; communication completes before computation; no GPU-speed-aware repartitioning |
| Memory hierarchy | Native-Tensor budgets, opt-in whole-Tensor LRU spill/reload, differentiable copy and snapshots | Does not manage Torch-owned storage or tile arbitrary oversized kernels |
| Tile-pruned attention | Explicitly selected approximate CUDA dense forward | Backward unsupported; approximation depends on the threshold and input |
| Visualization | `tg.visualize.gaussians`, volumes, meshes and fields | Gaussian geometry/rendering uses host work and is not differentiable |

[API recipes](api-examples.md), [memory and storage](memory.md), and
[distributed execution](memory-and-distributed.md) provide runnable entry points.
[Performance and scalability](experiments.md) records the corresponding measured workloads.

## Compiler capabilities { #implemented-vertical-slices }

The compiler retains relation structure and reducer algebra through Domain,
Iter and Kernel IR, derives supported VJPs, and lowers to CPU LLVM or GPU TTIR.
Task/Storage IR describes dependencies, placement and data movement.
[The IR walkthrough](ir-walkthrough.md) follows an executable program through these layers.

## Planned extensions { #active-closure-items }

| Area | Planned work |
|---|---|
| Generated relations | Larger-k selection, general metric lowering, budgeted scratch storage and generated backward |
| Distributed graphs | Compute-aware partitioning, partition-local topology at larger scales and failure recovery |
| Paging | Reduce host staging copies, assemble outputs directly on device and overlap page movement with execution |
| Graph algorithms | Device-side convergence, structured reverse loops and frontier/intersection traversal |
| Solvers | Multi-state CUDA loops, distributed reductions and implicit adjoint differentiation |
| Providers | Additional vendor backends with hardware correctness and performance coverage |
| Packaging | Broader platform/Python coverage and versioned public documentation |

## Development priorities { #engineering-order }

Extend operator and gradient coverage, improve large-graph staging and partitioning,
and maintain reproducible tests for each supported backend.
Each additional provider requires its own toolchain and hardware validation.

Contributor build and test instructions are in [development](development.md).
Report problems through the [support guide](support.md#bug-report).
