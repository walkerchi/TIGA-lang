# Current status and roadmap

This page is a public summary. The authoritative completion ledger is
`PROJECT.md §15.3`; when a statement here conflicts with that ledger, the
ledger wins. A feature is not promoted from **partial** until correctness,
inspectable compiler artifacts and the declared performance/conformance gate
all exist.

## Support matrix

| Target | Status | Executable path | Remaining release gate |
|---|---|---|---|
| NVIDIA CUDA | local alpha | `gf.domain/tensor → gf.iter → gf.kernel → serialized TTIR → vendor Triton → PTX/cubin`; native CUDA Driver runtime | broader shape/layout/dtype and multi-device matrix |
| CPU | local alpha | `gf_tensor → Vector/SCF/MemRef → LLVM → ExecutionEngine`; fused relation range parallelism | ragged/power-law NUMA and broader dtype/layout matrix |
| AMD ROCm / Hygon DCU | provider pending | provider ABI and conformance inputs exist | vendor TTIR→hsaco plugin plus real-hardware artifacts |
| Apple Metal | provider pending | provider ABI and conformance inputs exist | legal IR→MSL/metallib plugin plus Apple hardware CI |
| PPU | provider pending | provider ABI and conformance inputs exist | vendor compiler/runtime plugin plus hardware CI |

Detection by Torch or Triton is not support. A provider must pass correctness,
artifact inspection, cold/warm compilation, runtime and matched performance on
the target hardware.

## Implemented vertical slices

<div class="gf-feature-grid">
  <div class="gf-card"><div class="gf-card__label">Compiler</div><h3>Inspectable native IR</h3><p>Tensor, relation and reducer capture; Domain→Iter→Kernel; automatic VJP; CPU LLVM and GPU TTIR translation.</p></div>
  <div class="gf-card"><div class="gf-card__label">Runtime</div><h3>Torch-independent execution</h3><p>CPU buffers/ExecutionEngine and CUDA Driver allocation, streams, events, modules, launch and pinned DMA.</p></div>
  <div class="gf-card"><div class="gf-card__label">Structure</div><h3>Static + generated relations</h3><p>CSR, dense/triangular, generated Euclidean radius, degree-aware sparse scheduling and builder–consumer fusion.</p></div>
  <div class="gf-card"><div class="gf-card__label">Differentiation</div><h3>Compiler-derived VJP</h3><p>Tensor views/broadcast/reduce/scan/matmul, complex values and relation/reducer families without user backward kernels.</p></div>
  <div class="gf-card"><div class="gf-card__label">Memory</div><h3>Physical planning</h3><p>Capacity, versions, checkpoint/spill, pinned↔HBM DMA and RAM↔NVMe execution represented in IR.</p></div>
  <div class="gf-card"><div class="gf-card__label">Distribution</div><h3>Owned / ghost / halo tasks</h3><p>Two-process MPI correctness, CPU overlap and CUDA stream dependency ordering below the user kernel.</p></div>
</div>

## Active closure items

| Ledger | Status | What exists | Required to close |
|---|---|---|---|
| K0 · exact procedural kNN | **partial — ranked M0 measured** | `gf.ranked_relation` → ranked-pairs → ranked launch; stable candidate-tile top-k, masked arbitrary k≤64, hierarchical merge, selected-edge fusion, live-coordinate rebind, three passing N/D/k gates | large k, general metric UDF, memory-budgeted spill/task plan, backward lowering |
| X0 · distributed execution | **partial** | typed halo Event DAG, MPI two-process forward/VJP, CPU overlap, NCCL rank-one binding and CUDA submission-order fixture | real 2+ GPU NCCL correctness/profiler overlap/performance and RCCL evidence |
| P0 · release engineering | **partial** | pinned LLVM build, local wheel audit, no-Torch smoke, sdist rebuild and hosted compiler CI | complete CPython/Linux/macOS release matrix, trusted publishing and first PyPI release |
| G0 · graph-algorithm probes | **partial** | fixed-iteration PageRank compiler/control path and registered performance cases | reverse structured loop/tape performance, device-side convergence, representative frontier and sorted-intersection IR probes |
| L0 · matrix-free solvers | **partial** | `LinearOperator`, multi-result `repeat/while` + CPU typed double buffers and `scf.while`, MessagePassing FEM stiffness apply, fixed/residual-driven CG with optional preconditioner | multi-state CUDA loop plan, distributed reductions, structured reverse loop, implicit adjoint VJP, and matched forward/backward artifacts |
| Vendor providers | **pending** | provider ABI, plugin entry point and fail-closed conformance command | independently distributed provider plus target-hardware artifacts for each vendor |

The exact-kNN gates time live all-pairs rebuild plus consume in one launch;
cached spatial-directory reuse is not substituted for rebuild timing.

## Engineering order

1. Extend ranked-relation lowering beyond the measured FP32/k≤64 M0
   contract, including bounded scratch/spill tasks and generated backward.
2. Extend general sparse/vector/high-degree and nonlinear fusion coverage using
   registered natural degree distributions.
3. Validate real multi-device communication/compute overlap and add topology-
   aware partition cost models.
4. Run the full release matrix and publish the first signed PyPI artifacts.
5. Add vendor providers only when their toolchain and hardware conformance can
   run continuously.
6. Use the FEM/solver probe to add bounded multi-state control flow and
   residual-guarded implicit differentiation without introducing
   workload-named kernels.

## Claim discipline

- FLA/FSA, attention, PageRank and visualization remain examples/benchmarks;
  the compiler core contains no workload-named kernel.
- Results apply only to registered shapes, dtypes, topology distributions,
  cache states and hardware.
- External dispatch is labeled as dispatch. A semantic evaluator is labeled as
  an oracle. Neither is reported as generated TTIR.
- One-GPU NCCL binding proves integration, not peer-link performance.
- Public status changes must update the ledger, tests, benchmark artifacts and
  this page together.
