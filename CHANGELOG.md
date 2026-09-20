# Changelog

All notable changes to Tiga are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-09-20

Initial alpha release. Listed capabilities have
path-specific limits documented in `docs/roadmap.md`.

### Added

- Bilingual typed API contracts and executable examples, full-page language/link
  checks, installation/input troubleshooting and GitHub issue templates.
- `tg.visualize.gaussians`; `splats` remains a compatibility alias.
- Local CPython 3.11/3.12 Linux wheels and isolated no-Torch CPU execution checks.
- `tg.stencil.von_neumann()` / `moore()` neighborhood macros and bilingual examples.
- CUDA Euclidean radius hash-directory lowering with exact distance filtering;
  periodic geometry retains its separate dense directory.
- Module-level API navigation, Read the Docs configuration and bundled dependency notices.

### Fixed

- Ordinary CUDA results and generated index snapshots retain independent storage,
  including detached, DLPack and storage aliases across subsequent calls.
- Inference tensors invalidate data-dependent caches without accessing a missing
  mutation counter; editable compiler discovery matches the current Python ABI.
- Release upload requires CPU/MLIR/docs and installed-wheel GPU gates, exact
  source/tag/archive metadata, and wheels rebuilt from the source archive.
- Graph transpose across Torch-backed CSR and dynamic relation snapshots.
- Invalid video frame rates, non-integer symbolic bounds and fractional/bool
  legacy offload budgets now fail before work begins.
- Gradient comparisons cover every parameter in deeper edge-NN networks.
- Documentation distinguishes GIF buffering, fixed-topology gradients, native
  prepared launches and historical benchmark timing boundaries.

### Compiler and runtime capabilities

- MessagePassing UDF compiler: declare relations with `Graph`, per-edge and
  per-node logic with `MessagePassing`. Supported paths lower to CPU (LLVM)
  or NVIDIA CUDA (TTIR → vendor Triton). Native and supported edge-NN paths
  derive VJPs; some Torch CSR backwards instead replay fixed-topology semantics.
- Relations: static CSR, dense, triangular, `cu_seqlens`, `Graph.cat`,
  stencil, generated Euclidean radius, ranked kNN (FP32, k ≤ 64), and
  disk-paged CSR.
- Reducers: `sum`, `mean`, `prod`, `online_softmax`, and custom
  associative+commutative subclasses.
- Memory hierarchy: budget-driven `tg.runtime.auto_offload`, paged CSR with a
  prefetch pipeline, disk-resident fields, and paged forward + backward.
- Visualization: `tg.visualize` heatmaps with colormaps, particles, Delaunay
  and mesh rendering, camera control, OBJ/PLY I/O, GIF/MP4 video, EWA Gaussian
  splatting, and density-volume ray marching.
- Default Torch Tensor inputs/outputs and Torch autograd, without user-side wrappers.
- Detailed CSR-to-Iter IR walkthrough, numerical example, diagram and field reference.
- MPI and CUDA halo exchange with one public communication-then-compute policy.

### Compiler correctness

- CUDA weighted CSR outputs use the destination count, including bipartite graphs.
- Torch scalar CSR and weighted-sum compiled forwards retain a Torch autograd
  connection via semantic replay; this backward is not a generated TTIR kernel.
- Gradient-bearing sparse wrappers are not reused across freed autograd tapes;
  live output views are not overwritten by cached CSR launches.
- Non-power-of-two weighted feature widths select the sparse provider instead
  of emitting an invalid direct TTIR tile.

### Distribution

- PyPI package `tiga-lang`; Python import `tiga` (conventionally `as tg`).
- Torch is installed separately, not pulled in by the default installation;
  `[torch]` is an explicit opt-in extra. Native execution works without Torch.
  Explicit `tg.tensor()` still creates the advanced native Tensor type.
- First-release wheels target `manylinux_2_38_x86_64` (glibc ≥ 2.38),
  CPython 3.11/3.12. Other Python/OS targets are outside this release.
- Optional adapter versions: Torch 2.11.x; Triton 3.6.x for CUDA.
