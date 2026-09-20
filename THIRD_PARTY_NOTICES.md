# Binary distribution notices

Tiga itself is licensed under Apache-2.0; see LICENSE.
The Linux release toolchain incorporates LLVM/MLIR 22.1.8 code into native
tools and the compiler extension. Repaired Linux wheels also bundle zstd
1.5.5 when required by the build host's shared-library dependency graph.

| Component | Upstream license text | Included copy |
|---|---|---|
| LLVM / MLIR 22.1.8 | [Apache-2.0 with LLVM exceptions and legacy notices](https://github.com/llvm/llvm-project/blob/llvmorg-22.1.8/llvm/LICENSE.TXT) | third_party/licenses/llvm-22.1.8.txt |
| zstd 1.5.5 | [BSD-3-Clause](https://github.com/facebook/zstd/blob/v1.5.5/LICENSE) | third_party/licenses/zstd-1.5.5.txt |

Wheels include these texts under `tiga/licenses/`. Alternate toolchains and
wheel repair may introduce different dependencies: inspect every repaired
wheel and update this inventory before distribution. Torch, Triton, Warp and
the CUDA driver are not redistributed inside the Tiga wheel. Optional packages
retain their own licenses; selecting an extra installs separate distributions.
