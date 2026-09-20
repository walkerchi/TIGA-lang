# Third-party distribution notices

This directory contains license texts, not vendored implementations or Git
submodules. `licenses/` is installed into `tiga/licenses/` in native wheels.
See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for the component inventory.

LLVM/MLIR is obtained as a versioned SDK, found by CMake through `MLIR_DIR` and
`LLVM_DIR`. The release baseline is pinned in
[LLVMRevision.cmake](../cmake/LLVMRevision.cmake); CI verifies the SDK archive's
SHA-256 before building. zstd may be copied into repaired Linux wheels by
auditwheel, and its license accompanies that binary.

Torch, Triton and other optional Python dependencies are installed separately;
they are not copied into Tiga's source tree or wheel. A Git submodule is useful
when a build consumes an external source checkout at a pinned commit. This
build uses an SDK and separately installed packages, so no recursive clone is
required. Keep the license copies here even when building from a source archive
without Git metadata.
