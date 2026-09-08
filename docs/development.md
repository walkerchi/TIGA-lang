# Development

## Repository layout

```text
include/graphforge/Dialect/  TableGen dialect definitions
include/graphforge/Dialect/Tensor/ canonical Tensor operations and contracts
lib/Dialect/                 verifiers and dialect implementation
lib/Transforms/              target-independent analyses and passes
lib/Target/                  provider translators
tools/                       gf-opt and gf-translate
python/tiga/           stable public façade only at package root
python/tiga/tensor/    Tensor metadata, views, capture and creation
python/tiga/autograd/  semantic reverse-mode transforms
python/tiga/graph/     logical relation plus dynamic builders
python/tiga/reducer/   reducer semantic definitions
python/tiga/message_passing/ capture, planning and execution façade
python/tiga/kernel/    compiled variants and artifact diagnostics
python/tiga/distributed/ mesh, partition and halo placement models
python/tiga/compiler/  typed capture, native binding, toolchain and JIT cache
python/tiga/interop/torch/ optional Torch oracle/provider boundary
python_bindings/            C++ OpBuilder and in-process ExecutionEngine binding
python/tiga/runtime/   native runtime binding
lib/Runtime/                 Torch-independent runtime C ABI
examples/                    user programs
benchmarks/                  workloads, protocol and reports
benchmarks/kernels/          handwritten oracles; never imported by core
tests/mlir/                  IR/verifier/pass/translation tests
tests/python/                frontend/runtime/differential tests
output/roofline/             checked benchmark artifacts by operation/case
```

`PROJECT.md` is the canonical design specification. Public documentation may
explain and demonstrate that design but must not introduce a conflicting IR or
runtime contract.

## Building and checks

Tiga is a standard out-of-tree MLIR project; it never downloads or
modifies the user's LLVM. Development builds require a prebuilt MLIR SDK:

```bash
cmake -S . -B build -G Ninja \
  -DMLIR_DIR=/opt/llvm-22.1.8/lib/cmake/mlir \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --target gf-opt gf-translate
```

The release toolchain is pinned to LLVM/MLIR 22.1.8 (`llvmorg-22.1.8`).
`TIGA_STRICT_LLVM_VERSION` defaults to `ON`; disabling it is reserved
for explicit API compatibility tests and changes neither the release ABI nor
compiler cache identity.

The provider handoff crosses a process boundary on purpose:
`gf-translate -gf-kernel-to-ttir` serializes verified Kernel IR to provider
TTIR plus a launch manifest, because a vendor Triton/MLIR revision may be
incompatible with the pinned Tiga revision and MLIR C++ objects must
never be shared in one address space.

```bash
PYTHONPATH=python python -m pytest tests/python -q
cmake --build build --target check-tiga
mkdocs build --strict
export CMAKE_ARGS="-DMLIR_DIR=/path/to/llvm-22.1.8/lib/cmake/mlir -DLLVM_DIR=/path/to/llvm-22.1.8/lib/cmake/llvm"
python -m build
```

The wheel's CMake install component bundles the native Python compiler
extension, `gf-opt` and `gf-translate`, the Tiga runtime, and the real
`libMLIR`/`libLLVM` SONAME files of the shared-SDK build; the extension and
tools carry relative RPATHs. Release CI must reproduce the local
clean-environment smoke test in a manylinux image, run `auditwheel
show/repair` (or the platform equivalent), and test installation without
developer SDK paths. The post-install command is:

```bash
/path/to/clean/venv/bin/python tests/wheel_smoke.py
```

## Contribution policy

Before submitting a change, verify:

- no workload-specific `@triton.jit` kernel enters `python/tiga/`;
  handwritten templates are isolated in `benchmarks/kernels/` and used only as
  performance oracles, and CI retains a source scan enforcing this boundary;
- no lowering matches Python class names or field names;
- every rewrite states legality conditions and preserves reducer/effect data;
- external-library dispatch is explicit in `explain()`;
- benchmark oracles are imported only by benchmark code;
- Tensor/autograd additions are generic IR/runtime facilities, never
  optimizer-, NN-, dataset- or workload-specific implementations;
- major subsystems remain packages with a small `__init__.py`; do not recreate
  the former flat `tensor.py`, `graph.py`, `kernel.py` or similar modules.

Claim discipline:

- FLA/FSA, attention, PageRank and visualization remain examples/benchmarks;
  the compiler core contains no workload-named kernel.
- Results apply only to registered shapes, dtypes, topology distributions,
  cache states and hardware.
- External dispatch is labeled as dispatch. A semantic evaluator is labeled as
  an oracle. Neither is reported as generated TTIR.
- One-GPU NCCL binding proves integration, not peer-link performance.
- Public status changes must update the ledger, tests, benchmark artifacts and
  the roadmap page together.

## Documentation

- Prefer semantic HTML for page structure and native SVG for architecture.
- Publish benchmark charts as interactive HTML when hover/filtering adds real
  value; keep a committed SVG fallback and PNG only for raster-only clients.
- Generate performance views from `benchmarks/evidence_manifest.json` and
  registered JSON. Do not hand-copy measurements into a new chart generator.
- Give each exact provider one corpus-stable color; vary marker/line style for
  conditions and input sizes.
- Every figure needs useful alt text, a caption and a full-size/fallback link.
- Run `mkdocs build --strict` and inspect desktop plus narrow layouts before
  publishing.
