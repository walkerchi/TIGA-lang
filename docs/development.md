# Development

## Repository layout

```text
include/graphforge/Dialect/  TableGen dialect definitions
include/graphforge/Dialect/Tensor/ canonical Tensor operations and contracts
lib/Dialect/                 verifiers and dialect implementation
lib/Transforms/              target-independent analyses and passes
lib/Target/                  provider translators
tools/                       gf-opt and gf-translate
python/graphforge/           stable public façade only at package root
python/graphforge/tensor/    Tensor metadata, views, capture and creation
python/graphforge/autograd/  semantic reverse-mode transforms
python/graphforge/graph/     logical relation plus dynamic builders
python/graphforge/reducer/   reducer semantic definitions
python/graphforge/message_passing/ capture, planning and execution façade
python/graphforge/kernel/    compiled variants and artifact diagnostics
python/graphforge/distributed/ mesh, partition and halo placement models
python/graphforge/compiler/  typed capture, native binding, toolchain and JIT cache
python/graphforge/interop/torch/ optional Torch oracle/provider boundary
python_bindings/            C++ OpBuilder and in-process ExecutionEngine binding
python/graphforge/runtime/   native runtime binding
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

## Compiler-only boundary

Before submitting a change, verify:

- no workload-specific `@triton.jit` kernel enters `python/graphforge/`;
- no lowering matches Python class names or field names;
- every rewrite states legality conditions and preserves reducer/effect data;
- external-library dispatch is explicit in `explain()`;
- benchmark oracles are imported only by benchmark code.
- Tensor/autograd additions are generic IR/runtime facilities, never
  optimizer-, NN-, dataset- or workload-specific implementations.
- major subsystems remain packages with a small `__init__.py`; do not recreate
  the former flat `tensor.py`, `graph.py`, `kernel.py` or similar modules.

## Checks

```bash
python -m unittest discover -s tests/python -v
cmake --build build --target check-graphforge
mkdocs build --strict
export CMAKE_ARGS="-DMLIR_DIR=/path/to/llvm-22.1.8/lib/cmake/mlir -DLLVM_DIR=/path/to/llvm-22.1.8/lib/cmake/llvm"
python -m build
```

The wheel build requires the pinned MLIR CMake package. CMake installs shared
MLIR/LLVM runtime targets by SONAME and gives the extension/tools relative
RPATHs; the local clean-environment smoke test covers this path. Release CI
must reproduce it in a manylinux image, run `auditwheel show/repair` (or the
platform equivalent), and test installation without developer SDK paths.
The post-install command is:

```bash
/path/to/clean/venv/bin/python tests/wheel_smoke.py
```

## Documentation and media

Public documentation has three layers: **Learn** for users, **Compiler** for
architecture/runtime, and **Performance** for measured evidence. Design notes
and research snapshots live under **Internals** and must not be used as the
current support matrix.

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
