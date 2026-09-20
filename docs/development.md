# Development

Prerequisite: [IR by example](ir-walkthrough.md) for the compiler model, or
[Getting started](getting-started.md) for installation alone. This page covers
changing the project, not requirements for running a first program.

## Make a first compiler change

1. Reproduce a problem with a small Python test or MLIR fixture.
2. Use the walkthrough's `gf-opt` commands to locate the first incorrect stage.
3. Change the relevant verifier or pass and add a regression test.
4. Run the Python/MLIR checks below; regenerate IR documentation only if the
   compiler output intentionally changed.

| Change | Implementation starting point |
|---|---|
| Public fields and call validation | `python/tiga/message_passing/` |
| Domain operations and verification | `include/tiga/Dialect/Domain/`, `lib/Dialect/Domain/` |
| Relation traversal | `lib/Transforms/LowerDomainToIter.cpp` |
| Kernel representation and schedule | `lib/Transforms/LowerIterToKernel.cpp`, `SelectKernelSchedule.cpp` |
| Distributed task dependencies | `lib/Transforms/PlanDistributedTasks.cpp` |
| Target TTIR emission | `lib/Target/Triton/Translate.cpp` |

## Repository layout

```text
CMakeLists.txt              native build entry point used by Python packaging
cmake/                      shared build settings, including the pinned LLVM SDK
include/tiga/               C++ headers and TableGen declarations
include/tiga/Dialect/  TableGen dialect definitions
include/tiga/Dialect/Tensor/ canonical Tensor operations and contracts
lib/Dialect/                 verifiers and dialect implementation
lib/Transforms/              target-independent analyses and passes
lib/Target/                  provider translators
tools/                       gf-opt and gf-translate
python/tiga/           public façade at package root
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
third_party/licenses/        license texts copied into native wheels, not dependency sources
assets/                     canonical logo sources and rendered variants
docs/archive/               historical design records, not current API contracts
output/roofline/             local benchmark outputs (ignored by Git)
```

`include/` and `lib/` separate declarations from implementations. TableGen reads
the `.td` definitions and writes generated C++ headers into the build directory;
these generated files are not source files to commit. `cmake/` is a set of
configuration helpers for the root build, not a separate project. Python
packaging invokes that same native build through scikit-build-core.

Dependencies are not Git submodules: CMake discovers a pinned LLVM/MLIR SDK,
while Python extras install optional packages separately. `third_party/licenses/`
contains only the notices required alongside redistributed binaries, not LLVM
or Torch source checkouts. It must also be present in source archives and wheels
that are installed without Git.

The C++ namespace and headers use `tiga`; the native extension is
`_tiga_compiler`, and the runtime library is `libtiga_runtime` on Linux.
The existing `gf.*` IR syntax, `gf-*` executables and `gfrt_*` C ABI symbols remain
stable. Historical experiment snapshots retain their recorded names and hashes.

Historical planning documents live in `docs/archive/` and are excluded from the
documentation site. The [API reference](api.md), [support matrix](roadmap.md) and
implementation tests define the current boundary.
`SECURITY.md` is the public policy for privately reporting vulnerabilities.

## Building and checks

Install an editable package and the test/documentation extras as described in
[Getting started](getting-started.md#source-build), replacing its install
command with:

```bash
python -m pip install -e ".[test,docs]"
python -m pytest tests/python -q
python -m mkdocs build --strict
python tools/render_api_reference.py --check
python tools/check_docs_links.py site
```

### MLIR tests

API tables and minimal examples are maintained in `tools/render_api_reference.py`;
`tests/python/test_api_reference.py` checks bilingual freshness and executes the
CPU examples. Add parameter contracts and usage examples together with API changes.
All published pages must have English and Chinese counterparts. Reporting and
publication requirements are in [support](support.md).

The editable wheel build disables compiler tests. Use a separate CMake
directory for the [lit](https://en.wikipedia.org/wiki/LLVM) suite.
Set `TIGA_LLVM_ROOT` to a prebuilt LLVM/MLIR 22.1.8 SDK first.

Some binary SDKs omit FileCheck, not, count and lit. The following helper
downloads the pinned LLVM source archive, verifies its checksum, and builds
only these test utilities into a separate directory; it does not modify the SDK.
An existing source archive can be supplied with `--archive`.

```bash
python tools/bootstrap_llvm_test_tools.py \
  --llvm-root "$TIGA_LLVM_ROOT" \
  --output "$PWD/build/llvm-test-tools"

cmake -S . -B build/compiler -G Ninja \
  -DMLIR_DIR="$TIGA_LLVM_ROOT/lib/cmake/mlir" \
  -DLLVM_DIR="$TIGA_LLVM_ROOT/lib/cmake/llvm" \
  -DCMAKE_BUILD_TYPE=Release \
  -DTIGA_INCLUDE_TESTS=ON \
  -DLLVM_EXTERNAL_LIT="$PWD/build/llvm-test-tools/lit/lit.py" \
  -DTIGA_LLVM_TEST_TOOLS_DIR="$PWD/build/llvm-test-tools/bin"
cmake --build build/compiler --target check-tiga --parallel 2
python tools/render_ir_docs.py --gf-opt build/compiler/bin/gf-opt --check
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

### CUDA and package checks

```bash
python -m pip install -e ".[cuda,test]"
python tools/gpu_gate.py
python -m pip install build
python -m build
```

These commands retain the `CMAKE_ARGS` SDK configuration from installation.
CUDA tests need a supported NVIDIA GPU and driver; they are not CPU setup
requirements. Compiler tool discovery can use the installed package tools.

`tools/gpu_gate.py` runs the CUDA forward/backward, cache, and diagnostics
suite. CUDA, Triton and the built compiler tools are required. Skipped,
deselected and expected-failure checks make the gate fail. GPU verification
also runs against the repaired release wheel on a trusted self-hosted runner
labelled `tiga-cuda`. Configure the `release-validation` environment and runner
before tagging: without them, publication remains blocked. GPU jobs run only
in the release workflow, not on untrusted pull requests.

Publication additionally requires the reusable CPU/MLIR/docs workflow,
wheels rebuilt from the same sdist, and exact tag/source/archive metadata
checks. Configure the `pypi` environment and PyPI trusted publisher separately;
local builds never publish. First-release artifacts cover Linux x86-64,
CPython 3.11/3.12 and glibc 2.38 or newer.

Read the Docs uses `.readthedocs.yaml` and `docs/requirements.txt` without
installing Tiga or LLVM. Import the GitHub repository into RTD after making it
public. `READTHEDOCS_CANONICAL_URL` selects the documentation base URL.
Local documentation can be viewed with `mkdocs serve`.

The wheel's CMake install component bundles the native Python compiler
extension, `gf-opt` and `gf-translate`, the Tiga runtime, and the real
`libMLIR`/`libLLVM` SONAME files when using a shared SDK. Static-SDK builds link
these libraries into the binaries instead. Shared dependencies use relative
RPATHs. Release CI must reproduce the
clean-environment smoke test in a manylinux image, run `auditwheel
show/repair` (or the platform equivalent), and test installation without
developer SDK paths. Install the repaired wheel and its declared dependencies into that clean
environment first. The Torch-free post-install check exercises tensor and
message-passing forward/backward execution and asserts the native CPU JIT.
Run it without developer `PYTHONPATH`, `LD_LIBRARY_PATH`, `TIGA_OPT`,
`TIGA_TRANSLATE`, `MLIR_DIR` or `LLVM_DIR` overrides:


```bash
# Run outside the checkout, in a fresh environment without Torch or SDK paths.
cd /tmp
/path/to/clean/venv/bin/python -I /path/to/tiga-lang/tests/smoke/wheel_without_torch.py
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

- Use `import tiga as tg` and `tg.*` in Python examples; preserve compiler IR
  namespaces and existing compatibility anchors.
- Keep the user learning path separate from compiler implementation; explain how
  to run a program before introducing internal IR.
- State prerequisites and next steps. Use reproducible compiler output for IR
  examples and verify the checked snapshots with `tools/render_ir_docs.py --check`
  (supplying the built `--gf-opt` path).
- Keep English and Chinese pages aligned; use neutral prose without second person.
- Preserve English technical terms and link their first occurrence to Wikipedia
  (Chinese terms to Baidu Baike).
- Put formulas in standalone `$$` blocks and link example commands to their source.
- Give each diagram one focus and inspect a rendered screenshot after changing it.
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
