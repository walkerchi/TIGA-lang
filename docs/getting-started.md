# Getting started

## Install

```bash
pip install tiga-lang   # PyPI's bare "tiga" is an unrelated project
```

```python
import tiga as gf
gf.__version__  # e.g. "0.1.0"
```

The wheel ships the compiler toolchain prebuilt; building from source is only
needed for development.

## Source build

```bash
git clone https://github.com/walkerchi/TIGA-lang.git
cd tiga-lang
export CMAKE_ARGS="-DMLIR_DIR=/path/to/llvm-22.1.8/lib/cmake/mlir"
python -m pip install -e ".[cuda,dev]"
python -m tiga
```

`python -m tiga` prints the package version, native runtime state and the
resolved compiler tools (plus optional Torch adapter/CUDA provider state when
installed). `.[cuda]` installs Triton but does not install Torch; add `.[torch]`
only for framework interop, and `.[nccl-cu12]` only on CUDA deployments that
need the bundled NCCL library.

## First differentiable JIT program

```python
--8<-- "examples/message_passing_autograd.py"
```

The first call creates a specialization from graph provenance, tensor dtype,
shape/stride, target and provider identity, and compiles it the first time
the result is observed (`tolist()`, `to_numpy()`, `realize()`). A warm call
with identical guards reuses the compiled executable. `kernel.cache_info`
reports exactly this: a `miss` is one compiled variant, a `hit` is a reuse;
`kernel.explain()` prints the lowering and the same counts, and
`kernel.ir(stage)` / `kernel.code("ptx")` expose the per-stage artifacts.

## Run tests

```bash
python -m pytest
cmake --build build --target check-tiga
```

The Python suite covers capture/runtime/differential behavior; the lit suite
checks dialect verifiers, transformations and provider translation.

## Where next

- [Programming model](programming-model.md) — relations, UDFs and reducers in one picture
- [Examples](examples.md) — runnable programs by category
- [Benchmark results](benchmark-results.md) — the measured evidence
