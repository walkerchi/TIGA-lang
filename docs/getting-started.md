# Getting started

## Package naming

The intended PyPI distribution is `graphforge-compiler`; the shorter
`graphforge` distribution name is already owned by an unrelated openCypher
database. Installation therefore uses:

```bash
pip install graphforge-compiler
```

The import stays concise:

```python
import graphforge as gf
print(gf.__version__)
```

Binary wheels are not published yet. Until the first release is published,
build from source with the pinned LLVM/MLIR toolchain.

## Source build

```bash
git clone https://github.com/walkerchi/graphforge.git
cd graphforge
export CMAKE_ARGS="-DMLIR_DIR=/path/to/llvm-22.1.8/lib/cmake/mlir"
python -m pip install -e ".[cuda,dev]"
python -m graphforge
```

`python -m graphforge` prints the package version, native runtime state and the
resolved compiler tools (plus optional Torch adapter/CUDA provider state when
installed). `.[cuda]` installs Triton but does not install Torch; add `.[torch]`
only for framework interop, and `.[nccl-cu12]` only on CUDA deployments that
need the bundled NCCL library.
A release wheel will bundle those tools under
`graphforge/bin`; environment variables remain useful for development builds.

## First differentiable JIT program

--8<-- "examples/message_passing_autograd.py"

The first call creates a specialization from graph provenance, tensor dtype,
shape/stride, target and provider identity. A warm call reuses the executable
only if its guards still hold.

## Run tests

```bash
python -m unittest discover -s tests/python -v
cmake --build build --target check-graphforge
```

The Python suite covers capture/runtime/differential behavior; the lit suite
checks dialect verifiers, transformations and provider translation.
