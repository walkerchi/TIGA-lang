# Installation and first program { #getting-started }

Goal: install Tiga, run a neighbor aggregation on an NVIDIA GPU, and obtain
results and gradients with ordinary Torch tensors. No compiler IR knowledge is required.
Basic Python is assumed; source installation additionally requires a build toolchain.

## Install

This alpha is preparing its first release. Use source installation or a
locally built wheel for now. The distribution name is `tiga-lang`;
the import name is `tiga`.

### From PyPI (after the first release) { #pypi-install }

The following commands are for the upcoming release, not a currently published
artifact. In a virtual environment, install Torch first for the examples below
(skip the first command if Torch is already installed or native-only execution is intended):

```bash
python -m pip install torch
python -m pip install tiga-lang
python -m tiga
```

For GPU use, install a hardware-compatible Torch build instead of the generic
Torch command above. Tiga does not install or upgrade Torch by default.
For the optional Triton provider, replace the Tiga install command with:

```bash
python -m pip install "tiga-lang[cuda]"
```

This extra adds neither Torch nor a GPU driver. A compatible prebuilt wheel
bundles the compiler tools and native libraries; no separate LLVM/MLIR SDK or
C++ build toolchain is needed to install it. The first-release wheel matrix is
Linux x86-64, CPython 3.11/3.12, `manylinux_2_38_x86_64` (glibc ≥ 2.38).
macOS, Windows and other Python versions are not first-release targets.
The Torch adapter targets Torch 2.11.x and Triton 3.6.x for CUDA; install a
hardware-compatible Torch distribution separately. If no compatible wheel is available, use the
[source build](#source-build) with its prerequisites; a source distribution
is not a precompiled wheel.

To install a locally built wheel before publication, use its actual file path:

```bash
python -m pip install "/path/to/tiga_lang-0.1.0-<python>-<abi>-<platform>.whl"
python -m tiga
```

Replace the placeholder filename with a wheel matching the Python version and
platform. No PyPI access is needed for the base wheel installation.

### Install from source { #source-build }

Requirements: Python 3.11–3.12, a C++ compiler, CMake, Ninja and a prebuilt
[LLVM](https://en.wikipedia.org/wiki/LLVM)/MLIR 22.1.8 SDK. Replace the SDK path.
The commands below use the supported Linux x86-64 / CPython 3.11–3.12 build environment.

```bash
git clone https://github.com/walkerchi/TIGA-lang.git tiga-lang
cd tiga-lang
python -m venv .venv
source .venv/bin/activate
# Torch examples: install Torch first; omit this for native-only use.
python -m pip install torch
export TIGA_LLVM_ROOT=/path/to/llvm-22.1.8
export CMAKE_ARGS="-DMLIR_DIR=$TIGA_LLVM_ROOT/lib/cmake/mlir -DLLVM_DIR=$TIGA_LLVM_ROOT/lib/cmake/llvm"
python -m pip install -e .
python -m tiga
```

Torch is optional. For GPU use, install the Torch build matching the intended
hardware before installing Tiga; reuse an existing Torch environment if available.
The default Tiga installation does not install or upgrade Torch. Without Torch,
use the [native Tensor and CPU JIT path](execution.md#native-execution).

`python -m tiga` should report the package version, native runtime state and
resolved compiler tools. Resolve missing tools before continuing:
[execution and troubleshooting](execution.md#diagnose-a-failure).

## First differentiable JIT program

Start with three nodes and five edges. Each edge sends its source temperature
multiplied by its conductivity; each destination sums the messages and adds a bias.
A [reducer](https://en.wikipedia.org/wiki/Fold_(higher-order_function)) specifies
how messages combine; this example uses sum.

Run these blocks in order. Inputs and outputs use `torch.Tensor`; Tiga captures
the message-passing computation without a public tensor wrapper.

```python
--8<-- "examples/torch_quickstart.py:quickstart"
```

Observe the results and gradients:

```python
print(output.tolist())        # approximately [11.1, 8.2, 17.3]
print(d_temperature.tolist()) # [7.0, 10.0, 3.0]
```

The first result, 11.1, comes from two messages arriving at destination 0 plus
its bias. The [programming model](programming-model.md) explains the mapping
between these edges, arrays and fields.

No explicit compile call or handwritten backward function is needed.
The output works with Torch operations, `torch.autograd.grad` and `.backward()`.
A successful result does not by itself identify a compiled provider; planning
and artifact inspection are covered in [execution and troubleshooting](execution.md).

Run the complete source with
[`python examples/torch_quickstart.py`](#first-differentiable-jit-program).
For the same small computation on CPU, replace `torch.device("cuda")` with
`torch.device("cpu")`. For execution without Torch, see the
[native example](execution.md#native-execution).

## Optional installation and dependencies { #optional-installation }

- Install Torch separately before Tiga to run the Torch examples. Native Tiga
  runs without Torch; `tiga-lang[torch]` is only an explicit opt-in dependency extra.
- PyPI extras use `tiga-lang[cuda]` or `tiga-lang[nccl-cu12]` (bundled NCCL).
  From a source checkout, use `python -m pip install -e ".[cuda]"` or
  `python -m pip install -e ".[nccl-cu12]"` instead. CUDA drivers remain an external prerequisite.

## Development checks { #run-tests }

Running a program does not require configuring compiler tests first.
For source changes, follow [Build and contribute](development.md) to install
test dependencies and configure a separate MLIR test build.

## Next { #where-next }

1. [Programming model](programming-model.md): graphs, fields, messages and aggregation.
2. [Execution and troubleshooting](execution.md): identify execution paths and failures.
3. [Worked examples](examples.md): apply the interface to a workload.

Compiler developers can then continue with the
[compiler introduction](compiler-pipeline.md) and [IR walkthrough](ir-walkthrough.md).
