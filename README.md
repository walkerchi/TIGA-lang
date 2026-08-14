# GraphForge v2

GraphForge is an experimental relation-oriented compiler with a minimal
Tensor/runtime/autograd substrate. It is deliberately not an operator library:
workloads live in `examples/` and `benchmarks/`; optimizer, NN and dataset
layers are out of scope.

Licensed under Apache-2.0.

```text
Python MessagePassing → gf.domain ───────────────→ gf.iter → gf.kernel
                           └→ gf.storage/gf.task ─┘
                              → serialized provider TTIR → PTX/cubin
Python Tensor → gf_tensor → SCF/MemRef → LLVM → in-process CPU JIT
```

`Graph.halo()` supplies a DTensor-like declarative distributed placement on the
same `Graph` type. The compiler IR already preserves Region/PhysicalInstance/
Event and partition/halo dependencies. A placed kernel is lowered to an
inspectable `pack → exchange → unpack` plus interior/boundary overlap task DAG;
the CPU stdlib provider executes rank-local contiguous Tensor forward/VJP through
the same `Graph.halo()` MessagePassing API. Versioned `.gfg` relations remain
paged on NVMe and each rank reads only its destination/edge shard. CUDA Buffer
shards also pass forward/VJP binding on the available single GPU. The MPI plugin
ABI is executable through an mpi4py-compatible communicator. A Torch-free NCCL
provider now binds native device-buffer slices and CUDA stream events below the
same graph API; its single-rank communicator plus local-device transport gate
passes locally. RCCL and true NCCL multi-device correctness/overlap/performance
remain fail-closed release gates.

Generated paths currently cover scalar fixed/bounded-ragged CSR, generated
radius distance aggregation, and a generic dense Cartesian contraction with a
structured streaming reducer. Handwritten performance kernels live only under
`benchmarks/kernels/` and are never imported by GraphForge runtime code.

The Torch-independent runtime owns CPU Buffer/Stream/Event objects. `gf.Tensor`
supports general broadcasting, views, axis reductions, complex dtypes and
functional conjugate-Wirtinger VJP. Supported CPU DAGs are constructed with
MLIR OpBuilder and lowered in-process through SCF/MemRef and the LLVM dialect to
an MLIR ExecutionEngine; there is no generated C/C++ source path. The Python
evaluator remains a correctness oracle. `Tensor.mlir()` emits verified canonical
`gf_tensor` IR, the JIT cache uses its stable semantic hash, and
`gf.autograd.grad_mlir()` exposes the native reverse-mode pass.
Torch is an optional adapter installed with `.[torch]`; it is not pulled in by
the Torch-free CUDA compiler/runtime extra `.[cuda]`. NCCL deployment support is
available separately as `.[nccl-cu12]` or through a system-provided library.
Static-CSR edge/node UDFs and structurally additive tuple reducers have
compiler-generated relation VJPs, so examples do not define backward methods.

```bash
export PYTHONPATH="$PWD/python"
python3 -m unittest discover -s tests/python -v
python3 examples/message_passing_autograd.py
python3 examples/torch_interop.py  # optional PyTorch adapter only
python3 -m benchmarks.autograd.message_passing_backward --quick
python3 -m benchmarks.autograd.message_passing_backward \
  --quick --gradient dweight --features 16
python3 -m benchmarks.neural_networks.dense_attention --quick
```

Runnable examples are indexed in [examples/README.md](examples/README.md).
Measured artifacts use `output/roofline/<operation>/<case>/`; the public
[benchmark guide](docs/performance.md) explains semantic matching, FLOP/byte
models, cold JIT accounting and performance gates.

The planned PyPI distribution name is `graphforge-compiler`, because the
`graphforge` distribution name belongs to an unrelated project. The Python
import remains `import graphforge as gf`.

## MLIR compiler

The out-of-tree compiler is pinned to LLVM/MLIR 22.1.8. Point `MLIR_DIR` at its
CMake package and build `gf-opt` plus the lit suite:

```bash
cmake -S . -B build -G Ninja \
  -DMLIR_DIR=/path/to/llvm-22.1.8/lib/cmake/mlir
cmake --build build --target check-graphforge
```

For local bring-up an explicitly configured compatibility SDK may be used with
`GRAPHFORGE_STRICT_LLVM_VERSION=OFF`; no `/tmp` tool path is part of the package
or test contract. The strict pinned LLVM/MLIR build remains the release/CI contract. See
[docs/COMPILER_BOOTSTRAP.md](docs/COMPILER_BOOTSTRAP.md).

Start with [PROJECT.md](PROJECT.md).  The earlier architecture discussion is in
[docs/rfcs/0001-architecture.md](docs/rfcs/0001-architecture.md), and the
literature/project survey is in [docs/RELATED_WORK.md](docs/RELATED_WORK.md).
The focused design note on tensor-axis, sparse/ragged, layout, storage, pipeline,
and distributed scheduling is in
[docs/SCHEDULING_ABSTRACTIONS.md](docs/SCHEDULING_ABSTRACTIONS.md).
The staged plan for naive Static/Dynamic Graph support followed by one measured
GPU work-tile optimization is in
[docs/GPU_GRAPH_OPTIMIZATION.md](docs/GPU_GRAPH_OPTIMIZATION.md).
The query/cache coverage matrix, roofline methodology, SOTA provider rules, and
measured RTX 5070 Ti baseline are in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md).
