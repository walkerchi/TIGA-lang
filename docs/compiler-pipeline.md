# Compiler introduction { #compiler-pipeline }

This is the entry point for compiler developers. Prerequisite:
the [Python programming model](programming-model.md). For running programs
or identifying an execution failure, start with [execution and troubleshooting](execution.md).

## Why internal representations exist { #the-pipeline-at-a-glance }

The Python program describes the result; the compiler must also decide how to
visit edges, divide work and move data. An
[intermediate representation (IR)](https://en.wikipedia.org/wiki/Intermediate_representation)
records those decisions in a form that later transformations can inspect.
IR is internal data, not an additional user API.

For a weighted neighbor sum, the multiplication and sum define the computation.
Choosing CSR row traversal or allocating work to a GPU block does not change
that definition. Different IR layers keep these concerns separate.

| Layer | Actual operation examples | Question being answered |
|---|---|---|
| Domain IR | `gf.relation`, `gf.apply`, `gf.reducer` | What graph and computation does the program describe? |
| Iter IR | `gf_iter.traverse` | In what order should the relation be visited? |
| Kernel IR | `gf_kernel.launch` | How should work and local resources be arranged? |
| Task / Storage IR | `gf_task.launch`, `gf_storage.transfer` | Which tasks depend on which data or events? |
| Tensor IR | `gf_tensor.mul`, `gf_tensor.reduce_sum` | What tensor expressions and gradients must be evaluated? |

The older labels `gf.domain`, `gf.iter` and `gf.kernel` in architectural
material are not literal operation names. A
[dialect](https://en.wikipedia.org/wiki/MLIR_(software)) groups related
operations; an operation such as `gf_iter.traverse` is one concrete instruction
in that dialect. Some inspection APIs also accept stage aliases such as
`"gf.iter"`; those strings are API keys, not MLIR syntax.

**Continue with [IR by example](ir-walkthrough.md)** for actual input/output,
notation, and commands that reproduce each transformation.

## Not every program follows the same route { #execution-paths }

- **Native Tensor programs**, including native CSR message passing, capture
  tensor expressions. Supported compiled CPU paths lower `gf_tensor`
  operations toward LLVM; gradient expressions are also represented here.
- **Relation compiler paths** retain `gf.apply` before
  [lowering](https://en.wikipedia.org/wiki/Compiler#Back_end) through
  `gf_iter` and `gf_kernel`. The walkthrough exercises this route directly
  from a checked compiler fixture.
- **Task and storage planning** add dependencies where the selected path
  requires them. Task IR is not an obligatory fourth stage of every call.
- **Reference evaluation** may be selected for small native expressions under
  `auto`. It does not demonstrate that any compiled route executed.

The [support matrix](roadmap.md) records which entry points serve which cases.
A diagram of the compiler's capabilities is not an execution trace.

## What is a provider?

A [backend](https://en.wikipedia.org/wiki/Compiler#Back_end) targets an execution
environment. In Tiga, a provider is the adapter to a target toolchain/runtime,
not a cloud service.

The current NVIDIA path hands serialized Triton IR (TTIR) to Triton, which
continues toward device code. The CPU path uses
[LLVM](https://en.wikipedia.org/wiki/LLVM).
Serialization keeps Tiga's and a provider's potentially different MLIR builds
separate. Earlier schedule selection still uses target capabilities; target
decisions do not all begin at the provider boundary.

## Inspecting a compiled program { #inspecting-a-compiled-program }

Start with the result, not an assumed pipeline:

| Question | Inspection |
|---|---|
| Did this native result execute through the JIT? | Realize it, then inspect `output.execution` |
| What native Tensor code was generated? | `output.generated_code()`; `output.mlir(verify=True)` for verified Tensor IR |
| What plan did a MessagePassing instance select? | `kernel.explain()` |
| Does a selected variant contain Iter or Kernel IR? | `kernel.ir("iter")` / `kernel.ir("kernel")`, only when those artifacts exist |

`kernel.ir("domain")` returns the variant's semantic plan, which is not
necessarily parseable MLIR. `"gf.kernel.ttir"` is a supported inspection key,
not a literal operation name. Missing artifacts raise an error; a loop asking
for every stage is not a portable inspection recipe.

Native gradients use a
[VJP](https://en.wikipedia.org/wiki/Automatic_differentiation), derived by
reverse-mode differentiation. The
[runtime and autograd chapter](runtime-and-autograd.md) explains its representation.

## Where to go deeper { #details }

| Development question | Next chapter |
|---|---|
| How do real operations change between passes? | [IR by example](ir-walkthrough.md) |
| Where are the files, and how are changes tested? | [Build and contribute](development.md) |
| How are generated neighbors and irregular degrees handled? | [Dynamic relations](dynamic-graphs.md) |
| How are gradients and deferred tensors implemented? | [Tensor runtime and autograd](runtime-and-autograd.md) |
| How are transfers and communication represented? | [Memory and distributed execution](memory-and-distributed.md) |
| How do solver loops and implicit gradients work? | [Linear solvers](linear-solvers.md) |
