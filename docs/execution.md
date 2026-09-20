# Execution and troubleshooting

After [running the first program](getting-started.md), use this page to answer:
when does computation happen, did the JIT run, and where should a failure be investigated?
Compiler IR is not a prerequisite.

## Default Torch execution { #torch-execution }

Torch inputs produce Torch outputs and use Torch autograd. Planning diagnostics
belong to the kernel:

```python
--8<-- "examples/torch_quickstart.py:quickstart"
print(kernel.explain())
```

Torch results do not expose Tiga's `.execution`, `.realize()` or `.generated_code()`.
`TIGA_TENSOR_BACKEND` controls native-Tensor evaluation, not every Torch provider.
Reference and compiled coverage depend on the relation and selected provider;
a numerically correct result alone does not prove compilation.

The Torch bridge for compiled weighted-sum and scalar CSR forwards currently
uses fixed-topology semantic replay for backward; `kernel.explain()` labels it
as non-TTIR backward. Weighted feature widths outside the direct tile's
power-of-two constraint select a sparse provider without changing Tensor types.

## Require native execution (advanced) { #native-execution }

Native tensors record deferred expressions. Observing a result triggers the
selected execution policy. A Python
[oracle](https://en.wikipedia.org/wiki/Test_oracle) is a reference evaluator
used to check results; it is not generated machine code.

```python
import os
os.environ["TIGA_TENSOR_BACKEND"] = "native"

import tiga as tg

x = tg.tensor([1.0, 2.0, 3.0], requires_grad=True)
out = (x * x).sum()
dx = tg.autograd.grad(out, x)

print(out.tolist())                 # 14.0
print(dx.tolist())                  # [2.0, 4.0, 6.0]
print(out.execution["backend"])     # cpu-llvm-jit
```

`native` requires native execution and reports unsupported compilation.
The default `auto` policy may use `python-oracle` for small expressions.
Changing the policy does not make unsupported operations or devices supported.

## Read the right diagnostics

| Observation | Meaning |
|---|---|
| A native program returned a Tensor | An expression may have been captured; computation may still be deferred |
| `tolist()`, `to_numpy()` or `realize()` completed | The value is available; inspect execution metadata to identify how it was obtained |
| `out.execution["backend"] == "cpu-llvm-jit"` | This result used the CPU JIT path |
| `out.execution` is `None` | It may be unexecuted, or an input/constant/view needing no execution |
| `kernel.cache_info` reports a hit | A capture variant was found; this does not prove compilation or execution |
| Native `out.execution` includes `cache_hit` / `compile_ms` | These describe actual compilation-cache behavior, not capture lookups |

The [backend](https://en.wikipedia.org/wiki/Compiler#Back_end) field identifies
the execution path. `kernel.explain()` describes planning;
`out.generated_code()` exposes native artifacts after compilation.
Neither a successful call nor a capture-cache hit is a performance measurement.

## Diagnose a failure

| Symptom | First check |
|---|---|
| Import or native tool discovery fails | Run `python -m tiga`; check the active environment and [source installation](getting-started.md#source-build) |
| A call reports a missing field or argument | Check `src`, `dst`, `edge` and leading dimensions in the [message-passing contract](message-passing.md#field-namespaces) |
| A graph constructs but cannot execute | Check the specific entry point in the [support matrix](roadmap.md#feature-support) |
| A tiny example reports `python-oracle` | This can be expected under `auto`; explicitly require `native` to validate JIT execution |
| CUDA or Torch interop is unavailable | Install a suitable Torch build separately; check the GPU/driver and `cuda` extra |
| `kernel.ir("iter")` raises `KeyError` | The selected variant may not have that artifact; see [inspection boundaries](compiler-pipeline.md#inspecting-a-compiled-program) |
| First-call latency is much larger than repeated execution | Separate compilation, transfers and execution; see the [measurement protocol](performance.md) |

Do not hide a failed native compilation by reporting a reference result as a
compiled result. A useful bug record includes the minimal program, exception,
`python -m tiga` output and available execution diagnostics, without private data.

## Next

- [Examples](examples.md): apply the API to a workload.
- [Python API](api.md): look up signatures and exact constraints.
- [Compiler introduction](compiler-pipeline.md): understand implementation decisions.
