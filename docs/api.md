# Python API

API reference organized by module. The side navigation selects a module;
each module starts with a compact method index. Full signatures, parameter
types and examples follow each entry. Complete executable recipes are collected in [API examples](api-examples.md).

Examples use `import tiga as tg`; `tg` is the Python alias, not an IR namespace.

## Default application interface: Torch { #torch-default }

Install Torch separately before Tiga to use this interface. Torch is not a
default dependency. Without Torch, use native `tg.tensor`, `tg.MessagePassing`
and `tg.autograd`; see [native execution](execution.md#native-execution).

Create fields with `torch.tensor` / `torch.randn`, pass them directly to
`tg.Graph` and a `tg.MessagePassing` instance, and use PyTorch autograd on its
Torch result. No wrapper or second Tensor API is needed for ordinary calls.
The [complete CPU recipe](api-examples.md#torch-default) includes an empty destination
row and checks gradients of both source and edge fields.

| Intent | Ordinary application API | Native compiler API (advanced) |
|---|---|---|
| Create values | `torch.tensor(...)` | `tg.tensor(...)` |
| Differentiate | `torch.autograd.grad` / `loss.backward()` | `tg.autograd.grad` |
| Move device | Torch `.to(device)` | Native `Tensor.to(device)` |
| Inspect compilation | Kernel `explain()` / available `ir()` artifacts | Native `Tensor.mlir()` / `generated_code()` |
| Storage budget | PyTorch-owned allocations are outside Tiga LRU | `tg.execution` manages scope-owned native buffers |

`tg.Tensor` remains a distinct standalone/advanced type, not an alias for
`torch.Tensor`. The `Tensor.*` entries below refer to **that native type**.
Raw Tensor programs, staged loop drivers and distributed/storage internals still
have native-only contracts; adopting Torch at the application boundary does not
silently make those APIs accept every Torch operation.

Some CUDA Torch CSR providers use a compiled forward and replay the fixed-topology
expression with Torch in backward. This may materialize edge intermediates and
is not a generated TTIR backward; inspect `kernel.explain()`. Native VJP and
the dedicated compiled edge-nn VJP are separate execution paths.

## Native Tensor and autograd (advanced) { #tensor-and-autograd }

`float16/float32/float64`, `complex64/complex128`, `int32/int64` and `bool`
are native Tiga dtypes. Only floating and complex values may require
gradients. Broadcasting follows right-aligned NumPy semantics.

### tg.tensor(data, *, dtype=None, device=None, requires_grad=False) { #gftensor }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `data` | <code>bool &#124; int &#124; float &#124; complex &#124; Sequence</code> | Scalar or rectangular nested values. |
| `dtype` | <code>tg.DType &#124; None</code> | Native value dtype; None infers from input data where supported. |
| `device` | <code>str &#124; tg.Device &#124; None</code> | Execution device; None uses the constructor's documented default. |
| `requires_grad` | <code>bool</code> | Record differentiable floating/complex inputs; default False. |

Return type: `tg.Tensor`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    x = tg.tensor([1., 2.], requires_grad=True)
    ```

<!-- typed-contract:end -->

Creates a Tensor backed by the native Tiga runtime from nested Python
sequences or a scalar. The dtype is inferred from the data when omitted. With `device=None`, the active `tg.execution` default is used, or CPU outside a context.

### tg.empty(shape, *, dtype=tg.float32, device=None, requires_grad=False) { #gfempty }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `shape` | <code>int &#124; Sequence[int]</code> | Non-negative dimension sizes; elements are uninitialized. |
| `dtype` | <code>tg.DType</code> | Native element dtype, default tg.float32; None is not accepted. |
| `device` | <code>str &#124; tg.Device &#124; None</code> | Execution device; None uses the constructor's documented default. |
| `requires_grad` | <code>bool</code> | Record differentiable floating/complex inputs; default False. |

Return type: `tg.Tensor`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    buffer = tg.empty((2, 3), dtype=tg.float32)
    ```

<!-- typed-contract:end -->

Allocates an uninitialized contiguous Tensor.

### tg.zeros_like(value, *, requires_grad=False) { #gfzeros_like }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `value` | <code>tg.Tensor</code> | Template for shape, dtype and device. |
| `requires_grad` | <code>bool</code> | Record differentiable floating/complex inputs; default False. |

Return type: `tg.Tensor`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    result = tg.zeros_like(x)
    ```

<!-- typed-contract:end -->

Allocates a zero-filled Tensor with the shape, dtype and device of `value`.

### tg.ones_like(value, *, requires_grad=False) { #gfones_like }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `value` | <code>tg.Tensor</code> | Template for shape, dtype and device. |
| `requires_grad` | <code>bool</code> | Record differentiable floating/complex inputs; default False. |

Return type: `tg.Tensor`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    result = tg.ones_like(x)
    ```

<!-- typed-contract:end -->

Allocates a one-filled Tensor with the shape, dtype and device of `value`.

### tg.from_torch(value, *, requires_grad=None) { #gffrom_torch }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `value` | <code>torch.Tensor</code> | Dense Torch-owned Tensor; wrapping shares storage, not autograd history. |
| `requires_grad` | <code>bool &#124; None</code> | None inherits the source flag; otherwise set the native leaf flag. |

Return type: `tg.Tensor`.

??? example "Minimal usage"

    ```python
    import torch
    import tiga as tg
    native = tg.from_torch(torch.tensor([1., 2.]))
    ```

<!-- typed-contract:end -->

Wraps a dense Torch tensor without copying its storage. Example:
[Torch interface and native storage bridge](examples/programs-and-interop.md#optional-torch-interoperability).

### Tensor.reshape(*shape) { #tensorreshape }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `*shape` | <code>int &#124; Sequence[int]</code> | New extents; at most one -1, with unchanged element count. |

Return type: `tg.Tensor`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    result = x.reshape(4)
    ```

<!-- typed-contract:end -->

Returns a lazy view with a new shape; one `-1` extent is inferred.

### Tensor.permute(*axes) { #tensorpermute }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `*axes` | <code>int &#124; Sequence[int]</code> | A permutation containing each axis exactly once. |

Return type: `tg.Tensor`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    result = x.permute(1, 0)
    ```

<!-- typed-contract:end -->

Returns a lazy view with axes reordered.

### Tensor.transpose(dim0, dim1) { #tensortranspose }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `dim0 / dim1` | <code>int</code> | The two axes to swap; negative axes are normalized. |

Return type: `tg.Tensor`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    result = x.transpose(0, 1)
    ```

<!-- typed-contract:end -->

Returns a lazy view with two axes swapped. `Tensor.T` reverses all axes.

### Tensor.squeeze(dim=None) { #tensorsqueeze }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `dim` | <code>int &#124; None</code> | Remove singleton axis dim, or all singleton axes when None. |

Return type: `tg.Tensor`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    result = x.squeeze()
    ```

<!-- typed-contract:end -->

Removes size-one extents, at `dim` only when given.

### Tensor.unsqueeze(dim) { #tensorunsqueeze }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `dim` | <code>int</code> | Insertion position of a new size-one axis. |

Return type: `tg.Tensor`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    result = x.unsqueeze(0)
    ```

<!-- typed-contract:end -->

Inserts a size-one extent at `dim`.

### Tensor.broadcast_to(shape) { #tensorbroadcast_to }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `shape` | <code>Sequence[int]</code> | Broadcast-compatible target extents, aligned from the right. |

Return type: `tg.Tensor`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    result = x.broadcast_to((3, 2, 2))
    ```

<!-- typed-contract:end -->

Returns a lazy broadcast view. `Tensor.expand(*shape)` is the variadic
spelling of the same view.

### Arithmetic, comparison and matmul operators { #tensor-operators }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `other` | <code>tg.Tensor &#124; bool &#124; int &#124; float &#124; complex</code> | Compatible scalar or Tensor operand; @ requires matrix operands. |

Return type: `tg.Tensor`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    result = (x + 1) * x
    mask = x > 2
    ```

<!-- typed-contract:end -->

`+`, `-`, `*`, `/`, unary `-`, comparisons and `@` build lazy expression
nodes. A `Tensor` used as a Python boolean (`if tensor:`) raises `TypeError`.

### Tensor.matmul(other) { #tensormatmul }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `other` | <code>tg.Tensor</code> | Rank-2 right operand with compatible contraction dimension. |

Return type: `tg.Tensor`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    result = x.matmul(x)
    ```

<!-- typed-contract:end -->

Rank-two matrix multiplication with a conjugate-Wirtinger VJP for complex
inputs. Example:
[Matrix multiplication with derived gradients](examples/programs-and-interop.md#matrix-multiplication-with-derived-gradients).

### Tensor.exp() { #tensorexp }

<!-- typed-contract:start -->

No explicit parameters (`self` omitted for instance methods).

Return type: `tg.Tensor`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    result = x.exp()
    ```

<!-- typed-contract:end -->

Elementwise exponential.

### Tensor.sqrt() { #tensorsqrt }

<!-- typed-contract:start -->

No explicit parameters (`self` omitted for instance methods).

Return type: `tg.Tensor`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    result = x.sqrt()
    ```

<!-- typed-contract:end -->

Elementwise square root.

### Tensor.conj() { #tensorconj }

<!-- typed-contract:start -->

No explicit parameters (`self` omitted for instance methods).

Return type: `tg.Tensor`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    result = x.conj()
    ```

<!-- typed-contract:end -->

Elementwise complex conjugate; identity for real dtypes. Example:
[Complex VJP](examples/programs-and-interop.md#complex-vjp).

### Tensor.cumsum(dim, *, reverse=False) { #tensorcumsum }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `dim` | <code>int</code> | Axis for the inclusive scan. |
| `reverse` | <code>bool</code> | Scan from the last element when True; default False. |

Return type: `tg.Tensor`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    result = x.cumsum(1, reverse=True)
    ```

<!-- typed-contract:end -->

Inclusive scan along one axis. Example:
[Linear recurrence from map/cumsum/contract](examples/programs-and-interop.md#linear-recurrence-from-mapcumsumcontract).

### Tensor.sum(axis=None, *, keepdims=False) { #tensorsum }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `axis` | <code>int &#124; Sequence[int] &#124; None</code> | Reduction axes; None reduces all axes. |
| `keepdims` | <code>bool</code> | Keep reduced axes with length one; default False. |

Return type: `tg.Tensor`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    result = x.sum(axis=1, keepdims=True)
    ```

<!-- typed-contract:end -->

Reduction over one axis, several axes, or all axes. These native arguments are
`axis` / `keepdims`; ordinary Torch tensors use `dim` / `keepdim`.

### Tensor.gather(index) { #tensorgather }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `index` | <code>tg.Tensor</code> | 1-D integer row IDs on the same device; repeats are allowed. |

Return type: `tg.Tensor`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    result = x.gather(tg.tensor([1, 0, 1], dtype=tg.int64))
    ```

<!-- typed-contract:end -->

Selects rows of a non-scalar native Tensor. `index` must be a rank-1 native
integer Tensor on the same device. Input `(N, *F)` and `K` indices produce
`(K, *F)`; indices must be in `[0, N)`. The VJP is a segment sum, so repeated
indices accumulate gradients.

### Tensor.segment_sum(index, num_segments) { #tensorsegment_sum }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `index` | <code>tg.Tensor</code> | One integer segment ID per input row, same device. |
| `num_segments` | <code>int</code> | Non-negative output row count; indices must be in range. |

Return type: `tg.Tensor`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    result = x.segment_sum(tg.tensor([0, 0], dtype=tg.int64), 2)
    ```

<!-- typed-contract:end -->

Sums `(N, *F)` rows into `(num_segments, *F)` output buckets. `index` must be a
rank-1 native integer Tensor of length `N` on the same device. `num_segments`
is a nonnegative integer; indices must be in `[0, num_segments)`. Empty buckets
produce zero.

### Tensor.checkpoint() { #tensorcheckpoint }

<!-- typed-contract:start -->

No explicit parameters (`self` omitted for instance methods).

Return type: `tg.Tensor`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    result = x.checkpoint()
    ```

<!-- typed-contract:end -->

Marks a non-leaf value as an explicit backward save
(`gf_tensor.checkpoint` in IR), overriding the checkpoint planner for this
value.

### Tensor.realize() { #tensorrealize }

<!-- typed-contract:start -->

No explicit parameters (`self` omitted for instance methods).

Return type: `tg.Tensor`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    result = x.realize()
    ```

<!-- typed-contract:end -->

Materializes the deferred expression using the selected execution policy.
`TIGA_TENSOR_BACKEND=native` requires JIT compilation; the default `auto`
policy may use the Python oracle for small expressions. Returns the Tensor.

### Tensor.prepare() { #tensorprepare }

<!-- typed-contract:start -->

No explicit parameters (`self` omitted for instance methods).

Return type: `Callable[[], tg.Tensor]`.

Before running this example, configure the required device, files or distributed environment as described below.

??? example "Minimal usage"

    ```python
    # Requires a CUDA provider supporting prepared launches.
    import tiga as tg
    x = tg.tensor([1., 2.], device='cuda')
    prepared = (x * 2).prepare()
    result = prepared()
    ```

<!-- typed-contract:end -->

Returns a zero-argument hot callable that resubmits the compiled DAG with
stable input/output buffers, after a first `realize()`. Cold compilation
stays visible in `Tensor.execution`.

Raises `RuntimeError` when the selected backend has no prepared submission,
or inside `tg.execution(eviction="lru")`: fixed addresses cannot be evicted.
The callable reuses storage and is not a fresh immutable output per invocation.

### Tensor.execution { #tensorexecution }

<!-- typed-contract:start -->

No explicit parameters (`self` omitted for instance methods).

Return type: `dict[str, object] | None`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    result = (x * 2).realize()
    print(result.execution)
    ```

<!-- typed-contract:end -->

Property returning codegen/launch diagnostics after realization: semantic
hash, compile/launch/materialization timing, the MLIR checkpoint plan, saved
bytes and the selected backend. `None` before execution, and also for
already-materialized inputs or constant/view results that need no execution.

### Tensor.generated_code(kind=None) { #tensorgenerated_code }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `kind` | <code>str &#124; None</code> | Artifact name such as llvm or ptx; None selects the default. Availability depends on the executed path. |

Return type: `str | bytes | None`.

Before running this example, configure the required device, files or distributed environment as described below.

??? example "Minimal usage"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    result = (x * 2).realize()
    print(result.generated_code())
    ```

<!-- typed-contract:end -->

Returns a real compiler stage artifact such as `"gf_tensor"`, `"cpu_loop"`,
`"llvm"` or `"ptx"`; the default picks the most lowered available stage.
Guide: [Compiler pipeline](compiler-pipeline.md#inspecting-a-compiled-program).

### Tensor.mlir(*, verify=False) { #tensormlir }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `verify` | <code>bool</code> | Run native IR verification when True; default False. |

Return type: `str`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    print((x * 2).mlir(verify=True))
    ```

<!-- typed-contract:end -->

Returns canonical target-independent `gf_tensor` IR; `verify=True`
round-trips it through the native parser and C++ verifiers. This is the
compiler contract; `Tensor.expression()` is informal debug capture.

### Tensor.tolist() / Tensor.to_numpy() / Tensor.to_torch(*, copy=False) { #tensor-host-interop }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `copy` | <code>bool</code> | to_torch only: False shares Torch-owned storage; True permits a copy from native contiguous storage. |

Return type: `Python scalar/list / numpy.ndarray / torch.Tensor`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    values = x.tolist()
    array = x.to_numpy()
    torch_copy = x.to_torch(copy=True)
    ```

<!-- typed-contract:end -->

`tolist()` returns nested Python values (a scalar for shape `()`); `to_numpy()`
returns a host copy. Both observe/realize the value. NumPy is optional.
`to_torch(copy=False)` only shares **Torch-owned** storage, such as an unchanged
`tg.from_torch(x)` wrapper; native-owned results raise `RuntimeError`.
`copy=True` permits a copy of contiguous native storage onto the same device.
Storage sharing alone does not connect the native autograd DAG to PyTorch autograd.

### tg.autograd.grad(output, inputs, *, grad_output=None, allow_unused=False, checkpoint="auto") { #gfautogradgrad }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `output` | <code>tg.Tensor</code> | Native differentiable output, not a Torch Tensor. |
| `inputs` | <code>tg.Tensor &#124; Sequence[tg.Tensor]</code> | Differentiation targets with requires_grad=True; sequence returns a tuple. |
| `grad_output` | <code>tg.Tensor &#124; None</code> | Output cotangent with matching shape/dtype/device; required for non-scalar or complex output. |
| `allow_unused` | <code>bool</code> | Return None for disconnected inputs when True; otherwise raise. |
| `checkpoint` | <code>Literal[&#x27;auto&#x27;, &#x27;save&#x27;, &#x27;recompute&#x27;]</code> | Forward-value storage policy for backward. |

Return type: `tg.Tensor | None | tuple[tg.Tensor | None, ...]`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    dx = tg.autograd.grad((x * x).sum(), x)
    assert dx.tolist() == [[2., 4.], [6., 8.]]
    ```

<!-- typed-contract:end -->

Builds functional reverse-mode VJP expressions without mutating inputs.
`checkpoint` selects backward primal storage: `save` marks non-leaf values
as saved, `recompute` retains the full primal expression, and `auto` emits
`gf_tensor.checkpoint_candidate` for the target-independent checkpoint
planner. Non-scalar or complex outputs require an explicit `grad_output`
matching output shape, dtype and device. Only native tensors are accepted, and
each requested input must have `requires_grad=True`. One input returns one
gradient; a sequence returns a tuple. Disconnected inputs raise by default or
return `None` with `allow_unused=True`; no `.grad` fields are mutated. Ordinary
Torch values use `torch.autograd.grad`.
Example:
[Matrix multiplication with derived gradients](examples/programs-and-interop.md#matrix-multiplication-with-derived-gradients).

### tg.autograd.value_and_grad(function, *, argnums=0) { #gfautogradvalue_and_grad }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `function` | <code>Callable</code> | Native Tensor function whose selected arguments are differentiable. |
| `argnums` | <code>int &#124; Sequence[int]</code> | Positional argument indices to differentiate; default 0. |

Return type: `Callable`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    f = tg.autograd.value_and_grad(lambda a: (a * a).sum())
    value, dx = f(x)
    ```

<!-- typed-contract:end -->

Returns a transform of `function` that produces the primal output and the
gradients of the positional arguments selected by `argnums`, without `.grad`
mutation.

### tg.autograd.joint_plan(output, inputs, *, checkpoint="auto") { #gfautogradjoint_plan }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `output` | <code>tg.Tensor</code> | Native differentiable output, not a Torch Tensor. |
| `inputs` | <code>tg.Tensor &#124; Sequence[tg.Tensor]</code> | Differentiation targets with requires_grad=True; sequence returns a tuple. |
| `checkpoint` | <code>Literal[&#x27;auto&#x27;, &#x27;save&#x27;, &#x27;recompute&#x27;]</code> | Forward-value storage policy for backward. |

Return type: `JointAutogradPlan`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    plan = tg.autograd.joint_plan((x * x).sum(), x)
    value, dx = plan.run()
    ```

<!-- typed-contract:end -->

Builds an executable, inspectable joint forward/backward task DAG with one
snapshot version. `plan.run()` returns `(output, gradients)`;
`plan.explain()` renders the bundle. Example:
[Joint forward/backward DAG](examples/programs-and-interop.md#joint-forwardbackward-dag).

### tg.autograd.grad_mlir(output, input, *, grad_output=None, lower=True) { #gfautogradgrad_mlir }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `output` | <code>tg.Tensor</code> | Native differentiable output, not a Torch Tensor. |
| `input` | <code>tg.Tensor</code> | Single requires_grad input to differentiate. |
| `grad_output` | <code>tg.Tensor &#124; None</code> | Output cotangent with matching shape/dtype/device; required for non-scalar or complex output. |
| `lower` | <code>bool</code> | Apply the VJP lowering pass; False retains the explicit request. |

Return type: `str`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    print(tg.autograd.grad_mlir((x * x).sum(), x))
    ```

<!-- typed-contract:end -->

Returns the explicit VJP request, or with `lower=True` the `gf-tensor-vjp`
result, as MLIR text for inspection.

## Graph { #graph }

An immutable logical relation connecting source and destination nodes, accepting
Torch or native index tensors. Materialized CSR owns
two index tensors; dense, triangular, radius and kNN relations stay implicit
or procedural and are never silently materialized into O(N²) adjacency.

### Graph.from_csr(row_ptr, col_idx, *, num_src=None, sorted_by_dst=True, validate="basic") { #graphfrom_csr }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `row_ptr` | <code>torch.Tensor &#124; tg.Tensor</code> | 1-D int32/int64 destination-row boundaries, length num_dst + 1. |
| `col_idx` | <code>torch.Tensor &#124; tg.Tensor</code> | 1-D source IDs in edge order, same device/dtype as row_ptr. |
| `num_src` | <code>int &#124; None</code> | Source-node count; None infers from indices. Supply it to retain isolated sources. |
| `sorted_by_dst` | <code>bool</code> | Destination grouping declaration, default True; CSR rows already group destinations. |
| `validate` | <code>Literal[&#x27;basic&#x27;, &#x27;full&#x27;]</code> | Basic structural checks or full index/boundary validation. |

Return type: `tg.Graph`.

??? example "Minimal usage"

    ```python
    import torch
    import tiga as tg
    row_ptr = torch.tensor([0, 2, 3, 3], dtype=torch.int64)
    col_idx = torch.tensor([0, 1, 1], dtype=torch.int64)
    graph = tg.Graph.from_csr(row_ptr, col_idx, num_src=2)
    assert graph.schema.num_dst == 3
    ```

<!-- typed-contract:end -->

Creates a frozen destination-row CSR relation. Use rank-1 integer Torch index
tensors on the same device; native tensors are also accepted. Destination `i`
receives edges `row_ptr[i]:row_ptr[i+1]`, whose sources are `col_idx` entries.
Edge fields follow this exact order; equal consecutive row pointers denote an
empty destination. `num_dst = len(row_ptr) - 1`. `num_src` defaults to
`max(col_idx) + 1`, or `num_dst` for empty columns; specify it explicitly to
retain isolated source nodes. `validate="full"` checks endpoints, monotonicity
and index ranges, raising `ValueError` for invalid CSR. Example:
[GCN aggregation](examples/message-passing.md#gcn-aggregation).

### Graph.from_coo(src, dst, *, num_src=None, num_dst=None) { #graphfrom_coo }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `src / dst` | <code>torch.Tensor &#124; tg.Tensor</code> | Same-length 1-D source/destination integer IDs, same device and dtype. |
| `num_src` | <code>int &#124; None</code> | Source-node count; None infers from indices. Supply it to retain isolated sources. |
| `num_dst` | <code>int &#124; None</code> | Destination-node count; see the constructor's inference rule. |

Return type: `tg.Graph`.

??? example "Minimal usage"

    ```python
    import torch
    import tiga as tg
    graph = tg.Graph.from_coo(torch.tensor([0, 1]), torch.tensor([1, 0]), num_src=3, num_dst=3)
    ```

<!-- typed-contract:end -->

Creates the same frozen relation from coordinate lists, performing a stable
destination sort into canonical CSR.

### Graph.regular(num_nodes, degree, *, device="cpu") { #graphregular }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `num_nodes` | <code>int</code> | Node count. |
| `degree` | <code>int</code> | Fixed incoming-neighbor count per node. |
| `device` | <code>str &#124; tg.Device &#124; None</code> | Execution device; None uses the constructor's documented default. |

Return type: `tg.Graph`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    graph = tg.Graph.regular(4, 2)
    ```

<!-- typed-contract:end -->

Creates a deterministic fixed-degree relation — destination `i` gathers from
sources `(i * degree + k) % num_nodes` — so benchmarks and smoke tests do not
hand-assemble `arange`/`%` CSR arrays.

### Graph.dense(num_src, num_dst=None, *, device=None, index_dtype=tg.int64) { #graphdense }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `num_src` | <code>int</code> | Non-negative source count. |
| `num_dst` | <code>int &#124; None</code> | Destination count; None uses num_src. |
| `device` | <code>str &#124; tg.Device &#124; None</code> | Execution device; None uses the constructor's documented default. |
| `index_dtype` | <code>tg.DType</code> | tg.int32 or tg.int64; default tg.int64. |

Return type: `tg.Graph`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    graph = tg.Graph.dense(3, 2)
    assert graph.num_edges == 6
    ```

<!-- typed-contract:end -->

Creates an implicit Cartesian relation; no `N×N` index tensor is allocated.
`num_dst` defaults to `num_src`. Example:
[Full attention, no mask](examples/attention.md#full-attention-no-mask).

### Graph.triangular(num_entities, *, device=None, index_dtype=tg.int64) { #graphtriangular }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `num_entities` | <code>int</code> | Non-negative node count; destination i receives from 0 through i. |
| `device` | <code>str &#124; tg.Device &#124; None</code> | Execution device; None uses the constructor's documented default. |
| `index_dtype` | <code>tg.DType</code> | tg.int32 or tg.int64; default tg.int64. |

Return type: `tg.Graph`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    graph = tg.Graph.triangular(3)
    assert graph.num_edges == 6
    ```

<!-- typed-contract:end -->

Creates an implicit lower-inclusive relation (`src <= dst`). It remains a
graph topology in IR; any MessagePassing reducer may consume it, and dense
lowering uses bounded source tiles plus a pair mask without materializing
CSR. Example:
[Causal dense relation](examples/attention.md#causal-dense-relation).

### Graph.cu_seqlens(cu_seqlens, *, causal=True) { #graphcu_seqlens }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `cu_seqlens` | <code>torch.Tensor &#124; tg.Tensor</code> | 1-D cumulative sequence boundaries, starting at zero and nondecreasing. |
| `causal` | <code>bool</code> | True uses triangular blocks; False uses dense blocks. |

Return type: `tg.Graph`.

??? example "Minimal usage"

    ```python
    import torch
    import tiga as tg
    graph = tg.Graph.cu_seqlens(torch.tensor([0, 2, 5]), causal=True)
    ```

<!-- typed-contract:end -->

Creates a materialized block-diagonal relation for flash-attn style packed
variable-length sequences: position `i` in sequence `k` gathers from sources
`[s_k, i]` (`causal=True`) or from its whole sequence (`causal=False`).
`cu_seqlens` must start at zero and be monotonic. Example:
[Varlen causal attention with cu_seqlens](examples/attention.md#varlen-causal-attention-with-cu_seqlens).

### Graph.cat(graphs) { #graphcat }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `graphs` | <code>list[tg.Graph] &#124; tuple[tg.Graph, ...]</code> | Non-empty same-device blocks; offsets source and destination IDs independently, adding no cross-block edges. |

Return type: `tg.Graph`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    graph = tg.Graph.cat([tg.Graph.triangular(2), tg.Graph.triangular(3)])
    assert graph.num_edges == 9
    ```

<!-- typed-contract:end -->

Composes relations into one block-diagonal relation: a destination of block
`k` is related exactly to the sources of block `k`, offset by the cumulative
source count. `cat([Graph.triangular(n) for n in lengths])` equals
`Graph.cu_seqlens([0, *cumsum(lengths)], causal=True)` after converting the boundaries to an integer Tensor; `Graph.dense` blocks give
the `causal=False` variant. The composition realizes one materialized CSR
snapshot. Example:
[Varlen causal attention with cu_seqlens](examples/attention.md#varlen-causal-attention-with-cu_seqlens).

### Graph.stencil(dims, offsets=None, *, periodic=False, device="cpu") { #graphstencil }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `dims` | <code>tuple[int, ...]</code> | Positive grid extents in row-major order. |
| `offsets` | <code>tuple[tuple[int, ...], ...] &#124; tg.stencil.Neighborhood &#124; None</code> | Integer source offsets or a neighborhood macro; None defaults to von_neumann(radius=1, include_center=True). |
| `periodic` | <code>bool</code> | Wrap out-of-grid coordinates when True; otherwise omit them (not zero padding). |
| `device` | <code>str &#124; tg.Device &#124; None</code> | Execution device; None uses the constructor's documented default. |

Return type: `tg.Graph`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    graph = tg.Graph.stencil((3, 3), tg.stencil.von_neumann())
    assert graph.num_edges == 33
    ```

<!-- typed-contract:end -->

Creates a materialized regular-grid stencil relation over row-major
linearized nodes; each destination gathers from `coord(dst) + offset` in
offsets order. Non-periodic boundaries truncate out-of-grid sources, periodic
boundaries wrap with modulo. None selects the radius-one von Neumann macro
including the center. Zero/constant padding is not yet supported. Examples:
[Neighborhood macros and five-point averaging](examples/dynamic-relations.md#stencil-neighborhood-macros),
[Matrix-free FEM operator and solver loop](examples/solvers.md#matrix-free-fem-operator-and-solver-loop),
[A disk-resident giant graph](examples/distributed-memory.md#a-disk-resident-giant-graph).

### stencil.von_neumann(radius=1, *, include_center=True) { #stencilvon_neumann }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `radius` | <code>int</code> | Positive inclusive integer radius; default 1, bool rejected. |
| `include_center` | <code>bool</code> | Include the zero offset; default True. |

Return type: `tg.stencil.Neighborhood`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    macro = tg.stencil.von_neumann(include_center=False)
    graph = tg.Graph.stencil((4, 4), macro)
    ```

<!-- typed-contract:end -->

Returns a dimension-independent [von Neumann neighborhood](https://en.wikipedia.org/wiki/Von_Neumann_neighborhood)
macro: sum of absolute offset components ≤ radius. The default contains five
points in 2-D and seven in 3-D. `Graph.stencil` infers dimension from `dims`.

### stencil.moore(radius=1, *, include_center=True) { #stencilmoore }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `radius` | <code>int</code> | Positive inclusive integer radius; default 1, bool rejected. |
| `include_center` | <code>bool</code> | Include the zero offset; default True. |

Return type: `tg.stencil.Neighborhood`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    macro = tg.stencil.moore(include_center=False)
    graph = tg.Graph.stencil((4, 4), macro)
    ```

<!-- typed-contract:end -->

Returns a dimension-independent [Moore neighborhood](https://en.wikipedia.org/wiki/Moore_neighborhood)
macro: maximum absolute offset component ≤ radius. The default contains nine
points in 2-D and 27 in 3-D. Use `include_center=False` to remove the zero offset.

### stencil.Neighborhood.offsets(ndim) { #stenciloffsets }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `ndim` | <code>int</code> | Positive number of grid axes; bool rejected. |

Return type: `tuple[tuple[int, ...], ...]`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    offsets = tg.stencil.von_neumann(include_center=False).offsets(2)
    assert offsets == ((-1, 0), (0, -1), (0, 1), (1, 0))
    ```

<!-- typed-contract:end -->

Expands a macro to integer offsets in lexicographic order, without allocating
device data or importing Torch. The frozen value object exposes `kind`, `radius`
and `include_center`. Explicit tuples preserve their supplied order instead.
Macro size grows with radius and dimension; this API materializes offsets.

### Graph.radius(positions, cutoff, *, exclude_self=True, fields=None, metric=None, select=None, periodic=None) { #graphradius }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `positions` | <code>torch.Tensor &#124; tg.Tensor</code> | Floating coordinates with shape (N, D), on the target graph device. |
| `cutoff` | <code>float</code> | Positive distance cutoff. |
| `exclude_self` | <code>bool &#124; None</code> | Exclude self edges; default depends on this constructor. |
| `fields` | <code>Mapping[str, Tensor] &#124; None</code> | Builder fields, distinct from MessagePassing bindings. |
| `metric` | <code>Callable &#124; None</code> | Optional distance UDF; None uses Euclidean distance. |
| `select` | <code>Callable &#124; None</code> | Optional edge-selection UDF; unsupported compilation combinations fail closed. |
| `periodic` | <code>Tensor &#124; Sequence &#124; None</code> | Box lengths (D,) or lattice (D,D); None disables periodic boundaries. |

Return type: `tg.Graph`.

??? example "Minimal usage"

    ```python
    import torch
    import tiga as tg
    graph = tg.Graph.radius(torch.tensor([[0., 0.], [0.5, 0.]]), cutoff=1.)
    ```

<!-- typed-contract:end -->

Creates a rebuildable procedural relation from rank-two floating positions;
adjacency is recomputed from the current snapshot. `metric`/`select` are
custom builder UDFs, `periodic` supplies box lengths `[D]` or lattice vectors
`[D,D]`. Example:
[Differentiable radius relation](examples/dynamic-relations.md#differentiable-radius-relation).
For the callback's `(P,D)` inputs, `(P,)` return value, cosine example and
all-pairs performance limitation, see [distance metrics](examples/dynamic-relations.md#distance-metrics).
`metric` accepts a callable or None, not a distance-name string.

### Graph.knn(positions, k, *, candidates=None, exclude_self=None) { #graphknn }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `positions` | <code>torch.Tensor &#124; tg.Tensor</code> | Floating coordinates with shape (N, D), on the target graph device. |
| `k` | <code>int</code> | Number of nearest candidates per destination, subject to count/backend limits. |
| `candidates` | <code>torch.Tensor &#124; tg.Tensor &#124; None</code> | Source coordinates (M,D); None reuses positions for self-kNN. |
| `exclude_self` | <code>bool &#124; None</code> | Exclude self edges; default depends on this constructor. |

Return type: `tg.Graph`.

??? example "Minimal usage"

    ```python
    import torch
    import tiga as tg
    graph = tg.Graph.knn(torch.tensor([[0., 0.], [1., 0.], [3., 0.]]), k=1)
    ```

<!-- typed-contract:end -->

Creates an exact procedural kNN relation; adjacency is not allocated until an
explicit materialization or a compiler-selected consumer requires it.
Omitting `candidates` creates self-kNN and defaults `exclude_self=True`;
passing candidates creates a bipartite query→candidate relation. Example:
[Exact kNN feeding MessagePassing](examples/dynamic-relations.md#exact-knn-feeding-messagepassing).
Currently Euclidean-only: passing `metric=` is not supported. For cosine kNN,
normalize nonzero queries and candidates first; see [distance metrics](examples/dynamic-relations.md#distance-metrics).

### Graph.open(path, *, device="cpu") { #graphopen }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `path` | <code>str &#124; os.PathLike</code> | Filesystem path, not a URL; parent and overwrite rules are described below. |
| `device` | <code>str &#124; tg.Device &#124; None</code> | Execution device; None uses the constructor's documented default. |

Return type: `tg.Graph`.

Before running this example, configure the required device, files or distributed environment as described below.

??? example "Minimal usage"

    ```python
    import tiga as tg
    graph = tg.Graph.open('saved-graph.gfg')
    ```

<!-- typed-contract:end -->

Opens a persistent graph without eagerly loading its CSR arrays; the result
is an ordinary `Graph` backed by paged storage that reads only the manifest
until a bounded row range is requested. `tg.load(path, *, device="cpu")` is
the module-level alias. Example:
[A disk-resident giant graph](examples/distributed-memory.md#a-disk-resident-giant-graph).

### tg.save(graph, path, *, fields=None) { #gfsave }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `graph` | <code>tg.Graph</code> | Relation consumed by this call; device and entity counts must match fields. |
| `path` | <code>str &#124; os.PathLike</code> | Filesystem path, not a URL; parent and overwrite rules are described below. |
| `fields` | <code>Mapping[str, Mapping[str, tg.Tensor]] &#124; None</code> | Optional src/dst/edge native field payloads for a graph snapshot. |

Return type: `None`.

Before running this example, configure the required device, files or distributed environment as described below.

??? example "Minimal usage"

    ```python
    import torch
    import tiga as tg
    row_ptr = torch.tensor([0, 2, 3, 3], dtype=torch.int64)
    col_idx = torch.tensor([0, 1, 1], dtype=torch.int64)
    graph = tg.Graph.from_csr(row_ptr, col_idx, num_src=2)
    tg.save(graph, 'new-graph.gfg')
    ```

<!-- typed-contract:end -->

Persists a static CSR graph in the versioned `.gfg` format (manifest plus
`row_ptr.bin`/`col_idx.bin`); `tg.load` reopens it. `fields={"src": {...},
"dst": {...}, "edge": {...}}` additionally writes node/edge fields as
fixed-width rows so they page from disk alongside the topology.

### Graph.fields(role, *, requires_grad=None) { #graphfields }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `role` | <code>Literal[&#x27;src&#x27;, &#x27;dst&#x27;, &#x27;edge&#x27;]</code> | Role whose saved fields should be attached; paged graphs only. |
| `requires_grad` | <code>bool &#124; None</code> | Native leaf differentiation flag; None (default) is treated as False. |

Return type: `dict[str, tg.Tensor]`.

Before running this example, configure the required device, files or distributed environment as described below.

??? example "Minimal usage"

    ```python
    import tiga as tg
    graph = tg.load('saved-graph.gfg')
    source_fields = graph.fields('src', requires_grad=True)
    ```

<!-- typed-contract:end -->

Returned handles inherit the Graph's logical device and retain the backing reader.
Attachment does not allocate the complete field. CPU paged execution supports VJP;
CUDA paged execution currently supports forward only. See [memory limits](memory.md#large-graphs-current-boundary).

On a `paged_csr` graph saved with `fields=`, returns the per-role mapping of
lazily-read field shells (full shape, payload on disk) to pass straight into
a kernel call. `requires_grad=True` makes the shells differentiable; CPU paged
execution then supports both forward and `tg.autograd.grad`. CUDA paged execution
rejects gradient-requiring fields before launching work.

### graph.halo(mesh, *, partition=None, depth="auto") { #graphhalo }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `mesh` | <code>tg.DeviceMesh</code> | Logical device mesh; does not start worker processes. |
| `partition` | <code>tg.ByDestination &#124; None</code> | Destination partition policy; None uses the default policy. |
| `depth` | <code>int &#124; Literal[&#x27;auto&#x27;]</code> | Non-negative halo depth or compiler inference. |

Return type: `tg.Graph`.

??? example "Minimal usage"

    ```python
    import torch
    import tiga as tg
    row_ptr = torch.tensor([0, 2, 3, 3], dtype=torch.int64)
    col_idx = torch.tensor([0, 1, 1], dtype=torch.int64)
    graph = tg.Graph.from_csr(row_ptr, col_idx, num_src=2)
    placed = graph.halo(tg.DeviceMesh('cpu', 2))
    ```

<!-- typed-contract:end -->

Returns the same logical `Graph` type with owned/ghost requirements;
`partition` defaults to `tg.ByDestination()`. Declarative only: communication
is inserted below the user kernel as typed pack, exchange, unpack, interior
and boundary tasks. Example:
[Two-process halo exchange](examples/distributed-memory.md#two-process-halo-exchange).

### graph.paged_rows(begin, end) { #graphpaged_rows }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `begin / end` | <code>int</code> | Half-open destination-row range within the paged graph. |

Return type: `tuple[tuple[int, ...], tuple[int, ...]]`.

Before running this example, configure the required device, files or distributed environment as described below.

??? example "Minimal usage"

    ```python
    import tiga as tg
    graph = tg.load('saved-graph.gfg')
    row_ptr, col_idx = graph.paged_rows(0, 2)
    ```

<!-- typed-contract:end -->

Compiler/runtime hook returning one bounded CSR destination tile; valid only
on a `paged_csr` graph opened with `tg.load`.

### graph.resolve_csr() { #graphresolve_csr }

<!-- typed-contract:start -->

No explicit parameters (`self` omitted for instance methods).

Return type: `tuple[torch.Tensor | tg.Tensor, torch.Tensor | tg.Tensor]`.

??? example "Minimal usage"

    ```python
    import torch
    import tiga as tg
    row_ptr = torch.tensor([0, 2, 3, 3], dtype=torch.int64)
    col_idx = torch.tensor([0, 1, 1], dtype=torch.int64)
    graph = tg.Graph.from_csr(row_ptr, col_idx, num_src=2)
    result = graph.resolve_csr()
    ```

<!-- typed-contract:end -->

Explicitly materializes and returns `(row_ptr, col_idx)` — an inspection or
debug request, not part of normal compiler consumption.

### graph.transpose() { #graphtranspose }

<!-- typed-contract:start -->

No explicit parameters (`self` omitted for instance methods).

Return type: `tg.Graph`.

??? example "Minimal usage"

    ```python
    import torch
    import tiga as tg
    row_ptr = torch.tensor([0, 2, 3, 3], dtype=torch.int64)
    col_idx = torch.tensor([0, 1, 1], dtype=torch.int64)
    graph = tg.Graph.from_csr(row_ptr, col_idx, num_src=2)
    result = graph.transpose()
    ```

<!-- typed-contract:end -->

Returns one materialized snapshot with endpoint roles swapped.

### graph.explain() { #graphexplain }

<!-- typed-contract:start -->

No explicit parameters (`self` omitted for instance methods).

Return type: `str`.

??? example "Minimal usage"

    ```python
    import torch
    import tiga as tg
    row_ptr = torch.tensor([0, 2, 3, 3], dtype=torch.int64)
    col_idx = torch.tensor([0, 1, 1], dtype=torch.int64)
    graph = tg.Graph.from_csr(row_ptr, col_idx, num_src=2)
    result = graph.explain()
    ```

<!-- typed-contract:end -->

Renders a one-line summary: origin, lifecycle, realization, entity and edge
counts, device, and halo/paged backing when present. Example:
[A disk-resident giant graph](examples/distributed-memory.md#a-disk-resident-giant-graph).

### Graph introspection properties { #graph-properties }

<!-- typed-contract:start -->

No explicit parameters (`self` omitted for instance methods).

Return type: `GraphSchema / tg.Device / int | None / GraphPlacement | None / bool`.

??? example "Minimal usage"

    ```python
    import torch
    import tiga as tg
    row_ptr = torch.tensor([0, 2, 3, 3], dtype=torch.int64)
    col_idx = torch.tensor([0, 1, 1], dtype=torch.int64)
    graph = tg.Graph.from_csr(row_ptr, col_idx, num_src=2)
    print(graph.schema, graph.device, graph.num_edges)
    print(graph.placement, graph.is_distributed)
    ```

<!-- typed-contract:end -->

`graph.schema` (the `GraphSchema` dataclass), `graph.device`,
`graph.num_edges` (`None` for procedural relations until realized),
`graph.placement` and `graph.is_distributed`.

### Planner statistics { #graph-planner-statistics }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `samples` | <code>int</code> | source_index_span_ratio only: sampled row count, default 4096. |

Return type: `Planner statistics (method-specific)`.

??? example "Minimal usage"

    ```python
    import torch
    import tiga as tg
    row_ptr = torch.tensor([0, 2, 3, 3], dtype=torch.int64)
    col_idx = torch.tensor([0, 1, 1], dtype=torch.int64)
    graph = tg.Graph.from_csr(row_ptr, col_idx, num_src=2)
    print(graph.degree_bounds())
    print(graph.degree_statistics())
    print(graph.fixed_degree())
    ```

<!-- typed-contract:end -->

`graph.degree_bounds()`, `graph.degree_statistics()`,
`graph.degree_histogram()`, `graph.fixed_degree()` and
`graph.source_index_span_ratio(samples=4096)` expose the physical statistics
the planner consumes; results are cached per snapshot.

## MessagePassing { #messagepassing }

Subclass `MessagePassing`, set the class attribute `reducer`, and implement
`edge(src, dst, edge, **params)`; `node(dst, aggregate, **params)` is
optional and defaults to the identity. Node fields bind through
`src={...}`/`dst={...}` endpoint-role mappings, or through one `ndata={...}`
mapping on homogeneous graphs. Guide:
[Message passing](message-passing.md).

### program(graph=..., src=None, dst=None, edge=None, ndata=None, **params) { #messagepassing-call }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `graph` | <code>tg.Graph</code> | Relation consumed by this call; device and entity counts must match fields. |
| `src` | <code>Mapping[str, torch.Tensor &#124; tg.Tensor] &#124; None</code> | Source fields, leading dimension num_src; use {} for an unused role. |
| `dst` | <code>Mapping[str, torch.Tensor &#124; tg.Tensor] &#124; None</code> | Destination fields, leading dimension num_dst; use {} for an unused role. |
| `edge` | <code>Mapping[str, torch.Tensor &#124; tg.Tensor] &#124; None</code> | Edge fields in graph edge order; None means no explicit edge fields. |
| `ndata` | <code>Mapping[str, torch.Tensor &#124; tg.Tensor] &#124; None</code> | Homogeneous-node shorthand, mutually exclusive with src/dst. |
| `**params` | <code>object</code> | Named scalar/captured arguments declared by edge/node; not arbitrary Python objects. |

Return type: `torch.Tensor | tg.Tensor`.

??? example "Minimal usage"

    ```python
    import torch
    import tiga as tg
    row_ptr = torch.tensor([0, 2, 3, 3], dtype=torch.int64)
    col_idx = torch.tensor([0, 1, 1], dtype=torch.int64)
    graph = tg.Graph.from_csr(row_ptr, col_idx, num_src=2)
    class Sum(tg.MessagePassing):
        reducer = tg.sum()
        def edge(self, src, dst, edge):
            return src.x
    program = Sum()
    x = torch.tensor([2., 3.], requires_grad=True)
    out = program(graph=graph, src={'x': x}, dst={})
    out.sum().backward()
    assert x.grad.tolist() == [1., 2.]
    ```

<!-- typed-contract:end -->

Calling a kernel instance selects an execution path. Default Torch fields return
an ordinary `torch.Tensor` participating in Torch autograd; all-native fields
return `tg.Tensor`. Native CSR/radius calls capture a deferred Tensor expression;
observation selects JIT or oracle execution according to the backend policy.

| Argument | Contract |
|---|---|
| `graph` | Supported relation; constructor availability is not backend coverage |
| `src`, `dst` | Both mappings are required unless `ndata` is used; pass `{}` for an unused role |
| `edge` | Edge-field mapping; leading dimension equals edge count |
| `ndata` | Shared node-field mapping for homogeneous graphs only; cannot combine with `src/dst` |
| `**params` | Additional parameters named by the edge/node method signatures |

Node fields must match their role's entity count and graph device. Missing
role mappings or mixed `ndata/src/dst` raise `TypeError`; incompatible
field sizes/devices or bipartite `ndata` raise `ValueError`. `reducer` must
be an instance such as `tg.sum()`, not the factory `tg.sum`. The result's
leading dimension is the destination count. Complete [CSR recipe](api-examples.md#message-passing).

On a `paged_csr` graph the call streams
bounded destination-row pages and accepts three extra call-time parameters:
`page_rows` (page height, default 100 000 or `TIGA_PAGED_PAGE_ROWS`),
`prefetch=True` (overlap page reads with compute) and `prefetch_depth`
(pages read ahead concurrently, default 2 or
`TIGA_PAGED_PREFETCH_DEPTH`).
Examples:
[GCN aggregation](examples/message-passing.md#gcn-aggregation),
[A disk-resident giant graph](examples/distributed-memory.md#a-disk-resident-giant-graph).

### tg.runtime.auto_offload(ram) { #gfruntimeautoffload }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `ram` | <code>int</code> | Non-negative byte threshold for graph CSR offload; use tg.execution for IEC strings. |

Return type: `context manager`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    with tg.runtime.auto_offload(1024 ** 3):
        graph = tg.Graph.regular(4, 2)
    ```

<!-- typed-contract:end -->

Context manager: while active, every explicit CSR builder
(`Graph.from_csr`, `Graph.stencil`, `Graph.cat`, ...) persists the topology
as `.gfg` and returns a `paged_csr` graph when the CSR exceeds `ram` bytes,
so kernel calls page it with prefetch automatically. This legacy value is a per-topology threshold, not an aggregate allocation/RSS limit; construction happens before the offload decision. Use [execution policy](memory.md) for managed allocation accounting. The env var
`TIGA_GRAPH_RAM_BUDGET` sets a process-wide default; the context
manager wins. Offloads land in a per-process directory cleaned up at exit,
or under `TIGA_SPILL_DIR` when set. CPU graphs only; fields can join the
topology on disk (`tg.save(..., fields=...)`), and both forward and backward
run paged. Example:
[Automatic offload under a RAM budget](examples/distributed-memory.md#automatic-graph-offload).

### program.reference(**kwargs) { #messagepassingreference }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `**kwargs` | <code>Mapping[str, object]</code> | Same graph, field bindings and UDF arguments as a regular call; uses Torch. |

Return type: `torch.Tensor`.

??? example "Minimal usage"

    ```python
    import torch
    import tiga as tg
    row_ptr = torch.tensor([0, 2, 3, 3], dtype=torch.int64)
    col_idx = torch.tensor([0, 1, 1], dtype=torch.int64)
    graph = tg.Graph.from_csr(row_ptr, col_idx, num_src=2)
    class Sum(tg.MessagePassing):
        reducer = tg.sum()
        def edge(self, src, dst, edge):
            return src.x
    program = Sum()
    x = torch.tensor([2., 3.], requires_grad=True)
    out = program.reference(graph=graph, src={'x': x}, dst={})
    ```

<!-- typed-contract:end -->

Executes the semantic implementation explicitly through the Torch
interop oracle; it is a correctness reference, not a native backend.

### program.prepare(*, graph, src=None, dst=None, edge=None, ndata=None, **params) { #messagepassingprepare }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `graph` | <code>tg.Graph</code> | Relation consumed by this call; device and entity counts must match fields. |
| `src` | <code>Mapping[str, torch.Tensor &#124; tg.Tensor] &#124; None</code> | Source fields, leading dimension num_src; use {} for an unused role. |
| `dst` | <code>Mapping[str, torch.Tensor &#124; tg.Tensor] &#124; None</code> | Destination fields, leading dimension num_dst; use {} for an unused role. |
| `edge` | <code>Mapping[str, torch.Tensor &#124; tg.Tensor] &#124; None</code> | Edge fields in graph edge order; None means no explicit edge fields. |
| `ndata` | <code>Mapping[str, torch.Tensor &#124; tg.Tensor] &#124; None</code> | Homogeneous-node shorthand, mutually exclusive with src/dst. |
| `**params` | <code>object</code> | Named scalar/captured arguments declared by edge/node; not arbitrary Python objects. |

Return type: `Callable[[], Tensor]`.

Before running this example, configure the required device, files or distributed environment as described below.

??? example "Minimal usage"

    ```python
    # CUDA scalar CSR only; graph and native fields must already be validated.
    prepared = program.prepare(graph=graph, src={'x': x}, dst={})
    out = prepared()
    ```

<!-- typed-contract:end -->

Optionally freezes one validated scalar-CSR CUDA binding into a
zero-argument hot submission. Ordinary execution remains lazy JIT and never
requires this call.

### program.explain() { #messagepassingexplain }

<!-- typed-contract:start -->

No explicit parameters (`self` omitted for instance methods).

Return type: `str`.

??? example "Minimal usage"

    ```python
    import torch
    import tiga as tg
    row_ptr = torch.tensor([0, 2, 3, 3], dtype=torch.int64)
    col_idx = torch.tensor([0, 1, 1], dtype=torch.int64)
    graph = tg.Graph.from_csr(row_ptr, col_idx, num_src=2)
    class Sum(tg.MessagePassing):
        reducer = tg.sum()
        def edge(self, src, dst, edge):
            return src.x
    program = Sum()
    x = torch.tensor([2., 3.], requires_grad=True)
    out = program(graph=graph, src={'x': x}, dst={})
    print(program.explain())
    ```

<!-- typed-contract:end -->

Renders planner/lowering/cache information for the last variant as text.

### program.diagnostics { #messagepassingdiagnostics }

<!-- typed-contract:start -->

No explicit parameters (`self` omitted for instance methods).

Return type: `tuple[AnalysisFinding, ...]`.

??? example "Minimal usage"

    ```python
    import torch
    import tiga as tg
    row_ptr = torch.tensor([0, 2, 3, 3], dtype=torch.int64)
    col_idx = torch.tensor([0, 1, 1], dtype=torch.int64)
    graph = tg.Graph.from_csr(row_ptr, col_idx, num_src=2)
    class Sum(tg.MessagePassing):
        reducer = tg.sum()
        def edge(self, src, dst, edge):
            return src.x
    program = Sum()
    x = torch.tensor([2., 3.], requires_grad=True)
    out = program(graph=graph, src={'x': x}, dst={})
    print(program.diagnostics)
    ```

<!-- typed-contract:end -->

Property returning typed `AnalysisFinding` records — stage, disposition
(`accepted`/`rejected`/`warning`/`unknown`/`remark`) and optional resource,
target contract, estimate and suggested action. `explain()` is their
human-readable rendering.

### program.schedules { #messagepassingschedules }

<!-- typed-contract:start -->

No explicit parameters (`self` omitted for instance methods).

Return type: `tuple[MachineSchedule, ...]`.

??? example "Minimal usage"

    ```python
    import torch
    import tiga as tg
    row_ptr = torch.tensor([0, 2, 3, 3], dtype=torch.int64)
    col_idx = torch.tensor([0, 1, 1], dtype=torch.int64)
    graph = tg.Graph.from_csr(row_ptr, col_idx, num_src=2)
    class Sum(tg.MessagePassing):
        reducer = tg.sum()
        def edge(self, src, dst, edge):
            return src.x
    program = Sum()
    x = torch.tensor([2., 3.], requires_grad=True)
    out = program(graph=graph, src={'x': x}, dst={})
    print(program.schedules)
    ```

<!-- typed-contract:end -->

Property returning typed `MachineSchedule` records extracted by the native
compiler from verified `gf.kernel` IR: schedule family, row/neighbor tile,
subgroup count, pipeline depth, named resources, execution roles,
producer/consumer handoffs and admitted instruction classes.
`pipeline_stages == 1` is an explicit negative result — no
compiler-controlled asynchronous pipeline was admitted — so provider
instruction reordering must not be presented as proven overlap.

### program.ir(stage="domain") { #messagepassingir }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `stage` | <code>str</code> | IR artifact stage; default domain. Missing artifacts raise KeyError. |

Return type: `str`.

??? example "Minimal usage"

    ```python
    import torch
    import tiga as tg
    row_ptr = torch.tensor([0, 2, 3, 3], dtype=torch.int64)
    col_idx = torch.tensor([0, 1, 1], dtype=torch.int64)
    graph = tg.Graph.from_csr(row_ptr, col_idx, num_src=2)
    class Sum(tg.MessagePassing):
        reducer = tg.sum()
        def edge(self, src, dst, edge):
            return src.x
    program = Sum()
    x = torch.tensor([2., 3.], requires_grad=True)
    out = program(graph=graph, src={'x': x}, dst={})
    print(program.ir('domain'))
    ```

<!-- typed-contract:end -->

Returns the semantic plan for `domain` (not necessarily parseable MLIR),
or an available artifact for `iter`, `kernel`, `task` or a provider stage.
Stage aliases such as `"gf.iter"` are inspection keys, not operation names.
Missing artifacts raise `KeyError`. Guide:
[Inspection boundaries](compiler-pipeline.md#inspecting-a-compiled-program).

### program.code(kind="ptx") { #messagepassingcode }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `kind` | <code>str</code> | Generated artifact kind, default ptx; requires a matching compiled execution. |

Return type: `str`.

Before running this example, configure the required device, files or distributed environment as described below.

??? example "Minimal usage"

    ```python
    # After a CUDA-compiled program call:
    print(program.code('ptx'))
    ```

<!-- typed-contract:end -->

Returns code artifacts such as `ttgir`, `llir` or `ptx` for the last
variant.

### program.cache_info { #messagepassingcache_info }

<!-- typed-contract:start -->

No explicit parameters (`self` omitted for instance methods).

Return type: `dict[str, int]`.

??? example "Minimal usage"

    ```python
    import torch
    import tiga as tg
    row_ptr = torch.tensor([0, 2, 3, 3], dtype=torch.int64)
    col_idx = torch.tensor([0, 1, 1], dtype=torch.int64)
    graph = tg.Graph.from_csr(row_ptr, col_idx, num_src=2)
    class Sum(tg.MessagePassing):
        reducer = tg.sum()
        def edge(self, src, dst, edge):
            return src.x
    program = Sum()
    x = torch.tensor([2., 3.], requires_grad=True)
    out = program(graph=graph, src={'x': x}, dst={})
    print(program.cache_info)
    ```

<!-- typed-contract:end -->

Property returning kernel variant lookup hit/miss/variant counts. On the
native Tensor path these describe capture bindings, even before realization;
they are not compilation counts. After realization, `Tensor.execution`
reports the actual backend and native executable `cache_hit`/`compile_ms`.

### program.last_variant and program.variants { #messagepassinglast_variant }

<!-- typed-contract:start -->

No explicit parameters (`self` omitted for instance methods).

Return type: `CompiledVariant (variants: tuple[CompiledVariant, ...])`.

??? example "Minimal usage"

    ```python
    import torch
    import tiga as tg
    row_ptr = torch.tensor([0, 2, 3, 3], dtype=torch.int64)
    col_idx = torch.tensor([0, 1, 1], dtype=torch.int64)
    graph = tg.Graph.from_csr(row_ptr, col_idx, num_src=2)
    class Sum(tg.MessagePassing):
        reducer = tg.sum()
        def edge(self, src, dst, edge):
            return src.x
    program = Sum()
    x = torch.tensor([2., 3.], requires_grad=True)
    out = program(graph=graph, src={'x': x}, dst={})
    print(program.last_variant)
    ```

<!-- typed-contract:end -->

The `CompiledVariant` record of the most recent call, or of every cached
specialization: backend, provider, lowering, passes, artifacts, diagnostics
and schedules. `last_variant` raises before the first call.

## Reducers { #reducers }

A reducer is executable compiler input, not an eager tensor operation. The
[reducers guide](reducers.md) explains the algebra contract, attributes and
lowering selection in prose.

### tg.sum(*, identity=0, deterministic=False) { #gfsum }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `identity` | <code>int &#124; float</code> | Additive identity; default 0. |
| `deterministic` | <code>bool</code> | Request fixed reduction ordering; default False, subject to backend support. |

Return type: `SumReducer`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    reducer = tg.sum()
    ```

<!-- typed-contract:end -->

Associative, commutative additive reducer. Example:
[GCN aggregation](examples/message-passing.md#gcn-aggregation).

### tg.mean(*, deterministic=False) { #gfmean }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `deterministic` | <code>bool</code> | Request fixed reduction ordering; default False, subject to backend support. |

Return type: `MeanReducer`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    reducer = tg.mean()
    ```

<!-- typed-contract:end -->

Neighbor-mean reducer over built-in tuple state `(sum, count)`; a degree-0
row yields NaN. Guide: [tg.mean()](reducers.md#gfmean).

### tg.prod(*, deterministic=False) { #gfprod }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `deterministic` | <code>bool</code> | Request fixed reduction ordering; default False, subject to backend support. |

Return type: `ProductReducer`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    reducer = tg.prod()
    ```

<!-- typed-contract:end -->

Multiplicative reducer, promoted to a zero-safe product IR; a degree-0 row
yields the identity 1. Example:
[Product of edge gates](reducers.md#product-of-edge-gates).

### tg.online_softmax(*, accumulation_dtype=None, deterministic=False, block_prune_threshold=None) { #gfonline_softmax }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `accumulation_dtype` | <code>tg.DType &#124; torch.dtype &#124; None</code> | Accumulation precision hint; None uses the provider default. |
| `deterministic` | <code>bool</code> | Request fixed reduction ordering; default False, subject to backend support. |
| `block_prune_threshold` | <code>float &#124; None</code> | None is exact; (0,1] explicitly enables approximate tile pruning on supported CUDA forward paths. |

Return type: `OnlineSoftmaxReducer`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    reducer = tg.online_softmax()
    ```

<!-- typed-contract:end -->

Stable multi-value streaming reducer; the edge region returns
`reducer(score, value)` (an `OnlineSoftmaxItem`). `block_prune_threshold` in
`(0, 1]` is a semantic approximation policy for dense streaming tiles, not a
scheduling hint. Examples:
[Full attention, no mask](examples/attention.md#full-attention-no-mask),
[Tile-pruned sparse attention](examples/attention.md#tile-pruned-sparse-attention).

### class tg.Reducer { #gfreducer }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `deterministic` | <code>bool</code> | Request fixed reduction ordering; default False, subject to backend support. |

Return type: `tg.Reducer subclass instance`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    class Add(tg.Reducer):
        associative = True
        commutative = True
        def identity(self): return 0.
        def combine(self, left, right): return left + right
    reducer = Add()
    ```

<!-- typed-contract:end -->

User-defined reducer base class: subclass and implement `identity()`,
`lift(*messages)`, `combine(left, right)` and `finalize(state)`; state may
be a scalar or a tuple. Class attributes `associative`, `commutative` and
`deterministic` declare the algebra — parallel lowering requires an explicit
associativity declaration, which Tiga never proves from Python source.
Example:
[User-defined reducer](examples/message-passing.md#user-defined-reducer);
guide: [Inventing a reducer](reducers.md#inventing-your-own).

### reducer(*messages) { #reducer-call }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `*messages` | <code>Tensor expressions</code> | One or more edge-local inputs for lift; this packages messages, not an immediate reduction. |

Return type: `ReducerCall | OnlineSoftmaxItem`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    class Attention(tg.MessagePassing):
        reducer = tg.online_softmax()
        def edge(self, src, dst, edge):
            return self.reducer(edge.score, src.value)
    ```

<!-- typed-contract:end -->

Calling a reducer inside `edge()` binds it to one or more edge-local
messages as a staged `ReducerCall`.

### reducer.mlir(*, message_dtypes=None, symbol=None) { #reducermlir }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `message_dtypes` | <code>Sequence[tg.DType] &#124; None</code> | One native dtype per message; None uses one float32 message. |
| `symbol` | <code>str &#124; None</code> | IR symbol override; None uses the reducer name. |

Return type: `str`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    print(tg.sum().mlir(message_dtypes=(tg.float32,)))
    ```

<!-- typed-contract:end -->

Captures the four regions as a verified native `gf.reducer` op and returns
the MLIR text; `message_dtypes` defaults to a single `float32` message.
Guide: [Inspecting a reducer](reducers.md#inspecting-a-reducer).

??? info "Automatic reducer VJP coverage"

    Additive/stable algebras and captured associative scalar reducers lower to
    balanced or deterministic trees through CPU LLVM and CUDA TTIR. Product is
    promoted to zero-safe CSR product/VJP IR. Empty/ragged rows and ordered
    non-commutative tuple state are covered; unregistered reducer shapes remain
    correctness-only claims.

## nn { #nn }

### tg.nn.trace(module, *, block_e=None, num_warps=None) { #gfnntrace }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `module` | <code>torch.nn.Module</code> | Supported traceable module; original parameters retain Torch autograd connections. |
| `block_e` | <code>int &#124; None</code> | Optional tile size, power of two in [16,1024]. |
| `num_warps` | <code>int &#124; None</code> | Optional provider launch warp count; None uses lowering defaults. |

Return type: `TracedModule`.

??? example "Minimal usage"

    ```python
    import torch
    import tiga as tg
    module = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh())
    traced = tg.nn.trace(module)
    y = traced(torch.ones(2, 3))
    y.sum().backward()
    ```

<!-- typed-contract:end -->

Wraps a `torch.nn` module so it can be called inside an edge UDF: eager with
Torch tensors (inputs concatenated along the feature dimension) or captured
into a compilable subgraph for the fused edge-NN tile lowering. `block_e`
(edge-tile size, a power of two in [16, 1024]) and `num_warps` are launch
geometry only and never change numerics; the defaults are per-lowering,
128/4 for edge-centric sum tiles and 16/1 for row-centric attention tiles.
Examples:
[GAT edge attention with an nn score](examples/attention.md#gat-edge-attention-with-an-nn-score),
[Edge nn modules with a fused tile kernel](examples/programs-and-interop.md#edge-nn-modules-with-a-fused-tile-kernel);
guide: [Edge nn modules (CUDA, torch interop)](message-passing.md#edge-nn-modules-cuda-torch-interop).

## Control and jit { #control }

Structured compiler control flow over Tiga semantic values: the body
is captured once and lowered to `gf_control.repeat` / `gf_control.while`
ops, not executed as a Python loop.

### @tg.jit or @tg.jit(max_iterations=k) { #gfjit }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `function` | <code>Callable &#124; None</code> | Decorated function; source must be available for AST capture. |
| `max_iterations` | <code>int &#124; None</code> | Finite bound for while loops; None is allowed when no while loop is captured. |

Return type: `Callable`.

Before running this example, configure the required device, files or distributed environment as described below.

??? example "Minimal usage"

    ```python
    import tiga as tg
    @tg.jit
    def decay(x):
        for i in range(3):
            x = x * 0.5
        return x
    print(decay(tg.tensor([8.])).tolist())
    ```

<!-- typed-contract:end -->

AST route that captures `for i in range(k)` as `gf_control.repeat` and a
bare `while cond:` as `gf_control.while` bounded by the decorator argument.
A leading `if cond: break` in a `for` body is an early exit. Loop-carried
variables are Tensors assigned before the loop; `continue`, mid-body
`break`, `while True`, non-range iteration and loop `else` fail closed.
MessagePassing UDF regions are never AST-transformed. `@tg.jit` also
activates the composition context: sibling MessagePassing calls in the
straight-line body capture as `GraphProgram` SSA leaves and fuse where
legal, leaves participate in tensor arithmetic, and kernel calls inside a
staged loop body keep per-iteration semantics (inlined, never registered
as top-level leaves). Leaves are differentiable when a captured field
requires gradients: reverse mode re-expands each leaf inline (per leaf,
unfused) and accumulates contributions onto shared inputs; unsupported
leaf forms fail closed. Examples:
[Fixed-iteration PageRank probe](examples/message-passing.md#fixed-iteration-pagerank-probe),
[Matrix-free FEM operator and solver loop](examples/solvers.md#matrix-free-fem-operator-and-solver-loop).

### tg.repeat(initial, body, *, iterations) { #gfrepeat }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `initial` | <code>tg.Tensor &#124; Sequence[tg.Tensor]</code> | Loop-carried native values; their shapes, dtypes and devices must remain unchanged. |
| `body` | <code>Callable[..., tg.Tensor &#124; Sequence[tg.Tensor]]</code> | Captured once; returns one value per carried state. |
| `iterations` | <code>int</code> | Non-negative fixed iteration count; bool is rejected. |

Return type: `tg.Tensor | tuple[tg.Tensor, ...]`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    y = tg.repeat(tg.tensor([8.]), lambda x: x * 0.5, iterations=3)
    assert y.tolist() == [1.]
    ```

<!-- typed-contract:end -->

Anonymous shorthand capturing a fixed-count loop with one or more
Tensor-carried states; `body` is traced once and must return one Tensor per
carried state with preserved shape, dtype and device. CPU lowering allocates
two reusable buffers per carried value.

### tg.while_loop(initial, condition, body, *, max_iterations) { #gfwhile_loop }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `initial` | <code>tg.Tensor &#124; Sequence[tg.Tensor]</code> | Loop-carried native values; their shapes, dtypes and devices must remain unchanged. |
| `condition` | <code>Callable[..., tg.Tensor]</code> | Captured condition returning a scalar boolean Tensor. |
| `body` | <code>Callable[..., tg.Tensor &#124; Sequence[tg.Tensor]]</code> | Captured once; returns one value per carried state. |
| `max_iterations` | <code>int</code> | Non-negative finite loop bound, including zero. |

Return type: `tg.Tensor | tuple[tg.Tensor, ...]`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    y = tg.while_loop(tg.tensor(1.), lambda x: x < 4., lambda x: x + 1., max_iterations=8)
    assert y.tolist() == 4.
    ```

<!-- typed-contract:end -->

Anonymous shorthand capturing bounded data-dependent control without a host
scalar check; `condition` must return a rank-zero boolean Tensor and
`max_iterations` remains in IR as a finite resource guard. CPU lowers to
`scf.while`; CUDA fails closed until a provider loop plan exists.

### class tg.control.Repeat { #gfcontrolrepeat }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `initial` | <code>tg.Tensor &#124; Sequence[tg.Tensor]</code> | Loop-carried native values; their shapes, dtypes and devices must remain unchanged. |
| `iterations` | <code>int</code> | Non-negative fixed iteration count; bool is rejected. |

Return type: `tg.Tensor | tuple[tg.Tensor, ...]`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    class Decay(tg.control.Repeat):
        def body(self, x): return x * 0.5
    assert Decay()(tg.tensor([8.]), iterations=3).tolist() == [1.]
    ```

<!-- typed-contract:end -->

Builder form of the same `gf_control.repeat` op: subclass and override
`body`; captured constants are ordinary instance attributes. Calling the
instance as `kernel(initial, iterations=k)` traces `body` exactly once.

### class tg.control.While { #gfcontrolwhile }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `initial` | <code>tg.Tensor &#124; Sequence[tg.Tensor]</code> | Loop-carried native values; their shapes, dtypes and devices must remain unchanged. |
| `max_iterations` | <code>int</code> | Non-negative finite loop bound, including zero. |

Return type: `tg.Tensor | tuple[tg.Tensor, ...]`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    class Grow(tg.control.While):
        def condition(self, x): return x < 4.
        def body(self, x): return x + 1.
    assert Grow()(tg.tensor(1.), max_iterations=8).tolist() == 4.
    ```

<!-- typed-contract:end -->

Builder form of `gf_control.while`: subclass and override `condition` and
`body`; calling the instance as `kernel(initial, max_iterations=k)` applies
the same bounds as `tg.while_loop`.

Stationary solvers are grammar sugar over these primitives, not core API:
[examples/solvers.py](https://github.com/walkerchi/TIGA-lang/blob/main/examples/solvers.py)
provides `dot`, `vector_norm`, `richardson` and `cg`; see
[Linear solvers and control flow](examples/solvers.md).

## Distributed and halo { #distributed-and-halo }

### tg.DeviceMesh(device_type, shape, *, names=()) { #gfdevicemesh }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `device_type` | <code>str</code> | Logical device family such as cpu or cuda; does not initialize devices. |
| `shape` | <code>int &#124; tuple[int, ...]</code> | Positive mesh extents; product is rank count. |
| `names` | <code>tuple[str, ...]</code> | Optional axis names, one per mesh dimension. |

Return type: `tg.DeviceMesh`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    mesh = tg.DeviceMesh('cpu', (2,), names=('workers',))
    ```

<!-- typed-contract:end -->

Describes logical devices (an int or tuple `shape`) without initializing a
process group. Deployment binds it to Torch `DeviceMesh`, MPI, NCCL/RCCL or
a vendor communicator. Example:
[Two-process halo exchange](examples/distributed-memory.md#two-process-halo-exchange).

### tg.ByDestination(mesh_axis=0, balance="auto") { #gfbydestination }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `mesh_axis` | <code>int &#124; str</code> | Mesh axis index or name. |
| `balance` | <code>Literal[&#x27;auto&#x27;, &#x27;edges&#x27;, &#x27;entities&#x27;]</code> | Partition balancing objective. |

Return type: `tg.ByDestination`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    partition = tg.ByDestination(balance='edges')
    ```

<!-- typed-contract:end -->

Partition policy assigning destination entities and final reducer state to
mesh shards; `balance` is `"edges"`, `"entities"` or `"auto"`.

### tg.GraphPlacement(mesh, partition, halo_depth) { #gfgraphplacement }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `mesh` | <code>tg.DeviceMesh</code> | Logical device topology, not a running communicator. |
| `partition` | <code>tg.ByDestination</code> | Destination ownership policy. |
| `halo_depth` | <code>int &#124; Literal[&#x27;auto&#x27;]</code> | Non-negative depth or compiler inference. |

Return type: `tg.GraphPlacement`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    placement = tg.GraphPlacement(tg.DeviceMesh('cpu', 2), tg.ByDestination(), 'auto')
    ```

<!-- typed-contract:end -->

Logical ownership and ghost requirements attached to a Graph snapshot by
`graph.halo(...)`; `halo_depth` is a non-negative integer or `"auto"`.

### tg.HaloMap { #gfhalomap }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `rank` | <code>int</code> | Local rank in [0, world_size). |
| `world_size` | <code>int</code> | Positive rank count. |
| `owned_begin / owned_end` | <code>int</code> | Half-open interval of globally numbered owned entities. |
| `ghost_ids` | <code>tuple[int, ...]</code> | Sorted external source IDs required by owned rows. |
| `receive_from / send_to` | <code>tuple[tuple[int, tuple[int, ...]], ...]</code> | Peer rank and ordered global entity IDs. |
| `bytes_for: itemsize / trailing_elements` | <code>int</code> | Positive bytes per scalar and scalars per entity (default 1). |

Return type: `tg.HaloMap; bytes_for → int`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    halos = tg.collective_halo_maps([0, 1, 2], [1, 0], num_entities=2, world_size=2)
    print(halos[0].ghost_ids, halos[0].bytes_for(4))
    ```

<!-- typed-contract:end -->

Concrete per-rank owner/ghost map: `owned_begin`/`owned_end`, `ghost_ids`,
`receive_from` and `send_to`, with `owned_entities`, `ghost_entities` and
`bytes_for(itemsize, trailing_elements=1)` helpers.

### tg.derive_halo_map(row_ptr, col_idx, *, num_entities, world_size, rank, peer_requests=None) { #gfderive_halo_map }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `row_ptr / col_idx` | <code>Sequence[int]</code> | Global CSR row boundaries and source IDs as host integer sequences. |
| `num_entities` | <code>int</code> | Global entity count for the homogeneous CSR relation. |
| `world_size` | <code>int</code> | Positive rank count. |
| `rank` | <code>int</code> | Local rank in [0, world_size). |
| `peer_requests` | <code>Sequence[Sequence[int]] &#124; None</code> | IDs requested by each rank; None leaves send_to empty. |

Return type: `tg.HaloMap`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    halo = tg.derive_halo_map([0, 1, 2], [1, 0], num_entities=2, world_size=2, rank=0)
    ```

<!-- typed-contract:end -->

Derives exact ghosts from owned CSR rows without a framework dependency;
`peer_requests[p]` lists the locally owned entity IDs requested by peer `p`
and populates `send_to`. Guide:
[Ownership and ghosts, exactly](memory-and-distributed.md#ownership-and-ghosts-exactly).

### tg.collective_halo_maps(row_ptr, col_idx, *, num_entities, world_size) { #gfcollective_halo_maps }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `row_ptr / col_idx` | <code>Sequence[int]</code> | Global CSR row boundaries and source IDs as host integer sequences. |
| `num_entities` | <code>int</code> | Global entity count for the homogeneous CSR relation. |
| `world_size` | <code>int</code> | Positive rank count. |

Return type: `tuple[tg.HaloMap, ...]`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    halos = tg.collective_halo_maps([0, 1, 2], [1, 0], num_entities=2, world_size=2)
    ```

<!-- typed-contract:end -->

Creates mutually consistent receive/send maps for all ranks.

### tg.exchange_halo(halo, owned_data, *, element_bytes, transport) { #gfexchange_halo }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `halo` | <code>tg.HaloMap</code> | Per-rank ownership and communication map. |
| `owned_data` | <code>bytes &#124; bytearray &#124; memoryview</code> | Packed owned entities in global-ID order, starting at owned_begin. |
| `element_bytes` | <code>int</code> | Positive total bytes per entity, including feature dimensions. |
| `transport` | <code>NeighborTransport</code> | Bound transport implementing rank, world_size and exchange; must match the halo topology. |

Return type: `HaloBuffer`.

Before running this example, configure the required device, files or distributed environment as described below.

??? example "Minimal usage"

    ```python
    # Inside an initialized rank; peers must execute the matching exchange.
    received = tg.exchange_halo(halo, owned_data, element_bytes=4, transport=transport)
    ```

<!-- typed-contract:end -->

Packs, exchanges and unpacks one compiler-derived neighbor halo over a
transport and returns a `HaloBuffer`. `owned_data` is destination-owner
local byte storage ordered from `halo.owned_begin`; no framework tensor type
is involved.

### DistributedRuntime(transport, *, progress_threads=1) { #distributedruntime }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `transport` | <code>NeighborTransport</code> | Bound transport implementing rank, world_size and exchange; must match the halo topology. |
| `progress_threads` | <code>int</code> | Positive number of background progress workers; default 1. |

Return type: `tg.DistributedRuntime`.

Before running this example, configure the required device, files or distributed environment as described below.

??? example "Minimal usage"

    ```python
    # transport is a deployment-bound NeighborTransport.
    import tiga as tg
    with tg.DistributedRuntime(transport) as runtime:
        output = program(graph=placed_graph, src={'x': local_x}, dst={})
    ```

<!-- typed-contract:end -->

Owns asynchronous communication progress below graph kernels. Used as a
context manager it becomes the active runtime; calling a distributed graph
without one fails closed. `runtime.exchange_halo(halo, owned_data, *,
element_bytes)` starts halo progress and returns a provider-neutral
completion; `runtime.last_execution_trace` reports timings from the latest
automatic sharded execution.

Distributed execution uses one policy: complete halo exchange, then compute
all owned rows. No scheduling option is required. TCP uses host staging; NCCL
uses device buffers. The trace reports `host-staged-serialized` or
`device-direct-serialized`, with `measured_overlap_ms=0`.
TCP `timeout` must be finite and positive and bounds both handshakes and data
waits; allow extra time for cold JIT. See the [two-host validation commands](memory-and-distributed.md#two-host-cuda-sanity).

### DistributedRuntime.from_provider(name, *, progress_threads=1, **options) { #distributedruntimefrom_provider }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `name` | <code>str</code> | Registered transport: tcp, mpi, nccl, or an installed plugin. |
| `progress_threads` | <code>int</code> | Positive number of background progress workers; default 1. |
| `**options` | <code>provider-specific keyword arguments</code> | TCP: rank, world_size, host, port, bind, timeout. MPI: communicator, tag. NCCL requires a communicator/deployment setup; see the distributed guide. |
| `tcp/nccl: rank / world_size` | <code>int</code> | Required rank and positive group size; 0 <= rank < world_size. |
| `tcp: port` | <code>int</code> | Required rendezvous port; peers must agree and peer listener ports must be reachable. |
| `tcp: host / bind` | <code>str &#124; None / str</code> | host selects a rendezvous peer; None hosts it. bind defaults to the empty listening address. |
| `tcp: timeout` | <code>float</code> | Finite positive seconds for connection and data waits; default 30. |
| `mpi: communicator` | <code>mpi4py.MPI.Comm &#124; None</code> | None uses MPI.COMM_WORLD; mpi4py and an MPI runtime are required. |
| `mpi: tag` | <code>int</code> | Message tag on the supplied communicator; default 0. |
| `nccl: communicator_id` | <code>bytes &#124; None</code> | Shared 128-byte NCCL unique ID, distributed by the launcher; None is only valid for a one-rank group. |
| `nccl: device` | <code>str &#124; tg.Device</code> | Local CUDA device, default cuda:0; not a global rank identifier. |
| `nccl: library` | <code>str &#124; os.PathLike &#124; None</code> | Explicit libnccl path or automatic discovery. |

Return type: `tg.DistributedRuntime`.

Before running this example, configure the required device, files or distributed environment as described below.

??? example "Minimal usage"

    ```python
    # Run collectively with an MPI launcher and the mpi extra installed.
    import tiga as tg
    with tg.DistributedRuntime.from_provider('mpi') as runtime:
        print(runtime.transport.rank)
    ```

<!-- typed-contract:end -->

Binds a deployment transport plugin below the unchanged graph API:
`"mpi"` (mpi4py wrapper), `"tcp"` (dependency-free cross-machine sockets)
and `"nccl"` (grouped send/recv over `libnccl`) are registered by default;
third parties register through the `tiga.transport` entry-point group.
Examples:
[Two-process halo exchange](examples/distributed-memory.md#two-process-halo-exchange),
[Running across processes and machines](examples/distributed-memory.md#running-across-processes-and-machines);
guide: [Runtimes and transports](memory-and-distributed.md#runtimes-and-transports).

??? info "Current distributed execution boundary"

    CPU rank-local execution, reverse halo VJP, paged CSR shards and real
    two-process MPI transport are executable. CUDA native buffers use separate
    communication/compiler streams and event ordering. Two-host CUDA TCP/NCCL
    paths support forward and reverse halo VJP. Execution completes communication
    before computation; GPU-speed-aware repartitioning and RCCL execution are
    unsupported. See
    [Memory hierarchy and distributed execution](memory-and-distributed.md).

## Execution and storage { #spill-and-disk }

Complete, tested examples: [memory and storage](memory.md) and [API recipes](api-examples.md#storage).

### tg.execution(*, device="cpu", memory=None, spill_dir=None, page_rows=100000, prefetch_depth=2, eviction="error") { #tgexecution }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `device` | <code>str &#124; tg.Device &#124; None</code> | Execution device; None uses the constructor's documented default. |
| `memory` | <code>Mapping[str, int &#124; str] &#124; None</code> | Tier budgets in bytes or IEC sizes; omitted tiers are unbounded, not reserved. |
| `spill_dir` | <code>str &#124; os.PathLike &#124; None</code> | Temporary spill directory; required for explicit LRU policy. |
| `page_rows` | <code>int</code> | Positive destination rows per CPU CSR page; default 100000. |
| `prefetch_depth` | <code>int</code> | Positive in-flight page count; default 2. |
| `eviction` | <code>Literal[&#x27;error&#x27;, &#x27;lru&#x27;]</code> | Reject excess allocation or evict idle whole native Tensors. |

Return type: `ExecutionScope (context manager)`.

??? example "Minimal usage"

    ```python
    import tempfile
    import tiga as tg
    with tempfile.TemporaryDirectory() as directory:
        with tg.execution(memory={'ram': '1 MiB'}, spill_dir=directory) as run:
            x = tg.tensor([1., 2.])
            print(run.memory_report())
    ```

<!-- typed-contract:end -->

Returns a context manager. It selects the default device for `tg.tensor` / `tg.empty`, allocation limits and CPU paging defaults. Existing values do not migrate. Explicit device or paging call arguments override defaults.

| Parameter | Accepted values / default | Failure |
|---|---|---|
| `device` | Runtime device, e.g. `"cpu"` or `"cuda:0"`; CPU by default | Unsupported devices fail on use |
| `memory` | Mapping of `ram/host`, `device/hbm`, `host-pinned`, `nvme/ssd` to non-negative integer bytes or integer IEC sizes; absent tiers unlimited | `ValueError` for invalid sizes, duplicate aliases or non-allocatable tiers |
| `spill_dir` | Path for temporary spills and automatic graph offloads; managed temporary directory by default | Filesystem errors propagate |
| `page_rows` | Positive integer, default 100000 | `ValueError` for zero, negative, bool or non-integer |
| `prefetch_depth` | Positive integer, default 2 | Same validation |
| `eviction` | `"error"` rejects excess allocation; `"lru"` evicts idle native Tensors | `ValueError` for unknown policy or LRU without `spill_dir` |

`with tg.execution(...) as run:` rejects nesting with `RuntimeError`. Leaving the context resets defaults but does not destroy values. `run.memory_report()` returns `scope`, `excludes`, `live_bytes`, `peak_bytes`, `budget_bytes`, `automatic_tensor_eviction`, `eviction_policy`, `evictions` and `restores`. Reports use canonical tier keys; capacities are bytes. Only scope-created allocations are counted, including page-prefetch workers; this is not RSS accounting. LRU synchronously spills idle whole Tensors, cannot tile an oversized kernel working set, and rejects fixed-address `prepare()` calls. See the [runnable example and limitations](memory.md#automatic-eviction).

### Tensor.spill() { #tensorspill }

<!-- typed-contract:start -->

No explicit parameters (`self` omitted for instance methods).

Return type: `tg.Tensor`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    x.spill()
    assert not x.residency['resident']
    assert x.tolist() == [[1., 2.], [3., 4.]]
    ```

<!-- typed-contract:end -->

Returns the same Tensor after writing a temporary payload and dropping its own resident-buffer reference. Its shape, dtype, logical device, version and autograd history survive. Reads restore onto the original device. Existing views/launch owners may still retain the old allocation. Temporary files are removed on restore or collection. `MemoryError` means the temporary NVMe or restore tier budget was exceeded; calling spill twice without restoration raises `RuntimeError`.

### Tensor.to(device) { #tensorto }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `device` | <code>str &#124; tg.Device</code> | Destination device; same-device calls realize and return self. |

Return type: `tg.Tensor`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    host = x.to('cpu')
    ```

<!-- typed-contract:end -->

Returns a differentiable, realized copy on a different device without mutating the original. For the same device, realizes and returns the input itself. Destination allocation is budgeted; device/driver errors propagate. Copying is currently eager, not a scheduling annotation.

### Tensor.cpu() { #tensorcpu }

<!-- typed-contract:start -->

No explicit parameters (`self` omitted for instance methods).

Return type: `tg.Tensor`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    host = x.cpu()
    ```

<!-- typed-contract:end -->

Equivalent to `x.to("cpu")`. A CUDA input stays CUDA; the returned Tensor is CPU. A CPU input is realized in place.

### Tensor.residency { #tensorresidency }

<!-- typed-contract:start -->

No explicit parameters (`self` omitted for instance methods).

Return type: `dict[str, object]`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    print(x.residency)
    ```

<!-- typed-contract:end -->

Read-only status dictionary: `device` is the logical device string, `resident` means a buffer is attached, `backing` is `"snapshot"`, `"temporary-spill"` or `None`, and `bytes/version` describe the value. It is not a count of uniquely releasable bytes.

### Tensor.save(path, *, overwrite=False) / tg.save(tensor, path, *, overwrite=False) { #tensorsave }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `path` | <code>str &#124; os.PathLike</code> | Filesystem path, not a URL; parent and overwrite rules are described below. |
| `overwrite` | <code>bool</code> | Allow replacing an existing Tensor snapshot; default False. |

Return type: `pathlib.Path (method); None (tg.save)`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'value.tga'
        x.save(path)
        restored = tg.load(path)
        assert restored.tolist() == x.tolist()
    ```

<!-- typed-contract:end -->

Writes a contiguous logical-value snapshot, including non-contiguous views, without eviction. The Tensor method returns the path; the module helper returns `None`. An existing path raises `FileExistsError` unless `overwrite=True`; replacement is atomic. A failure cleans up its temporary output. Persistent files are not charged against the temporary NVMe budget. Autograd history is not serialized; same-process input history remains intact.

### tg.load(path, *, device="cpu") { #tgload }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `path` | <code>str &#124; os.PathLike</code> | Filesystem path, not a URL; parent and overwrite rules are described below. |
| `device` | <code>str &#124; tg.Device &#124; None</code> | Execution device; None uses the constructor's documented default. |

Return type: `tg.Tensor | tg.Graph`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'value.tga'
        x.save(path)
        restored = tg.load(path)
        assert restored.tolist() == x.tolist()
    ```

<!-- typed-contract:end -->

A file opens as a Tensor snapshot; a directory opens as a versioned Graph. Tensor attachment validates bounded metadata and payload length without allocating the full payload. First observation allocates on `device`. Returns a leaf Tensor with `requires_grad=False`, preserving shape, dtype and version. Missing paths raise filesystem errors; malformed snapshots raise `ValueError`. Tensor snapshots currently use the legacy native-endian format and are not a portable cross-endian interchange format.

### Tensor.disk(*, name=None) { #tensordisk }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `name` | <code>str &#124; None</code> | None spills temporarily; a safe basename persists to the legacy named store. |

Return type: `tg.Tensor`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    x = tg.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    x.disk()
    print(x.tolist())
    ```

<!-- typed-contract:end -->

Compatibility entry point. Without a name it performs a temporary spill; a name persists in `TIGA_SPILL_DIR` or `~/.cache/tiga/spill`. Names cannot begin with a dot or contain path separators; collisions raise `FileExistsError`. New persistence code should use explicit `save/load` paths. Spill preserves autograd but reattaching a named snapshot does not recreate that history.

### tg.from_disk(name) { #gffrom_disk }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `name` | <code>str</code> | Existing snapshot basename; no dots at the start or path separators. |

Return type: `tg.Tensor`.

Before running this example, configure the required device, files or distributed environment as described below.

??? example "Minimal usage"

    ```python
    import tiga as tg
    # Requires an existing named snapshot; prefer explicit tg.save/tg.load paths.
    x = tg.from_disk('existing-snapshot')
    ```

<!-- typed-contract:end -->

Lazily attaches the legacy named snapshot on CPU. `ValueError` rejects invalid names; `FileNotFoundError` rejects missing names. The file remains after loading. Prefer `tg.load(path)` for new code.

## GraphProgram { #program-and-visualize }

### @tg.program { #gfprogram }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `function` | <code>Callable</code> | Straight-line Python function containing MessagePassing calls. |

Return type: `Callable`.

Before running this example, configure the required device, files or distributed environment as described below.

??? example "Minimal usage"

    ```python
    import torch
    import tiga as tg
    row_ptr = torch.tensor([0, 2, 3, 3], dtype=torch.int64)
    col_idx = torch.tensor([0, 1, 1], dtype=torch.int64)
    graph = tg.Graph.from_csr(row_ptr, col_idx, num_src=2)
    class Sum(tg.MessagePassing):
        reducer = tg.sum()
        def edge(self, src, dst, edge):
            return src.x
    program = Sum()
    x = torch.tensor([2., 3.], requires_grad=True)
    @tg.program
    def composed(x):
        return program(graph=graph, src={'x': x}, dst={})
    value = composed(x)
    print(value.materialize())
    ```

<!-- typed-contract:end -->

Compatibility capture boundary kept for straight-line code that cannot
offer source access: MessagePassing calls inside the function become
applies of one `GraphProgram`, a typed acyclic SSA composition.
`@tg.jit` activates the same context automatically (and additionally
captures loops), so new code should reach for `@tg.jit`; stacking the two
is legal and idempotent. Ordinary leaf calls outside either decorator
remain automatic JIT. Example:
[cross-kernel SSA capture](examples/programs-and-interop.md#gfprogram-ssa-capture).

### GraphProgram.apply(kernel, *, graph, src, dst, edge=None, **params) { #graphprogramapply }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `kernel` | <code>tg.MessagePassing</code> | Leaf program to capture. |
| `graph` | <code>tg.Graph</code> | Relation consumed by this call; device and entity counts must match fields. |
| `src` | <code>Mapping[str, torch.Tensor &#124; tg.Tensor] &#124; None</code> | Source fields, leading dimension num_src; use {} for an unused role. |
| `dst` | <code>Mapping[str, torch.Tensor &#124; tg.Tensor] &#124; None</code> | Destination fields, leading dimension num_dst; use {} for an unused role. |
| `edge` | <code>Mapping[str, torch.Tensor &#124; tg.Tensor] &#124; None</code> | Edge fields in graph edge order; None means no explicit edge fields. |
| `**params` | <code>object</code> | Named scalar/captured arguments declared by edge/node; not arbitrary Python objects. |

Return type: `ProgramValue`.

Before running this example, configure the required device, files or distributed environment as described below.

??? example "Minimal usage"

    ```python
    import torch
    import tiga as tg
    row_ptr = torch.tensor([0, 2, 3, 3], dtype=torch.int64)
    col_idx = torch.tensor([0, 1, 1], dtype=torch.int64)
    graph = tg.Graph.from_csr(row_ptr, col_idx, num_src=2)
    class Sum(tg.MessagePassing):
        reducer = tg.sum()
        def edge(self, src, dst, edge):
            return src.x
    program = Sum()
    x = torch.tensor([2., 3.], requires_grad=True)
    composition = tg.GraphProgram()
    value = composition.apply(program, graph=graph, src={'x': x}, dst={})
    ```

<!-- typed-contract:end -->

Adds one MessagePassing leaf to the composition and returns its typed
`ProgramValue`.

### GraphProgram.outputs(*values) { #graphprogramoutputs }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `*values` | <code>ProgramValue &#124; tg.Tensor</code> | Outputs belonging to this composition; foreign program values are rejected. |

Return type: `GraphProgram (self)`.

Before running this example, configure the required device, files or distributed environment as described below.

??? example "Minimal usage"

    ```python
    import torch
    import tiga as tg
    row_ptr = torch.tensor([0, 2, 3, 3], dtype=torch.int64)
    col_idx = torch.tensor([0, 1, 1], dtype=torch.int64)
    graph = tg.Graph.from_csr(row_ptr, col_idx, num_src=2)
    class Sum(tg.MessagePassing):
        reducer = tg.sum()
        def edge(self, src, dst, edge):
            return src.x
    program = Sum()
    x = torch.tensor([2., 3.], requires_grad=True)
    composition = tg.GraphProgram()
    value = composition.apply(program, graph=graph, src={'x': x}, dst={})
    composition.outputs(value)
    ```

<!-- typed-contract:end -->

Selects the values observed at the composition boundary. Bare
`ProgramValue` leaves select their producers directly; a Tensor expression
contributes every program leaf it transitively references, so epilogue
arithmetic composed on top of leaves is a legal output spelling without
claiming fusion into the leaf kernels.

### GraphProgram.run() { #graphprogramrun }

<!-- typed-contract:start -->

No explicit parameters (`self` omitted for instance methods).

Return type: `Tensor | tuple[Tensor, ...]`.

Before running this example, configure the required device, files or distributed environment as described below.

??? example "Minimal usage"

    ```python
    import torch
    import tiga as tg
    row_ptr = torch.tensor([0, 2, 3, 3], dtype=torch.int64)
    col_idx = torch.tensor([0, 1, 1], dtype=torch.int64)
    graph = tg.Graph.from_csr(row_ptr, col_idx, num_src=2)
    class Sum(tg.MessagePassing):
        reducer = tg.sum()
        def edge(self, src, dst, edge):
            return src.x
    program = Sum()
    x = torch.tensor([2., 3.], requires_grad=True)
    composition = tg.GraphProgram()
    value = composition.apply(program, graph=graph, src={'x': x}, dst={})
    result = composition.run()
    ```

<!-- typed-contract:end -->

JITs and executes the selected outputs; independent leaves are fused
horizontally where legal, dependent applies run through a typed runtime
dependency DAG.

### GraphProgram.ir(stage="domain") { #graphprogramir }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `stage` | <code>str</code> | domain, fused, iteration/iter, or kernel. |

Return type: `str`.

Before running this example, configure the required device, files or distributed environment as described below.

??? example "Minimal usage"

    ```python
    import torch
    import tiga as tg
    row_ptr = torch.tensor([0, 2, 3, 3], dtype=torch.int64)
    col_idx = torch.tensor([0, 1, 1], dtype=torch.int64)
    graph = tg.Graph.from_csr(row_ptr, col_idx, num_src=2)
    class Sum(tg.MessagePassing):
        reducer = tg.sum()
        def edge(self, src, dst, edge):
            return src.x
    program = Sum()
    x = torch.tensor([2., 3.], requires_grad=True)
    composition = tg.GraphProgram()
    value = composition.apply(program, graph=graph, src={'x': x}, dst={})
    print(composition.ir('domain'))
    ```

<!-- typed-contract:end -->

Returns `domain`, `fused`, `iteration`/`iter` or `kernel` IR text. Calling
`ir` is the observation boundary that triggers composition, verification and
lowering.

### GraphProgram.code(kind="ptx") { #graphprogramcode }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `kind` | <code>str</code> | Provider artifact such as ptx or ttir; unavailable artifacts raise KeyError. |

Return type: `str | bytes | dict`.

Before running this example, configure the required device, files or distributed environment as described below.

??? example "Minimal usage"

    ```python
    import torch
    import tiga as tg
    row_ptr = torch.tensor([0, 2, 3, 3], dtype=torch.int64)
    col_idx = torch.tensor([0, 1, 1], dtype=torch.int64)
    graph = tg.Graph.from_csr(row_ptr, col_idx, num_src=2)
    class Sum(tg.MessagePassing):
        reducer = tg.sum()
        def edge(self, src, dst, edge):
            return src.x
    program = Sum()
    x = torch.tensor([2., 3.], requires_grad=True)
    composition = tg.GraphProgram()
    value = composition.apply(program, graph=graph, src={'x': x}, dst={})
    # Requires a matching provider; CPU paths do not produce PTX.
    print(composition.code('ptx'))
    ```

<!-- typed-contract:end -->

Returns provider artifacts such as `ttir` or `ptx`; for dependent applies a
mapping of per-apply artifacts.

### GraphProgram.explain() and GraphProgram.semantic_hash { #graphprogramexplain }

<!-- typed-contract:start -->

No explicit parameters (`self` omitted for instance methods).

Return type: `str (semantic_hash: str)`.

Before running this example, configure the required device, files or distributed environment as described below.

??? example "Minimal usage"

    ```python
    import torch
    import tiga as tg
    row_ptr = torch.tensor([0, 2, 3, 3], dtype=torch.int64)
    col_idx = torch.tensor([0, 1, 1], dtype=torch.int64)
    graph = tg.Graph.from_csr(row_ptr, col_idx, num_src=2)
    class Sum(tg.MessagePassing):
        reducer = tg.sum()
        def edge(self, src, dst, edge):
            return src.x
    program = Sum()
    x = torch.tensor([2., 3.], requires_grad=True)
    composition = tg.GraphProgram()
    value = composition.apply(program, graph=graph, src={'x': x}, dst={})
    result = composition.explain()
    ```

<!-- typed-contract:end -->

Render apply/fusion counts plus the content hash of the composed domain IR.

### ProgramValue.materialize() { #programvaluematerialize }

<!-- typed-contract:start -->

No explicit parameters (`self` omitted for instance methods).

Return type: `torch.Tensor | tg.Tensor`.

Before running this example, configure the required device, files or distributed environment as described below.

??? example "Minimal usage"

    ```python
    import torch
    import tiga as tg
    row_ptr = torch.tensor([0, 2, 3, 3], dtype=torch.int64)
    col_idx = torch.tensor([0, 1, 1], dtype=torch.int64)
    graph = tg.Graph.from_csr(row_ptr, col_idx, num_src=2)
    class Sum(tg.MessagePassing):
        reducer = tg.sum()
        def edge(self, src, dst, edge):
            return src.x
    program = Sum()
    x = torch.tensor([2., 3.], requires_grad=True)
    composition = tg.GraphProgram()
    value = composition.apply(program, graph=graph, src={'x': x}, dst={})
    result = value.materialize()
    ```

<!-- typed-contract:end -->

JITs and executes the connected program at this observation boundary and
returns the concrete value. `value.ir(stage)` inspects the owning program.

## visualize { #visualize }

### tg.visualize.heatmap(values, *, vmin=0.0, vmax=1.0, low=None, high=None, cmap=None) { #gfvisualizeheatmap }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `values` | <code>tg.Tensor</code> | Contiguous rank-two native floating Tensor. Torch input currently requires tg.from_torch; that conversion does not bridge autograd. |
| `vmin / vmax` | <code>float &#124; None</code> | Scalar color range; None derives bounds for geometry renderers. heatmap defaults to 0 and 1. |
| `low / high` | <code>tuple[float, float, float] &#124; None</code> | Optional RGB endpoints in [0,1] for a two-color ramp. |
| `cmap` | <code>str &#124; Sequence[RGB] &#124; Sequence[tuple[float, RGB]] &#124; None</code> | Named map or color stops in [0,1]; explicit cmap overrides low/high. |

Return type: `tg.visualize.Raster`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    raster = tg.visualize.heatmap(tg.tensor([[0., 0.5], [0.75, 1.]]))
    assert raster.to_numpy().shape == (2, 2, 3)
    ```

<!-- typed-contract:end -->

Compiles a colormapped heatmap of a contiguous rank-two floating Tensor into
one fused kernel and returns a lazy RGB `Raster`. The default colormap is
`viridis`; `cmap` selects another multi-stop colormap — a name from
`tg.visualize.colormaps()`, a list of evenly spaced RGB colors, or a list of
`(position, RGB)` stops with strictly increasing positions in [0, 1] — and
`low`/`high` select a plain two-color ramp instead (a missing end falls back
to the matching viridis endpoint). Two-color ramps leave values outside
`[vmin,vmax]` unclipped in the lazy Tensor, keeping the expression
differentiable; multi-stop colormaps clamp to the end colors like
Matplotlib. The color transform is ordinary Tensor broadcast/arithmetic
(`sqrt(x²)` tent functions for multi-stop ramps); there is no visualization
operation in compiler core. Example:
[GPU heatmap visualization prep](examples/visualization.md#gpu-heatmap-visualization-prep).

### tg.visualize.colormaps() { #gfvisualizecolormaps }

<!-- typed-contract:start -->

No explicit parameters (`self` omitted for instance methods).

Return type: `tuple[str, ...]`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    print(tg.visualize.colormaps())
    ```

<!-- typed-contract:end -->

Returns the names of the built-in heatmap colormaps: `viridis`, `magma`,
`plasma`, `inferno`, `jet`, `coolwarm`, `gray` — eight-stop samples of the
Matplotlib maps of the same name, interpolated piecewise-linearly inside the
fused kernel.

### Raster { #raster }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `pixels` | <code>tg.Tensor</code> | Lazy RGB pixel values; prefer a rendering factory over manual construction. |
| `width / height` | <code>int</code> | Positive image dimensions in pixels; default 512 each. |
| `to_numpy: clip` | <code>bool</code> | Clamp exported RGB to [0,1]; default True. |
| `save: path` | <code>str &#124; os.PathLike</code> | Image destination; Pillow encodes and existing files are replaced. |
| `mlir: verify` | <code>bool</code> | Run the IR verifier; default False. |
| `generated_code: kind` | <code>str &#124; None</code> | Artifact name; inherits Tensor.generated_code behavior. |
| `show: **imshow_options` | <code>object</code> | Keyword arguments forwarded to Matplotlib imshow. |

Return type: `Raster; to_numpy → numpy.ndarray; save → pathlib.Path; show → None`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    raster = tg.visualize.heatmap(tg.tensor([[0., 1.]]))
    print(raster.realize().execution)
    print(raster.to_numpy())
    ```

<!-- typed-contract:end -->

Lazy RGB raster whose pixels remain a Tiga Tensor: `realize()`,
`prepare()`, `execution`, `mlir(*, verify=False)` and `generated_code(kind)`
expose the same compiler/runtime path as `Tensor`; `to_numpy(*, clip=True)`,
`save(path)` (Pillow) and `show(**imshow_options)` (Matplotlib) are optional
host interop/encoding helpers.

### tg.visualize.Camera { #gfvisualizecamera }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `position / target / up` | <code>tuple[float, float, float]</code> | World-space camera position, look-at target and up vector; up defaults to (0,0,1). |
| `fov` | <code>float</code> | Vertical field of view in degrees; default 45. |
| `auto / from_angles: positions` | <code>array-like &#124; None</code> | Cloud used for framing; auto requires positions. |
| `auto: margin` | <code>float</code> | Bounding-sphere distance multiplier; default 1.2. |
| `from_angles: elevation / azimuth` | <code>float</code> | Camera angles in degrees; defaults 30 and -60. |
| `from_angles: distance` | <code>float &#124; None</code> | Camera-to-target distance; None fits positions or uses the default. |
| `world_to_ndc: points_xyz` | <code>numpy.ndarray</code> | World-space coordinates (N,3). |

Return type: `Camera; world_to_ndc → tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    positions = [[0., 0.], [1., 0.], [0., 1.]]
    camera = tg.visualize.Camera.auto(positions)
    ```

<!-- typed-contract:end -->

Host-side perspective camera for point-cloud rendering:
`Camera(position, target, fov=45.0, up=(0.0, 0.0, 1.0))`, with
`Camera.auto(positions, *, fov=45.0, margin=1.2)` fitting target and
distance from a bounding sphere — planar (N, 2) input is viewed top-down,
3-D input follows the smallest-variance principal axis and falls back to an
isometric direction for isotropic clouds — and
`Camera.from_angles(elevation=30.0, azimuth=-60.0, *, positions=None, distance=None, fov=45.0)`
placing the camera on a sphere around the target.
`camera.world_to_ndc(points_xyz)` projects (N, 3) points to
`(ndc, depth, visible)`. Example:
[particles/video](examples/visualization.md#particles-and-video).

### tg.visualize.particles(positions, values=None, *, camera="auto", width=512, height=512, point_radius=2.0, vmin=None, vmax=None, low=None, high=None, cmap=None) { #gfvisualizeparticles }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `positions` | <code>torch.Tensor &#124; tg.Tensor &#124; numpy.ndarray &#124; Sequence</code> | Coordinates (N,2) or (N,3), copied to host for geometry rendering/export. |
| `values` | <code>torch.Tensor &#124; tg.Tensor &#124; numpy.ndarray &#124; Sequence &#124; None</code> | One scalar per vertex; None uses the renderer's uniform/density default. |
| `camera` | <code>Camera &#124; Literal[&#x27;auto&#x27;] &#124; tuple[float, float]</code> | Explicit camera, automatic framing, or elevation/azimuth in degrees. |
| `width / height` | <code>int</code> | Positive image dimensions in pixels; default 512 each. |
| `point_radius` | <code>float</code> | Disk radius in pixels; default 2. |
| `vmin / vmax` | <code>float &#124; None</code> | Scalar color range; None derives bounds for geometry renderers. heatmap defaults to 0 and 1. |
| `low / high` | <code>tuple[float, float, float] &#124; None</code> | Optional RGB endpoints in [0,1] for a two-color ramp. |
| `cmap` | <code>str &#124; Sequence[RGB] &#124; Sequence[tuple[float, RGB]] &#124; None</code> | Named map or color stops in [0,1]; explicit cmap overrides low/high. |

Return type: `Raster`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    positions = [[0., 0.], [1., 0.], [0., 1.]]
    raster = tg.visualize.particles(positions, width=32, height=32)
    ```

<!-- typed-contract:end -->

Splats every point as a disk into a host-side scalar field — the point
value, or a unit density when `values=None` — then colors that field through
the `heatmap` Tensor expression and returns a lazy `Raster`. `camera`
accepts `"auto"`, a `Camera`, or an `(elevation, azimuth)` tuple;
`vmin`/`vmax` default to the field's data range; `cmap` picks a multi-stop
colormap (see [`heatmap`](#gfvisualizeheatmap)), overriding `low`/`high`.
Example:
[Delaunay mesh from points](examples/visualization.md#simulation-meshes-load-shade-orbit).

### tg.visualize.delaunay(positions, values=None, *, camera="auto", width=512, height=512, vmin=None, vmax=None, low=None, high=None, cmap=None, wireframe=False) { #gfvisualizedelaunay }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `positions` | <code>torch.Tensor &#124; tg.Tensor &#124; numpy.ndarray &#124; Sequence</code> | Coordinates (N,2) or (N,3), copied to host for geometry rendering/export. |
| `values` | <code>torch.Tensor &#124; tg.Tensor &#124; numpy.ndarray &#124; Sequence &#124; None</code> | One scalar per vertex; None uses the renderer's uniform/density default. |
| `camera` | <code>Camera &#124; Literal[&#x27;auto&#x27;] &#124; tuple[float, float]</code> | Explicit camera, automatic framing, or elevation/azimuth in degrees. |
| `width / height` | <code>int</code> | Positive image dimensions in pixels; default 512 each. |
| `vmin / vmax` | <code>float &#124; None</code> | Scalar color range; None derives bounds for geometry renderers. heatmap defaults to 0 and 1. |
| `low / high` | <code>tuple[float, float, float] &#124; None</code> | Optional RGB endpoints in [0,1] for a two-color ramp. |
| `cmap` | <code>str &#124; Sequence[RGB] &#124; Sequence[tuple[float, RGB]] &#124; None</code> | Named map or color stops in [0,1]; explicit cmap overrides low/high. |
| `wireframe` | <code>bool</code> | Draw edges instead of filled triangles; default False. |

Return type: `Raster`.

Before running this example, configure the required device, files or distributed environment as described below.

??? example "Minimal usage"

    ```python
    import tiga as tg
    positions = [[0., 0.], [1., 0.], [0., 1.]]
    # Requires Matplotlib for triangulation.
    raster = tg.visualize.delaunay(positions, width=32, height=32)
    ```

<!-- typed-contract:end -->

Triangulates positions — planar (N, 2) input directly, 3-D input after
camera projection — fills every triangle with the mean of its vertex values
(`values=None` uses a uniform value; `wireframe=True` draws the triangle
edges instead) and colors the resulting scalar field through `heatmap`
(`cmap` overrides `low`/`high`), returning a lazy `Raster`. Example:
[mesh rendering](examples/visualization.md#simulation-meshes-load-shade-orbit).

### tg.visualize.mesh(positions, faces, values=None, *, camera="auto", width=512, height=512, wireframe=False, vmin=None, vmax=None, low=None, high=None, cmap=None) { #gfvisualizemesh }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `positions` | <code>torch.Tensor &#124; tg.Tensor &#124; numpy.ndarray &#124; Sequence</code> | Coordinates (N,2) or (N,3), copied to host for geometry rendering/export. |
| `faces` | <code>numpy.ndarray &#124; Sequence[Sequence[int]]</code> | Triangle vertex IDs (M,3), all in range. |
| `values` | <code>torch.Tensor &#124; tg.Tensor &#124; numpy.ndarray &#124; Sequence &#124; None</code> | One scalar per vertex; None uses the renderer's uniform/density default. |
| `camera` | <code>Camera &#124; Literal[&#x27;auto&#x27;] &#124; tuple[float, float]</code> | Explicit camera, automatic framing, or elevation/azimuth in degrees. |
| `width / height` | <code>int</code> | Positive image dimensions in pixels; default 512 each. |
| `wireframe` | <code>bool</code> | Draw edges instead of filled triangles; default False. |
| `vmin / vmax` | <code>float &#124; None</code> | Scalar color range; None derives bounds for geometry renderers. heatmap defaults to 0 and 1. |
| `low / high` | <code>tuple[float, float, float] &#124; None</code> | Optional RGB endpoints in [0,1] for a two-color ramp. |
| `cmap` | <code>str &#124; Sequence[RGB] &#124; Sequence[tuple[float, RGB]] &#124; None</code> | Named map or color stops in [0,1]; explicit cmap overrides low/high. |

Return type: `Raster`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    positions = [[0., 0.], [1., 0.], [0., 1.]]
    raster = tg.visualize.mesh(positions, [[0, 1, 2]], width=32, height=32)
    ```

<!-- typed-contract:end -->

Renders a given triangle mesh — where `delaunay` computes the triangulation
from points, `mesh` takes explicit `faces` (an (M, 3) index array, e.g. a
simulation mesh or a model from `load_obj`). Triangles are painted
far-to-near by mean vertex depth (painter's algorithm) so nearer triangles
occlude farther ones; each is filled with the mean of its finite vertex
values (`values=None` uses a uniform value; all-non-finite triangles are
skipped), and `wireframe=True` draws edges instead. Camera handling, planar
input, and behind-camera clipping match `delaunay`. Example:
[simulation meshes: load, shade, orbit](examples/visualization.md#simulation-meshes-load-shade-orbit).

<span id="gfvisualizesplats"></span>

### tg.visualize.gaussians(positions, colors, scales, *, rotations=None, opacities=None, camera="auto", width=512, height=512, cutoff=3.0, background=(0,0,0)) { #tgvisualizegaussians }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `positions` | <code>array-like</code> | 3-D means (N,3); geometry is rendered on CPU, not differentiable. |
| `colors` | <code>numpy.ndarray &#124; Sequence</code> | RGB (N,3), values in [0,1]. |
| `scales` | <code>numpy.ndarray &#124; Sequence</code> | Positive standard deviations (N,) or (N,3). |
| `rotations` | <code>numpy.ndarray &#124; Sequence &#124; None</code> | Unit quaternions (N,4), ordered w,x,y,z; None is identity. |
| `opacities` | <code>numpy.ndarray &#124; Sequence &#124; None</code> | Opacity (N,) in [0,1]; None means opaque. |
| `camera` | <code>Camera &#124; Literal[&#x27;auto&#x27;] &#124; tuple[float, float]</code> | Explicit camera, automatic framing, or elevation/azimuth in degrees. |
| `width / height` | <code>int</code> | Positive image dimensions in pixels; default 512 each. |
| `cutoff` | <code>float</code> | Ellipse extent in standard deviations; default 3. |
| `background` | <code>tuple[float, float, float]</code> | Background RGB in [0,1]; default black. |

Return type: `Raster`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    raster = tg.visualize.gaussians([[0., 0., 0.]], [[1., 0., 0.]], [0.1], width=32, height=32)
    ```

<!-- typed-contract:end -->

`tg.visualize.splats` remains a compatibility alias; new code uses `gaussians`.

Renders anisotropic 3-D Gaussians into a lazy `Raster` (EWA splatting, as in
3-D Gaussian Splatting): the covariance `R·diag(scales²)·Rᵀ` projects
through the perspective Jacobian to a 2-D ellipse, evaluated within
`cutoff` sigma, and colors alpha-composite near-to-far with front-to-back
transmittance. `colors` (N, 3) RGB in [0, 1], `scales` (N, 3) or (N,)
positive standard deviations, `rotations` (N, 4) `(w, x, y, z)`
quaternions, `opacities` (N,) in [0, 1]. Gaussians behind the camera,
sub-pixel, or fainter than 1/255 are skipped. Example:
[Gaussian splats and volumes](examples/visualization.md#gaussian-splats-and-volumes).

### tg.visualize.volume(density, *, camera="auto", width=512, height=512, steps=128, cmap="viridis", vmin=None, vmax=None, scale=8.0, background=(0,0,0)) { #gfvisualizevolume }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `density` | <code>torch.Tensor &#124; tg.Tensor &#124; numpy.ndarray</code> | 3-D density grid; host ray marching is not differentiable. |
| `camera` | <code>Camera &#124; Literal[&#x27;auto&#x27;] &#124; tuple[float, float]</code> | Explicit camera, automatic framing, or elevation/azimuth in degrees. |
| `width / height` | <code>int</code> | Positive image dimensions in pixels; default 512 each. |
| `steps` | <code>int</code> | Positive sample count per ray; default 128. |
| `cmap` | <code>str &#124; Sequence[RGB] &#124; Sequence[tuple[float, RGB]] &#124; None</code> | Named map or color stops in [0,1]; explicit cmap overrides low/high. |
| `vmin / vmax` | <code>float &#124; None</code> | Scalar color range; None derives bounds for geometry renderers. heatmap defaults to 0 and 1. |
| `scale` | <code>float</code> | Absorption multiplier; default 8. |
| `background` | <code>tuple[float, float, float]</code> | Background RGB in [0,1]; default black. |

Return type: `Raster`.

??? example "Minimal usage"

    ```python
    import numpy as np
    import tiga as tg
    raster = tg.visualize.volume(np.ones((3, 3, 3)), width=8, height=8, steps=8)
    ```

<!-- typed-contract:end -->

Ray-marches a density grid (X, Y, Z) — mapped onto the unit cube — into a
lazy `Raster` with the emission-absorption model: trilinear samples along
each pixel ray emit the `cmap` color of the normalized density and absorb
with `alpha = 1 - exp(-density·scale·Δt)`. `vmin`/`vmax` default to the
data range; `camera` frames the cube when `"auto"`. Example:
[Gaussian splats and volumes](examples/visualization.md#gaussian-splats-and-volumes).

### tg.visualize.load_ply(path) { #gfvisualizeloadply }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `path` | <code>str &#124; os.PathLike</code> | Filesystem path, not a URL; parent and overwrite rules are described below. |

Return type: `tuple[numpy.ndarray, numpy.ndarray | None, dict[str, numpy.ndarray]]`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    positions = [[0., 0.], [1., 0.], [0., 1.]]
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'triangle.ply'
        tg.visualize.export_ply(path, positions, faces=[[0, 1, 2]])
        points, faces, extras = tg.visualize.load_ply(path)
    ```

<!-- typed-contract:end -->

Parses a PLY file (ASCII or binary_little_endian) into
`(positions, faces, extras)`: an (N, 3) float64 array from the vertex
`x y z` properties, an (M, K) int64 array from the face list property (or
`None`), and a dict of every remaining vertex property — the entry point
for 3-D Gaussian Splatting payloads (`f_dc_*`, `opacity`, `scale_*`,
`rot_*`). Example:
[Gaussian splats and volumes](examples/visualization.md#gaussian-splats-and-volumes).

### tg.visualize.load_obj(path) { #gfvisualizeloadobj }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `path` | <code>str &#124; os.PathLike</code> | Filesystem path, not a URL; parent and overwrite rules are described below. |

Return type: `tuple[numpy.ndarray, numpy.ndarray]`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    positions = [[0., 0.], [1., 0.], [0., 1.]]
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'triangle.obj'
        tg.visualize.export_obj(path, positions, faces=[[0, 1, 2]])
        points, faces = tg.visualize.load_obj(path)
    ```

<!-- typed-contract:end -->

Parses a minimal Wavefront OBJ file into `(positions, faces)`: `v x y z`
vertex rows become an (N, 3) float64 array, `f` face rows (including
`f a/b/c` token forms; only the vertex index is used) become an (M, 3)
integer array with 1-based indices resolved and polygons fan-triangulated.
Empty files, missing vertices/faces, and out-of-range indices raise
`ValueError`. Example:
[simulation meshes: load, shade, orbit](examples/visualization.md#simulation-meshes-load-shade-orbit).

### tg.visualize.export_ply(path, positions, *, faces=None, values=None, low=None, high=None, cmap=None) { #gfvisualizeexportply }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `path` | <code>str &#124; os.PathLike</code> | Filesystem path, not a URL; parent and overwrite rules are described below. |
| `positions` | <code>torch.Tensor &#124; tg.Tensor &#124; numpy.ndarray &#124; Sequence</code> | Coordinates (N,2) or (N,3), copied to host for geometry rendering/export. |
| `faces` | <code>array-like &#124; None</code> | Optional triangle vertex indices (M,3). |
| `values` | <code>torch.Tensor &#124; tg.Tensor &#124; numpy.ndarray &#124; Sequence &#124; None</code> | One scalar per vertex; None uses the renderer's uniform/density default. |
| `low / high` | <code>tuple[float, float, float] &#124; None</code> | Optional RGB endpoints in [0,1] for a two-color ramp. |
| `cmap` | <code>str &#124; Sequence[RGB] &#124; Sequence[tuple[float, RGB]] &#124; None</code> | Named map or color stops in [0,1]; explicit cmap overrides low/high. |

Return type: `pathlib.Path`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    positions = [[0., 0.], [1., 0.], [0., 1.]]
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'triangle.ply'
        tg.visualize.export_ply(path, positions, faces=[[0, 1, 2]])
        points, faces, extras = tg.visualize.load_ply(path)
    ```

<!-- typed-contract:end -->

Dumps geometry as binary_little_endian [PLY](https://en.wikipedia.org/wiki/PLY_(file_format))
for [Blender](https://en.wikipedia.org/wiki/Blender_(software)) import:
vertices carry `x y z` float32 (planar (N, 2) input gets `z = 0`), optional
`faces` (M, 3) become a `vertex_indices` list element, and `values` adds
`red green blue` uint8 (the same colormap as `heatmap` — `viridis` by
default, or `cmap`/`low`/`high` — scaled to the data range) plus
`scalar_value` float32 with the raw field. Positions and
values arrive through one host copy (GPU tensor included) and the payload is
assembled in a single structured buffer; latency depends on input size, device
transfer and filesystem. Example:
[simulation meshes: load, shade, orbit](examples/visualization.md#simulation-meshes-load-shade-orbit).

### tg.visualize.export_obj(path, positions, faces) { #gfvisualizeexportobj }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `path` | <code>str &#124; os.PathLike</code> | Filesystem path, not a URL; parent and overwrite rules are described below. |
| `positions` | <code>torch.Tensor &#124; tg.Tensor &#124; numpy.ndarray &#124; Sequence</code> | Coordinates (N,2) or (N,3), copied to host for geometry rendering/export. |
| `faces` | <code>numpy.ndarray &#124; Sequence[Sequence[int]]</code> | Triangle vertex IDs (M,3), all in range. |

Return type: `pathlib.Path`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    positions = [[0., 0.], [1., 0.], [0., 1.]]
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'triangle.obj'
        tg.visualize.export_obj(path, positions, faces=[[0, 1, 2]])
        points, faces = tg.visualize.load_obj(path)
    ```

<!-- typed-contract:end -->

Writes a text Wavefront OBJ — the exact inverse of `load_obj`: `v` rows at
full float64 round-trip precision and 1-based `f` rows, so
`load_obj(export_obj(...))` restores coordinates and faces identically.
Example:
[simulation meshes: load, shade, orbit](examples/visualization.md#simulation-meshes-load-shade-orbit).

### tg.visualize.export_vdb(path, positions, values, *, voxel_size=0.05) { #gfvisualizeexportvdb }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `path` | <code>str &#124; os.PathLike</code> | Filesystem path, not a URL; parent and overwrite rules are described below. |
| `positions` | <code>torch.Tensor &#124; tg.Tensor &#124; numpy.ndarray &#124; Sequence</code> | Coordinates (N,2) or (N,3), copied to host for geometry rendering/export. |
| `values` | <code>torch.Tensor &#124; tg.Tensor &#124; numpy.ndarray &#124; Sequence &#124; None</code> | One scalar per vertex; None uses the renderer's uniform/density default. |
| `voxel_size` | <code>float</code> | Positive world-space voxel width; default 0.05. |

Return type: `pathlib.Path`.

Before running this example, configure the required device, files or distributed environment as described below.

??? example "Minimal usage"

    ```python
    import tiga as tg
    positions = [[0., 0.], [1., 0.], [0., 1.]]
    # Requires pyopenvdb installed separately.
    tg.visualize.export_vdb('cloud.vdb', positions, [1., 2., 3.])
    ```

<!-- typed-contract:end -->

Rasterizes points plus a scalar field into an [OpenVDB](https://en.wikipedia.org/wiki/OpenVDB)
dense grid. Requires the optional `pyopenvdb` dependency — the
dependency-free PLY export already covers the Blender interchange path, so
OpenVDB is only pulled in when a volumetric grid is explicitly wanted;
without it the function raises `ModuleNotFoundError`.

### tg.visualize.save_video(frames, path, *, fps=30) { #gfvisualizesavevideo }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `frames` | <code>Iterable[Raster &#124; numpy.ndarray]</code> | Same-size RGB frames, arrays shaped (H,W,3) with values in [0,1]. |
| `path` | <code>str &#124; os.PathLike</code> | Filesystem path, not a URL; parent and overwrite rules are described below. |
| `fps` | <code>float</code> | Finite positive playback frame rate; bool and non-finite values are rejected. Default 30. |

Return type: `pathlib.Path`.

??? example "Minimal usage"

    ```python
    import tempfile
    from pathlib import Path
    import tiga as tg
    frame = tg.visualize.heatmap(tg.tensor([[0., 1.]]))
    with tempfile.TemporaryDirectory() as directory:
        tg.visualize.save_video([frame, frame], Path(directory) / 'demo.gif', fps=10)
    ```

<!-- typed-contract:end -->

Encodes any iterable of frames — `Raster` or float (H, W, 3) arrays in
[0, 1] — as `.gif` (Pillow) or `.mp4` (raw RGB piped into an ffmpeg
subprocess). MP4 consumes generators frame by frame; GIF currently buffers all
frames in host memory before encoding. Existing output files are replaced.
Example:
[particles/video](examples/visualization.md#particles-and-video).

## Compiler types { #compiler-types }

### tg.compiler.Dim / TensorSpec / ShapeSpecializer { #gfcompiler-symbolic-shapes }

<!-- typed-contract:start -->

| Parameter | Type | Meaning |
|---|---|---|
| `Dim: name` | <code>str</code> | Symbol identifier shared across argument shapes. |
| `Dim: minimum / maximum / multiple_of` | <code>int / int &#124; None / int</code> | Inclusive bounds and positive divisibility constraint; defaults 1, None, 1. |
| `TensorSpec: shape` | <code>Sequence[int &#124; Dim]</code> | Concrete non-negative extents or constrained dimensions. |
| `dtype` | <code>tg.DType &#124; None</code> | Native value dtype; None infers from input data where supported. |
| `device` | <code>str &#124; tg.Device &#124; None</code> | Execution device; None uses the constructor's documented default. |
| `ShapeSpecializer: specs` | <code>Sequence[TensorSpec]</code> | One specification per positional input. |
| `ShapeSpecializer: compiler` | <code>Callable[[ShapeBinding], Callable]</code> | Build a callable for a validated concrete shape binding. |
| `ShapeSpecializer.__call__: *values` | <code>tg.Tensor</code> | Inputs checked before specialization lookup/compilation. |

Return type: `Dim / TensorSpec / ShapeSpecializer; calling specializer → compiler-defined result`.

??? example "Minimal usage"

    ```python
    import tiga as tg
    n = tg.compiler.Dim('N', maximum=16)
    spec = tg.compiler.TensorSpec((n,))
    run = tg.compiler.ShapeSpecializer([spec], lambda binding: lambda x: x * 2)
    assert run(tg.tensor([1., 2.])).tolist() == [2., 4.]
    ```

<!-- typed-contract:end -->

Bounded symbolic signatures: `Dim(name, minimum=1, maximum=None,
multiple_of=1)`, `TensorSpec(shape, *, dtype=tg.float32, device=None)` and
`ShapeSpecializer(specs, compiler)`. Repeated symbols must bind to one
extent across arguments; minimum/maximum/divisibility guards are checked
before compilation, and each legal binding forms a concrete MLIR
specialization/cache key. Guide:
[Tensor runtime and autograd](runtime-and-autograd.md#current-alpha-slice).
