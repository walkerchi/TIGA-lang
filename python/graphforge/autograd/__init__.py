"""Reverse-mode transformation over the minimal Tensor expression graph."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from ..tensor import Tensor, ones_like


class _RealizeTensor:
    def __init__(self, tensor: Tensor) -> None:
        self.tensor = tensor

    def launch(self) -> None:
        self.tensor.realize()


class JointAutogradPlan:
    """Executable forward/backward dependency DAG with one snapshot version."""

    def __init__(self, output: Tensor, inputs: tuple[Tensor, ...], *, checkpoint: str):
        from ..runtime import ExecutableBundle, KernelInvocation, ResourceAccess

        self.output = output
        self.inputs = inputs
        built = grad(
            output, inputs[0] if len(inputs) == 1 else inputs,
            checkpoint=checkpoint,
        )
        self.gradients = (built,) if isinstance(built, Tensor) else tuple(built)
        version = max(
            (value.version for value in (output, *inputs, *self.gradients)),
            default=0,
        )
        invocations = [KernelInvocation(
            "forward", _RealizeTensor(output), arguments=(),
            accesses=(ResourceAccess("output", "write", version),),
            task_kind="autograd-forward",
        )]
        for index, gradient in enumerate(self.gradients):
            invocations.append(KernelInvocation(
                f"backward:{index}", _RealizeTensor(gradient), arguments=(),
                accesses=(
                    ResourceAccess("output", "read", version),
                    ResourceAccess(f"gradient:{index}", "write", version),
                ),
                depends_on=("forward",), task_kind="autograd-backward",
            ))
        self.bundle = ExecutableBundle(
            invocations,
            required_bindings=(
                "output", *(f"gradient:{index}" for index in range(len(inputs)))
            ),
            snapshot_version=version,
        )

    def run(self):
        from ..runtime import SynchronousProvider

        resources = {"output": self.output}
        resources.update({
            f"gradient:{index}": gradient
            for index, gradient in enumerate(self.gradients)
        })
        self.bundle.submit(SynchronousProvider(), resources).wait()
        gradients = self.gradients[0] if len(self.gradients) == 1 else self.gradients
        return self.output, gradients

    def explain(self) -> str:
        return self.bundle.explain() + (
            "\ncheckpoint/spill decisions are consumed while each backward "
            "executable is physicalized"
        )


def _topological_sort(output: Tensor) -> list[Tensor]:
    ordered: list[Tensor] = []
    visited: set[int] = set()

    def visit(value: Tensor) -> None:
        if id(value) in visited:
            return
        visited.add(id(value))
        if value._expr is not None:
            for operand in value._expr.operands:
                visit(operand)
        ordered.append(value)

    visit(output)
    return ordered


def _reduce_to_shape(value: Tensor, shape: tuple[int, ...]) -> Tensor:
    if value.shape == shape:
        return value
    if len(shape) > value.ndim:
        raise ValueError(f"cannot reduce gradient shape {value.shape} to {shape}")
    padded = (1,) * (value.ndim - len(shape)) + shape
    axes = tuple(
        axis
        for axis, (source_extent, target_extent) in enumerate(
            zip(value.shape, padded)
        )
        if target_extent == 1 and source_extent != 1
        or axis < value.ndim - len(shape)
    )
    if any(
        source_extent != target_extent and target_extent != 1
        for source_extent, target_extent in zip(value.shape, padded)
    ):
        raise ValueError(f"cannot reduce gradient shape {value.shape} to {shape}")
    # Canonicalize vector-to-scalar broadcast VJPs to the provider's row
    # reduction contract [1,N] -> [1,1]. This is semantically identical to
    # sum(axis=0) on [N] and avoids a separate rank-one reduction skeleton.
    if value.ndim == 1 and axes == (0,) and shape in {(), (1,)}:
        reduced = value.reshape((1, value.shape[0])).sum(1, keepdims=True)
    else:
        reduced = value.sum(axes, keepdims=True) if axes else value
    return reduced.reshape(shape) if reduced.shape != shape else reduced


def _accumulate(table: dict[int, Tensor], target: Tensor, contribution: Tensor) -> None:
    previous = table.get(id(target))
    table[id(target)] = contribution if previous is None else previous + contribution


def grad(
    output: Tensor,
    inputs: Tensor | Sequence[Tensor],
    *,
    grad_output: Tensor | None = None,
    allow_unused: bool = False,
    checkpoint: str = "auto",
) -> Tensor | tuple[Tensor, ...]:
    """Build symbolic reverse-mode expressions without mutating input Tensors.

    ``checkpoint`` controls backward primal storage. ``recompute`` retains the
    complete primal expression, ``save`` marks non-leaf local values as saved,
    and ``auto`` emits candidates for the target-independent MLIR checkpoint
    planner. The planner selects saves under the deployment byte budget; every
    decision remains visible in Tensor IR and execution diagnostics.
    """
    if not isinstance(output, Tensor):
        raise TypeError("output must be a graphforge.Tensor")
    single = isinstance(inputs, Tensor)
    requested = (inputs,) if single else tuple(inputs)
    if any(not isinstance(value, Tensor) for value in requested):
        raise TypeError("inputs must contain only graphforge.Tensor values")
    if any(not value.requires_grad for value in requested):
        raise ValueError("every differentiated input must have requires_grad=True")
    if checkpoint not in {"auto", "save", "recompute"}:
        raise ValueError("checkpoint must be 'auto', 'save' or 'recompute'")
    if grad_output is None:
        if output.shape != ():
            raise ValueError("non-scalar outputs require an explicit grad_output")
        if output.dtype.kind == "complex":
            raise ValueError("complex outputs require an explicit grad_output")
        grad_output = ones_like(output)
    if grad_output.shape != output.shape:
        raise ValueError("grad_output shape must match output shape")
    if grad_output.dtype is not output.dtype or grad_output.device != output.device:
        raise ValueError("grad_output dtype and device must match output")

    adjoints: dict[int, Tensor] = {id(output): grad_output}

    primal_cache: dict[int, Tensor] = {}

    def primal(value: Tensor) -> Tensor:
        # Saving a scalar cannot lower memory pressure and creates a separate
        # rank-zero provider executable on GPUs whose pointwise ABI is
        # intentionally rank-one/rank-two. Always recompute it in its consumer.
        if checkpoint == "recompute" or value._expr is None or value.numel <= 1:
            return value
        cached = primal_cache.get(id(value))
        if cached is not None:
            return cached
        if checkpoint == "save":
            selected = value.checkpoint()
            primal_cache[id(value)] = selected
            return selected
        # Auto creates a semantic identity candidate. Target-independent MLIR
        # planning decides whether it becomes a save or a recomputation under
        # the deployment memory budget.
        selected = value._checkpoint_candidate()
        primal_cache[id(value)] = selected
        return selected

    for value in reversed(_topological_sort(output)):
        upstream = adjoints.get(id(value))
        expression = value._expr
        if upstream is None or expression is None:
            continue
        operands = expression.operands
        if expression.op == "add":
            for operand in operands:
                _accumulate(
                    adjoints, operand, _reduce_to_shape(upstream, operand.shape))
        elif expression.op == "mul":
            lhs, rhs = operands
            _accumulate(
                adjoints, lhs,
                _reduce_to_shape(upstream * primal(rhs).conj(), lhs.shape))
            _accumulate(
                adjoints, rhs,
                _reduce_to_shape(upstream * primal(lhs).conj(), rhs.shape))
        elif expression.op == "matmul":
            lhs, rhs = operands
            if lhs.requires_grad:
                _accumulate(
                    adjoints, lhs, upstream @ primal(rhs).conj().T)
            if rhs.requires_grad:
                _accumulate(
                    adjoints, rhs, primal(lhs).conj().T @ upstream)
        elif expression.op == "div":
            lhs, rhs = operands
            _accumulate(
                adjoints, lhs,
                _reduce_to_shape(upstream / rhs.conj(), lhs.shape))
            rhs_local = -(lhs / (rhs * rhs))
            _accumulate(
                adjoints, rhs,
                _reduce_to_shape(upstream * rhs_local.conj(), rhs.shape))
        elif expression.op == "neg":
            _accumulate(adjoints, operands[0], -upstream)
        elif expression.op == "exp":
            _accumulate(adjoints, operands[0], upstream * primal(value))
        elif expression.op == "sqrt":
            _accumulate(
                adjoints, operands[0], upstream / (primal(value) + primal(value)))
        elif expression.op == "cumsum":
            _accumulate(
                adjoints,
                operands[0],
                upstream.cumsum(
                    int(expression.attr("axis")),
                    reverse=not bool(expression.attr("reverse")),
                ),
            )
        elif expression.op == "conj":
            operand = operands[0]
            _accumulate(adjoints, operand, upstream.conj())
        elif expression.op in {"checkpoint", "checkpoint_candidate"}:
            _accumulate(adjoints, operands[0], upstream)
        elif expression.op == "sum":
            operand = operands[0]
            axes = expression.attr("axes")
            keepdims = expression.attr("keepdims")
            expanded = upstream
            if not keepdims:
                expanded_shape = tuple(
                    1 if axis in axes else operand.shape[axis]
                    for axis in range(operand.ndim)
                )
                expanded = upstream.reshape(expanded_shape)
            _accumulate(adjoints, operand, expanded.broadcast_to(operand.shape))
        elif expression.op == "gather":
            operand, index = operands
            _accumulate(
                adjoints, operand,
                upstream.segment_sum(index, operand.shape[0]))
        elif expression.op == "segment_sum":
            operand, index = operands
            _accumulate(adjoints, operand, upstream.gather(index))
        elif expression.op == "csr_expand_rows":
            operand, row_ptr = operands
            _accumulate(
                adjoints, operand,
                upstream.csr_segment_sum(row_ptr, operand.shape[0]))
        elif expression.op == "csr_segment_sum":
            operand, row_ptr = operands
            _accumulate(
                adjoints, operand,
                upstream.csr_expand_rows(row_ptr, operand.shape[0]))
        elif expression.op == "csr_segment_product":
            operand, row_ptr, destination = operands
            contribution = operand._csr_segment_product_vjp(
                row_ptr, destination, upstream,
                int(expression.attr("num_rows")),
                int(expression.attr("max_degree")),
                bool(expression.attr("uniform_degree")),
            )
            _accumulate(adjoints, operand, contribution)
        elif expression.op == "csr_segment_product_vjp":
            raise NotImplementedError(
                "higher-order differentiation of product-reducer VJP is not supported")
        elif expression.op == "csr_segment_max_stop_gradient":
            # The private online-softmax shift is translation invariant and
            # intentionally detached. It is not a public max-reduction VJP.
            continue
        elif expression.op == "distributed_halo_snapshot":
            operand = operands[0]
            contribution = upstream._distributed_halo_reverse(
                expression.attr("halo"), expression.attr("owned_rows"))
            _accumulate(adjoints, operand, contribution)
        elif expression.op == "broadcast":
            operand = operands[0]
            _accumulate(adjoints, operand, _reduce_to_shape(upstream, operand.shape))
        elif expression.op == "permute":
            operand = operands[0]
            axes = expression.attr("axes")
            inverse = [0] * len(axes)
            for output_axis, input_axis in enumerate(axes):
                inverse[input_axis] = output_axis
            _accumulate(adjoints, operand, upstream.permute(inverse))
        elif expression.op == "reshape":
            operand = operands[0]
            _accumulate(adjoints, operand, upstream.reshape(operand.shape))
        else:
            raise NotImplementedError(
                f"no GraphForge VJP rule is registered for {expression.op!r}")

    results: list[Tensor] = []
    for value in requested:
        result = adjoints.get(id(value))
        if result is None:
            if allow_unused:
                results.append(None)  # type: ignore[arg-type]
                continue
            raise ValueError("an input is not connected to the output expression")
        results.append(result)
    return results[0] if single else tuple(results)


def value_and_grad(
    function: Callable[..., Tensor],
    *,
    argnums: int | Sequence[int] = 0,
) -> Callable[..., tuple[Tensor, Tensor | tuple[Tensor, ...]]]:
    """Return a functional transform that builds primal and VJP expressions."""
    selected = (argnums,) if isinstance(argnums, int) else tuple(argnums)

    def transformed(*args: Any, **kwargs: Any):
        output = function(*args, **kwargs)
        inputs = tuple(args[index] for index in selected)
        gradients = grad(output, inputs[0] if len(inputs) == 1 else inputs)
        return output, gradients

    return transformed


def joint_plan(
    output: Tensor,
    inputs: Tensor | Sequence[Tensor],
    *,
    checkpoint: str = "auto",
) -> JointAutogradPlan:
    """Build an executable, inspectable joint forward/backward task DAG."""
    selected = (inputs,) if isinstance(inputs, Tensor) else tuple(inputs)
    if not selected:
        raise ValueError("joint autograd plan requires at least one input")
    return JointAutogradPlan(output, selected, checkpoint=checkpoint)


def grad_mlir(
    output: Tensor,
    input: Tensor,
    *,
    grad_output: Tensor | None = None,
    lower: bool = True,
) -> str:
    """Inspect the canonical compiler VJP for one differentiated input."""
    if grad_output is None:
        if output.shape != () or output.dtype.kind == "complex":
            raise ValueError("non-scalar or complex outputs require grad_output")
        grad_output = ones_like(output)
    from ..compiler.tensor_mlir import lower_tensor_vjp_mlir, tensor_vjp_mlir

    if lower:
        return lower_tensor_vjp_mlir(output, input, grad_output)
    return tensor_vjp_mlir(output, input, grad_output)


__all__ = [
    "JointAutogradPlan", "grad", "grad_mlir", "joint_plan", "value_and_grad"
]
