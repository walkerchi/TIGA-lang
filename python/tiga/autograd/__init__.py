"""Reverse-mode transformation over the minimal Tensor expression graph."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from ..tensor import Tensor, ones_like
from ..tensor.core import _Expr


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


def _expand_fixed_repeats(output: Tensor) -> tuple[Tensor, dict[int, Tensor]]:
    """Inline fixed control regions for the correctness-first VJP path.

    Forward capture stays as one bounded ``gf_control.repeat``. Reverse-mode
    currently reuses the existing per-operation VJP rules by specializing the
    body at each logical state. The returned replacement map keeps requests
    for a repeated value itself well-defined. A later control-autodiff pass can
    replace this O(iterations) transformation without changing public APIs.
    """
    inspected: set[int] = set()

    def reject_data_dependent_control(value: Tensor, visited: set[int]) -> None:
        if id(value) in visited:
            return
        visited.add(id(value))
        expression = value._expr
        if expression is None:
            return
        if expression.op == "while":
            raise NotImplementedError(
                "autograd through tg.while_loop requires structured control "
                "VJP or an implicit-solve rule; Tiga does not unroll a "
                "data-dependent loop through host scalar synchronization")
        for operand in expression.operands:
            reject_data_dependent_control(operand, visited)
        if expression.region is not None:
            outputs = expression.region.output
            for item in ((outputs,) if isinstance(outputs, Tensor) else outputs):
                reject_data_dependent_control(item, visited)
            if expression.region.condition is not None:
                reject_data_dependent_control(
                    expression.region.condition, visited)

    reject_data_dependent_control(output, set())

    def contains_repeat(value: Tensor) -> bool:
        if id(value) in inspected:
            return False
        inspected.add(id(value))
        expression = value._expr
        if expression is None:
            return False
        if expression.op == "repeat":
            return True
        if any(contains_repeat(operand) for operand in expression.operands):
            return True
        if expression.region is None:
            return False
        outputs = expression.region.output
        return any(contains_repeat(item) for item in (
            (outputs,) if isinstance(outputs, Tensor) else outputs
        ))

    if not contains_repeat(output):
        return output, {}

    memo: dict[int, Tensor] = {}
    replacements: dict[int, Tensor] = {}
    repeat_groups: dict[int, tuple[Tensor, ...]] = {}

    def rebuild(value: Tensor) -> Tensor:
        cached = memo.get(id(value))
        if cached is not None:
            return cached
        expression = value._expr
        if expression is None:
            memo[id(value)] = value
            return value
        if expression.op == "loop_argument":
            raise RuntimeError("a loop argument escaped its repeat region")
        if expression.op == "paged_message_passing":
            # Eager execution boundary: keep the materialized node intact so
            # downstream primal saves read its buffer instead of re-running
            # the paged forward symbolically.
            memo[id(value)] = value
            return value
        if expression.op == "repeat":
            region = expression.region
            if region is None or not region.arguments:
                raise RuntimeError("repeat expression is missing its body region")
            grouped = repeat_groups.get(id(region))
            if grouped is not None:
                return grouped[int(expression.attr("result_index"))]
            operands = tuple(rebuild(operand) for operand in expression.operands)
            num_carried = int(expression.attr("num_carried"))
            current = list(operands[:num_carried])
            captures = operands[num_carried:]
            outputs = (region.output,) if isinstance(region.output, Tensor) \
                else region.output

            def instantiate(region_value: Tensor, local: dict[int, Tensor]) -> Tensor:
                found = local.get(id(region_value))
                if found is not None:
                    return found
                nested = region_value._expr
                if nested is None:
                    resolved = rebuild(region_value)
                elif nested.op == "repeat":
                    raise NotImplementedError(
                        "autograd of nested tg.repeat regions is not supported yet")
                elif nested.region is not None:
                    raise NotImplementedError(
                        "autograd supports only flat fixed repeat regions")
                else:
                    resolved = Tensor(
                        region_value.shape,
                        dtype=region_value.dtype,
                        device=region_value.device,
                        requires_grad=region_value.requires_grad,
                        expression=_Expr(
                            nested.op,
                            tuple(instantiate(item, local)
                                  for item in nested.operands),
                            nested.attrs,
                        ),
                        version=region_value.version,
                    )
                local[id(region_value)] = resolved
                return resolved

            iterations = int(expression.attr("iterations"))
            for _ in range(iterations):
                local = {
                    id(argument): operand
                    for argument, operand in zip(
                        region.arguments, (*current, *captures), strict=True)
                }
                current = [instantiate(item, local) for item in outputs]
            resolved = tuple(current)
            repeat_groups[id(region)] = resolved
            for handle, item in zip(region.results, resolved, strict=True):
                memo[id(handle)] = item
                replacements[id(handle)] = item
            return resolved[int(expression.attr("result_index"))]
        if expression.region is not None:
            raise NotImplementedError(
                f"autograd cannot expand control op {expression.op!r}")
        rebuilt = Tensor(
            value.shape,
            dtype=value.dtype,
            device=value.device,
            requires_grad=value.requires_grad,
            expression=_Expr(
                expression.op,
                tuple(rebuild(operand) for operand in expression.operands),
                expression.attrs,
            ),
            version=value.version,
        )
        memo[id(value)] = rebuilt
        replacements[id(value)] = rebuilt
        return rebuilt

    return rebuild(output), replacements


def _expand_program_leaves(output: Tensor) -> Tensor:
    """Replace program SSA leaves with their inline kernel expansions.

    Reverse mode through a program boundary is per leaf: each leaf's kernel
    re-expands into ordinary Tensor IR (no program capture), so the existing
    per-op VJP rules compose with the epilogue, and leaves sharing one input
    field accumulate contributions through the usual adjoint table. The
    forward fusion plan is untouched. Leaves whose kernels have no inline
    expansion (unsupported relation realization, provider-owned fields)
    fail closed with a clear error instead of a silent wrong gradient.
    """
    from ..program import leaf_expression

    # Inline expansions are temporary DAGs. Retain each source node alongside
    # its replacement: a bare id -> replacement cache lets an expansion die,
    # then mistakes a later leaf's recycled Python id for the previous node.
    memo: dict[int, tuple[Tensor, Tensor]] = {}

    def rebuild(value: Tensor) -> Tensor:
        cached = memo.get(id(value))
        if cached is not None:
            return cached[1]
        expression = value._expr
        if expression is None:
            memo[id(value)] = (value, value)
            return value
        if expression.op == "program_value":
            program = expression.attr("program")
            leaf = program._values[int(expression.attr("producer"))]
            try:
                expanded = leaf_expression(leaf)
            except NotImplementedError as error:
                raise NotImplementedError(
                    f"autograd cannot differentiate through program leaf "
                    f"{type(leaf.program._apply_specs[leaf.producer].kernel).__name__}: "
                    f"{error}"
                ) from error
            # A dependent apply re-expands to an expression that itself
            # references earlier leaves; recurse so they expand too.
            rebuilt = rebuild(expanded)
            memo[id(value)] = (value, rebuilt)
            return rebuilt
        if expression.op == "paged_message_passing":
            # Eager execution boundary with a materialized buffer; there are
            # no program leaves below it that need expansion.
            memo[id(value)] = (value, value)
            return value
        if expression.region is not None:
            raise NotImplementedError(
                f"autograd cannot expand control op {expression.op!r}")
        rebuilt = Tensor(
            value.shape,
            dtype=value.dtype,
            device=value.device,
            requires_grad=value.requires_grad,
            expression=_Expr(
                expression.op,
                tuple(rebuild(operand) for operand in expression.operands),
                expression.attrs,
            ),
            version=value.version,
        )
        memo[id(value)] = (value, rebuilt)
        return rebuilt

    return rebuild(output)


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
        raise TypeError("output must be a tiga.Tensor")
    single = isinstance(inputs, Tensor)
    requested = (inputs,) if single else tuple(inputs)
    if any(not isinstance(value, Tensor) for value in requested):
        raise TypeError("inputs must contain only tiga.Tensor values")
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

    differentiated_output, control_replacements = _expand_fixed_repeats(output)
    differentiated_output = _expand_program_leaves(differentiated_output)
    adjoints: dict[int, Tensor] = {id(differentiated_output): grad_output}

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

    for value in reversed(_topological_sort(differentiated_output)):
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
        elif expression.op == "device_copy":
            _accumulate(adjoints, operands[0], upstream.to(operands[0].device))
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
        elif expression.op == "scatter_rows":
            operand, destination, _inverse = operands
            _accumulate(adjoints, operand, upstream.gather(destination))
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
        elif expression.op == "paged_message_passing":
            # Eager execution boundary: the paged executor replays the
            # recorded plan page by page, differentiating against page-local
            # field views so no adjoint exceeds one page; overlapping source
            # adjoints scatter-accumulate into a disk-resident gradient file.
            from ..message_passing.paged import execute_paged_vjp

            contributions = execute_paged_vjp(expression.attr("plan"), upstream)
            if len(contributions) != len(operands):
                raise RuntimeError(
                    "paged MessagePassing VJP returned a wrong contribution count")
            for operand, contribution in zip(operands, contributions):
                if contribution is not None:
                    _accumulate(adjoints, operand, contribution)
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
                f"no Tiga VJP rule is registered for {expression.op!r}")

    results: list[Tensor] = []
    for value in requested:
        differentiated_value = control_replacements.get(id(value), value)
        result = adjoints.get(id(differentiated_value))
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
