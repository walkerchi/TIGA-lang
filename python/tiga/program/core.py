"""Native-MLIR-backed straight-line composition of MessagePassing leaves."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
import hashlib
from typing import Any, Mapping

from ..compiler.domain_capture import capture_message_passing
from ..compiler.native import domain_ir, program_ir
from ..tensor import complex64, complex128, float16, float32, float64


_ACTIVE: ContextVar[Any] = ContextVar(
    "graphforge_active_program", default=None
)


@dataclass(frozen=True)
class ProgramValue:
    """One typed SSA result owned by a :class:`GraphProgram`.

    A leaf also participates in tensor arithmetic: binary operations adapt
    it into the Tensor expression system through :func:`leaf_tensor`, where
    the expression node references ``(program, producer)``. Observation of
    such an expression runs the program first (fusing the leaves it selects)
    and then evaluates the epilogue on the leaf results.
    """

    program: "GraphProgram"
    producer: int
    shape: tuple[int, ...]
    dtype: object
    version: int = 0

    def ir(self, stage: str = "domain") -> str:
        return self.program.ir(stage)

    @property
    def semantic_hash(self) -> str:
        return self.program.semantic_hash

    def _as_tensor(self):
        """View this SSA leaf as a Tensor expression leaf for arithmetic."""
        return leaf_tensor(self)

    def __add__(self, other):
        return self._as_tensor() + other

    def __radd__(self, other):
        return self._as_tensor().__radd__(other)

    def __sub__(self, other):
        return self._as_tensor() - other

    def __rsub__(self, other):
        return self._as_tensor().__rsub__(other)

    def __mul__(self, other):
        return self._as_tensor() * other

    def __rmul__(self, other):
        return self._as_tensor().__rmul__(other)

    def __truediv__(self, other):
        return self._as_tensor() / other

    def __rtruediv__(self, other):
        return self._as_tensor().__rtruediv__(other)

    def __neg__(self):
        return -self._as_tensor()

    def __pos__(self):
        return self._as_tensor()

    def materialize(self):
        """JIT and execute the connected program at this observation boundary."""
        self.program.outputs(self)
        return self.program.run()


def leaf_tensor(value: ProgramValue):
    """Adapt one program SSA leaf into the Tensor expression system.

    The resulting Tensor is an expression leaf whose ``program_value`` node
    references ``(program, producer)``; evaluating it runs the owning program
    and takes the producer's result. No arithmetic is folded into the leaf
    kernels — epilogue math stays in the Tensor expression, honestly outside
    the fused launch. The leaf is differentiable when any captured field
    requires gradients; reverse mode re-expands the leaf's kernel inline
    (see :func:`leaf_expression`).
    """
    from ..tensor.core import _Expr, Tensor

    spec = value.program._apply_specs[value.producer]
    requires_grad = any(
        isinstance(field, Tensor) and field.requires_grad
        for namespace in (spec.src, spec.dst, spec.edge, spec.params)
        for field in namespace.values()
    )
    return Tensor(
        value.shape,
        dtype=value.dtype,
        device=spec.graph.device,
        requires_grad=requires_grad,
        expression=_Expr(
            "program_value",
            (),
            (("program", value.program), ("producer", value.producer)),
        ),
        version=value.version,
    )


def leaf_expression(value: ProgramValue):
    """Re-expand one leaf's kernel as an inline Tensor expression.

    Autograd differentiates through a program boundary per leaf: the leaf's
    recorded fields replay through the ordinary inline MessagePassing path
    (no program capture), so the standard per-op VJP rules apply and
    contributions onto shared input fields accumulate. The forward fusion
    plan is untouched — reverse mode is honestly one expansion per leaf.
    """
    from ..tensor import Tensor

    spec = value.program._apply_specs[value.producer]

    def rebind(mapping):
        return {
            name: leaf_tensor(field) if isinstance(field, ProgramValue) else field
            for name, field in mapping.items()
        }

    for namespace in (spec.src, spec.dst, spec.edge, spec.params):
        for field in namespace.values():
            if not isinstance(field, (Tensor, ProgramValue)):
                raise NotImplementedError(
                    "autograd through a GraphProgram leaf requires native "
                    "tiga.Tensor fields; provider-owned (e.g. torch) "
                    "storage has no symbolic VJP across the program boundary")
    token = _ACTIVE.set(None)
    try:
        expanded = spec.kernel(
            graph=spec.graph,
            src=rebind(spec.src),
            dst=rebind(spec.dst),
            edge=rebind(spec.edge),
            **dict(spec.params),
        )
    finally:
        _ACTIVE.reset(token)
    if not isinstance(expanded, Tensor):
        raise NotImplementedError(
            "autograd through a GraphProgram leaf requires the inline "
            "Tensor expansion of its kernel")
    return expanded


@dataclass(frozen=True)
class _ApplySpec:
    kernel: object
    graph: object
    src: Mapping[str, object]
    dst: Mapping[str, object]
    edge: Mapping[str, object]
    params: Mapping[str, object]


class GraphProgram:
    """Capture a typed, acyclic sequence of coarse relation operations.

    Calling ``ir`` is the observation boundary: native MLIR composes the leaf
    captures, verifies SSA types, CSEs identical relation snapshots, forms
    legal horizontal fusion groups, and lowers through Kernel IR.
    """

    def __init__(self) -> None:
        self._modules: list[str] = []
        self._bindings: list[tuple[int, ...]] = []
        self._values: list[ProgramValue] = []
        self._apply_specs: list[_ApplySpec] = []
        self._external_slots: dict[tuple[object, ...], int] = {}
        self._external_values: list[object] = []
        self._outputs: tuple[int, ...] = ()
        self._stages: tuple[str, str, str, str] | None = None
        self._executable = None
        self._artifacts: dict[str, str | bytes] = {}

    def _external(self, key: tuple[object, ...], value: object) -> int:
        slot = self._external_slots.get(key)
        if slot is None:
            slot = len(self._external_slots)
            self._external_slots[key] = slot
            self._external_values.append(value)
        return -(slot + 1)

    def _operand(self, value: object, key: tuple[object, ...]) -> int:
        if isinstance(value, ProgramValue):
            if value.program is not self:
                raise ValueError("a ProgramValue cannot cross GraphProgram instances")
            expected = self._values[value.producer]
            if value.version != expected.version:
                raise ValueError(
                    "stale ProgramValue snapshot version: "
                    f"expected {expected.version}, got {value.version}"
                )
            return value.producer
        return self._external((key, id(value)), value)

    def apply(
        self,
        kernel,
        *,
        graph,
        src: Mapping[str, object],
        dst: Mapping[str, object],
        edge: Mapping[str, object] | None = None,
        **params: Any,
    ) -> ProgramValue:
        edge = {} if edge is None else edge
        descriptor = capture_message_passing(
            kernel=kernel, graph=graph, src=src, dst=dst, edge=edge,
            params=params,
            kernel_name=f"program_apply_{len(self._values)}_{type(kernel).__name__}",
        )
        bindings: list[int] = []
        relation_arguments = {
            "materialized_csr": 2,
            "generated_radius": 7,
            "implicit_dense": 0,
        }[descriptor.graph.realization]
        if descriptor.graph.realization == "materialized_csr":
            relation_values = graph.resolve_csr()
        elif descriptor.graph.realization == "generated_radius":
            directory = graph.generated_cell_directory()
            relation_values = (
                graph.euclidean_positions(), directory.cell_ptr,
                directory.particle_order, directory.cell_coordinates,
                directory.extents, directory.strides,
                directory.neighbor_offsets,
            )
        else:
            relation_values = ()
        for index in range(relation_arguments):
            bindings.append(self._external(
                ("relation", id(graph), index), relation_values[index]
            ))
        namespaces = {"src": src, "dst": dst, "edge": edge}
        for field in descriptor.fields:
            value = namespaces[field.role][field.name]
            bindings.append(
                self._operand(value, ("field",))
            )
        for parameter in descriptor.params:
            value = params[parameter.name]
            bindings.append(
                self._operand(value, ("param", parameter.name, len(self._values)))
            )

        result_spelling = descriptor.reducer.result_dtypes[0]
        dtype_name = result_spelling
        result_shape = (descriptor.graph.num_dst,)
        if result_spelling.startswith("vector:"):
            _, width, dtype_name = result_spelling.split(":")
            result_shape += (int(width),)
        dtype = {
            "float16": float16,
            "float32": float32,
            "float64": float64,
            "complex64": complex64,
            "complex128": complex128,
        }.get(dtype_name)
        if dtype is None:
            raise NotImplementedError(
                f"GraphProgram output dtype {result_spelling!r} is unsupported"
            )
        value = ProgramValue(
            self, len(self._values), result_shape, dtype, version=0
        )
        self._modules.append(domain_ir(descriptor))
        self._bindings.append(tuple(bindings))
        self._apply_specs.append(_ApplySpec(
            kernel, graph, dict(src), dict(dst), dict(edge), dict(params)))
        self._values.append(value)
        self._outputs = (value.producer,)
        self._stages = None
        self._executable = None
        self._artifacts = {}
        return value

    def _output_producers(
        self, value: object, _seen: set[int] | None = None
    ) -> list[int]:
        """Collect the producers a boundary value transitively references.

        Bare leaves contribute themselves; a Tensor expression contributes
        every ``program_value`` leaf it references, so epilogue arithmetic
        composed on top of leaves is a legal output spelling.
        """
        from ..tensor import Tensor

        if isinstance(value, ProgramValue):
            if value.program is not self:
                raise ValueError("a ProgramValue cannot cross GraphProgram instances")
            return [value.producer]
        if not isinstance(value, Tensor):
            raise ValueError("every output must belong to this GraphProgram")
        seen = set() if _seen is None else _seen
        if id(value) in seen:
            return []
        seen.add(id(value))
        expression = value._expr
        if expression is None:
            return []
        if expression.op == "program_value":
            program = expression.attr("program")
            if program is not self:
                raise ValueError("a ProgramValue cannot cross GraphProgram instances")
            return [int(expression.attr("producer"))]
        producers: list[int] = []
        for operand in expression.operands:
            producers.extend(self._output_producers(operand, seen))
        return producers

    def outputs(self, *values: ProgramValue) -> "GraphProgram":
        """Select the values observed at the composition boundary.

        Bare leaves select their producers directly; Tensor expressions
        contribute every referenced program leaf, keeping the leaf set —
        and with it the fusion plan — identical to the bare-leaf spelling.
        """
        if not values:
            raise ValueError("GraphProgram requires at least one output")
        producers = [
            producer
            for value in values
            for producer in self._output_producers(value)
        ]
        if not producers:
            raise ValueError("GraphProgram outputs must reference a captured leaf")
        selected = tuple(producers)
        if selected != self._outputs:
            self._outputs = selected
            self._stages = None
            self._executable = None
            self._artifacts = {}
        return self

    def _lower(self) -> tuple[str, str, str, str]:
        if not self._modules:
            raise RuntimeError("cannot lower an empty GraphProgram")
        if self._stages is None:
            self._stages = program_ir(
                tuple(self._modules), tuple(self._bindings), self._outputs
            )
        return self._stages

    def ir(self, stage: str = "domain") -> str:
        index = {
            "domain": 0,
            "fused": 1,
            "iteration": 2,
            "iter": 2,
            "kernel": 3,
            "gf.kernel": 3,
        }.get(stage)
        if index is None:
            raise KeyError(
                "GraphProgram stage must be domain, fused, iteration, or kernel"
            )
        return self._lower()[index]

    def _compile_product(self):
        if self._executable is not None:
            return self._executable
        from ..codegen import prepare_ttir_csr_product
        from ..interop.torch.compiler_bridge import (
            lower_kernel_to_ttir,
            parse_kernel_ttir_plan,
        )

        kernel = self.ir("kernel")
        ttir = lower_kernel_to_ttir(kernel)
        manifest = parse_kernel_ttir_plan(ttir)
        if manifest.entry != "gf_csr_product_additive_tile":
            raise NotImplementedError(
                "GraphProgram execution currently requires one fused CSR "
                "multi-result product launch"
            )
        def provider_value(value):
            return value.to_torch() if hasattr(value, "to_torch") else value

        first = self._bindings[0]
        row_slot, col_slot = -first[0] - 1, -first[1] - 1
        plan = prepare_ttir_csr_product(
            ttir,
            row_ptr=provider_value(self._external_values[row_slot]),
            col_idx=provider_value(self._external_values[col_slot]),
            num_rows=self._values[0].shape[0],
            block_rows=manifest.block_rows,
            num_results=len(self._modules),
            num_warps=manifest.num_warps,
        )
        runtime_inputs: list[object] = []
        for bindings in self._bindings:
            for binding in bindings[2:]:
                if binding >= 0:
                    raise NotImplementedError(
                        "cross-apply GraphProgram execution requires task-DAG "
                        "lowering; its SSA IR is available for inspection"
                    )
                runtime_inputs.append(
                    provider_value(self._external_values[-binding - 1])
                )
        self._executable = (plan, tuple(runtime_inputs))
        self._artifacts = {"ttir": ttir, **dict(plan.result.artifacts)}
        return self._executable

    def _product_path_available(self) -> bool:
        """The fused multi-result product launch is a CUDA/Triton kernel.

        Without CUDA-backed torch fields there is no fused executable; the
        program then falls back to executing each apply as an ordinary
        kernel through the task DAG, and the fusion remains an IR-level plan.
        The missing CPU multi-result fused kernel is an unimplemented
        performance feature, not a structural limitation: on CPU the task
        DAG executes per kernel with identical result semantics.
        """
        try:
            import torch
        except ImportError:
            return False
        from ..tensor import Tensor

        for value in self._external_values:
            if isinstance(value, torch.Tensor):
                if not value.is_cuda:
                    return False
            elif isinstance(value, Tensor):
                if not str(value.device).startswith("cuda"):
                    return False
            # scalars and other non-tensor externals ride the launch by value
        return True

    def run(self):
        """Automatically JIT and execute the currently selected outputs."""
        if any(binding >= 0 for bindings in self._bindings for binding in bindings):
            return self._run_task_dag()
        if len(self._modules) == 1:
            return self._run_single()
        if not self._product_path_available():
            return self._run_task_dag()
        plan, inputs = self._compile_product()
        all_results = plan.run(*inputs)
        results = tuple(all_results[index] for index in self._outputs)
        return results[0] if len(results) == 1 else results

    def _run_single(self):
        """Execute a one-leaf program through the ordinary kernel path.

        Fusing a single leaf is the identity, so the leaf runs as the same
        lazy-JIT MessagePassing call it was captured from instead of the
        multi-result product launch.
        """
        spec = self._apply_specs[0]
        result = spec.kernel(
            graph=spec.graph,
            src=dict(spec.src),
            dst=dict(spec.dst),
            edge=dict(spec.edge),
            **dict(spec.params),
        )
        self._executable = result
        self._artifacts = {}
        return result

    def _observed_results(self) -> dict[int, object]:
        """Run the selected outputs once; return a producer → value mapping."""
        if not self._outputs:
            self.outputs(*self._values)
        selected = self._outputs
        result = self.run()
        values = (result,) if len(selected) == 1 else tuple(result)
        return dict(zip(selected, values, strict=True))

    def _run_task_dag(self):
        """Execute dependent applies through a typed runtime dependency DAG.

        Each leaf remains an ordinary lazy-JIT MessagePassing kernel. The DAG
        owns inter-apply ordering and live SSA values; independent leaves take
        the horizontal product lowering when ``_product_path_available`` and
        otherwise run per-kernel through this same DAG (e.g. on CPU, where no
        fused product kernel exists).
        """
        from ..runtime import (
            ExecutableBundle, KernelInvocation, ResourceAccess,
            SynchronousProvider,
        )

        produced: list[object | None] = [None] * len(self._apply_specs)

        def resolve(value):
            if not isinstance(value, ProgramValue):
                return value
            result = produced[value.producer]
            if result is None:
                raise RuntimeError(
                    f"GraphProgram value {value.producer} was consumed before production")
            return result

        invocations = []
        resources = {
            f"value:{index}": object() for index in range(len(self._apply_specs))
        }
        for index, spec in enumerate(self._apply_specs):
            dependencies = sorted({
                binding for binding in self._bindings[index] if binding >= 0
            })

            def execute(index=index, spec=spec):
                produced[index] = spec.kernel(
                    graph=spec.graph,
                    src={name: resolve(value) for name, value in spec.src.items()},
                    dst={name: resolve(value) for name, value in spec.dst.items()},
                    edge={name: resolve(value) for name, value in spec.edge.items()},
                    **{name: resolve(value) for name, value in spec.params.items()},
                )

            accesses = tuple(
                ResourceAccess(f"value:{dependency}", "read")
                for dependency in dependencies
            ) + (ResourceAccess(f"value:{index}", "write"),)
            invocations.append(KernelInvocation(
                f"apply_{index}", execute,
                arguments=(),
                depends_on=tuple(f"apply_{item}" for item in dependencies),
                accesses=accesses,
                task_kind="program-apply",
            ))
        bundle = ExecutableBundle(
            tuple(invocations), required_bindings=tuple(resources))
        bundle.submit(SynchronousProvider(), resources).wait()
        self._executable = bundle
        self._artifacts = {}
        for index, spec in enumerate(self._apply_specs):
            variant = getattr(spec.kernel, "last_variant", None)
            if variant is None:
                continue
            for kind, artifact in variant.artifacts.items():
                if isinstance(artifact, (str, bytes)):
                    self._artifacts[f"apply_{index}.{kind}"] = artifact
        results = tuple(produced[index] for index in self._outputs)
        return results[0] if len(results) == 1 else results

    def code(self, kind: str = "ptx") -> str | bytes | Mapping[str, str | bytes]:
        dependent = any(
            binding >= 0 for bindings in self._bindings for binding in bindings)
        if dependent:
            if not self._artifacts:
                self._run_task_dag()
            selected = {
                name: value for name, value in self._artifacts.items()
                if name.endswith(f".{kind}")
            }
            if selected:
                return selected
        else:
            self._compile_product()
        try:
            return self._artifacts[kind]
        except KeyError as error:
            raise KeyError(
                f"GraphProgram artifact {kind!r} is unavailable; have "
                f"{sorted(self._artifacts)}"
            ) from error

    def explain(self) -> str:
        fused = self.ir("fused").count('"gf.apply"')
        return (
            f"GraphProgram applies={len(self._modules)}, post_fusion={fused}\n"
            f"semantic_hash={self.semantic_hash}\n"
            "observation triggers native Domain→Kernel→provider JIT"
        )

    @property
    def semantic_hash(self) -> str:
        return hashlib.sha256(self.ir("domain").encode()).hexdigest()


def active_program() -> GraphProgram | None:
    return _ACTIVE.get()


def capture_boundary(function):
    """Activate a fresh GraphProgram context for the duration of one call.

    This is the shared capture boundary behind ``@gf.program`` (kept as a
    compatibility entry point) and the automatic capture in ``@gf.jit``.
    Stacking the two is legal and idempotent: an already-active program is
    reused instead of replaced. When the call registers no kernel leaves the
    boundary is a no-op and the result passes through unchanged; when the
    result carries leaves (bare ``ProgramValue`` or inside a Tensor
    expression), the referenced leaves become the program's outputs.
    """

    @wraps(function)
    def captured(*args, **kwargs):
        if active_program() is not None:
            return function(*args, **kwargs)
        graph_program = GraphProgram()
        token = _ACTIVE.set(graph_program)
        try:
            result = function(*args, **kwargs)
        finally:
            _ACTIVE.reset(token)
        if not graph_program._values:
            # No kernel call was captured: the boundary is a no-op.
            return result
        from ..tensor import Tensor

        values = result if isinstance(result, tuple) else (result,)
        producers = [
            producer
            for value in values
            if isinstance(value, (ProgramValue, Tensor))
            for producer in graph_program._output_producers(value)
        ]
        if producers:
            graph_program.outputs(*(
                value
                for value in values
                if isinstance(value, (ProgramValue, Tensor))
            ))
        return result

    return captured


def program(function):
    """Compatibility capture boundary; ``@gf.jit`` now captures automatically.

    Straight-line code that cannot offer source access (or simply predates
    the unified entry point) can still use ``@gf.program`` to activate the
    program context without AST rewriting. New code should use ``@gf.jit``.
    """
    return capture_boundary(function)


__all__ = [
    "GraphProgram", "ProgramValue", "active_program", "capture_boundary",
    "leaf_tensor", "program",
]
