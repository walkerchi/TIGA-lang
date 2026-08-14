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
    """One typed SSA result owned by a :class:`GraphProgram`."""

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

    def materialize(self):
        """JIT and execute the connected program at this observation boundary."""
        self.program.outputs(self)
        return self.program.run()


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

    def outputs(self, *values: ProgramValue) -> "GraphProgram":
        if not values:
            raise ValueError("GraphProgram requires at least one output")
        for value in values:
            if not isinstance(value, ProgramValue) or value.program is not self:
                raise ValueError("every output must belong to this GraphProgram")
        selected = tuple(value.producer for value in values)
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

    def run(self):
        """Automatically JIT and execute the currently selected outputs."""
        if any(binding >= 0 for bindings in self._bindings for binding in bindings):
            return self._run_task_dag()
        plan, inputs = self._compile_product()
        all_results = plan.run(*inputs)
        results = tuple(all_results[index] for index in self._outputs)
        return results[0] if len(results) == 1 else results

    def _run_task_dag(self):
        """Execute dependent applies through a typed runtime dependency DAG.

        Each leaf remains an ordinary lazy-JIT MessagePassing kernel. The DAG
        owns inter-apply ordering and live SSA values; independent leaves are
        still handled by the horizontal product lowering in ``_compile_product``.
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


def program(function):
    """Optional capture boundary; ordinary leaf calls remain automatic JIT."""

    @wraps(function)
    def captured(*args, **kwargs):
        graph_program = GraphProgram()
        token = _ACTIVE.set(graph_program)
        try:
            result = function(*args, **kwargs)
        finally:
            _ACTIVE.reset(token)
        values = result if isinstance(result, tuple) else (result,)
        graph_program.outputs(*values)
        return result

    return captured


__all__ = ["GraphProgram", "ProgramValue", "active_program", "program"]
