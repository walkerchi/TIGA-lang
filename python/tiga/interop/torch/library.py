"""Dynamically registered ``torch.library`` facade for Tiga UDFs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping


_LIBRARIES: list[Any] = []
_REGISTERED: dict[str, "RegisteredMessagePassingOp"] = {}


def _ordered_fields(*groups: Mapping[str, Any]):
    return tuple(
        (role, name)
        for role, group in zip(("src", "dst", "edge"), groups, strict=True)
        for name in sorted(group)
    )


@dataclass(frozen=True)
class RegisteredMessagePassingOp:
    """One immutable Torch dispatcher binding for a compiled UDF/schema."""

    qualified_name: str
    operation: Any
    topology: tuple[Any, Any]
    bindings: tuple[tuple[str, str], ...]
    static_params: Mapping[str, Any]

    def __call__(
        self,
        *,
        src: Mapping[str, Any],
        dst: Mapping[str, Any],
        edge: Mapping[str, Any] | None = None,
    ):
        edge = {} if edge is None else edge
        groups = {"src": src, "dst": dst, "edge": edge}
        expected = {
            role: {name for candidate, name in self.bindings if candidate == role}
            for role in ("src", "dst", "edge")
        }
        for role, group in groups.items():
            if set(group) != expected[role]:
                raise ValueError(
                    f"{role} fields must be {sorted(expected[role])}, "
                    f"got {sorted(group)}")
        return self.operation(
            *self.topology,
            *(groups[role][name] for role, name in self.bindings),
        )

    def opcheck(self, *values: Any) -> dict[str, str]:
        """Run PyTorch schema/autograd/FakeTensor/AOT conformance checks."""
        import torch

        return torch.library.opcheck(
            self.operation, (*self.topology, *values))


def register_message_passing(
    kernel,
    *,
    graph,
    src: Mapping[str, Any],
    dst: Mapping[str, Any],
    edge: Mapping[str, Any] | None = None,
    name: str | None = None,
    **params: Any,
) -> RegisteredMessagePassingOp:
    """Register a Tiga UDF specialization as a functional Torch op.

    Tensor arguments remain explicit dispatcher operands.  Graph topology,
    field names and non-Tensor parameters form the immutable specialization.
    The real implementation enters Tiga's normal compiler path; the
    backward replays the semantic oracle under autograd, so users never write
    a separate backward while compiler-native VJPs can replace that adapter
    implementation independently.
    """
    import torch

    edge = {} if edge is None else edge
    if graph.schema.realization != "materialized_csr" or \
            graph.schema.lifecycle != "static":
        raise NotImplementedError(
            "torch.library registration currently requires a static CSR graph")
    row_ptr, col_idx = graph.resolve_csr()
    if not isinstance(row_ptr, torch.Tensor):
        row_ptr = row_ptr.to_torch()
    if not isinstance(col_idx, torch.Tensor):
        col_idx = col_idx.to_torch()
    bindings = _ordered_fields(src, dst, edge)
    groups = {"src": src, "dst": dst, "edge": edge}
    values = tuple(groups[role][field] for role, field in bindings)
    if not values or any(not isinstance(value, torch.Tensor) for value in values):
        raise TypeError("registered MessagePassing fields must be torch.Tensor values")
    if any(isinstance(value, torch.Tensor) for value in params.values()):
        raise TypeError("Tensor parameters must be expressed as src/dst/edge fields")

    # The sample establishes result pytree/shape/dtype without hiding a copy.
    with torch.no_grad():
        sample = kernel.reference(
            graph=graph, src=src, dst=dst, edge=edge, **params)
    if not isinstance(sample, torch.Tensor):
        raise NotImplementedError(
            "the initial torch.library facade requires one Tensor result")
    output_shape = tuple(sample.shape)
    output_dtype = sample.dtype
    signature = repr((
        type(kernel).__module__, type(kernel).__qualname__,
        kernel.reducer.specialization_key(), graph.planning_key(), bindings,
        tuple((tuple(value.shape), str(value.dtype)) for value in values),
        tuple(sorted((key, repr(value)) for key, value in params.items())),
        output_shape, str(output_dtype),
    ))
    digest = hashlib.sha256(signature.encode()).hexdigest()[:20]
    op_name = name or f"message_passing_{digest}"
    if not op_name.replace("_", "").isalnum() or not op_name[0].isalpha():
        raise ValueError("Torch op name must begin with a letter and be alphanumeric/_")
    qualified = f"tiga::{op_name}"
    found = _REGISTERED.get(qualified)
    if found is not None:
        return found

    library = torch.library.Library("tiga", "FRAGMENT")
    arguments = ", ".join(
        ["Tensor row_ptr", "Tensor col_idx", *(
            f"Tensor input_{index}" for index in range(len(bindings))
        )]
    )
    library.define(f"{op_name}({arguments}) -> Tensor")

    def unpack(flat):
        materialized = {"src": {}, "dst": {}, "edge": {}}
        for (role, field), value in zip(bindings, flat, strict=True):
            materialized[role][field] = value
        return materialized

    def runtime_graph(flat):
        from ...graph import Graph

        return Graph.from_csr(
            flat[0], flat[1], num_src=graph.schema.num_src,
            validate="basic")

    def implementation(*flat):
        materialized = unpack(flat[2:])
        return kernel(
            graph=runtime_graph(flat),
            src=materialized["src"], dst=materialized["dst"],
            edge=materialized["edge"], **params)

    library.impl(op_name, implementation, "CompositeExplicitAutograd")

    def fake(*flat):
        return flat[2].new_empty(output_shape, dtype=output_dtype)

    torch.library.register_fake(qualified, fake, lib=library)

    def setup_context(ctx, inputs, output):
        del output
        ctx.save_for_backward(*inputs)

    def backward(context, grad_output):
        replay = []
        differentiable = []
        positions = []
        for index, (value, needed) in enumerate(
            zip(context.saved_tensors, context.needs_input_grad, strict=True)
        ):
            local = value.detach()
            if needed and (local.is_floating_point() or local.is_complex()):
                local.requires_grad_(True)
                differentiable.append(local)
                positions.append(index)
            replay.append(local)
        if not differentiable:
            return (None,) * len(replay)
        materialized = unpack(replay[2:])
        with torch.enable_grad():
            output = kernel.reference(
                graph=runtime_graph(replay), src=materialized["src"],
                dst=materialized["dst"], edge=materialized["edge"], **params)
            computed = torch.autograd.grad(
                output, differentiable, grad_output, allow_unused=True)
        gradients = [None] * len(replay)
        for position, gradient in zip(positions, computed, strict=True):
            gradients[position] = gradient
        return tuple(gradients)

    torch.library.register_autograd(
        qualified, backward, setup_context=setup_context, lib=library)
    operation = getattr(getattr(torch.ops.tiga, op_name), "default")
    registered = RegisteredMessagePassingOp(
        qualified_name=qualified,
        operation=operation,
        topology=(row_ptr, col_idx),
        bindings=bindings,
        static_params=dict(params),
    )
    _LIBRARIES.append(library)  # dispatcher registrations follow Library lifetime
    _REGISTERED[qualified] = registered
    return registered


__all__ = ["RegisteredMessagePassingOp", "register_message_passing"]
