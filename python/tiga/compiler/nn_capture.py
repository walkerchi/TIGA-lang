"""Capture torch.nn modules called inside MessagePassing edge UDFs.

``gf.nn.trace(module)`` wraps a module so one source drives two paths:

- called with Torch tensors (eager/reference execution) the inputs are
  concatenated along the feature dimension and forwarded to the module
  unchanged;
- called with compiler capture proxies (``Expr`` field expressions) the call
  becomes an ``Expr("nn_subgraph", ...)`` node carrying a translated
  ``MessageDAG`` — the module's ``torch.fx`` graph restricted to the op set
  the fused edge-NN tile lowering supports.

The DAG is a typed frontend descriptor, not generated source.  Structure
proofs and TTIR emission live in ``compiler.edge_nn_tile``.

Supported structure (all strictly edge-local: edge ``e``'s activations never
depend on another edge):

- ``nn.Linear`` layers (bias optional);
- component-wise activations, module or functional form: relu, sigmoid,
  tanh, gelu (exact), silu, elu, leaky_relu, hardtanh/clamp (relu6 included),
  hardsigmoid, hardswish, mish, selu, softplus, exp, log, sqrt, rsqrt, abs,
  sin, cos, square;
- scalar-constant arithmetic ``x * c``, ``x + c``, ``c - x``, ``x / c``,
  ``c / x``, ``x ** p``, ``-x`` (temperature scaling, affine shifts);
- feature-wise ``nn.LayerNorm`` — normalizes one edge's message vector over
  its feature axis, which never couples edges;
- ``nn.Identity`` (no-op, skipped).

Anything else raises at capture and the eager oracle owns the semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .capture import Expr

BLOCK_E = 128
NUM_WARPS = 4


@dataclass(frozen=True)
class LinearLayer:
    """One ``y = x @ weight.T + bias`` tile op in the message chain."""

    module_path: str
    in_features: int
    out_features: int
    weight_name: str
    bias_name: str | None


@dataclass(frozen=True)
class ActivationLayer:
    """One component-wise tile op in the message chain.

    ``params`` carries the float constants of parameterized ops
    (``(negative_slope,)`` for leaky_relu, ``(min, max)`` for hardtanh with
    ``None`` for an open side, ``(c,)`` for scalar-constant arithmetic, …).
    """

    kind: str
    params: tuple[float | None, ...] = ()


@dataclass(frozen=True)
class NormLayer:
    """Feature-wise LayerNorm over one edge's message vector (edge-local)."""

    module_path: str
    width: int
    eps: float
    weight_name: str | None
    bias_name: str | None

    @property
    def in_features(self) -> int:
        return self.width

    @property
    def out_features(self) -> int:
        return self.width


@dataclass(frozen=True)
class MessageDAG:
    """A linear chain of tile ops plus the weight source for the chain."""

    layers: tuple[LinearLayer | ActivationLayer | NormLayer, ...]
    in_features: int
    out_features: int
    # None = the consuming lowering picks its own default (edge-centric sum
    # tiles favor 128/4, row-centric attention tiles favor 16/1).
    block_e: int | None = None
    num_warps: int | None = None
    # The traced GraphModule owns the parameters; it is identity metadata,
    # not part of the structural key.
    graph_module: Any = field(compare=False, hash=False, default=None)


def _module_activation(module) -> tuple[str, tuple[float | None, ...]] | None:
    from torch import nn

    simple = {
        nn.ReLU: "relu",
        nn.Sigmoid: "sigmoid",
        nn.Tanh: "tanh",
        nn.SiLU: "silu",
        nn.Hardsigmoid: "hardsigmoid",
        nn.Hardswish: "hardswish",
        nn.Mish: "mish",
        nn.SELU: "selu",
    }
    kind = simple.get(type(module))
    if kind is not None:
        return (kind, ())
    # nn.ReLU6 subclasses nn.Hardtanh; check it first.
    if isinstance(module, nn.ReLU6):
        return ("hardtanh", (0.0, 6.0))
    if isinstance(module, nn.Hardtanh):
        return ("hardtanh", (float(module.min_val), float(module.max_val)))
    if isinstance(module, nn.GELU):
        if module.approximate != "none":
            raise NotImplementedError(
                "edge nn subgraphs support exact gelu only "
                f"(approximate={module.approximate!r})")
        return ("gelu", ())
    if isinstance(module, nn.LeakyReLU):
        return ("leaky_relu", (float(module.negative_slope),))
    if isinstance(module, nn.ELU):
        return ("elu", (float(module.alpha),))
    if isinstance(module, nn.Softplus):
        return ("softplus", (float(module.beta), float(module.threshold)))
    return None


def _scalar_pair(args, previous) -> tuple[float, bool] | None:
    """Return ``(constant, reversed)`` for ``op(stream, c)``/``op(c, stream)``."""
    if len(args) != 2:
        return None
    first, second = args
    if first is previous and isinstance(second, (int, float)):
        return (float(second), False)
    if second is previous and isinstance(first, (int, float)):
        return (float(first), True)
    return None


def _function_activation(
    target, args, kwargs, previous
) -> tuple[str, tuple[float | None, ...]] | None:
    import operator

    import torch
    from torch.nn import functional

    # F.relu/F.silu take an optional positional inplace flag; drop it.
    if len(args) == 2 and isinstance(args[1], bool):
        args = args[:1]
    simple = {
        torch.relu: "relu",
        functional.relu: "relu",
        torch.sigmoid: "sigmoid",
        torch.tanh: "tanh",
        torch.exp: "exp",
        functional.silu: "silu",
        torch.log: "log",
        torch.sqrt: "sqrt",
        torch.rsqrt: "rsqrt",
        torch.abs: "abs",
        torch.sin: "sin",
        torch.cos: "cos",
        torch.square: "square",
        functional.hardsigmoid: "hardsigmoid",
        functional.hardswish: "hardswish",
        functional.mish: "mish",
        functional.selu: "selu",
    }
    kind = simple.get(target)
    if kind is not None:
        if args != (previous,):
            return None
        return (kind, ())

    # Scalar-constant arithmetic: one side is the stream, the other a float.
    alpha = float(kwargs.get("alpha", 1.0))
    pair = _scalar_pair(args, previous)
    if target in (operator.mul, torch.mul):
        if pair is None:
            return None
        return ("mul_const", (pair[0],))
    if target in (operator.add, torch.add):
        if pair is None:
            return None
        return ("add_const", (alpha * pair[0],))
    if target in (operator.sub, torch.sub):
        if pair is None:
            return None
        constant, reversed_ = pair
        if reversed_:
            if alpha != 1.0:
                return None  # c - alpha * x is affine in x, not constant-x
            return ("rsub_const", (constant,))
        return ("add_const", (-alpha * constant,))
    if target in (operator.truediv, torch.div, torch.true_divide):
        if pair is None:
            return None
        constant, reversed_ = pair
        return ("rdiv_const" if reversed_ else "div_const", (constant,))
    if target in (operator.pow, torch.pow):
        if pair is None or pair[1]:
            return None  # only x ** const
        return ("pow_const", (pair[0],))
    if target is operator.neg:
        if args != (previous,):
            return None
        return ("mul_const", (-1.0,))

    # Parameterized unary functions consume the stream as their first arg.
    if not args or args[0] is not previous:
        return None

    def kwarg(name, position, default):
        if name in kwargs:
            return kwargs[name]
        if len(args) > position and args[position] is not previous:
            return args[position]
        return default

    if target is functional.gelu:
        approximate = kwargs.get("approximate", "none")
        if approximate != "none":
            raise NotImplementedError(
                "edge nn subgraphs support exact gelu only "
                f"(approximate={approximate!r})")
        return ("gelu", ())
    if target is functional.leaky_relu:
        return ("leaky_relu", (float(kwarg("negative_slope", 1, 0.01)),))
    if target is functional.elu:
        return ("elu", (float(kwarg("alpha", 1, 1.0)),))
    if target is functional.hardtanh:
        return ("hardtanh", (
            float(kwarg("min_val", 1, -1.0)),
            float(kwarg("max_val", 2, 1.0))))
    if target is functional.softplus:
        return ("softplus", (
            float(kwarg("beta", 1, 1)), float(kwarg("threshold", 2, 20))))
    if target is torch.clamp:
        low = kwargs.get("min", args[1] if len(args) > 1 else None)
        high = kwargs.get("max", args[2] if len(args) > 2 else None)
        if low is None and high is None:
            return None
        return ("hardtanh", (
            None if low is None else float(low),
            None if high is None else float(high)))
    return None


def _translate_graph_module(graph_module) -> MessageDAG:
    """Translate a traced module graph into a linear MessageDAG chain."""
    layers: list[LinearLayer | ActivationLayer | NormLayer] = []
    previous = None
    width: int | None = None
    for node in graph_module.graph.nodes:
        if node.op == "placeholder":
            if previous is not None:
                raise NotImplementedError(
                    "edge nn subgraphs take exactly one placeholder")
            previous = node
            continue
        if node.op == "output":
            break
        if node.op == "call_module":
            if node.args != (previous,):
                raise NotImplementedError(
                    "edge nn subgraphs support single-input linear chains; "
                    f"node {node.name!r} has arguments {node.args!r}")
            target = graph_module.get_submodule(node.target)
            from torch import nn

            if isinstance(target, nn.Linear):
                if width is not None and target.in_features != width:
                    raise NotImplementedError(
                        "edge nn subgraph chain width mismatch at "
                        f"{node.target!r}")
                bias_name = (
                    f"{node.target}.bias" if target.bias is not None else None)
                layers.append(LinearLayer(
                    module_path=node.target,
                    in_features=target.in_features,
                    out_features=target.out_features,
                    weight_name=f"{node.target}.weight",
                    bias_name=bias_name,
                ))
                width = target.out_features
            elif isinstance(target, nn.Identity):
                pass  # no-op layer: skipped, stream unchanged
            elif isinstance(target, nn.LayerNorm):
                shape = target.normalized_shape
                if len(shape) != 1:
                    raise NotImplementedError(
                        "edge nn subgraphs support feature-wise LayerNorm "
                        f"only, got normalized_shape={tuple(shape)!r}")
                if width is not None and shape[0] != width:
                    raise NotImplementedError(
                        "edge nn subgraph LayerNorm width mismatch at "
                        f"{node.target!r}")
                layers.append(NormLayer(
                    module_path=node.target,
                    width=shape[0],
                    eps=float(target.eps),
                    weight_name=(
                        f"{node.target}.weight"
                        if target.elementwise_affine else None),
                    bias_name=(
                        f"{node.target}.bias"
                        if target.elementwise_affine else None),
                ))
            else:
                activation = _module_activation(target)
                if activation is None:
                    raise NotImplementedError(
                        f"unsupported edge nn module {type(target).__name__}")
                layers.append(ActivationLayer(
                    kind=activation[0], params=activation[1]))
        elif node.op == "call_function":
            activation = _function_activation(
                node.target, node.args, node.kwargs, previous)
            if activation is None:
                raise NotImplementedError(
                    f"unsupported edge nn function {node.target!r} with "
                    f"arguments {node.args!r}")
            layers.append(ActivationLayer(
                kind=activation[0], params=activation[1]))
        else:
            raise NotImplementedError(
                f"unsupported edge nn graph node op {node.op!r}")
        previous = node
    linears = [layer for layer in layers if isinstance(layer, LinearLayer)]
    if not linears:
        raise NotImplementedError(
            "edge nn subgraph requires at least one Linear layer")
    # Post-pass width check: activations preserve width, norms pin it, and a
    # norm ahead of the first Linear must match the chain input width.
    running = linears[0].in_features
    for layer in layers:
        if isinstance(layer, LinearLayer):
            if layer.in_features != running:
                raise NotImplementedError(
                    f"edge nn subgraph chain width mismatch at "
                    f"{layer.module_path!r}")
            running = layer.out_features
        elif isinstance(layer, NormLayer) and layer.width != running:
            raise NotImplementedError(
                f"edge nn subgraph LayerNorm width mismatch at "
                f"{layer.module_path!r}")
    return MessageDAG(
        layers=tuple(layers),
        in_features=linears[0].in_features,
        out_features=linears[-1].out_features,
        graph_module=graph_module,
    )


class TracedModule:
    """An nn.Module wrapper that is eager-callable and capture-visible."""

    def __init__(
        self, module, *, block_e: int | None = None, num_warps: int | None = None
    ) -> None:
        import torch.fx

        if block_e is not None and (
                block_e < 16 or block_e > 1024 or block_e & (block_e - 1)):
            raise ValueError(
                f"block_e must be a power of two in [16, 1024], got {block_e}")
        if num_warps is not None and (
                num_warps < 1 or num_warps > 16 or num_warps & (num_warps - 1)):
            raise ValueError(
                f"num_warps must be a power of two in [1, 16], got {num_warps}")
        self._module = module
        self._graph_module = torch.fx.symbolic_trace(module)
        dag = _translate_graph_module(self._graph_module)
        self._dag = MessageDAG(
            layers=dag.layers,
            in_features=dag.in_features,
            out_features=dag.out_features,
            block_e=block_e,
            num_warps=num_warps,
            graph_module=dag.graph_module,
        )

    @property
    def dag(self) -> MessageDAG:
        return self._dag

    def __call__(self, *inputs):
        if any(isinstance(value, Expr) for value in inputs):
            if not all(isinstance(value, Expr) for value in inputs):
                raise TypeError(
                    "edge nn subgraph inputs must all be capture expressions")
            return Expr("nn_subgraph", (self._dag, tuple(inputs)))
        if len(inputs) != 1:
            import torch

            inputs = (torch.cat(list(inputs), dim=-1),)
        return self._graph_module(*inputs)

    def __getattr__(self, name: str):
        # Optimizers, state_dict and device moves see the wrapped module.
        if name in {"_module", "_graph_module", "_dag", "dag"}:
            raise AttributeError(name)
        return getattr(self._module, name)


def trace(module, *, block_e: int | None = None, num_warps: int | None = None):
    """Wrap *module* so MessagePassing edge UDF calls are capturable.

    ``block_e`` is the edge-tile size (edges per program/chunk) of the fused
    kernel and ``num_warps`` its warp count; both are launch geometry only
    and never change numerics.  The defaults are per-lowering: 128/4 for
    edge-centric sum tiles, 16/1 for row-centric attention tiles.
    """
    if isinstance(module, TracedModule):
        return module
    return TracedModule(module, block_e=block_e, num_warps=num_warps)


__all__ = [
    "BLOCK_E",
    "NUM_WARPS",
    "ActivationLayer",
    "LinearLayer",
    "MessageDAG",
    "NormLayer",
    "TracedModule",
    "trace",
]
