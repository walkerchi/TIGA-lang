"""Framework-independent MessagePassing UDF kernel frontend."""

from __future__ import annotations

from collections.abc import Mapping
import inspect
from types import SimpleNamespace
from typing import Any

from ..graph import Graph
from ..kernel import Kernel
from ..reducer import (
    OnlineSoftmaxItem, OnlineSoftmaxReducer, Reducer, ReducerCall, SumReducer,
)
from ..tensor import Tensor


def _validate_native_fields(role, fields, entities, device) -> None:
    if not isinstance(fields, Mapping):
        raise TypeError(f"{role} fields must be a mapping")
    for name, value in fields.items():
        if not isinstance(value, Tensor):
            raise TypeError(f"{role}.{name} must be a graphforge.Tensor")
        if value.ndim == 0 or value.shape[0] != entities:
            raise ValueError(
                f"{role}.{name} leading dimension must be {entities}, "
                f"got {value.shape}")
        if value.device != device:
            raise ValueError(f"{role}.{name} must be on {device}")


def _call_region(method, positional, params):
    signature = inspect.signature(method)
    accepts_rest = any(
        item.kind is inspect.Parameter.VAR_KEYWORD
        for item in signature.parameters.values())
    selected = params if accepts_rest else {
        name: value for name, value in params.items()
        if name in signature.parameters
    }
    return method(*positional, **selected)


def _contains_torch(values: object) -> bool:
    if type(values).__module__.split(".")[0] == "torch":
        return True
    if isinstance(values, Mapping):
        return any(_contains_torch(value) for value in values.values())
    if isinstance(values, (tuple, list)):
        return any(_contains_torch(value) for value in values)
    return False


def _is_zero_constant(expression) -> bool:
    return (
        expression.op in {"constant", "typed_constant"}
        and expression.args
        and isinstance(expression.args[0], (int, float))
        and expression.args[0] == 0
    )


def _additive_state_reducer(descriptor) -> bool:
    """Prove that combine is a component-wise additive product monoid."""
    state_count = len(descriptor.state_dtypes)
    if len(descriptor.identity.results) != state_count or any(
        not _is_zero_constant(item) for item in descriptor.identity.results
    ):
        return False
    for index, result in enumerate(descriptor.combine.results):
        if result.op != "add":
            return False
        expected = {
            ("reducer_arg", (index,)),
            ("reducer_arg", (state_count + index,)),
        }
        actual = {(item.op, item.args) for item in result.args}
        if actual != expected:
            return False
    return True


def _multiplicative_state_reducer(descriptor) -> bool:
    """Prove the product monoid from captured reducer expression trees."""
    if (
        len(descriptor.message_dtypes) != 1
        or len(descriptor.state_dtypes) != 1
        or len(descriptor.result_dtypes) != 1
        or len(descriptor.identity.results) != 1
        or len(descriptor.lift.results) != 1
        or len(descriptor.combine.results) != 1
        or len(descriptor.finalize.results) != 1
    ):
        return False
    identity = descriptor.identity.results[0]
    lift = descriptor.lift.results[0]
    combine = descriptor.combine.results[0]
    finalize = descriptor.finalize.results[0]
    return (
        identity.op in {"constant", "typed_constant"}
        and identity.args and float(identity.args[0]) == 1.0
        and lift.op == "reducer_arg" and lift.args == (0,)
        and combine.op == "mul"
        and {(item.op, item.args) for item in combine.args} == {
            ("reducer_arg", (0,)), ("reducer_arg", (1,)),
        }
        and finalize.op == "reducer_arg" and finalize.args == (0,)
    )


def _stable_weighted_state_reducer(descriptor) -> bool:
    """Prove stable (max, denominator, numerator) algebra from Expr trees."""
    if (
        len(descriptor.message_dtypes) != 2
        or len(descriptor.state_dtypes) != 3
        or len(descriptor.result_dtypes) != 1
        or len(descriptor.identity.results) != 3
        or len(descriptor.lift.results) != 3
        or len(descriptor.combine.results) != 3
        or len(descriptor.finalize.results) != 1
    ):
        return False

    def arg(index):
        from ..compiler.capture import Expr

        return Expr("reducer_arg", (index,))

    def constant(expression, value):
        return (
            expression.op in {"constant", "typed_constant"}
            and expression.args
            and expression.args[0] == value
        )

    def unordered(expression, op, left, right):
        return expression.op == op and (
            expression.args == (left, right)
            or expression.args == (right, left)
        )

    def maximum(expression):
        return unordered(expression, "maximum", arg(0), arg(3))

    def scale(expression, score, factor):
        if expression.op != "mul":
            return False
        for exponential, candidate_factor in (
            expression.args, reversed(expression.args)
        ):
            if candidate_factor != factor or exponential.op != "exp":
                continue
            shifted = exponential.args[0]
            if (
                shifted.op == "sub"
                and shifted.args[0] == score
                and maximum(shifted.args[1])
            ):
                return True
        return False

    def scaled_sum(expression, left_factor, right_factor):
        if expression.op != "add":
            return False
        left, right = expression.args
        return (
            scale(left, arg(0), left_factor)
            and scale(right, arg(3), right_factor)
        ) or (
            scale(right, arg(0), left_factor)
            and scale(left, arg(3), right_factor)
        )

    identity = descriptor.identity.results
    lift = descriptor.lift.results
    combine = descriptor.combine.results
    finalize = descriptor.finalize.results[0]
    return (
        identity[0].op in {"constant", "typed_constant"}
        and float(identity[0].args[0]) < 0.0
        and constant(identity[1], 0.0)
        and constant(identity[2], 0.0)
        and lift[0] == arg(0)
        and constant(lift[1], 1.0)
        and lift[2] == arg(1)
        and maximum(combine[0])
        and scaled_sum(combine[1], arg(1), arg(4))
        and scaled_sum(combine[2], arg(2), arg(5))
        and finalize.op == "div"
        and finalize.args == (arg(2), arg(1))
    )


def _evaluate_reducer_expr(expression, arguments, *, like, lift=False):
    """Interpret captured reducer algebra as differentiable Tensor nodes."""
    op = expression.op
    if op == "reducer_arg":
        return arguments[expression.args[0]]
    if op in {"constant", "typed_constant"}:
        value = expression.args[0]
        result = Tensor.scalar(value, dtype=like.dtype, device=like.device)
        return result.broadcast_to(like.shape) if lift else result
    if op == "neg":
        return -_evaluate_reducer_expr(
            expression.args[0], arguments, like=like, lift=lift)
    if op == "exp":
        return _evaluate_reducer_expr(
            expression.args[0], arguments, like=like, lift=lift).exp()
    if op in {"add", "sub", "mul", "div"}:
        lhs = _evaluate_reducer_expr(
            expression.args[0], arguments, like=like, lift=lift)
        rhs = _evaluate_reducer_expr(
            expression.args[1], arguments, like=like, lift=lift)
        if op == "add":
            return lhs + rhs
        if op == "sub":
            return lhs + (-rhs)
        if op == "mul":
            return lhs * rhs
        return lhs / rhs
    raise NotImplementedError(
        f"native differentiable reducer expression {op!r} is not supported")


def _apply_general_reducer_tree(
    reducer, descriptor, messages, row_ptr, num_dst
):
    """Build an explicit differentiable tree for a captured scalar reducer.

    The tree is a semantic/native-autograd fallback, not a traversal kernel
    specialization.  Provider lowering can still recognize the original
    reducer regions for the fused forward path.  Keeping every lift/combine/
    finalize scalar operation in Tensor IR lets ordinary reverse-mode derive
    the backward without a user-authored kernel.
    """
    if not reducer.associative:
        raise NotImplementedError(
            "parallel user reducer lowering requires associative=True")
    if not messages or any(
        not isinstance(item, Tensor) or item.ndim < 1 for item in messages
    ):
        raise NotImplementedError(
            "general reduction-tree autograd requires edge-domain messages")
    edge_count = messages[0].shape[0]
    feature_shape = messages[0].shape[1:]
    if any(item.shape != (edge_count, *feature_shape)
           for item in messages[1:]):
        raise NotImplementedError(
            "general reduction-tree autograd currently requires all messages "
            "to share one trailing feature shape")
    if len(descriptor.result_dtypes) != 1:
        raise NotImplementedError(
            "general reduction-tree autograd currently returns one scalar result")

    rows = tuple(int(value) for value in row_ptr.tolist())
    if len(rows) != num_dst + 1 or rows[0] != 0 or rows[-1] != edge_count:
        raise ValueError("static CSR row pointer does not match reducer messages")
    if any(right < left for left, right in zip(rows, rows[1:])):
        raise ValueError("static CSR row pointer must be monotonic")

    # GPU pointwise lowering requires gather sources to be ABI leaves. Save
    # the edge-local message expression once, then let every tree leaf gather
    # from that compiler-managed checkpoint. CPU scalar lowering can inline it.
    tree_messages = (
        tuple(item.checkpoint() for item in messages)
        if messages[0].device.type.name == "CUDA"
        else tuple(messages)
    )
    like = tree_messages[0]
    state_like = Tensor.scalar(
        0, dtype=like.dtype, device=like.device)
    if feature_shape:
        state_like = state_like.broadcast_to(feature_shape)

    def identity_state():
        return tuple(
            _evaluate_reducer_expr(
                expression, (), like=state_like,
                lift=bool(feature_shape))
            for expression in descriptor.identity.results
        )

    def edge_state(edge_index):
        # Index Tensors are immutable compiler constants.  A one-element
        # gather keeps topology in canonical Tensor IR and its VJP is the
        # existing segment-sum/scatter rule.
        from ..tensor import tensor

        index = tensor(
            [edge_index], dtype=row_ptr.dtype, device=row_ptr.device)
        scalar_messages = tuple(
            message.gather(index).reshape(feature_shape)
            for message in tree_messages)
        return tuple(
            _evaluate_reducer_expr(
                expression, scalar_messages, like=scalar_messages[0])
            for expression in descriptor.lift.results
        )

    def combine(left, right):
        arguments = (*left, *right)
        return tuple(
            _evaluate_reducer_expr(
                expression, arguments, like=arguments[0])
            for expression in descriptor.combine.results
        )

    def reduce_row(begin, end):
        states = [edge_state(edge) for edge in range(begin, end)]
        if not states:
            return identity_state()
        if reducer.deterministic:
            state = states[0]
            for item in states[1:]:
                state = combine(state, item)
            return state
        # A stable balanced tree matches parallel reduction depth and avoids
        # an O(degree) expression critical path.
        while len(states) > 1:
            next_level = [
                combine(states[index], states[index + 1])
                for index in range(0, len(states) - 1, 2)
            ]
            if len(states) % 2:
                next_level.append(states[-1])
            states = next_level
        return states[0]

    row_results = []
    for begin, end in zip(rows, rows[1:]):
        state = reduce_row(begin, end)
        finalized = tuple(
            _evaluate_reducer_expr(expression, state, like=state[0])
            for expression in descriptor.finalize.results
        )
        row_results.append(finalized[0])

    # Assemble scalar rows with existing broadcast/mul/add primitives.  This
    # avoids introducing a Python-only stack op or textual IR shortcut.
    from ..tensor import tensor

    output_shape = (num_dst, *feature_shape)
    output = Tensor.scalar(
        0, dtype=like.dtype, device=like.device).broadcast_to(output_shape)
    for row, result in enumerate(row_results):
        selector_values = [
            1 if index == row else 0 for index in range(num_dst)]
        selector_data = (
            [[value] for value in selector_values]
            if len(feature_shape) == 1 else selector_values
        )
        selector = tensor(
            selector_data,
            dtype=like.dtype, device=like.device,
        )
        if selector.shape != (num_dst,) + (1,) * len(feature_shape):
            selector = selector.reshape(
                (num_dst,) + (1,) * len(feature_shape))
        selector = selector.broadcast_to(output_shape)
        output = output + result.reshape((1, *feature_shape)).broadcast_to(
            output_shape) * selector
    return output


def _apply_additive_reducer(reducer, message, row_ptr, num_dst):
    from ..compiler.reducer_capture import capture_reducer

    binding = message if isinstance(message, ReducerCall) else None
    messages = binding.messages if binding is not None else (message,)
    if binding is not None and binding.reducer is not reducer:
        raise ValueError("edge() returned a binding for a different reducer")
    if not messages or any(not isinstance(item, Tensor) for item in messages):
        raise TypeError("native reducer messages must be gf.Tensor values")
    descriptor = capture_reducer(
        reducer, message_dtypes=tuple(item.dtype for item in messages))
    if _stable_weighted_state_reducer(descriptor):
        return _apply_stable_weighted_tensors(
            messages[0], messages[1], row_ptr, num_dst)
    if _multiplicative_state_reducer(descriptor):
        return messages[0]._csr_segment_product(row_ptr, num_dst)
    if not _additive_state_reducer(descriptor):
        return _apply_general_reducer_tree(
            reducer, descriptor, messages, row_ptr, num_dst)
    like = messages[0]
    lifted = tuple(
        _evaluate_reducer_expr(
            expression, messages, like=like, lift=True)
        for expression in descriptor.lift.results
    )
    states = tuple(
        item.csr_segment_sum(row_ptr, num_dst) for item in lifted)
    finalized = tuple(
        _evaluate_reducer_expr(expression, states, like=states[0])
        for expression in descriptor.finalize.results
    )
    if len(finalized) != 1:
        raise NotImplementedError(
            "native differentiable MessagePassing currently returns one reducer result")
    return finalized[0]


def _apply_stable_weighted_tensors(score, value, row_ptr, num_dst):
    """Expand proven stable weighted algebra to differentiable primitives."""
    if not isinstance(score, Tensor) or not isinstance(value, Tensor):
        raise TypeError("online_softmax score/value must be gf.Tensor values")
    if score.shape != value.shape[:1]:
        raise ValueError("online_softmax score must have one scalar per edge")
    row_max = score._csr_segment_max_stop_gradient(row_ptr, num_dst)
    shifted = score + (-row_max.csr_expand_rows(row_ptr, score.shape[0]))
    weight = shifted.exp()
    denominator = weight.csr_segment_sum(row_ptr, num_dst)
    if value.ndim == 1:
        numerator = (weight * value).csr_segment_sum(row_ptr, num_dst)
        return numerator / denominator
    expanded_weight = weight.reshape((weight.shape[0],) + (1,) * (value.ndim - 1))
    numerator = (expanded_weight * value).csr_segment_sum(row_ptr, num_dst)
    expanded_denominator = denominator.reshape(
        (denominator.shape[0],) + (1,) * (value.ndim - 1))
    return numerator / expanded_denominator


def _apply_online_softmax(message, row_ptr, num_dst):
    """Expand stable online softmax into differentiable Tensor primitives."""
    if not isinstance(message, OnlineSoftmaxItem):
        raise TypeError(
            "online_softmax edge() must return reducer(score, value)")
    return _apply_stable_weighted_tensors(
        message.score, message.value, row_ptr, num_dst)


class MessagePassing(Kernel):
    """A relation traversal plus edge UDF and reducer kernel."""

    reducer: Reducer = SumReducer()

    def __init__(self) -> None:
        super().__init__()
        self._torch_executor = None
        # Optional framework adapters may install a guarded, already-compiled
        # executable here after the first lazy-JIT call. Keeping the hook on
        # the native kernel avoids repeated framework discovery and graph-view
        # conversion; a failed guard always returns to normal dispatch below.
        self._torch_fast_path = None

    def edge(self, src, dst, edge, **params):
        raise NotImplementedError

    def node(self, dst, aggregate, **params):
        return aggregate

    def prepare(
        self,
        *,
        graph: Graph,
        src: Mapping[str, Tensor | object],
        dst: Mapping[str, Tensor | object],
        edge: Mapping[str, Tensor | object] | None = None,
        **params: Any,
    ):
        """Freeze one already-specialized submission as a zero-argument call.

        Normal calls remain lazy-JIT and need no explicit compilation. This
        optional capture removes Python mapping/guard work from repeated
        microbenchmark, CUDA-graph, and inner-solver submissions. Tensor
        contents remain live; graph identity, bindings, shapes and strides are
        frozen at preparation time.
        """
        edge = {} if edge is None else edge
        if _contains_torch((src, dst, edge, params)):
            from ..interop.torch.message_passing import prepare

            return prepare(
                self, graph=graph, src=src, dst=dst, edge=edge, params=params)
        raise NotImplementedError(
            "prepared submission currently requires the CUDA Torch interop "
            "adapter; native CPU Tensor expressions are already lazy values")

    def __call__(
        self,
        *,
        graph: Graph,
        src: Mapping[str, Tensor | object],
        dst: Mapping[str, Tensor | object],
        edge: Mapping[str, Tensor | object] | None = None,
        **params: Any,
    ):
        if not getattr(graph, "_graphforge_graph", False):
            raise TypeError("graph must be a graphforge.Graph")
        edge = {} if edge is None else edge
        from ..program import active_program

        capture = active_program()
        if capture is not None:
            return capture.apply(
                self, graph=graph, src=src, dst=dst, edge=edge, **params
            )
        fast_path = self._torch_fast_path
        if fast_path is not None:
            hit, result = fast_path(graph, src, dst, edge, params)
            if hit:
                return result
        if graph.is_distributed:
            from ..distributed.transport import (
                current_distributed_runtime,
                execute_sharded_message_passing,
            )

            if _contains_torch((src, dst, edge, params)):
                if current_distributed_runtime() is None:
                    # Preserve task-stage inspection on a failed dry submit;
                    # the compatibility planner records the distributed
                    # variant before issuing the active-runtime diagnostic.
                    from ..interop.torch.message_passing import execute

                    return execute(
                        self,
                        graph=graph,
                        src=src,
                        dst=dst,
                        edge=edge,
                        params=params,
                    )
                # Preserve the same Graph/MessagePassing surface for the
                # optional adapter.  Transport remains runtime-owned; Torch
                # tensors are zero-copy Buffer views at the boundary.
                from ..tensor import from_torch

                def adapt(values):
                    return {
                        name: value if isinstance(value, Tensor)
                        else from_torch(value)
                        for name, value in values.items()
                    }

                native_params = {
                    name: (
                        value if isinstance(value, Tensor)
                        else from_torch(value)
                        if type(value).__module__.startswith("torch")
                        else value
                    )
                    for name, value in params.items()
                }
                result = execute_sharded_message_passing(
                    self,
                    graph=graph,
                    src=adapt(src),
                    dst=adapt(dst),
                    edge=adapt(edge),
                    params=native_params,
                )
                # Distributed halo assembly is runtime-owned.  Returning to
                # the optional adapter is an explicit device-to-device copy;
                # the public zero-copy Tensor.to_torch() contract stays strict.
                return result.to_torch(copy=True)
            return execute_sharded_message_passing(
                self, graph=graph, src=src, dst=dst, edge=edge, params=params,
            )
        if _contains_torch((src, dst, edge, params)):
            from ..interop.torch.message_passing import execute

            return execute(
                self, graph=graph, src=src, dst=dst, edge=edge, params=params
            )
        _validate_native_fields("src", src, graph.schema.num_src, graph.device)
        _validate_native_fields("dst", dst, graph.schema.num_dst, graph.device)
        if graph.schema.realization not in {
            "materialized_csr", "procedural_radius"
        }:
            raise NotImplementedError(
                "native differentiable MessagePassing currently requires CSR "
                "or a default Euclidean radius relation")
        row_ptr, col_idx = graph.resolve_csr()
        _validate_native_fields("edge", edge, col_idx.numel, graph.device)
        src_at_edge = SimpleNamespace(**{
            name: value.gather(col_idx) for name, value in src.items()
        })
        dst_at_edge = SimpleNamespace(**{
            name: value.csr_expand_rows(row_ptr, col_idx.numel)
            for name, value in dst.items()
        })
        implicit_edge = graph.implicit_edge_fields(row_ptr, col_idx)
        overlap = implicit_edge.keys() & edge.keys()
        if overlap:
            raise ValueError(
                "edge fields shadow graph-provided fields: "
                + ", ".join(sorted(overlap)))
        edge_at_edge = SimpleNamespace(**{**implicit_edge, **dict(edge)})
        message = _call_region(
            self.edge, (src_at_edge, dst_at_edge, edge_at_edge), params)
        if isinstance(self.reducer, SumReducer):
            if not isinstance(message, Tensor):
                raise TypeError(
                    "native sum MessagePassing edge() must return one gf.Tensor")
            aggregate = message.csr_segment_sum(row_ptr, graph.schema.num_dst)
            reducer_lowering = "builtin-additive-state"
        elif isinstance(self.reducer, OnlineSoftmaxReducer):
            if self.reducer.block_prune_threshold is not None:
                raise NotImplementedError(
                    "block_prune_threshold is currently a generated CUDA "
                    "DenseGraph approximation; native CSR execution refuses "
                    "to silently substitute exact online softmax"
                )
            aggregate = _apply_online_softmax(
                message, row_ptr, graph.schema.num_dst)
            reducer_lowering = "stable-online-softmax-tensor-algebra"
        else:
            aggregate = _apply_additive_reducer(
                self.reducer, message, row_ptr, graph.schema.num_dst)
            from ..compiler.reducer_capture import capture_reducer

            binding = message if isinstance(message, ReducerCall) else None
            messages = binding.messages if binding is not None else (message,)
            descriptor = capture_reducer(
                self.reducer,
                message_dtypes=tuple(item.dtype for item in messages),
            )
            reducer_lowering = (
                "proved-stable-weighted-udf"
                if _stable_weighted_state_reducer(descriptor)
                else (
                    "proved-product-monoid"
                    if _multiplicative_state_reducer(descriptor)
                    else (
                        "proved-componentwise-additive-udf"
                        if _additive_state_reducer(descriptor)
                        else "explicit-reduction-tree-autograd"
                    )
                )
            )
        output = (
            aggregate if type(self).node is MessagePassing.node
            else _call_region(
                self.node, (SimpleNamespace(**dict(dst)), aggregate), params)
        )
        if not isinstance(output, Tensor):
            raise TypeError("native MessagePassing node() must return a gf.Tensor")
        key = (
            "native-differentiable-relation", graph.planning_key(),
            tuple((name, value.shape, value.dtype.name) for name, value in src.items()),
            tuple((name, value.shape, value.dtype.name) for name, value in dst.items()),
            tuple((name, value.shape, value.dtype.name) for name, value in edge.items()),
            self.reducer.specialization_key(),
        )
        from ..kernel import CompiledVariant

        self._cache_misses += 1
        self._record_variant(CompiledVariant(
            key=key,
            backend=str(graph.device),
            provider="graphforge-runtime",
            lowering="gf-tensor-relation-autograd",
            domain_plan=output.expression(),
            passes=(
                "materialize-fixed-radius-snapshot"
                if graph.schema.realization == "procedural_radius"
                else "bind-static-csr-snapshot",
                "capture-message-passing-udf",
                "analyze-reducer-algebra",
                "lower-csr-to-gather-segment",
                "fuse-edge-node-regions",
            ),
            remarks=(
                "edge/node UDFs remain in the differentiable Tensor DAG",
                "gf-tensor-vjp generates CSR gather/segment-sum adjoints",
                f"reducer lowering: {reducer_lowering}",
            ),
        ))
        return output

    def reference(self, **kwargs):
        if not _contains_torch(kwargs):
            raise RuntimeError(
                "reference() is an optional Torch interop oracle, not a native backend"
            )
        from ..interop.torch.message_passing import execute_reference

        return execute_reference(self, **kwargs)


__all__ = ["MessagePassing"]
