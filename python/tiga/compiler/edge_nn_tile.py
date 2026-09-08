"""Structural proof and Python TTIR emission for fused edge-NN tile kernels.

This is the Phase-1 Python emission backend for edge-local nn subgraphs
(see ``compiler.nn_capture``).  It proves that a captured ``nn_subgraph``
message is a linear chain of tile ops whose inputs are per-edge field
gathers, then emits one edge-centric tile kernel as canonical TTIR text:

- one ``BLOCK_E``-edge block per program, no row pointer required;
- segment gathers assemble the padded ``[BLOCK_E, K_PAD]`` input tile
  (columns outside each segment stay zero);
- every Linear layer is a ``tt.dot`` with the bias broadcast folded into the
  accumulator; contraction/output dims pad up to multiples of 16 and padded
  weight rows/columns are zero, so padded lanes contribute nothing;
- the message tile is segment-reduced by destination with a masked
  ``tt.atomic_rmw fadd`` — no O(E) message tensor ever exists.

The emitted text crosses the same serialized-TTIR provider boundary as
``gf-translate`` output (``codegen.ttir.compile_ttir``); a later phase moves
this emitter into the C++ ``gf-kernel-to-ttir`` translation.  Anything the
proof rejects falls back to the eager oracle, which remains the semantic
ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

from .capture import Expr
from .nn_capture import (
    BLOCK_E,
    NUM_WARPS,
    ActivationLayer,
    LinearLayer,
    MessageDAG,
    NormLayer,
)

ENTRY = "gf_edge_nn_tile"

_FIELD_KINDS = {"src": "src_field", "dst": "dst_field", "edge": "edge_field"}


@dataclass(frozen=True)
class Segment:
    """One column range of the input tile sourced from a per-edge gather."""

    kind: str  # "src_field" | "dst_field" | "edge_field" | "displacement"
    name: str
    width: int
    offset: int


@dataclass(frozen=True)
class EdgeNNTileSpec:
    """The proved, hashable kernel structure for one edge nn subgraph."""

    segments: tuple[Segment, ...]
    layers: tuple[LinearLayer | ActivationLayer | NormLayer, ...]
    in_width: int
    k_pad: int
    out_width: int
    out_pad: int
    needs_positions: bool
    block_e: int = BLOCK_E
    num_warps: int = NUM_WARPS


def _pad16(width: int) -> int:
    """Pad a tile dimension to a power of two of at least 16.

    ``tt.make_range`` and ``tt.dot`` only accept power-of-two extents, so a
    multiple-of-16 padding is not sufficient (e.g. 48 must become 64).
    """
    return max(16, 1 << (width - 1).bit_length())


def build_edge_nn_spec(
    message: Expr,
    *,
    field_widths: dict[tuple[str, str], int],
    position_dim: int | None,
) -> EdgeNNTileSpec | None:
    """Prove that *message* is an edge nn subgraph with gatherable inputs."""
    if not isinstance(message, Expr) or message.op != "nn_subgraph":
        return None
    dag, inputs = message.args
    if not isinstance(dag, MessageDAG) or not inputs:
        return None
    segments: list[Segment] = []
    offset = 0
    for expr in inputs:
        if not isinstance(expr, Expr) or expr.op != "field":
            return None
        role, name = expr.args
        if role == "edge" and name == "displacement":
            if position_dim is None:
                return None
            segments.append(Segment(
                "displacement", name, position_dim, offset))
            offset += position_dim
            continue
        if role == "edge" and name == "distance":
            return None  # v1 keeps distance out of nn inputs
        kind = _FIELD_KINDS.get(role)
        width = field_widths.get((role, name))
        if kind is None or width is None:
            return None
        segments.append(Segment(kind, name, width, offset))
        offset += width
    if offset != dag.in_features:
        return None
    k_pad = _pad16(offset)
    out_pad = _pad16(dag.out_features)
    return EdgeNNTileSpec(
        segments=tuple(segments),
        layers=dag.layers,
        in_width=offset,
        k_pad=k_pad,
        out_width=dag.out_features,
        out_pad=out_pad,
        needs_positions=any(
            segment.kind == "displacement" for segment in segments),
        block_e=dag.block_e if dag.block_e is not None else BLOCK_E,
        num_warps=dag.num_warps if dag.num_warps is not None else NUM_WARPS,
    )


class _Emit:
    """Minimal SSA text builder for the single fixed kernel shape."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.counter = 0

    def v(self) -> str:
        self.counter += 1
        return f"%v{self.counter}"

    def emit(self, text: str) -> None:
        self.lines.append(text)

    def assign(self, text: str) -> str:
        name = self.v()
        self.emit(f"    {name} = {text}")
        return name


def _linear_tile(
    e: _Emit, index: int, layer: LinearLayer, current: str,
    block_e: int, in_pad: int,
) -> tuple[str, str, int]:
    """Emit ``current @ weight.T + bias`` for one Linear layer.

    ``%w{index}``/``%b{index}`` must be the matching kernel parameters.
    Returns ``(result, weight_tile, out_pad)`` — the weight tile is kept for
    the symbolic VJP (``d_a = d @ W`` reuses it transposed).
    """
    out_pad = _pad16(layer.out_features)
    kr = _range(e, in_pad)
    kc = _expand(e, kr, 1, f"{in_pad}xi32", f"{in_pad}x1xi32")
    c_no = _dense(e, out_pad, f"{in_pad}x1", "i32")
    wrow = e.assign(f"arith.muli {kc}, {c_no} : tensor<{in_pad}x1xi32>")
    wp = _splat_ptr(e, f"%w{index}", f"{in_pad}x1")
    wp2 = _addptr(e, wp, wrow, f"{in_pad}x1")
    nr = _range(e, out_pad)
    nr2 = _expand(e, nr, 0, f"{out_pad}xi32", f"1x{out_pad}xi32")
    wp3 = _broadcast(
        e, wp2, f"{in_pad}x1x!tt.ptr<f32>", f"{in_pad}x{out_pad}x!tt.ptr<f32>")
    nr3 = _broadcast(
        e, nr2, f"1x{out_pad}xi32", f"{in_pad}x{out_pad}xi32")
    wp4 = _addptr(e, wp3, nr3, f"{in_pad}x{out_pad}")
    wt = e.assign(
        f"tt.load {wp4} : tensor<{in_pad}x{out_pad}x!tt.ptr<f32>>")
    if layer.bias_name is not None:
        bp = _splat_ptr(e, f"%b{index}", str(out_pad))
        bp2 = _addptr(e, bp, nr, str(out_pad))
        bv = e.assign(f"tt.load {bp2} : tensor<{out_pad}x!tt.ptr<f32>>")
        bv2 = _expand(e, bv, 0, f"{out_pad}xf32", f"1x{out_pad}xf32")
        acc = _broadcast(
            e, bv2, f"1x{out_pad}xf32", f"{block_e}x{out_pad}xf32")
    else:
        acc = _dense(e, "0.0", f"{block_e}x{out_pad}", "f32")
    result = e.assign(
        f"tt.dot {current}, {wt}, {acc} : tensor<{block_e}x{in_pad}xf32> "
        f"* tensor<{in_pad}x{out_pad}xf32> -> tensor<{block_e}x{out_pad}xf32>")
    return result, wt, out_pad


def _replay_chain(
    e: _Emit,
    layers,
    in_width: int,
    k_pad: int,
    current: str,
    block_e: int,
    keep_records: bool = False,
):
    """Emit the nn layer chain over a ``[block_e, k_pad]`` input tile.

    With ``keep_records`` the VJP records are collected per layer
    (``("act", kind, params, pre, post, width, pad)``,
    ``("norm", index, xhat, rstd2, mf, gamma2, width, pad)``,
    ``("linear", index, input, in_pad, out_pad, wt)``).  Returns
    ``(output_tile, records, out_pad)``.
    """
    records = []
    in_pad = k_pad
    width = in_width
    for index, layer in enumerate(layers):
        if isinstance(layer, ActivationLayer):
            post = _activation(
                e, layer.kind, current, f"{block_e}x{in_pad}",
                layer.params, width)
            if keep_records:
                records.append(
                    ("act", layer.kind, layer.params, current, post, width,
                     in_pad))
            current = post
            continue
        if isinstance(layer, NormLayer):
            post, (xhat, rstd2, mf, gamma2) = _norm_tile(
                e, index, layer, current, block_e, in_pad)
            if keep_records:
                records.append(
                    ("norm", index, xhat, rstd2, mf, gamma2, width, in_pad))
            current = post
            continue
        pre, wt, out_pad = _linear_tile(
            e, index, layer, current, block_e, in_pad)
        if keep_records:
            records.append(("linear", index, current, in_pad, out_pad, wt))
        current = pre
        in_pad = out_pad
        width = layer.out_features
    return current, records, in_pad


def _dense(e: _Emit, value: float | str, shape: str, ty: str) -> str:
    if not isinstance(value, str) and ty.startswith("f"):
        text = repr(float(value))
        # MLIR float literals need a decimal point: 1e-05 → 1.0e-05.
        if "." not in text and "e" in text:
            mantissa, exponent = text.split("e")
            text = f"{mantissa}.0e{exponent}"
        value = text
    return e.assign(f"arith.constant dense<{value}> : tensor<{shape}x{ty}>")


def _scalar(e: _Emit, value: int) -> str:
    return e.assign(f"arith.constant {value} : i32")


def _range(e: _Emit, size: int) -> str:
    return e.assign(
        f"tt.make_range {{end = {size} : i32, start = 0 : i32}} "
        f": tensor<{size}xi32>")


def _expand(e: _Emit, value: str, axis: int, src: str, dst: str) -> str:
    return e.assign(
        f"tt.expand_dims {value} {{axis = {axis} : i32}} "
        f": tensor<{src}> -> tensor<{dst}>")


def _broadcast(e: _Emit, value: str, src: str, dst: str) -> str:
    return e.assign(f"tt.broadcast {value} : tensor<{src}> -> tensor<{dst}>")


def _splat_ptr(e: _Emit, ptr: str, shape: str) -> str:
    return e.assign(
        f"tt.splat {ptr} : !tt.ptr<f32> -> tensor<{shape}x!tt.ptr<f32>>")


def _addptr(e: _Emit, ptr: str, offset: str, shape: str) -> str:
    return e.assign(
        f"tt.addptr {ptr}, {offset} : tensor<{shape}x!tt.ptr<f32>>, "
        f"tensor<{shape}xi32>")


def _splat_int(e: _Emit, value: str, shape: str) -> str:
    return e.assign(f"tt.splat {value} : i32 -> tensor<{shape}xi32>")


def _gather_columns(
    e: _Emit,
    *,
    ptr: str,
    row_index: str,
    width: int,
    offset: int,
    k_pad: int,
    block_e: int,
    rows_b: str,
    k2: str,
    zero_2d: str,
    subtract: str | None = None,
) -> str:
    """Load the ``[block_e, k_pad]`` tile of one segment column range."""
    c_off = _dense(e, offset, f"1x{k_pad}", "i32")
    c_end = _dense(e, offset + width, f"1x{k_pad}", "i32")
    m_lo = e.assign(f"arith.cmpi sge, {k2}, {c_off} : tensor<1x{k_pad}xi32>")
    m_hi = e.assign(f"arith.cmpi slt, {k2}, {c_end} : tensor<1x{k_pad}xi32>")
    m_seg = e.assign(f"arith.andi {m_lo}, {m_hi} : tensor<1x{k_pad}xi1>")
    m_seg2 = _broadcast(e, m_seg, f"1x{k_pad}xi1", f"{block_e}x{k_pad}xi1")
    mask = e.assign(f"arith.andi {rows_b}, {m_seg2} : tensor<{block_e}x{k_pad}xi1>")
    c_w = _dense(e, width, f"{block_e}x1", "i32")
    idx2 = _expand(e, row_index, 1, f"{block_e}xi32", f"{block_e}x1xi32")
    row = e.assign(f"arith.muli {idx2}, {c_w} : tensor<{block_e}x1xi32>")
    base = _splat_ptr(e, ptr, f"{block_e}x1")
    base2 = _addptr(e, base, row, f"{block_e}x1")
    base3 = _broadcast(
        e, base2, f"{block_e}x1x!tt.ptr<f32>", f"{block_e}x{k_pad}x!tt.ptr<f32>")
    col = e.assign(f"arith.subi {k2}, {c_off} : tensor<1x{k_pad}xi32>")
    col2 = _broadcast(e, col, f"1x{k_pad}xi32", f"{block_e}x{k_pad}xi32")
    base4 = _addptr(e, base3, col2, f"{block_e}x{k_pad}")
    tile = e.assign(
        f"tt.load {base4}, {mask}, {zero_2d} : "
        f"tensor<{block_e}x{k_pad}x!tt.ptr<f32>>")
    if subtract is not None:
        tile = e.assign(
            f"arith.subf {tile}, {subtract} : tensor<{block_e}x{k_pad}xf32>")
    return tile


def _mul(e: _Emit, a: str, b: str, shape: str) -> str:
    return e.assign(f"arith.mulf {a}, {b} : tensor<{shape}xf32>")


def _add(e: _Emit, a: str, b: str, shape: str) -> str:
    return e.assign(f"arith.addf {a}, {b} : tensor<{shape}xf32>")


def _sub(e: _Emit, a: str, b: str, shape: str) -> str:
    return e.assign(f"arith.subf {a}, {b} : tensor<{shape}xf32>")


def _div(e: _Emit, a: str, b: str, shape: str) -> str:
    return e.assign(f"arith.divf {a}, {b} : tensor<{shape}xf32>")


def _math(e: _Emit, name: str, x: str, shape: str) -> str:
    return e.assign(f"math.{name} {x} : tensor<{shape}xf32>")


def _where(e: _Emit, cond: str, a: str, b: str, shape: str) -> str:
    return e.assign(
        f"arith.select {cond}, {a}, {b} : "
        f"tensor<{shape}xi1>, tensor<{shape}xf32>")


def _cmp_const(e: _Emit, predicate: str, x: str, constant, shape: str) -> str:
    bound = _dense(e, constant, shape, "f32")
    return e.assign(
        f"arith.cmpf {predicate}, {x}, {bound} : tensor<{shape}xf32>")


def _sigmoid_tile(e: _Emit, x: str, shape: str) -> str:
    zero = _dense(e, "0.0", shape, "f32")
    one = _dense(e, "1.0", shape, "f32")
    neg = _sub(e, zero, x, shape)
    ex = _math(e, "exp", neg, shape)
    den = _add(e, ex, one, shape)
    return _div(e, one, den, shape)


def _tanh_tile(e: _Emit, x: str, shape: str) -> str:
    # math.tanh is illegal in TTIR; tanh(x) = 2·sigmoid(2x) − 1.
    two = _dense(e, "2.0", shape, "f32")
    one = _dense(e, "1.0", shape, "f32")
    doubled = _mul(e, x, two, shape)
    sig = _sigmoid_tile(e, doubled, shape)
    scaled = _mul(e, sig, two, shape)
    return _sub(e, scaled, one, shape)


def _elu_tile(e: _Emit, x: str, shape: str, alpha: float) -> str:
    pos = _cmp_const(e, "ogt", x, "0.0", shape)
    ex = _math(e, "exp", x, shape)
    one = _dense(e, "1.0", shape, "f32")
    em = _mul(e, _sub(e, ex, one, shape), _dense(e, alpha, shape, "f32"), shape)
    return _where(e, pos, x, em, shape)


def _clamp_tile(
    e: _Emit, x: str, shape: str, low: float | None, high: float | None
) -> str:
    if low is not None:
        x = e.assign(
            f"arith.maxnumf {x}, {_dense(e, low, shape, 'f32')} : "
            f"tensor<{shape}xf32>")
    if high is not None:
        x = e.assign(
            f"arith.minnumf {x}, {_dense(e, high, shape, 'f32')} : "
            f"tensor<{shape}xf32>")
    return x


def _softplus_tile(
    e: _Emit, x: str, shape: str, beta: float, threshold: float
) -> str:
    # Stable softplus: x where beta·x > threshold, else log1p(exp(beta·x))/beta.
    # math.log1p has no lowering in this Triton; log(1 + e) matches it to
    # f32 precision everywhere except beta·x < ~-16, where both are ~0.
    beta_t = _dense(e, beta, shape, "f32")
    bx = _mul(e, x, beta_t, shape)
    ex = _math(e, "exp", bx, shape)
    one = _dense(e, "1.0", shape, "f32")
    sp = _div(e, _math(e, "log", _add(e, one, ex, shape), shape), beta_t, shape)
    big = _cmp_const(e, "ogt", bx, threshold, shape)
    return _where(e, big, x, sp, shape)


def _pow_tile(e: _Emit, x: str, shape: str, p: float) -> str:
    """``x ** p`` without math.powf (no lowering in this Triton).

    Small integer/half-integer exponents lower to multiply chains and
    sqrt/rsqrt; anything else becomes exp(p·log(x)), which shares powf's
    domain (NaN for negative x at non-integer p).
    """
    if p == 1.0:
        return x
    if p == 2.0:
        return _mul(e, x, x, shape)
    if p == 3.0:
        return _mul(e, _mul(e, x, x, shape), x, shape)
    if p == 4.0:
        return _mul(e, _mul(e, x, x, shape), _mul(e, x, x, shape), shape)
    if p == 0.5:
        return _math(e, "sqrt", x, shape)
    if p == -0.5:
        return _math(e, "rsqrt", x, shape)
    one = _dense(e, "1.0", shape, "f32")
    if p == 0.0:
        return one
    if p == -1.0:
        return _div(e, one, x, shape)
    if p == -2.0:
        return _div(e, one, _mul(e, x, x, shape), shape)
    log = _math(e, "log", x, shape)
    return _math(e, "exp", _mul(e, log, _dense(e, p, shape, "f32"), shape),
                 shape)


def _mask_columns(e: _Emit, value: str, width: int, block_e: int, pad: int) -> str:
    """Zero the columns at/after ``width`` of a ``[block_e, pad]`` f32 tile.

    The tile invariant is that padded lanes are exactly zero; it keeps
    weight rows/columns of padded lanes inert in ``tt.dot`` and keeps
    row-wise reductions (LayerNorm) free of upstream bias garbage.
    """
    if width >= pad:
        return value
    shape = f"{block_e}x{pad}"
    r = _range(e, pad)
    r2 = _expand(e, r, 0, f"{pad}xi32", f"1x{pad}xi32")
    c_w = _dense(e, width, f"1x{pad}", "i32")
    m = e.assign(f"arith.cmpi slt, {r2}, {c_w} : tensor<1x{pad}xi32>")
    m2 = _broadcast(e, m, f"1x{pad}xi1", f"{shape}xi1")
    zero = _dense(e, "0.0", shape, "f32")
    return _where(e, m2, value, zero, shape)


def _row_sum(e: _Emit, value: str, rows: int, cols: int) -> str:
    """Sum a ``tensor<rows x cols xf32>`` over axis 1 into ``tensor<rows>``."""
    result, lhs, rhs, combined = e.v(), e.v(), e.v(), e.v()
    e.emit(f'    {result} = "tt.reduce"({value}) <{{axis = 1 : i32}}> ({{')
    e.emit(f"    ^bb0({lhs}: f32, {rhs}: f32):")
    e.emit(f"      {combined} = arith.addf {lhs}, {rhs} : f32")
    e.emit(f"      tt.reduce.return {combined} : f32")
    e.emit(
        f"    }}) : (tensor<{rows}x{cols}xf32>) -> tensor<{rows}xf32>")
    return result


def _broadcast_1d(e: _Emit, value: str, size: int, block_e: int) -> str:
    """Broadcast a ``tensor<size xf32>`` row vector to ``[block_e, size]``."""
    expanded = _expand(e, value, 0, f"{size}xf32", f"1x{size}xf32")
    return _broadcast(
        e, expanded, f"1x{size}xf32", f"{block_e}x{size}xf32")


def _broadcast_col(e: _Emit, value: str, block_e: int, size: int) -> str:
    """Broadcast a ``tensor<block_e xf32>`` column to ``[block_e, size]``."""
    expanded = _expand(e, value, 1, f"{block_e}xf32", f"{block_e}x1xf32")
    return _broadcast(
        e, expanded, f"{block_e}x1xf32", f"{block_e}x{size}xf32")


_SELU_ALPHA = 1.6732632423543772
_SELU_SCALE = 1.0507009873554805


def _activation(
    e: _Emit,
    kind: str,
    value: str,
    shape: str,
    params: tuple[float | None, ...] = (),
    width: int | None = None,
) -> str:
    """Apply one component-wise op to a ``tensor<shape xf32>``.

    ``width`` is the true (unpadded) feature count; when given, padded lanes
    of the result are zeroed so the tile invariant survives ops whose value
    at zero is not zero (sigmoid, exp, …) or that diverge at zero (log,
    sqrt, reciprocal).
    """
    if kind == "relu":
        zero = _dense(e, "0.0", shape, "f32")
        result = e.assign(f"arith.maxnumf {value}, {zero} : tensor<{shape}xf32>")
    elif kind == "exp":
        result = _math(e, "exp", value, shape)
    elif kind == "sigmoid":
        result = _sigmoid_tile(e, value, shape)
    elif kind == "silu":
        result = _mul(e, value, _sigmoid_tile(e, value, shape), shape)
    elif kind == "tanh":
        result = _tanh_tile(e, value, shape)
    elif kind == "gelu":
        # exact gelu: x/2 · (1 + erf(x/√2)); math.tanh is unavailable.
        half = _dense(e, "0.5", shape, "f32")
        inv_sqrt2 = _dense(e, "0.7071067811865476", shape, "f32")
        one = _dense(e, "1.0", shape, "f32")
        scaled = _mul(e, value, inv_sqrt2, shape)
        erf = _math(e, "erf", scaled, shape)
        plus = _add(e, erf, one, shape)
        halfx = _mul(e, value, half, shape)
        result = _mul(e, halfx, plus, shape)
    elif kind == "leaky_relu":
        slope = _dense(e, params[0], shape, "f32")
        pos = _cmp_const(e, "ogt", value, "0.0", shape)
        result = _where(e, pos, value, _mul(e, value, slope, shape), shape)
    elif kind == "elu":
        result = _elu_tile(e, value, shape, params[0])
    elif kind == "hardtanh":
        result = _clamp_tile(e, value, shape, params[0], params[1])
    elif kind == "abs":
        result = _math(e, "absf", value, shape)
    elif kind == "sqrt":
        result = _math(e, "sqrt", value, shape)
    elif kind == "rsqrt":
        result = _math(e, "rsqrt", value, shape)
    elif kind == "log":
        result = _math(e, "log", value, shape)
    elif kind == "sin":
        result = _math(e, "sin", value, shape)
    elif kind == "cos":
        result = _math(e, "cos", value, shape)
    elif kind == "square":
        result = _mul(e, value, value, shape)
    elif kind == "softplus":
        result = _softplus_tile(e, value, shape, params[0], params[1])
    elif kind == "mish":
        sp = _softplus_tile(e, value, shape, 1.0, 20.0)
        result = _mul(e, value, _tanh_tile(e, sp, shape), shape)
    elif kind == "hardswish":
        three = _dense(e, "3.0", shape, "f32")
        sixth = _dense(e, 1.0 / 6.0, shape, "f32")
        gate = _clamp_tile(e, _add(e, value, three, shape), shape, 0.0, 6.0)
        result = _mul(e, _mul(e, value, gate, shape), sixth, shape)
    elif kind == "hardsigmoid":
        three = _dense(e, "3.0", shape, "f32")
        sixth = _dense(e, 1.0 / 6.0, shape, "f32")
        gate = _clamp_tile(e, _add(e, value, three, shape), shape, 0.0, 6.0)
        result = _mul(e, gate, sixth, shape)
    elif kind == "selu":
        elu = _elu_tile(e, value, shape, _SELU_ALPHA)
        result = _mul(e, elu, _dense(e, _SELU_SCALE, shape, "f32"), shape)
    elif kind == "mul_const":
        result = _mul(e, value, _dense(e, params[0], shape, "f32"), shape)
    elif kind == "add_const":
        result = _add(e, value, _dense(e, params[0], shape, "f32"), shape)
    elif kind == "rsub_const":
        result = _sub(e, _dense(e, params[0], shape, "f32"), value, shape)
    elif kind == "div_const":
        result = _div(e, value, _dense(e, params[0], shape, "f32"), shape)
    elif kind == "rdiv_const":
        result = _div(e, _dense(e, params[0], shape, "f32"), value, shape)
    elif kind == "pow_const":
        result = _pow_tile(e, value, shape, params[0])
    else:
        raise NotImplementedError(f"unsupported edge nn activation {kind!r}")
    if width is not None:
        block_e, pad = (int(part) for part in shape.split("x"))
        result = _mask_columns(e, result, width, block_e, pad)
    return result


def _norm_tile(
    e: _Emit,
    index: int,
    layer: NormLayer,
    x: str,
    block_e: int,
    w_pad: int,
) -> tuple[str, tuple[str, str, str, str | None]]:
    """Feature-wise LayerNorm over a ``[block_e, w_pad]`` tile.

    Returns the normalized tile plus the SSA values the symbolic VJP needs
    (``x_hat``, ``rstd`` broadcast, the f32 column mask, gamma broadcast).
    Row statistics divide by the true width; padded lanes are masked to
    zero before reducing and again at the end.
    """
    shape = f"{block_e}x{w_pad}"
    nr = _range(e, w_pad)
    gamma = beta = None
    if layer.weight_name is not None:
        gp = _splat_ptr(e, f"%w{index}", str(w_pad))
        gp2 = _addptr(e, gp, nr, str(w_pad))
        gamma = e.assign(f"tt.load {gp2} : tensor<{w_pad}x!tt.ptr<f32>>")
    if layer.bias_name is not None:
        bp = _splat_ptr(e, f"%b{index}", str(w_pad))
        bp2 = _addptr(e, bp, nr, str(w_pad))
        beta = e.assign(f"tt.load {bp2} : tensor<{w_pad}x!tt.ptr<f32>>")
    nr2 = _expand(e, nr, 0, f"{w_pad}xi32", f"1x{w_pad}xi32")
    c_w = _dense(e, layer.width, f"1x{w_pad}", "i32")
    m = e.assign(f"arith.cmpi slt, {nr2}, {c_w} : tensor<1x{w_pad}xi32>")
    m2 = _broadcast(e, m, f"1x{w_pad}xi1", f"{shape}xi1")
    one = _dense(e, "1.0", shape, "f32")
    zero = _dense(e, "0.0", shape, "f32")
    mf = _where(e, m2, one, zero, shape)
    xm = _mul(e, x, mf, shape)
    inv_w = _dense(e, 1.0 / layer.width, str(block_e), "f32")
    mean = _mul(e, _row_sum(e, xm, block_e, w_pad), inv_w, str(block_e))
    mean2 = _broadcast_col(e, mean, block_e, w_pad)
    diff = _mul(e, _sub(e, x, mean2, shape), mf, shape)
    sq = _mul(e, diff, diff, shape)
    var = _mul(e, _row_sum(e, sq, block_e, w_pad), inv_w, str(block_e))
    eps = _dense(e, layer.eps, str(block_e), "f32")
    one_1d = _dense(e, "1.0", str(block_e), "f32")
    rstd = _div(
        e, one_1d, _math(e, "sqrt", _add(e, var, eps, str(block_e)),
                         str(block_e)), str(block_e))
    rstd2 = _broadcast_col(e, rstd, block_e, w_pad)
    xhat = _mul(e, diff, rstd2, shape)
    y = xhat
    gamma2 = None
    if gamma is not None:
        gamma2 = _broadcast_1d(e, gamma, w_pad, block_e)
        y = _mul(e, y, gamma2, shape)
    if beta is not None:
        y = _add(e, y, _broadcast_1d(e, beta, w_pad, block_e), shape)
    y = _mul(e, y, mf, shape)
    return y, (xhat, rstd2, mf, gamma2)


def emit_edge_nn_tile_ttir(spec: EdgeNNTileSpec) -> str:
    """Emit the canonical TTIR text for one proved edge nn tile kernel."""
    block_e = spec.block_e
    k_pad = spec.k_pad
    e = _Emit()

    segment_params = []
    for segment in spec.segments:
        if segment.kind == "displacement":
            continue
        segment_params.append(
            f"%seg{len(segment_params)}: !tt.ptr<f32> "
            "{tt.divisibility = 16 : i32}")
    if spec.needs_positions:
        segment_params.append("%positions: !tt.ptr<f32> {tt.divisibility = 16 : i32}")
    weight_params = []
    for index, layer in enumerate(spec.layers):
        if getattr(layer, "weight_name", None) is None:
            continue
        weight_params.append(
            f"%w{index}: !tt.ptr<f32> {{tt.divisibility = 16 : i32}}")
        if layer.bias_name is not None:
            weight_params.append(
                f"%b{index}: !tt.ptr<f32> {{tt.divisibility = 16 : i32}}")
    signature = ", ".join([
        "%dst_idx: !tt.ptr<i32> {tt.divisibility = 16 : i32}",
        "%src_idx: !tt.ptr<i32> {tt.divisibility = 16 : i32}",
        *segment_params,
        *weight_params,
        "%out: !tt.ptr<f32> {tt.divisibility = 16 : i32}",
        "%num_edges: i32",
    ])

    header = (
        f"// tiga.launch entry={ENTRY} block_e={block_e} "
        f"num_warps={spec.num_warps} abi=edge-nn-tile\n"
        "// emitted by tiga.compiler.edge_nn_tile (Python emission "
        "backend, phase 1)\n")
    e.emit("module {")
    e.emit(
        f"  tt.func public @{ENTRY}({signature}) "
        "attributes {noinline = false} {")

    zero_1d = _dense(e, 0, str(block_e), "i32")
    zero_2d = _dense(e, "0.0", f"{block_e}x{k_pad}", "f32")
    c_b = _scalar(e, block_e)
    pid = e.assign("tt.get_program_id x : i32")
    base = e.assign(f"arith.muli {pid}, {c_b} : i32")
    edge_range = _range(e, block_e)
    base_v = _splat_int(e, base, str(block_e))
    edges = e.assign(
        f"arith.addi {base_v}, {edge_range} : tensor<{block_e}xi32>")
    ne_v = _splat_int(e, "%num_edges", str(block_e))
    emask = e.assign(
        f"arith.cmpi slt, {edges}, {ne_v} : tensor<{block_e}xi32>")

    def load_index(ptr: str) -> str:
        splat = e.assign(
            f"tt.splat {ptr} : !tt.ptr<i32> -> tensor<{block_e}x!tt.ptr<i32>>")
        addr = e.assign(
            f"tt.addptr {splat}, {edges} : tensor<{block_e}x!tt.ptr<i32>>, "
            f"tensor<{block_e}xi32>")
        return e.assign(
            f"tt.load {addr}, {emask}, {zero_1d} : "
            f"tensor<{block_e}x!tt.ptr<i32>>")

    dst = load_index("%dst_idx")
    src = load_index("%src_idx")
    k_range = _range(e, k_pad)
    k2 = _expand(e, k_range, 0, f"{k_pad}xi32", f"1x{k_pad}xi32")
    rows = _expand(e, emask, 1, f"{block_e}xi1", f"{block_e}x1xi1")
    rows_b = _broadcast(e, rows, f"{block_e}x1xi1", f"{block_e}x{k_pad}xi1")

    tiles = []
    seg_index = 0
    for segment in spec.segments:
        if segment.kind == "displacement":
            from_src = _gather_columns(
                e, ptr="%positions", row_index=src, width=segment.width,
                offset=segment.offset, k_pad=k_pad, block_e=block_e,
                rows_b=rows_b, k2=k2, zero_2d=zero_2d)
            from_dst = _gather_columns(
                e, ptr="%positions", row_index=dst, width=segment.width,
                offset=segment.offset, k_pad=k_pad, block_e=block_e,
                rows_b=rows_b, k2=k2, zero_2d=zero_2d)
            tiles.append(e.assign(
                f"arith.subf {from_src}, {from_dst} : "
                f"tensor<{block_e}x{k_pad}xf32>"))
            continue
        row_index = {
            "src_field": src,
            "dst_field": dst,
            "edge_field": edges,
        }[segment.kind]
        tiles.append(_gather_columns(
            e, ptr=f"%seg{seg_index}", row_index=row_index,
            width=segment.width, offset=segment.offset, k_pad=k_pad,
            block_e=block_e, rows_b=rows_b, k2=k2, zero_2d=zero_2d))
        seg_index += 1
    current = tiles[0]
    for tile in tiles[1:]:
        current = e.assign(
            f"arith.addf {current}, {tile} : tensor<{block_e}x{k_pad}xf32>")

    current, _records, _pad = _replay_chain(
        e, spec.layers, spec.in_width, k_pad, current, block_e)

    out_pad = spec.out_pad
    nr_last = _range(e, out_pad)
    nr2_last = _expand(e, nr_last, 0, f"{out_pad}xi32", f"1x{out_pad}xi32")
    c_ow = _dense(e, spec.out_width, f"1x{out_pad}", "i32")
    om = e.assign(
        f"arith.cmpi slt, {nr2_last}, {c_ow} : tensor<1x{out_pad}xi32>")
    om2 = _broadcast(e, om, f"1x{out_pad}xi1", f"{block_e}x{out_pad}xi1")
    rows_b2 = _broadcast(
        e, rows, f"{block_e}x1xi1", f"{block_e}x{out_pad}xi1")
    omask = e.assign(f"arith.andi {rows_b2}, {om2} : tensor<{block_e}x{out_pad}xi1>")
    dst2 = _expand(e, dst, 1, f"{block_e}xi32", f"{block_e}x1xi32")
    c_ow2 = _dense(e, spec.out_width, f"{block_e}x1", "i32")
    orow = e.assign(f"arith.muli {dst2}, {c_ow2} : tensor<{block_e}x1xi32>")
    op = _splat_ptr(e, "%out", f"{block_e}x1")
    op2 = _addptr(e, op, orow, f"{block_e}x1")
    op3 = _broadcast(
        e, op2, f"{block_e}x1x!tt.ptr<f32>", f"{block_e}x{out_pad}x!tt.ptr<f32>")
    nr3_last = _broadcast(
        e, nr2_last, f"1x{out_pad}xi32", f"{block_e}x{out_pad}xi32")
    op4 = _addptr(e, op3, nr3_last, f"{block_e}x{out_pad}")
    e.assign(
        f"tt.atomic_rmw fadd, relaxed, gpu, {op4}, {current}, {omask} : "
        f"(tensor<{block_e}x{out_pad}x!tt.ptr<f32>>, "
        f"tensor<{block_e}x{out_pad}xf32>, tensor<{block_e}x{out_pad}xi1>) "
        f"-> tensor<{block_e}x{out_pad}xf32>")
    e.emit("    tt.return")
    e.emit("  }")
    e.emit("}")
    return header + "\n".join(e.lines) + "\n"


__all__ = [
    "BLOCK_E",
    "ENTRY",
    "NUM_WARPS",
    "EdgeNNTileSpec",
    "Segment",
    # shared with compiler.edge_nn_vjp (same emission backend)
    "_activation",
    "_broadcast_1d",
    "_broadcast_col",
    "_cmp_const",
    "_linear_tile",
    "_mask_columns",
    "_math",
    "_mul",
    "_norm_tile",
    "_pow_tile",
    "_row_sum",
    "_sigmoid_tile",
    "_softplus_tile",
    "_tanh_tile",
    "_where",
    "build_edge_nn_spec",
    "emit_edge_nn_tile_ttir",
]
