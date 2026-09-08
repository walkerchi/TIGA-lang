"""Symbolic VJP and Python TTIR emission for fused edge-NN tile backward.

The forward tile kernel (``compiler.edge_nn_tile``) materializes no O(E)
tensor; the backward must not either, or the memory bound the fusion was
built to remove comes back through autograd's saved activations.  This
emitter therefore produces one recompute-style kernel (the FlashAttention
pattern): per edge block it replays the forward chain in registers, walks
the same chain in reverse with the symbolic VJP rules

- Linear ``h = a @ W.T + b``: ``dW += a.T @ d_h`` (tile outer product),
  ``db += sum_edges d_h`` (column reduction), ``d_a = d_h @ W``;
- component-wise activations: ``d_a = d_h * sigma'(pre/post)``;
- the sum reducer's adjoint is a broadcast: ``d_message[e] = d_out[dst(e)]``;
- the concatenated input's adjoint scatters back: src/dst/edge field grads
  are masked atomic adds, and a displacement segment contributes
  ``d_pos[src] += d_in`` and ``d_pos[dst] -= d_in``.

All gradient accumulators are per-tile partials merged with relaxed atomics;
weights/biases accumulate into small padded buffers the host slices back to
parameter layout.  No per-edge activation ever leaves a tile.

The emitted text crosses the same serialized-TTIR provider boundary as the
forward kernel; anything the proof rejects still falls back to the eager
oracle, which remains the semantic ground truth.
"""

from __future__ import annotations

from .edge_nn_tile import (
    EdgeNNTileSpec,
    _add,
    _addptr,
    _broadcast,
    _broadcast_col,
    _cmp_const,
    _dense,
    _div,
    _Emit,
    _expand,
    _gather_columns,
    _mask_columns,
    _math,
    _mul,
    _pow_tile,
    _range,
    _replay_chain,
    _row_sum,
    _scalar,
    _sigmoid_tile,
    _softplus_tile,
    _splat_int,
    _splat_ptr,
    _sub,
    _tanh_tile,
    _where,
)

ENTRY_VJP = "gf_edge_nn_vjp"


def _atomic_add(e: _Emit, ptr: str, value: str, shape: str, mask: str | None = None) -> None:
    arguments = f"{ptr}, {value}" + (f", {mask}" if mask is not None else "")
    types = (
        f"(tensor<{shape}x!tt.ptr<f32>>, tensor<{shape}xf32>"
        + (f", tensor<{shape}xi1>" if mask is not None else "")
        + f") -> tensor<{shape}xf32>"
    )
    e.assign(f"tt.atomic_rmw fadd, relaxed, gpu, {arguments} : {types}")


def _trans(e: _Emit, value: str, src: str, dst: str) -> str:
    return e.assign(
        f"tt.trans {value} {{order = array<i32: 1, 0>}} : "
        f"tensor<{src}xf32> -> tensor<{dst}xf32>")


def _column_sum(e: _Emit, value: str, rows: int, cols: int) -> str:
    """Sum a ``tensor<rows x cols xf32>`` over axis 0 into ``tensor<cols>``."""
    result, lhs, rhs, combined = e.v(), e.v(), e.v(), e.v()
    e.emit(f'    {result} = "tt.reduce"({value}) <{{axis = 0 : i32}}> ({{')
    e.emit(f"    ^bb0({lhs}: f32, {rhs}: f32):")
    e.emit(f"      {combined} = arith.addf {lhs}, {rhs} : f32")
    e.emit(f"      tt.reduce.return {combined} : f32")
    e.emit(
        f"    }}) : (tensor<{rows}x{cols}xf32>) -> tensor<{cols}xf32>")
    return result


def _activation_grad(
    e: _Emit,
    kind: str,
    pre: str,
    post: str,
    d: str,
    shape: str,
    params: tuple[float | None, ...] = (),
) -> str:
    """Multiply ``d`` by the op's derivative evaluated at ``pre``/``post``.

    Padded lanes of ``pre`` are exactly zero (the forward masks every op's
    output), so derivatives that diverge at zero (log, sqrt, rdiv, …) can
    produce inf/NaN there; the caller re-masks the result with
    ``_mask_columns`` so the zero-padded-lane invariant holds in reverse.
    """
    if kind == "add_const":
        return d
    if kind in ("mul_const", "rsub_const"):
        sign = params[0] if kind == "mul_const" else -1.0
        return _mul(e, d, _dense(e, sign, shape, "f32"), shape)
    if kind == "div_const":
        return _div(e, d, _dense(e, params[0], shape, "f32"), shape)
    if kind == "rdiv_const":
        sq = _mul(e, pre, pre, shape)
        factor = _div(
            e, _dense(e, -params[0], shape, "f32"), sq, shape)
        return _mul(e, d, factor, shape)
    if kind == "pow_const":
        power = _pow_tile(e, pre, shape, params[0] - 1.0)
        factor = _mul(e, power, _dense(e, params[0], shape, "f32"), shape)
        return _mul(e, d, factor, shape)

    one = _dense(e, "1.0", shape, "f32")
    zero = _dense(e, "0.0", shape, "f32")
    if kind == "relu":
        positive = _cmp_const(e, "ogt", pre, "0.0", shape)
        factor = _where(e, positive, one, zero, shape)
    elif kind == "exp":
        factor = post
    elif kind == "sigmoid":
        complement = _sub(e, one, post, shape)
        factor = _mul(e, post, complement, shape)
    elif kind == "tanh":
        squared = _mul(e, post, post, shape)
        factor = _sub(e, one, squared, shape)
    elif kind == "silu":
        # d/dx [x * sigmoid(x)] = s * (1 + x * (1 - s))
        s = _sigmoid_tile(e, pre, shape)
        complement = _sub(e, one, s, shape)
        inner = _mul(e, pre, complement, shape)
        inner1 = _add(e, one, inner, shape)
        factor = _mul(e, s, inner1, shape)
    elif kind == "gelu":
        # exact gelu': Phi(x) + x * pdf(x); pdf(x) = exp(-x^2/2) / sqrt(2 pi)
        half = _dense(e, "0.5", shape, "f32")
        inv_sqrt2 = _dense(e, "0.7071067811865476", shape, "f32")
        inv_sqrt2pi = _dense(e, "0.3989422804014327", shape, "f32")
        scaled = _mul(e, pre, inv_sqrt2, shape)
        erf = _math(e, "erf", scaled, shape)
        cdf = _mul(e, half, _add(e, erf, one, shape), shape)
        xx = _mul(e, pre, pre, shape)
        neg_half_xx = _mul(e, xx, _dense(e, "-0.5", shape, "f32"), shape)
        pdf = _mul(e, inv_sqrt2pi, _math(e, "exp", neg_half_xx, shape), shape)
        xpdf = _mul(e, pre, pdf, shape)
        factor = _add(e, cdf, xpdf, shape)
    elif kind == "leaky_relu":
        positive = _cmp_const(e, "ogt", pre, "0.0", shape)
        slope = _dense(e, params[0], shape, "f32")
        factor = _where(e, positive, one, slope, shape)
    elif kind == "elu":
        positive = _cmp_const(e, "ogt", pre, "0.0", shape)
        ex = _math(e, "exp", pre, shape)
        branch = _mul(e, ex, _dense(e, params[0], shape, "f32"), shape)
        factor = _where(e, positive, one, branch, shape)
    elif kind == "hardtanh":
        inside = None
        if params[0] is not None:
            inside = _cmp_const(e, "ogt", pre, params[0], shape)
        if params[1] is not None:
            below = _cmp_const(e, "olt", pre, params[1], shape)
            inside = below if inside is None else e.assign(
                f"arith.andi {inside}, {below} : tensor<{shape}xi1>")
        factor = _where(e, inside, one, zero, shape)
    elif kind == "abs":
        positive = _cmp_const(e, "ogt", pre, "0.0", shape)
        negative = _cmp_const(e, "olt", pre, "0.0", shape)
        minus_one = _dense(e, "-1.0", shape, "f32")
        factor = _where(
            e, positive, one, _where(e, negative, minus_one, zero, shape),
            shape)
    elif kind == "sqrt":
        half = _dense(e, "0.5", shape, "f32")
        factor = _div(e, half, post, shape)
    elif kind == "rsqrt":
        # d/dx x^(-1/2) = -1/2 · x^(-3/2) = -1/2 · post^3
        cubed = _mul(e, _mul(e, post, post, shape), post, shape)
        factor = _mul(e, cubed, _dense(e, "-0.5", shape, "f32"), shape)
    elif kind == "log":
        factor = _div(e, one, pre, shape)
    elif kind == "sin":
        factor = _math(e, "cos", pre, shape)
    elif kind == "cos":
        factor = _sub(e, zero, _math(e, "sin", pre, shape), shape)
    elif kind == "square":
        factor = _mul(e, pre, _dense(e, "2.0", shape, "f32"), shape)
    elif kind == "softplus":
        beta = _dense(e, params[0], shape, "f32")
        bx = _mul(e, pre, beta, shape)
        big = _cmp_const(e, "ogt", bx, params[1], shape)
        factor = _where(e, big, one, _sigmoid_tile(e, bx, shape), shape)
    elif kind == "mish":
        # y = x·t, t = tanh(softplus(x)); dy = t + x·(1−t²)·softplus'(x)
        sp = _softplus_tile(e, pre, shape, 1.0, 20.0)
        t = _tanh_tile(e, sp, shape)
        big = _cmp_const(e, "ogt", pre, 20.0, shape)
        ds = _where(e, big, one, _sigmoid_tile(e, pre, shape), shape)
        tt = _mul(e, t, t, shape)
        dt = _mul(e, _sub(e, one, tt, shape), ds, shape)
        factor = _add(e, t, _mul(e, pre, dt, shape), shape)
    elif kind == "hardswish":
        above_lo = _cmp_const(e, "ogt", pre, "-3.0", shape)
        below_hi = _cmp_const(e, "olt", pre, "3.0", shape)
        mid = e.assign(
            f"arith.andi {above_lo}, {below_hi} : tensor<{shape}xi1>")
        third = _dense(e, 1.0 / 3.0, shape, "f32")
        half = _dense(e, "0.5", shape, "f32")
        slope = _add(e, _mul(e, pre, third, shape), half, shape)
        past = _cmp_const(e, "ogt", pre, "3.0", shape)
        factor = _where(
            e, mid, slope, _where(e, past, one, zero, shape), shape)
    elif kind == "hardsigmoid":
        above_lo = _cmp_const(e, "ogt", pre, "-3.0", shape)
        below_hi = _cmp_const(e, "olt", pre, "3.0", shape)
        mid = e.assign(
            f"arith.andi {above_lo}, {below_hi} : tensor<{shape}xi1>")
        factor = _where(e, mid, _dense(e, 1.0 / 6.0, shape, "f32"), zero,
                        shape)
    elif kind == "selu":
        from .edge_nn_tile import _SELU_ALPHA, _SELU_SCALE

        positive = _cmp_const(e, "ogt", pre, "0.0", shape)
        ex = _math(e, "exp", pre, shape)
        branch = _mul(e, ex, _dense(e, _SELU_ALPHA, shape, "f32"), shape)
        inner = _where(e, positive, one, branch, shape)
        factor = _mul(e, inner, _dense(e, _SELU_SCALE, shape, "f32"), shape)
    else:
        raise NotImplementedError(f"unsupported edge nn activation {kind!r}")
    return _mul(e, d, factor, shape)


def _scatter_columns(
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
    value: str,
    negate: bool = False,
) -> None:
    """Atomically add one column range of the ``[block_e, k_pad]`` grad tile."""
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
    if negate:
        zero = _dense(e, "0.0", f"{block_e}x{k_pad}", "f32")
        value = e.assign(
            f"arith.subf {zero}, {value} : tensor<{block_e}x{k_pad}xf32>")
    _atomic_add(e, base4, value, f"{block_e}x{k_pad}", mask)


def _reverse_chain(e: _Emit, records, layers, block_e: int, d: str) -> str:
    """Walk recompute records in reverse with the symbolic VJP rules.

    ``d`` is the chain-output adjoint tile; returns the chain-input adjoint
    tile.  Weight/bias gradients accumulate into the padded ``%dw{index}`` /
    ``%db{index}`` kernel parameters with relaxed atomics.
    """
    for record in reversed(records):
        if record[0] == "act":
            _t, kind, params, pre, post, width, pad = record
            d = _activation_grad(
                e, kind, pre, post, d, f"{block_e}x{pad}", params)
            d = _mask_columns(e, d, width, block_e, pad)
            continue
        if record[0] == "norm":
            _t, index, xhat, rstd2, mf, gamma2, width, pad = record
            layer = layers[index]
            shape = f"{block_e}x{pad}"
            dnr = _range(e, pad)
            if layer.weight_name is not None:
                # dγ += Σ_edges d ⊙ x̂ (padded lanes are zero throughout)
                dgamma = _column_sum(e, _mul(e, d, xhat, shape), block_e, pad)
                dgp = _splat_ptr(e, f"%dw{index}", str(pad))
                dgp2 = _addptr(e, dgp, dnr, str(pad))
                _atomic_add(e, dgp2, dgamma, str(pad))
            if layer.bias_name is not None:
                dbeta = _column_sum(e, d, block_e, pad)
                dbp = _splat_ptr(e, f"%db{index}", str(pad))
                dbp2 = _addptr(e, dbp, dnr, str(pad))
                _atomic_add(e, dbp2, dbeta, str(pad))
            # d_x = rstd · (dx̂ − mean(dx̂) − x̂ · mean(dx̂ ⊙ x̂)), masked
            dxh = _mul(e, d, gamma2, shape) if gamma2 is not None else d
            inv_w = _dense(e, 1.0 / width, str(block_e), "f32")
            row_a = _mul(
                e, _row_sum(e, dxh, block_e, pad), inv_w, str(block_e))
            row_b = _mul(
                e, _row_sum(e, _mul(e, dxh, xhat, shape), block_e, pad),
                inv_w, str(block_e))
            a2 = _broadcast_col(e, row_a, block_e, pad)
            b2 = _broadcast_col(e, row_b, block_e, pad)
            centered = _sub(
                e, _sub(e, dxh, a2, shape), _mul(e, xhat, b2, shape), shape)
            d = _mul(e, _mul(e, rstd2, centered, shape), mf, shape)
            continue
        _tag, index, a, layer_in_pad, layer_out_pad, wt = record
        # dW += a.T @ d (zero rows/cols of padded lanes contribute nothing)
        a_t = _trans(
            e, a, f"{block_e}x{layer_in_pad}", f"{layer_in_pad}x{block_e}")
        dw_acc = _dense(e, "0.0", f"{layer_in_pad}x{layer_out_pad}", "f32")
        dw_tile = e.assign(
            f"tt.dot {a_t}, {d}, {dw_acc} : "
            f"tensor<{layer_in_pad}x{block_e}xf32> * "
            f"tensor<{block_e}x{layer_out_pad}xf32> -> "
            f"tensor<{layer_in_pad}x{layer_out_pad}xf32>")
        dwr = _range(e, layer_in_pad)
        dwr2 = _expand(e, dwr, 1, f"{layer_in_pad}xi32", f"{layer_in_pad}x1xi32")
        c_dwo = _dense(e, layer_out_pad, f"{layer_in_pad}x1", "i32")
        dwrow = e.assign(f"arith.muli {dwr2}, {c_dwo} : tensor<{layer_in_pad}x1xi32>")
        dwp = _splat_ptr(e, f"%dw{index}", f"{layer_in_pad}x1")
        dwp2 = _addptr(e, dwp, dwrow, f"{layer_in_pad}x1")
        dnr = _range(e, layer_out_pad)
        dnr2 = _expand(e, dnr, 0, f"{layer_out_pad}xi32", f"1x{layer_out_pad}xi32")
        dwp3 = _broadcast(
            e, dwp2, f"{layer_in_pad}x1x!tt.ptr<f32>",
            f"{layer_in_pad}x{layer_out_pad}x!tt.ptr<f32>")
        dnr3 = _broadcast(
            e, dnr2, f"1x{layer_out_pad}xi32",
            f"{layer_in_pad}x{layer_out_pad}xi32")
        dwp4 = _addptr(e, dwp3, dnr3, f"{layer_in_pad}x{layer_out_pad}")
        _atomic_add(e, dwp4, dw_tile, f"{layer_in_pad}x{layer_out_pad}")
        layer = layers[index]
        if layer.bias_name is not None:
            db_vec = _column_sum(e, d, block_e, layer_out_pad)
            dbp = _splat_ptr(e, f"%db{index}", str(layer_out_pad))
            dbp2 = _addptr(e, dbp, dnr, str(layer_out_pad))
            _atomic_add(e, dbp2, db_vec, str(layer_out_pad))
        # d_a = d @ W (W transposed back from its forward tile layout)
        wt_t = _trans(
            e, wt, f"{layer_in_pad}x{layer_out_pad}",
            f"{layer_out_pad}x{layer_in_pad}")
        da_acc = _dense(e, "0.0", f"{block_e}x{layer_in_pad}", "f32")
        d = e.assign(
            f"tt.dot {d}, {wt_t}, {da_acc} : "
            f"tensor<{block_e}x{layer_out_pad}xf32> * "
            f"tensor<{layer_out_pad}x{layer_in_pad}xf32> -> "
            f"tensor<{block_e}x{layer_in_pad}xf32>")
    return d


def emit_edge_nn_vjp_ttir(spec: EdgeNNTileSpec) -> str:
    """Emit the canonical TTIR text for one fused edge nn backward kernel."""
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
    grad_field_params = [
        f"%dseg{index}: !tt.ptr<f32> {{tt.divisibility = 16 : i32}}"
        for index in range(len(segment_params) - spec.needs_positions)
    ]
    if spec.needs_positions:
        grad_field_params.append(
            "%d_positions: !tt.ptr<f32> {tt.divisibility = 16 : i32}")
    grad_weight_params = []
    for index, layer in enumerate(spec.layers):
        if getattr(layer, "weight_name", None) is None:
            continue
        grad_weight_params.append(
            f"%dw{index}: !tt.ptr<f32> {{tt.divisibility = 16 : i32}}")
        if layer.bias_name is not None:
            grad_weight_params.append(
                f"%db{index}: !tt.ptr<f32> {{tt.divisibility = 16 : i32}}")
    signature = ", ".join([
        "%dst_idx: !tt.ptr<i32> {tt.divisibility = 16 : i32}",
        "%src_idx: !tt.ptr<i32> {tt.divisibility = 16 : i32}",
        *segment_params,
        *weight_params,
        "%d_out: !tt.ptr<f32> {tt.divisibility = 16 : i32}",
        *grad_field_params,
        *grad_weight_params,
        "%num_edges: i32",
    ])

    header = (
        f"// tiga.launch entry={ENTRY_VJP} block_e={block_e} "
        f"num_warps={spec.num_warps} abi=edge-nn-vjp\n"
        "// emitted by tiga.compiler.edge_nn_vjp (Python emission "
        "backend, phase 2)\n")
    e.emit("module {")
    e.emit(
        f"  tt.func public @{ENTRY_VJP}({signature}) "
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

    # --- recompute the forward chain, keeping what the VJP rules need ----
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

    # --- recompute the forward chain, keeping what the VJP rules need ----
    _current, records, _pad = _replay_chain(
        e, spec.layers, spec.in_width, k_pad, current, block_e,
        keep_records=True)

    # --- adjoint of the sum reducer: d_message[e] = d_out[dst(e)] --------
    out_pad = spec.out_pad
    nr_last = _range(e, out_pad)
    nr2_last = _expand(e, nr_last, 0, f"{out_pad}xi32", f"1x{out_pad}xi32")
    c_ow = _dense(e, spec.out_width, f"1x{out_pad}", "i32")
    om = e.assign(
        f"arith.cmpi slt, {nr2_last}, {c_ow} : tensor<1x{out_pad}xi32>")
    om2 = _broadcast(e, om, f"1x{out_pad}xi1", f"{block_e}x{out_pad}xi1")
    rows_b2 = _broadcast(
        e, rows, f"{block_e}x1xi1", f"{block_e}x{out_pad}xi1")
    dmask = e.assign(f"arith.andi {rows_b2}, {om2} : tensor<{block_e}x{out_pad}xi1>")
    dst2 = _expand(e, dst, 1, f"{block_e}xi32", f"{block_e}x1xi32")
    c_ow2 = _dense(e, spec.out_width, f"{block_e}x1", "i32")
    drow = e.assign(f"arith.muli {dst2}, {c_ow2} : tensor<{block_e}x1xi32>")
    dp = _splat_ptr(e, "%d_out", f"{block_e}x1")
    dp2 = _addptr(e, dp, drow, f"{block_e}x1")
    dp3 = _broadcast(
        e, dp2, f"{block_e}x1x!tt.ptr<f32>", f"{block_e}x{out_pad}x!tt.ptr<f32>")
    nr3_last = _broadcast(
        e, nr2_last, f"1x{out_pad}xi32", f"{block_e}x{out_pad}xi32")
    dp4 = _addptr(e, dp3, nr3_last, f"{block_e}x{out_pad}")
    zero_dout = _dense(e, "0.0", f"{block_e}x{out_pad}", "f32")
    d = e.assign(
        f"tt.load {dp4}, {dmask}, {zero_dout} : "
        f"tensor<{block_e}x{out_pad}x!tt.ptr<f32>>")

    # --- reverse the chain with the symbolic VJP rules -------------------
    d = _reverse_chain(e, records, spec.layers, block_e, d)

    # --- scatter the input grad tile back to fields and positions --------
    seg_index = 0
    for segment in spec.segments:
        if segment.kind == "displacement":
            _scatter_columns(
                e, ptr="%d_positions", row_index=src, width=segment.width,
                offset=segment.offset, k_pad=k_pad, block_e=block_e,
                rows_b=rows_b, k2=k2, value=d)
            _scatter_columns(
                e, ptr="%d_positions", row_index=dst, width=segment.width,
                offset=segment.offset, k_pad=k_pad, block_e=block_e,
                rows_b=rows_b, k2=k2, value=d, negate=True)
            continue
        row_index = {
            "src_field": src,
            "dst_field": dst,
            "edge_field": edges,
        }[segment.kind]
        _scatter_columns(
            e, ptr=f"%dseg{seg_index}", row_index=row_index,
            width=segment.width, offset=segment.offset, k_pad=k_pad,
            block_e=block_e, rows_b=rows_b, k2=k2, value=d)
        seg_index += 1

    e.emit("    tt.return")
    e.emit("  }")
    e.emit("}")
    return header + "\n".join(e.lines) + "\n"


__all__ = ["ENTRY_VJP", "emit_edge_nn_vjp_ttir"]
