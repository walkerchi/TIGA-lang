"""Structural proof and Python TTIR emission for fused edge-NN attention.

The edge-centric tile kernel (``compiler.edge_nn_tile``) segment-reduces
with additive atomics; a softmax couples the edges of one destination row
(running-max rescaling), which atomics cannot express.  This emitter is
therefore **row-centric**: one program owns one CSR row and streams its
edges in ``BLOCK_N`` chunks, FlashAttention-style:

- per chunk, the nn score chain (proved by ``build_edge_nn_spec``,
  ``out_features == 1``) is evaluated on a ``[BLOCK_N, K_PAD]`` tile of
  gathered fields — no O(E) score tensor exists;
- the online-softmax state ``(m, l, acc)`` rides ``scf.for`` iter_args:
  ``m' = max(m, max_e s_e)``, ``l' = l·e^{m−m'} + Σ_e e^{s_e−m'}``,
  ``acc' = acc·e^{m−m'} + Σ_e e^{s_e−m'}·v_e`` for the value field ``v``;
- the epilogue writes ``acc / l`` (zero for degree-0 rows) directly — no
  atomics, no cross-program communication, no score matrix anywhere.

The emitted text crosses the same serialized-TTIR provider boundary as the
other Python emission backends.  Grad-mode calls still take the eager
oracle (the semantic ground truth); a fused VJP is a later phase.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .capture import Expr
from .edge_nn_tile import (
    _FIELD_KINDS,
    EdgeNNTileSpec,
    _broadcast,
    _broadcast_1d,
    _broadcast_col,
    _dense,
    _Emit,
    _expand,
    _gather_columns,
    _mask_columns,
    _mul,
    _pad16,
    _range,
    _replay_chain,
    _row_sum,
    _scalar,
    _splat_int,
    _where,
    build_edge_nn_spec,
)
from .edge_nn_vjp import _column_sum, _reverse_chain, _scatter_columns
from .nn_capture import ActivationLayer, NormLayer

ENTRY = "gf_edge_nn_attention"
NEG_INF = "-1.0e+30"


@dataclass(frozen=True)
class EdgeNNAttentionSpec:
    """The proved, hashable structure of one nn-scored attention kernel."""

    score: EdgeNNTileSpec  # nn chain with out_features == 1
    value_kind: str  # "src_field" | "dst_field" | "edge_field"
    value_name: str
    value_width: int
    value_pad: int


def build_edge_nn_attention_spec(
    item, *, field_widths: dict[tuple[str, str], int], position_dim: int | None
) -> EdgeNNAttentionSpec | None:
    """Prove an ``OnlineSoftmaxItem`` is nn-scored with a gatherable value."""
    score = getattr(item, "score", None)
    value = getattr(item, "value", None)
    if not isinstance(score, Expr) or score.op != "nn_subgraph":
        return None
    dag = score.args[0]
    if dag.out_features != 1:
        return None  # v1: one attention score per edge
    score_spec = build_edge_nn_spec(
        score, field_widths=field_widths, position_dim=position_dim)
    if score_spec is None:
        return None
    if dag.block_e is None or dag.num_warps is None:
        # Row-centric tiles waste (block_n − degree) lanes per chunk; small
        # chunks with one warp win for typical attention degrees (≲ 64).
        score_spec = replace(
            score_spec,
            block_e=dag.block_e if dag.block_e is not None else 16,
            num_warps=dag.num_warps if dag.num_warps is not None else 1)
    if not isinstance(value, Expr) or value.op != "field":
        return None
    role, name = value.args
    if role == "edge" and name in ("displacement", "distance"):
        return None  # v1 keeps implicit geometry out of the payload
    kind = _FIELD_KINDS.get(role)
    width = field_widths.get((role, name))
    if kind is None or width is None:
        return None
    return EdgeNNAttentionSpec(
        score=score_spec,
        value_kind=kind,
        value_name=name,
        value_width=width,
        value_pad=_pad16(width),
    )


def _splat_f32(e: _Emit, value: str, shape: str) -> str:
    return e.assign(f"tt.splat {value} : f32 -> tensor<{shape}xf32>")


def _reduce_1d(e: _Emit, value: str, size: int, combiner: str) -> str:
    """Reduce a ``tensor<size xf32>`` to a scalar f32 with addf/maximumf."""
    result, lhs, rhs, combined = e.v(), e.v(), e.v(), e.v()
    e.emit(f'    {result} = "tt.reduce"({value}) <{{axis = 0 : i32}}> ({{')
    e.emit(f"    ^bb0({lhs}: f32, {rhs}: f32):")
    e.emit(f"      {combined} = arith.{combiner} {lhs}, {rhs} : f32")
    e.emit(f"      tt.reduce.return {combined} : f32")
    e.emit(f"    }}) : (tensor<{size}xf32>) -> f32")
    return result


def _select_scalar(e: _Emit, cond: str, a: str, b: str) -> str:
    return e.assign(
        f'"arith.select"({cond}, {a}, {b}) : (i1, f32, f32) -> f32')


def emit_edge_nn_attention_ttir(spec: EdgeNNAttentionSpec) -> str:
    """Emit the canonical TTIR text for one proved nn-attention kernel."""
    score = spec.score
    block_n = score.block_e
    k_pad = score.k_pad
    w_pad = spec.value_pad
    e = _Emit()

    segment_params = []
    for segment in score.segments:
        if segment.kind == "displacement":
            continue
        segment_params.append(
            f"%seg{len(segment_params)}: !tt.ptr<f32> "
            "{tt.divisibility = 16 : i32}")
    if score.needs_positions:
        segment_params.append(
            "%positions: !tt.ptr<f32> {tt.divisibility = 16 : i32}")
    weight_params = []
    for index, layer in enumerate(score.layers):
        if getattr(layer, "weight_name", None) is None:
            continue
        weight_params.append(
            f"%w{index}: !tt.ptr<f32> {{tt.divisibility = 16 : i32}}")
        if layer.bias_name is not None:
            weight_params.append(
                f"%b{index}: !tt.ptr<f32> {{tt.divisibility = 16 : i32}}")
    signature = ", ".join([
        "%row_ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32}",
        "%col_idx: !tt.ptr<i32> {tt.divisibility = 16 : i32}",
        *segment_params,
        *weight_params,
        "%value: !tt.ptr<f32> {tt.divisibility = 16 : i32}",
        "%out: !tt.ptr<f32> {tt.divisibility = 16 : i32}",
        "%m_out: !tt.ptr<f32> {tt.divisibility = 16 : i32}",
        "%l_out: !tt.ptr<f32> {tt.divisibility = 16 : i32}",
    ])

    header = (
        f"// tiga.launch entry={ENTRY} block_e={block_n} "
        f"num_warps={score.num_warps} abi=edge-nn-attention\n"
        "// emitted by tiga.compiler.edge_nn_attention (Python "
        "emission backend, phase 3)\n")
    e.emit("module {")
    e.emit(
        f"  tt.func public @{ENTRY}({signature}) "
        "attributes {noinline = false} {")

    # --- row bounds (scalar pointer idiom) -------------------------------
    c1 = _scalar(e, 1)
    row = e.assign("tt.get_program_id x : i32")
    start_ptr = e.assign(f"tt.addptr %row_ptr, {row} : !tt.ptr<i32>, i32")
    row_start = e.assign(f"tt.load {start_ptr} : !tt.ptr<i32>")
    end_ptr = e.assign(f"tt.addptr {start_ptr}, {c1} : !tt.ptr<i32>, i32")
    row_end = e.assign(f"tt.load {end_ptr} : !tt.ptr<i32>")
    degree = e.assign(f"arith.subi {row_end}, {row_start} : i32")

    c0 = _scalar(e, 0)
    c_bn = _scalar(e, block_n)
    neg_inf = e.assign(f"arith.constant {NEG_INF} : f32")
    zero_s = e.assign("arith.constant 0.0 : f32")
    acc0 = _dense(e, "0.0", str(w_pad), "f32")
    e.emit(
        f"    %attn:3 = scf.for %cs = {c0} to {degree} step {c_bn} "
        f"iter_args(%m = {neg_inf}, %l = {zero_s}, %acc = {acc0}) -> "
        f"(f32, f32, tensor<{w_pad}xf32>) : i32 {{")

    # --- chunk geometry ----------------------------------------------------
    zero_1d = _dense(e, 0, str(block_n), "i32")
    zero_2d = _dense(e, "0.0", f"{block_n}x{k_pad}", "f32")
    lane = _range(e, block_n)
    off = e.assign(
        f"arith.addi {_splat_int(e, '%cs', str(block_n))}, {lane} : "
        f"tensor<{block_n}xi32>")
    emask = e.assign(
        f"arith.cmpi slt, {off}, {_splat_int(e, degree, str(block_n))} : "
        f"tensor<{block_n}xi32>")
    edges = e.assign(
        f"arith.addi {_splat_int(e, row_start, str(block_n))}, {off} : "
        f"tensor<{block_n}xi32>")
    col_base = e.assign(
        "tt.splat %col_idx : !tt.ptr<i32> -> "
        f"tensor<{block_n}x!tt.ptr<i32>>")
    col_ptr = e.assign(
        f"tt.addptr {col_base}, {edges} : tensor<{block_n}x!tt.ptr<i32>>, "
        f"tensor<{block_n}xi32>")
    src = e.assign(
        f"tt.load {col_ptr}, {emask}, {zero_1d} : "
        f"tensor<{block_n}x!tt.ptr<i32>>")
    rowsplat = _splat_int(e, row, str(block_n))
    k_range = _range(e, k_pad)
    k2 = _expand(e, k_range, 0, f"{k_pad}xi32", f"1x{k_pad}xi32")
    rows = _expand(e, emask, 1, f"{block_n}xi1", f"{block_n}x1xi1")
    rows_b = _broadcast(e, rows, f"{block_n}x1xi1", f"{block_n}x{k_pad}xi1")

    # --- gather the score-chain input tile --------------------------------
    tiles = []
    seg_index = 0
    for segment in score.segments:
        if segment.kind == "displacement":
            from_src = _gather_columns(
                e, ptr="%positions", row_index=src, width=segment.width,
                offset=segment.offset, k_pad=k_pad, block_e=block_n,
                rows_b=rows_b, k2=k2, zero_2d=zero_2d)
            from_dst = _gather_columns(
                e, ptr="%positions", row_index=rowsplat,
                width=segment.width, offset=segment.offset, k_pad=k_pad,
                block_e=block_n, rows_b=rows_b, k2=k2, zero_2d=zero_2d)
            tiles.append(e.assign(
                f"arith.subf {from_src}, {from_dst} : "
                f"tensor<{block_n}x{k_pad}xf32>"))
            continue
        row_index = {
            "src_field": src,
            "dst_field": rowsplat,
            "edge_field": edges,
        }[segment.kind]
        tiles.append(_gather_columns(
            e, ptr=f"%seg{seg_index}", row_index=row_index,
            width=segment.width, offset=segment.offset, k_pad=k_pad,
            block_e=block_n, rows_b=rows_b, k2=k2, zero_2d=zero_2d))
        seg_index += 1
    current = tiles[0]
    for tile in tiles[1:]:
        current = e.assign(
            f"arith.addf {current}, {tile} : tensor<{block_n}x{k_pad}xf32>")

    # --- the nn score chain, in registers ----------------------------------
    current, _records, in_pad = _replay_chain(
        e, score.layers, score.in_width, k_pad, current, block_n)
    # score tile is [block_n, 16] with the score in lane 0 and padded lanes
    # exactly zero, so a row sum extracts the per-edge scalar.
    score_vec = _row_sum(e, current, block_n, in_pad)
    score_vec = _where(
        e, emask, score_vec, _splat_f32(e, neg_inf, str(block_n)),
        str(block_n))

    # --- online softmax update ---------------------------------------------
    chunk_max = _reduce_1d(e, score_vec, block_n, "maximumf")
    new_m = e.assign(f"arith.maximumf %m, {chunk_max} : f32")
    diff_m = e.assign(f"arith.subf %m, {new_m} : f32")
    corr = e.assign(f"math.exp {diff_m} : f32")
    new_m_v = _splat_f32(e, new_m, str(block_n))
    diff_s = e.assign(
        f"arith.subf {score_vec}, {new_m_v} : tensor<{block_n}xf32>")
    p = e.assign(f"math.exp {diff_s} : tensor<{block_n}xf32>")
    l_scaled = e.assign(f"arith.mulf %l, {corr} : f32")
    l_new = e.assign(
        f"arith.addf {l_scaled}, {_reduce_1d(e, p, block_n, 'addf')} : f32")

    # --- value gather and weighted accumulate ------------------------------
    w_range = _range(e, w_pad)
    w2 = _expand(e, w_range, 0, f"{w_pad}xi32", f"1x{w_pad}xi32")
    rows_bv = _broadcast(e, rows, f"{block_n}x1xi1", f"{block_n}x{w_pad}xi1")
    zero_2dv = _dense(e, "0.0", f"{block_n}x{w_pad}", "f32")
    value_index = {
        "src_field": src,
        "dst_field": rowsplat,
        "edge_field": edges,
    }[spec.value_kind]
    vtile = _gather_columns(
        e, ptr="%value", row_index=value_index, width=spec.value_width,
        offset=0, k_pad=w_pad, block_e=block_n, rows_b=rows_bv, k2=w2,
        zero_2d=zero_2dv)
    pv = _mul(
        e, _broadcast_col(e, p, block_n, w_pad), vtile, f"{block_n}x{w_pad}")
    chunk_acc = _column_sum(e, pv, block_n, w_pad)
    corr_v = _splat_f32(e, corr, str(w_pad))
    acc_scaled = e.assign(f"arith.mulf %acc, {corr_v} : tensor<{w_pad}xf32>")
    acc_new = e.assign(
        f"arith.addf {acc_scaled}, {chunk_acc} : tensor<{w_pad}xf32>")
    e.emit(
        f"      scf.yield {new_m}, {l_new}, {acc_new} : "
        f"f32, f32, tensor<{w_pad}xf32>")
    e.emit("    }")

    # --- epilogue: acc / l, zero for empty rows ----------------------------
    l_pos = e.assign(f"arith.cmpf ogt, %attn#1, {zero_s} : f32")
    one_s = e.assign("arith.constant 1.0 : f32")
    l_safe = _select_scalar(e, l_pos, "%attn#1", one_s)
    result = e.assign(
        f"arith.divf %attn#2, {_splat_f32(e, l_safe, str(w_pad))} : "
        f"tensor<{w_pad}xf32>")
    zero_v = _dense(e, "0.0", str(w_pad), "f32")
    result = e.assign(
        f'"arith.select"({l_pos}, {result}, {zero_v}) : '
        f"(i1, tensor<{w_pad}xf32>, tensor<{w_pad}xf32>) -> "
        f"tensor<{w_pad}xf32>")
    c_ow = _scalar(e, spec.value_width)
    row_off = e.assign(f"arith.muli {row}, {c_ow} : i32")
    obase = e.assign(f"tt.addptr %out, {row_off} : !tt.ptr<f32>, i32")
    osplat = e.assign(
        f"tt.splat {obase} : !tt.ptr<f32> -> tensor<{w_pad}x!tt.ptr<f32>>")
    cols_out = _range(e, w_pad)
    optr = e.assign(
        f"tt.addptr {osplat}, {cols_out} : tensor<{w_pad}x!tt.ptr<f32>>, "
        f"tensor<{w_pad}xi32>")
    c_vw = _dense(e, spec.value_width, str(w_pad), "i32")
    omask = e.assign(
        f"arith.cmpi slt, {cols_out}, {c_vw} : tensor<{w_pad}xi32>")
    e.emit(
        f"    tt.store {optr}, {result}, {omask} : "
        f"tensor<{w_pad}x!tt.ptr<f32>>")
    # Persist the online-softmax row state for the fused backward.
    m_ptr = e.assign(f"tt.addptr %m_out, {row} : !tt.ptr<f32>, i32")
    e.emit(f"    tt.store {m_ptr}, %attn#0 : !tt.ptr<f32>")
    l_ptr = e.assign(f"tt.addptr %l_out, {row} : !tt.ptr<f32>, i32")
    e.emit(f"    tt.store {l_ptr}, %attn#1 : !tt.ptr<f32>")
    e.emit("    tt.return")
    e.emit("  }")
    e.emit("}")
    return header + "\n".join(e.lines) + "\n"


ENTRY_VJP = "gf_edge_nn_attention_vjp"


def emit_edge_nn_attention_vjp_ttir(spec: EdgeNNAttentionSpec) -> str:
    """Emit the fused backward for one nn-attention kernel.

    Per row (one program): replay the score chain per chunk with VJP
    records, rebuild the softmax weights from the forward's persisted
    ``(m, l)``, then

    - ``d_value[e] = w_e · d_out[row]`` scattered to the value field;
    - ``d_s[e] = w_e · (⟨d_out, v_e⟩ − ⟨d_out, out[row]⟩)`` — the softmax
      Jacobian adjoint — feeding the symbolic chain VJP (weight grads via
      tile outer products, field/position grads via masked atomics).

    No O(E) tensor exists in either direction.
    """
    score = spec.score
    # The backward replays and reverses the chain inside one chunk, roughly
    # doubling live tiles versus the forward; 128-edge chunks exceed the
    # shared-memory budget (110 KB > 101 KB), so the VJP chunks smaller.
    block_n = min(score.block_e, 64)
    k_pad = score.k_pad
    out_pad = score.out_pad
    w_pad = spec.value_pad
    e = _Emit()

    segment_params = []
    for segment in score.segments:
        if segment.kind == "displacement":
            continue
        segment_params.append(
            f"%seg{len(segment_params)}: !tt.ptr<f32> "
            "{tt.divisibility = 16 : i32}")
    if score.needs_positions:
        segment_params.append(
            "%positions: !tt.ptr<f32> {tt.divisibility = 16 : i32}")
    weight_params = []
    for index, layer in enumerate(score.layers):
        if getattr(layer, "weight_name", None) is None:
            continue
        weight_params.append(
            f"%w{index}: !tt.ptr<f32> {{tt.divisibility = 16 : i32}}")
        if layer.bias_name is not None:
            weight_params.append(
                f"%b{index}: !tt.ptr<f32> {{tt.divisibility = 16 : i32}}")
    grad_field_params = [
        f"%dseg{index}: !tt.ptr<f32> {{tt.divisibility = 16 : i32}}"
        for index in range(len(segment_params) - score.needs_positions)
    ]
    if score.needs_positions:
        grad_field_params.append(
            "%d_positions: !tt.ptr<f32> {tt.divisibility = 16 : i32}")
    grad_weight_params = []
    for index, layer in enumerate(score.layers):
        if getattr(layer, "weight_name", None) is None:
            continue
        grad_weight_params.append(
            f"%dw{index}: !tt.ptr<f32> {{tt.divisibility = 16 : i32}}")
        if layer.bias_name is not None:
            grad_weight_params.append(
                f"%db{index}: !tt.ptr<f32> {{tt.divisibility = 16 : i32}}")
    signature = ", ".join([
        "%row_ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32}",
        "%col_idx: !tt.ptr<i32> {tt.divisibility = 16 : i32}",
        *segment_params,
        *weight_params,
        "%value: !tt.ptr<f32> {tt.divisibility = 16 : i32}",
        "%fwd_out: !tt.ptr<f32> {tt.divisibility = 16 : i32}",
        "%m_buf: !tt.ptr<f32> {tt.divisibility = 16 : i32}",
        "%l_buf: !tt.ptr<f32> {tt.divisibility = 16 : i32}",
        "%d_out: !tt.ptr<f32> {tt.divisibility = 16 : i32}",
        *grad_field_params,
        *grad_weight_params,
        "%dvalue: !tt.ptr<f32> {tt.divisibility = 16 : i32}",
    ])

    header = (
        f"// tiga.launch entry={ENTRY_VJP} block_e={block_n} "
        f"num_warps={score.num_warps} abi=edge-nn-attention-vjp\n"
        "// emitted by tiga.compiler.edge_nn_attention (Python "
        "emission backend, phase 3)\n")
    e.emit("module {")
    e.emit(
        f"  tt.func public @{ENTRY_VJP}({signature}) "
        "attributes {noinline = false} {")

    # --- row state ---------------------------------------------------------
    c1 = _scalar(e, 1)
    row = e.assign("tt.get_program_id x : i32")
    start_ptr = e.assign(f"tt.addptr %row_ptr, {row} : !tt.ptr<i32>, i32")
    row_start = e.assign(f"tt.load {start_ptr} : !tt.ptr<i32>")
    end_ptr = e.assign(f"tt.addptr {start_ptr}, {c1} : !tt.ptr<i32>, i32")
    row_end = e.assign(f"tt.load {end_ptr} : !tt.ptr<i32>")
    degree = e.assign(f"arith.subi {row_end}, {row_start} : i32")
    m_ptr = e.assign(f"tt.addptr %m_buf, {row} : !tt.ptr<f32>, i32")
    m_val = e.assign(f"tt.load {m_ptr} : !tt.ptr<f32>")
    l_ptr = e.assign(f"tt.addptr %l_buf, {row} : !tt.ptr<f32>, i32")
    l_val = e.assign(f"tt.load {l_ptr} : !tt.ptr<f32>")
    one_s = e.assign("arith.constant 1.0 : f32")
    inv_l = e.assign(f"arith.divf {one_s}, {l_val} : f32")

    def load_row(ptr: str, zero_pad: str, cols: str, mask: str) -> str:
        c_ow = _scalar(e, spec.value_width)
        row_off = e.assign(f"arith.muli {row}, {c_ow} : i32")
        base = e.assign(f"tt.addptr {ptr}, {row_off} : !tt.ptr<f32>, i32")
        splat = e.assign(
            f"tt.splat {base} : !tt.ptr<f32> -> tensor<{w_pad}x!tt.ptr<f32>>")
        addr = e.assign(
            f"tt.addptr {splat}, {cols} : tensor<{w_pad}x!tt.ptr<f32>>, "
            f"tensor<{w_pad}xi32>")
        return e.assign(
            f"tt.load {addr}, {mask}, {zero_pad} : "
            f"tensor<{w_pad}x!tt.ptr<f32>>")

    w_range = _range(e, w_pad)
    c_vw = _dense(e, spec.value_width, str(w_pad), "i32")
    wmask = e.assign(
        f"arith.cmpi slt, {w_range}, {c_vw} : tensor<{w_pad}xi32>")
    zero_pad = _dense(e, "0.0", str(w_pad), "f32")
    d_out_tile = load_row("%d_out", zero_pad, w_range, wmask)
    fwd_tile = load_row("%fwd_out", zero_pad, w_range, wmask)
    row_dot = _reduce_1d(
        e, _mul(e, d_out_tile, fwd_tile, str(w_pad)), w_pad, "addf")

    c0 = _scalar(e, 0)
    c_bn = _scalar(e, block_n)
    e.emit(
        f"    scf.for %cs = {c0} to {degree} step {c_bn} : i32 {{")

    # --- chunk geometry (same as the forward) ------------------------------
    zero_1d = _dense(e, 0, str(block_n), "i32")
    zero_2d = _dense(e, "0.0", f"{block_n}x{k_pad}", "f32")
    lane = _range(e, block_n)
    off = e.assign(
        f"arith.addi {_splat_int(e, '%cs', str(block_n))}, {lane} : "
        f"tensor<{block_n}xi32>")
    emask = e.assign(
        f"arith.cmpi slt, {off}, {_splat_int(e, degree, str(block_n))} : "
        f"tensor<{block_n}xi32>")
    edges = e.assign(
        f"arith.addi {_splat_int(e, row_start, str(block_n))}, {off} : "
        f"tensor<{block_n}xi32>")
    col_base = e.assign(
        "tt.splat %col_idx : !tt.ptr<i32> -> "
        f"tensor<{block_n}x!tt.ptr<i32>>")
    col_ptr = e.assign(
        f"tt.addptr {col_base}, {edges} : tensor<{block_n}x!tt.ptr<i32>>, "
        f"tensor<{block_n}xi32>")
    src = e.assign(
        f"tt.load {col_ptr}, {emask}, {zero_1d} : "
        f"tensor<{block_n}x!tt.ptr<i32>>")
    rowsplat = _splat_int(e, row, str(block_n))
    k_range = _range(e, k_pad)
    k2 = _expand(e, k_range, 0, f"{k_pad}xi32", f"1x{k_pad}xi32")
    rows = _expand(e, emask, 1, f"{block_n}xi1", f"{block_n}x1xi1")
    rows_b = _broadcast(e, rows, f"{block_n}x1xi1", f"{block_n}x{k_pad}xi1")

    # --- replay the score chain with VJP records ---------------------------
    tiles = []
    seg_index = 0
    for segment in score.segments:
        if segment.kind == "displacement":
            from_src = _gather_columns(
                e, ptr="%positions", row_index=src, width=segment.width,
                offset=segment.offset, k_pad=k_pad, block_e=block_n,
                rows_b=rows_b, k2=k2, zero_2d=zero_2d)
            from_dst = _gather_columns(
                e, ptr="%positions", row_index=rowsplat,
                width=segment.width, offset=segment.offset, k_pad=k_pad,
                block_e=block_n, rows_b=rows_b, k2=k2, zero_2d=zero_2d)
            tiles.append(e.assign(
                f"arith.subf {from_src}, {from_dst} : "
                f"tensor<{block_n}x{k_pad}xf32>"))
            continue
        row_index = {
            "src_field": src,
            "dst_field": rowsplat,
            "edge_field": edges,
        }[segment.kind]
        tiles.append(_gather_columns(
            e, ptr=f"%seg{seg_index}", row_index=row_index,
            width=segment.width, offset=segment.offset, k_pad=k_pad,
            block_e=block_n, rows_b=rows_b, k2=k2, zero_2d=zero_2d))
        seg_index += 1
    current = tiles[0]
    for tile in tiles[1:]:
        current = e.assign(
            f"arith.addf {current}, {tile} : tensor<{block_n}x{k_pad}xf32>")
    current, records, _pad = _replay_chain(
        e, score.layers, score.in_width, k_pad, current, block_n,
        keep_records=True)

    # --- softmax weights from the persisted (m, l) -------------------------
    score_vec = _row_sum(e, current, block_n, out_pad)
    neg_inf = e.assign(f"arith.constant {NEG_INF} : f32")
    score_vec = _where(
        e, emask, score_vec, _splat_f32(e, neg_inf, str(block_n)),
        str(block_n))
    m_v = _splat_f32(e, m_val, str(block_n))
    shifted = e.assign(
        f"arith.subf {score_vec}, {m_v} : tensor<{block_n}xf32>")
    p = e.assign(f"math.exp {shifted} : tensor<{block_n}xf32>")
    w = e.assign(
        f"arith.mulf {p}, {_splat_f32(e, inv_l, str(block_n))} : "
        f"tensor<{block_n}xf32>")

    # --- value grad: d_value[e] = w_e · d_out[row] -------------------------
    w2 = _expand(e, w_range, 0, f"{w_pad}xi32", f"1x{w_pad}xi32")
    rows_bv = _broadcast(e, rows, f"{block_n}x1xi1", f"{block_n}x{w_pad}xi1")
    zero_2dv = _dense(e, "0.0", f"{block_n}x{w_pad}", "f32")
    value_index = {
        "src_field": src,
        "dst_field": rowsplat,
        "edge_field": edges,
    }[spec.value_kind]
    vtile = _gather_columns(
        e, ptr="%value", row_index=value_index, width=spec.value_width,
        offset=0, k_pad=w_pad, block_e=block_n, rows_b=rows_bv, k2=w2,
        zero_2d=zero_2dv)
    d_out_b = _broadcast_1d(e, d_out_tile, w_pad, block_n)
    d_v = _mul(e, _broadcast_col(e, w, block_n, w_pad), d_out_b,
               f"{block_n}x{w_pad}")
    _scatter_columns(
        e, ptr="%dvalue", row_index=value_index, width=spec.value_width,
        offset=0, k_pad=w_pad, block_e=block_n, rows_b=rows_bv, k2=w2,
        value=d_v)

    # --- score adjoint: d_s = w ⊙ (⟨d_out, v⟩ − ⟨d_out, out⟩) -------------
    dot_ev = _row_sum(
        e, _mul(e, d_out_b, vtile, f"{block_n}x{w_pad}"), block_n, w_pad)
    centered = e.assign(
        f"arith.subf {dot_ev}, {_splat_f32(e, row_dot, str(block_n))} : "
        f"tensor<{block_n}xf32>")
    d_s = e.assign(f"arith.mulf {w}, {centered} : tensor<{block_n}xf32>")

    # --- chain VJP from the per-edge score adjoint --------------------------
    d1 = _expand(e, d_s, 1, f"{block_n}xf32", f"{block_n}x1xf32")
    d_tile = _broadcast(
        e, d1, f"{block_n}x1xf32", f"{block_n}x{out_pad}xf32")
    d_tile = _mask_columns(e, d_tile, 1, block_n, out_pad)
    d = _reverse_chain(e, records, score.layers, block_n, d_tile)

    seg_index = 0
    for segment in score.segments:
        if segment.kind == "displacement":
            _scatter_columns(
                e, ptr="%d_positions", row_index=src, width=segment.width,
                offset=segment.offset, k_pad=k_pad, block_e=block_n,
                rows_b=rows_b, k2=k2, value=d)
            _scatter_columns(
                e, ptr="%d_positions", row_index=rowsplat,
                width=segment.width, offset=segment.offset, k_pad=k_pad,
                block_e=block_n, rows_b=rows_b, k2=k2, value=d, negate=True)
            continue
        row_index = {
            "src_field": src,
            "dst_field": rowsplat,
            "edge_field": edges,
        }[segment.kind]
        _scatter_columns(
            e, ptr=f"%dseg{seg_index}", row_index=row_index,
            width=segment.width, offset=segment.offset, k_pad=k_pad,
            block_e=block_n, rows_b=rows_b, k2=k2, value=d)
        seg_index += 1

    e.emit("    }")
    e.emit("    tt.return")
    e.emit("  }")
    e.emit("}")
    return header + "\n".join(e.lines) + "\n"


__all__ = [
    "ENTRY",
    "ENTRY_VJP",
    "EdgeNNAttentionSpec",
    "build_edge_nn_attention_spec",
    "emit_edge_nn_attention_ttir",
    "emit_edge_nn_attention_vjp_ttir",
]
