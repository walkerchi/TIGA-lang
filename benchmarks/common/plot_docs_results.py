"""Generate the benchmark-results documentation figures.

Every figure on docs/benchmark-results.md is produced here from the measured
artifacts under ``output/roofline/`` and ``output/distributed/`` (the two
hand-transcribed summary figures cite the registered results table as their
source). One shared style keeps the whole page visually unified; figures are
written to ``docs/assets/results/`` as SVG with a PNG fallback. The data-
generated "The full matrix" section is rebuilt at the same time into
``docs/includes/`` by :mod:`benchmarks.common.full_matrix`.

Run from the repository root:

    PYTHONPATH="$PWD/python" python -m benchmarks.common.plot_docs_results
"""

from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.common import docs_style, full_matrix
from benchmarks.common.docs_style import BASE, GF, GF_SOFT, SLATE, TEXT

ROOT = Path("output")
OUT = Path("docs/assets/results")

# --- unified style (benchmarks/common/docs_style.py) ------------------------
INK = TEXT             # dual-mode-safe ink for labels and value annotations
MUTED = TEXT
ORACLE = GF_SOFT       # handwritten oracle kernel: same class as Tiga
PEER = SLATE           # vendor/library peer

PROVIDER_COLORS = docs_style.PROVIDER_COLORS

_style = docs_style.apply
_finish_axes = docs_style.finish_axes
_headline = docs_style.headline
_save = docs_style.save
_label_bars = docs_style.label_bars


def _inject_tooltips(svg_path: Path, tooltips: dict[str, str]) -> None:
    """Attach native ``<title>`` hover tooltips to gid-tagged SVG groups.

    matplotlib writes each gid-tagged artist as ``<g id="...">``; a ``<title>``
    child makes browsers show a native tooltip when the SVG is embedded with
    ``<object>`` (an ``<img>`` embed rasterizes interactivity away).
    """
    text = svg_path.read_text(encoding="utf-8")
    for gid, tooltip in tooltips.items():
        tag = f'<g id="{gid}"'
        start = text.find(tag)
        if start < 0:
            raise ValueError(f"{svg_path}: gid {gid!r} not found")
        close = text.find(">", start)
        text = (text[:close + 1] + f"<title>{escape(tooltip)}</title>"
                + text[close + 1:])
    svg_path.write_text(text, encoding="utf-8")


# --- figures ---------------------------------------------------------------


def figure_transformations() -> None:
    """Registered high-water speedup per compiler transformation.

    Source: the registered "Compiler transformations" table in
    docs/benchmark-results.md (hand-transcribed; the table is canonical).
    """
    rows = [
        ("Periodic radius rebuild", 6.064, 3.861, "range"),
        ("Generated radius build + consume", 4.425, 4.397, None),
        ("Fixed vector CSR traversal", 4.322, 4.245, None),
        ("Bounded-ragged vector traversal", 4.213, 4.153, None),
        ("Fixed-iteration PageRank", 2.337, 2.312, None),
    ]
    names = [row[0] for row in rows]
    values = np.array([row[1] for row in rows])
    ci_low = np.array([row[2] for row in rows])
    # One-sided whisker down to the CI low (range row: down to the
    # registered-range bottom) so the bar end stays the measured value.
    xerr = np.zeros((2, len(rows)))
    xerr[0] = values - ci_low

    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    fig.subplots_adjust(top=0.78)
    bars = ax.barh(
        np.arange(len(rows)), values, height=0.62, color=GF, zorder=3,
        xerr=xerr,
        error_kw={"ecolor": SLATE, "elinewidth": 1.1, "capsize": 3})
    ax.invert_yaxis()
    ax.set_yticks(np.arange(len(rows)), names)
    ax.axvline(1.0, color=SLATE, linestyle="--", linewidth=1.0)
    ax.set_xlim(0, 7.4)
    ax.set_xlabel("speedup over matched peer (×)")
    for bar, value, low, kind in zip(bars, values, ci_low,
                                     [row[3] for row in rows]):
        label = (f"{low:.2f}–{value:.2f}×" if kind == "range"
                 else f"{value:.2f}×")
        ax.text(
            value + 0.06, bar.get_y() + bar.get_height() / 2, label,
            va="center", ha="left", fontsize=8.5, color=INK)
    _finish_axes(ax, horizontal=True)
    _headline(
        fig,
        "Compiler transformations change the execution plan",
        "registered high-water GPU case per transformation · "
        "error bar: CI low · higher is better")
    _save(fig, OUT / "transformations")


def figure_cpu_relations() -> None:
    """CPU fused relation loop vs the host sparse libraries.

    Source: output/roofline/cpu_relation roofline.json artifacts
    (regular_permuted_i32_n{16384,131072}_degree16_t16); the registered
    N=131072 gate carries CI low 5.540 against scipy.csr_matvec.
    """
    rows = [
        ("N=131k · vs torch.sparse.mm (CPU)", 2.5536 / 0.1236,
         f"{2.5536 / 0.1236:.2f}× · 0.124 vs 2.554 ms", GF),
        ("N=131k · vs scipy.csr_matvec", 6.318,
         "6.32× · 0.124 vs 0.781 ms", GF),
        ("N=16k · vs scipy.csr_matvec", 0.088379 / 0.054535,
         f"{0.088379 / 0.054535:.2f}× · 0.055 vs 0.088 ms", GF),
        ("N=16k · vs torch.sparse.mm (CPU)", 0.022950 / 0.054535,
         f"{0.022950 / 0.054535:.2f}× · 0.055 vs 0.023 ms", PEER),
    ]
    names = [row[0] for row in rows]
    values = [row[1] for row in rows]
    colors = [row[3] for row in rows]

    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    fig.subplots_adjust(top=0.74)
    bars = ax.barh(
        np.arange(len(rows)), values, height=0.62, color=colors, zorder=3)
    ax.invert_yaxis()
    ax.set_yticks(np.arange(len(rows)), names)
    ax.axvline(1.0, color=SLATE, linestyle="--", linewidth=1.0, zorder=4)
    ax.set_xlim(0, 26)
    ax.set_xlabel("speedup of the generated LLVM relation loop (×)")
    for bar, value, label in zip(bars, values, [r[2] for r in rows]):
        ax.text(
            value + 26 * 0.008, bar.get_y() + bar.get_height() / 2, label,
            va="center", ha="left", fontsize=8.5, color=INK)
    _finish_axes(ax, horizontal=True)
    _headline(
        fig,
        "CPU: fused relation loop vs the host sparse libraries",
        "weighted CSR SpMV · 16 threads · degree 16 · FP32 hot · "
        "two host baselines, never averaged")
    _save(fig, OUT / "cpu-relations")


def figure_primitive_parity() -> None:
    """Parity with mature specialized implementations.

    Source: the registered "Mature primitive parity" table in
    docs/benchmark-results.md (hand-transcribed; the table is canonical).
    """
    rows = [
        ("Exact kNN build + consume", 1.327),
        ("Tile-pruned sparse attention", 1.182),
        ("GPU visualization prep", 1.153),
        ("Causal dense attention", 1.117),
        ("Grouped-query attention", 1.084),
        ("Exact dense attention", 1.074),
        ("Linear attention", 1.037),
        ("Dense matmul", 1.005),
    ]
    names = [row[0] for row in rows]
    values = [row[1] for row in rows]

    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    fig.subplots_adjust(top=0.74)
    bars = ax.barh(
        np.arange(len(rows)), values, height=0.62, color=GF, zorder=3)
    ax.invert_yaxis()
    ax.set_yticks(np.arange(len(rows)), names)
    ax.axvline(1.0, color=SLATE, linestyle="--", linewidth=1.0,
               zorder=4)
    ax.set_xlim(0, 1.6)
    ax.set_xlabel("speedup over matched peer (×)")
    _label_bars(ax, bars)
    _finish_axes(ax, horizontal=True)
    _headline(
        fig,
        "Abstraction costs little against mature implementations",
        "registered parity cases · ≥1.0 means Tiga is not slower")
    _save(fig, OUT / "primitive-parity")


def _case_json(
    operation: str, case_hint: str, filename: str = "roofline.json"
) -> dict:
    for path in sorted((ROOT / "roofline" / operation).glob("*/")):
        if case_hint in path.name and (path / filename).is_file():
            return json.loads((path / filename).read_text())
    raise FileNotFoundError(f"{operation}/{case_hint}/{filename}")


def figure_sparse_relations() -> None:
    """Registered sparse-relation speedups vs torch.sparse.mm.

    Source: the registered "Sparse relations and reducers" table in
    docs/benchmark-results.md (hand-transcribed; the table is canonical).
    The local artifact subset does not contain every registered row, so the
    figure transcribes the registered values instead of re-deriving them.
    """
    rows = [
        ("Regular random · F16", 4.322, None),
        ("Irregular 0–32 · F16", 4.213, None),
        ("Regular random · F64", 1.966, None),
        ("Regular local · F1", 1.67, None),
        ("Irregular 0–32 · F64", 1.595, None),
        ("Power-law slice · F1", 1.525, 1.353),
        ("Exponential slice · F1 (cold)", 1.439, None),
        ("Log-normal slice · F1", 1.196, None),
        ("Scalar degree 4 · F1", 1.0 / 0.824, None),
    ]
    # Per-row hover definitions: every row is the same weighted in-edge sum;
    # only the graph slice, feature width and cache state change.  Descriptions
    # follow benchmarks/sparse_compute/cases.py.
    slice_notes = {
        "Regular random · F16":
            "fixed degree 16, neighbors uniform at random, 16 feature "
            "channels — no degree skew; the tiling-friendly slice.",
        "Irregular 0–32 · F16":
            "degrees uniform in 0–32 (mean 16), random neighbors, 16 feature "
            "channels — bounded skew stresses row-length variance.",
        "Regular random · F64":
            "fixed degree 16, random neighbors, 64 feature channels — wider "
            "rows, more compute per byte.",
        "Regular local · F1":
            "fixed degree 16, neighbors are the next 16 consecutive nodes "
            "(perfect locality), scalar features — the cache-friendliest "
            "slice.",
        "Irregular 0–32 · F64":
            "degrees uniform in 0–32, random neighbors, 64 feature channels "
            "— skewed row lengths with wide features.",
        "Power-law slice · F1":
            "social-network tail: 90% of rows degree 8, 9% degree 64, 1% "
            "degree 256, scalar features — hub rows force chunked "
            "scheduling; the label spans the 95% CI.",
        "Exponential slice · F1 (cold)":
            "degrees drawn from an exponential law rescaled to mean 16 "
            "(cap 256), scalar features, cold cache — the "
            "DRAM-bandwidth-bound slice.",
        "Log-normal slice · F1":
            "degrees log-normal (sigma 1.25) rescaled to mean 16 (cap 256), "
            "scalar features — a continuous heavy tail.",
        "Scalar degree 4 · F1":
            "fixed degree 4, scalar features — tiny rows where launch and "
            "scheduling overhead dominates.",
    }
    names = [row[0] for row in rows]
    values = np.array([row[1] for row in rows])

    fig, ax = plt.subplots(figsize=(7.0, 3.1))
    fig.subplots_adjust(top=0.78)
    bars = ax.barh(np.arange(len(rows)), values, height=0.62, color=GF,
                   zorder=3)
    ax.invert_yaxis()
    ax.set_yticks(np.arange(len(rows)), names)
    for index, bar in enumerate(bars):
        bar.set_gid(f"slice-{index}-bar")
    for index, label in enumerate(ax.get_yticklabels()):
        label.set_gid(f"slice-{index}-label")
    ax.axvline(1.0, color=SLATE, linestyle="--", linewidth=1.0)
    ax.set_xlim(0, 5.0)
    ax.set_xlabel("speedup over matched peer (×)")
    for bar, value, low in zip(bars, values, [row[2] for row in rows]):
        label = (f"{low:.2f}–{value:.2f}×" if low is not None
                 else f"{value:.2f}×")
        ax.text(
            value + 0.04, bar.get_y() + bar.get_height() / 2, label,
            va="center", ha="left", fontsize=8.5, color=INK)
    _finish_axes(ax, horizontal=True)
    _headline(
        fig,
        "Sparse relations: compiler tiles beat the sparse library",
        "131,072 rows · degree 16 unless noted · vs torch.sparse.mm, "
        "hot cache unless noted")
    _save(fig, OUT / "sparse-relations")
    common = ("Every row computes the same weighted in-edge sum printed "
              "above the chart; only the graph slice, feature width and "
              "cache state change — ")
    tooltips = {}
    for index, (name, _value, _low) in enumerate(rows):
        tooltips[f"slice-{index}-bar"] = common + slice_notes[name]
        tooltips[f"slice-{index}-label"] = common + slice_notes[name]
    _inject_tooltips(OUT / "sparse-relations.svg", tooltips)


def _ratio_figure(
    stem: Path,
    title: str,
    subtitle: str,
    rows: list[tuple[str, float, str, str]],
    *,
    x_max: float,
    x_label: str = "speedup over matched peer (×)",
) -> None:
    """One horizontal speedup-ratio bar chart in the unified page style.

    ``rows`` are ``(name, ratio, end-label, color)``; a dashed 1.0 baseline
    marks parity. All comparison figures on the results page share this
    geometry so they read identically at a glance.
    """
    names = [row[0] for row in rows]
    values = [row[1] for row in rows]
    colors = [row[3] for row in rows]
    fig, ax = plt.subplots(figsize=(7.0, max(2.2, 0.52 * len(rows) + 1.6)))
    fig.subplots_adjust(top=0.74)
    bars = ax.barh(
        np.arange(len(rows)), values, height=0.6, color=colors, zorder=3)
    ax.invert_yaxis()
    ax.set_yticks(np.arange(len(rows)), names)
    ax.axvline(1.0, color=SLATE, linestyle="--", linewidth=1.0, zorder=4)
    ax.set_xlim(0, x_max)
    ax.set_xlabel(x_label)
    for bar, value, _name, label, _color in zip(bars, values, names,
                                                [r[2] for r in rows], colors):
        ax.text(
            value + x_max * 0.008, bar.get_y() + bar.get_height() / 2,
            label, va="center", ha="left", fontsize=8.5, color=INK)
    _finish_axes(ax, horizontal=True)
    _headline(fig, title, subtitle)
    _save(fig, stem)


def figure_edge_nn() -> None:
    case = json.loads(
        (ROOT / "roofline" / "radius_edge_mlp" / "n262144" / "results.json")
        .read_text())
    latency = {
        item["provider"]: item["milliseconds"] for item in case["results"]}
    # The user-facing baselines are the external tools: PyTorch eager and
    # NVIDIA Warp. Tiga's own eager fallback is an internal dispatch
    # detail and is not a baseline on the public page.
    torch = latency["torch-eager"]
    warp = latency["warp-fused"]

    _ratio_figure(
        OUT / "edge-nn-message-passing",
        "Edge-local MLP on a radius relation: fused tile vs the field",
        "262k particles · 8.1M edges · MLP 11→16→8, FP32, degree ≈ 31 · "
        "compiled = handwritten oracle within 1.01×",
        [
            ("Tiga compiled tile",
             torch / latency["tiga-compiled-tile"],
             f"{torch / latency['tiga-compiled-tile']:.2f}× · "
             f"{latency['tiga-compiled-tile']:.3f} ms", GF),
            ("handwritten Triton oracle",
             torch / latency["triton-fused-tile"],
             f"{torch / latency['triton-fused-tile']:.2f}× · "
             f"{latency['triton-fused-tile']:.3f} ms", ORACLE),
            ("NVIDIA Warp (baseline)",
             torch / warp,
             f"{torch / warp:.2f}× · {warp:.2f} ms", PEER),
            ("PyTorch eager (baseline)",
             1.0, f"1.00× · {torch:.2f} ms", BASE),
        ],
        x_max=14,
        x_label="speedup over PyTorch eager (×)",
    )

    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    fig.subplots_adjust(top=0.70)
    scales = []
    for name, directory in (("16k · 0.5M edges", "quick"),
                            ("131k · 4.0M edges", "default"),
                            ("262k · 8.1M edges", "n262144")):
        payload = json.loads(
            (ROOT / "roofline" / "radius_edge_mlp" / directory
             / "results.json").read_text())
        scales.append(
            (name, payload["memory"]["eager_message_activation_bytes"] / 2**30))
    rows = [(
        f"eager (PyTorch / Tiga) · {name}", gb, f"{gb:.2f} GB", BASE)
        for name, gb in scales]
    rows.append(("fused tile (Tiga + oracle) · every scale",
                 0.0, "0 at every scale", GF))
    bars = ax.barh(
        np.arange(len(rows)), [row[1] for row in rows], height=0.55,
        color=[row[3] for row in rows], zorder=3)
    ax.invert_yaxis()
    ax.set_yticks(np.arange(len(rows)), [row[0] for row in rows])
    ax.set_xlim(0, max(gb for _name, gb in scales) * 1.3)
    ax.set_xlabel("O(E) message activations (GB)")
    for bar, value, label in zip(
            bars, [row[1] for row in rows], [row[2] for row in rows]):
        ax.text(
            value + max(gb for _n, gb in scales) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            label, va="center", fontsize=8.5, color=INK)
    _finish_axes(ax, horizontal=True)
    _headline(
        fig,
        "Per-edge memory grows with E — except in the fused kernels",
        "eager materializes [E, ·] activations (gather input + hidden + "
        "message); the fused kernels materialize none")
    _save(fig, OUT / "edge-nn-memory")


def figure_edge_nn_backward() -> None:
    """Edge-NN training step: fused recompute VJP vs eager Torch autograd.

    Source: output/roofline/edge_nn_backward/cuda_n131072_degree32
    (measured; one forward+backward step, compile excluded).
    """
    payload = json.loads(
        (ROOT / "roofline" / "edge_nn_backward" / "cuda_n131072_degree32"
         / "roofline.json").read_text())
    latency = {
        item["provider"]: item["milliseconds"] for item in payload["results"]}
    memory = payload["memory"]["step_peak_bytes"]
    eager_ms = latency["torch.eager_autograd"]
    fused_ms = latency["tiga.fused_recompute_vjp"]
    eager_gb = memory["torch.eager_autograd"] / 2**30
    fused_mb = memory["tiga.fused_recompute_vjp"] / 2**20

    _ratio_figure(
        OUT / "edge-nn-backward",
        "Edge-NN training: fused recompute VJP vs eager autograd",
        "131k particles · 4.0M edges · MLP 11→16→8, FP32 · "
        "one forward+backward step",
        [
            ("Training step (fwd+bwd)",
             eager_ms / fused_ms,
             f"{eager_ms / fused_ms:.2f}× · {fused_ms:.2f} vs "
             f"{eager_ms:.2f} ms", GF),
            ("Peak step memory",
             memory["torch.eager_autograd"]
             / memory["tiga.fused_recompute_vjp"],
             f"{memory['torch.eager_autograd'] / memory['tiga.fused_recompute_vjp']:.1f}× lower · "
             f"{fused_mb:.0f} vs {eager_gb * 1024:.0f} MiB", GF),
        ],
        x_max=22,
        x_label="improvement over eager Torch autograd (×)",
    )


def figure_gat_attention() -> None:
    """GAT edge attention: fused online-softmax kernels vs eager autograd.

    Source: output/roofline/gat_attention/cuda_n131072_degree32
    (measured; one forward+backward step, compile excluded).
    """
    payload = json.loads(
        (ROOT / "roofline" / "gat_attention" / "cuda_n131072_degree32"
         / "roofline.json").read_text())
    latency = {
        item["provider"]: item["milliseconds"] for item in payload["results"]}
    memory = payload["memory"]["step_peak_bytes"]
    eager_ms = latency["torch.eager_autograd"]
    fused_ms = latency["tiga.fused_online_softmax"]
    eager_mb = memory["torch.eager_autograd"] / 2**20
    fused_mb = memory["tiga.fused_online_softmax"] / 2**20

    _ratio_figure(
        OUT / "gat-attention",
        "GAT edge attention: fused online-softmax kernels vs eager autograd",
        "131k particles · 4.0M edges · score MLP 19→16→1, value width 8, "
        "FP32 · one forward+backward step",
        [
            ("Training step (fwd+bwd)",
             eager_ms / fused_ms,
             f"{eager_ms / fused_ms:.2f}× · {fused_ms:.2f} vs "
             f"{eager_ms:.2f} ms", GF),
            ("Peak step memory",
             eager_mb / fused_mb,
             f"{eager_mb / fused_mb:.1f}× lower · "
             f"{fused_mb:.0f} vs {eager_mb:.0f} MiB", GF),
        ],
        x_max=40,
        x_label="improvement over eager Torch autograd (×)",
    )


def figure_dynamic_boundaries() -> None:
    payload = _case_json(
        "radius_distance_aggregation", "cuda_n32768_d3_degree32",
        filename="pipeline.json")

    def ms(provider: str, phase: str) -> float:
        return next(
            item["milliseconds"] for item in payload["results"]
            if item["provider"] == provider and item["phase"] == phase)

    build_gf = ms("tiga.cell_directory_build", "directory-build-only")
    build_peer = ms("tiga.materialized_radius_build", "build-only")
    consume_gf = ms("tiga.auto", "consume-only")
    consume_peer = ms("torch.sparse.mm", "consume-only")
    reuse_gf = ms("tiga.dynamic.auto", "relation-reuse")
    reuse_peer = ms("torch.sparse.precomputed_distance", "relation-reuse")
    fresh_gf = ms("tiga.dynamic.auto", "fresh-build+consume")
    fresh_peer = ms("tiga.builder+torch.sparse.mm", "fresh-build+consume")
    _ratio_figure(
        OUT / "dynamic-boundaries",
        "Dynamic radius graphs: build is separate from consume",
        "N=32,768 · D=3 · degree ≈ 32 · generated cell directory "
        "vs materialized pipeline",
        [
            ("Topology build", build_peer / build_gf,
             f"{build_peer / build_gf:.2f}× · "
             f"{build_gf:.3f} vs {build_peer:.2f} ms", GF),
            ("Consume only", consume_peer / consume_gf,
             f"{consume_peer / consume_gf:.2f}× · "
             f"{consume_gf:.3f} vs {consume_peer:.3f} ms", GF),
            ("Relation reuse", reuse_peer / reuse_gf,
             f"{reuse_peer / reuse_gf:.2f}× · "
             f"{reuse_gf:.3f} vs {reuse_peer:.3f} ms", GF),
            ("Fresh build + consume", fresh_peer / fresh_gf,
             f"{fresh_peer / fresh_gf:.2f}× · "
             f"{fresh_gf:.3f} vs {fresh_peer:.2f} ms", GF),
        ],
        x_max=11,
        x_label="speedup of the generated path over materializing (×)",
    )


def figure_attention() -> None:
    panels = [
        ("Exact dense attention · vs Flash SDPA",
         _case_json("dense_attention", "b1_h16_n4096_d64_fp16"),
         ("tiga.direct_ttir", "torch.sdpa.flash")),
        ("Linear attention · vs official FLA",
         _case_json("linear_attention", "l64_t512"),
         ("tiga.compiler_scan_contract", "fla.chunk")),
        ("Tile-pruned sparse attention · vs official FSA",
         _case_json("sparse_attention", "b1_h16_n4096"),
         ("tiga.compiler_tile_pruned", "flash_sparse_attn.official")),
    ]
    rows = []
    for name, payload, (gf_key, peer_key) in panels:
        gf_ms = next(
            item["milliseconds"] for item in payload["results"]
            if item["provider"] == gf_key)
        peer_ms = next(
            item["milliseconds"] for item in payload["results"]
            if item["provider"] == peer_key)
        rows.append((
            name, peer_ms / gf_ms,
            f"{peer_ms / gf_ms:.3f}× · {gf_ms:.3f} vs {peer_ms:.3f} ms", GF))
    _ratio_figure(
        OUT / "attention",
        "Attention: three workloads, three different peers",
        "B1/H16/N4096/D64 FP16 (linear: L64/T512 FP32) · "
        "each audited against its own matched peer, never averaged",
        rows,
        x_max=1.45,
    )


def figure_distributed_overlap() -> None:
    payload = json.loads(
        (ROOT / "distributed" / "automatic_cpu_overlap" / "results.json")
        .read_text())
    sample = payload["ranks"][0]["samples"]["tiga.auto"][0]
    t0 = sample["interior_started_ns"]
    interior = (
        (sample["interior_started_ns"] - t0) / 1e6,
        (sample["interior_finished_ns"] - t0) / 1e6)
    comm = (
        (sample["communication_started_ns"] - t0) / 1e6,
        (sample["communication_finished_ns"] - t0) / 1e6)

    fig, ax_time = plt.subplots(figsize=(7.0, 2.3))
    fig.subplots_adjust(top=0.68)
    ax_time.broken_barh(
        [(interior[0], interior[1] - interior[0])], (0.46, 0.26),
        facecolors=GF, zorder=3)
    ax_time.broken_barh(
        [(comm[0], comm[1] - comm[0])], (0.12, 0.26),
        facecolors=PEER, zorder=3)
    overlap = (max(interior[0], comm[0]), min(interior[1], comm[1]))
    ax_time.axvspan(*overlap, color=ORACLE, alpha=0.14, zorder=2)
    ax_time.text(
        interior[0] + 0.15, 0.86,
        "halo exchange hidden behind interior compute\n"
        f"median overlap {payload['median_measured_overlap_ms']:.2f} ms",
        fontsize=8.5, color=MUTED)
    ax_time.text(
        (interior[0] + interior[1]) / 2, 0.53, "interior compute",
        ha="center", fontsize=8.5, color="white", fontweight="bold")
    ax_time.text(
        (comm[0] + comm[1]) / 2, 0.19, "halo exchange",
        ha="center", fontsize=8.5, color="white", fontweight="bold")
    ax_time.set_ylim(0, 1.05)
    ax_time.set_yticks([])
    ax_time.set_xlabel("time within one iteration (ms)")
    _finish_axes(ax_time)

    _headline(
        fig,
        "Distributed message passing overlaps communication by default",
        f"stdlib transport · {payload['entities']:,} entities · "
        f"degree {payload['degree']} · F={payload['features']} · "
        f"boundary {payload['boundary_fraction']:.0%} · 2 processes · "
        f"end-to-end {payload['median_end_to_end_ms']['tiga.auto']:.2f} ms "
        f"vs serialized "
        f"{payload['median_end_to_end_ms']['tiga.serialized']:.2f} ms "
        f"({payload['speedup_vs_serialized']:.3f}×)")
    _save(fig, OUT / "distributed-overlap")


def main() -> None:
    _style()
    figure_transformations()
    figure_cpu_relations()
    figure_primitive_parity()
    figure_sparse_relations()
    figure_edge_nn()
    figure_edge_nn_backward()
    figure_gat_attention()
    figure_dynamic_boundaries()
    figure_attention()
    figure_distributed_overlap()
    full_matrix.main()


if __name__ == "__main__":
    main()
