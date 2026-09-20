"""Generate the benchmark-results documentation figures.

Every figure on docs/benchmark-results.md is produced here from the measured
artifacts under ``output/roofline/`` and ``output/distributed/`` (the two
archived summary figures use the same checked JSON inputs). One shared style keeps the whole page visually unified; figures are
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

ROOT = Path("benchmarks/evidence_snapshot/output")
SOURCES = set()
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


def _point(operation, case, candidate, baseline, **filters):
    payload = _case_json(operation, case)
    def select(provider):
        matches = [r for r in payload["results"] if r["provider"] == provider
                   and all(r.get(k) == v for k, v in filters.items())]
        if len(matches) != 1:
            raise ValueError(f"ambiguous/missing chart point: {operation}/{case}/{provider}/{filters}")
        return float(matches[0]["milliseconds"])
    candidate_ms, baseline_ms = select(candidate), select(baseline)
    if min(candidate_ms, baseline_ms) <= 0:
        raise ValueError("non-positive chart latency")
    return baseline_ms / candidate_ms, f"{candidate_ms:.4g} vs {baseline_ms:.4g} ms"


def figure_transformations() -> None:
    selections = [
        ("Radius topology build", "radius_graph_build", "cuda_n32768_d3_degree32",
         "tiga.compact_cell_directory", "tiga.materialized_cell_list_csr", {}),
        ("Local CSR · F16 · prepared", "weighted_aggregation",
         "regular_local_cuda_i64_n131072_degree16_f1_16_64",
         "tiga.prepared_auto", "torch.sparse.mm", {"features": 16, "cache": "hot"}),
        ("Ragged CSR · F16 · prepared", "weighted_aggregation",
         "irregular_random_cuda_i32_n131072_degree16_f16_64",
         "tiga.prepared_auto", "torch.sparse.mm", {"features": 16, "cache": "hot"}),
        ("PageRank · 20 iterations", "pagerank", "cuda_fp32_iter20",
         "tiga.control", "torch.sparse.mm", {"nodes": 262144, "edges": 4194304, "cache": "hot"}),
    ]
    rows = []
    for label, op, case, candidate, baseline, filters in selections:
        ratio, times = _point(op, case, candidate, baseline, **filters)
        rows.append((label, ratio, f"{ratio:.2f}× · {times}", GF))
    _ratio_figure(OUT / "transformations", "Selected compiler transformations",
                  "Fixed archived cases · matched baseline per row · not an aggregate",
                  rows, x_max=max(r[1] for r in rows) * 1.2)


def figure_cpu_relations() -> None:
    rows = []
    for n, label in ((131072, "131k"), (16384, "16k")):
        for baseline in ("scipy.csr_matvec", "torch.sparse.mm"):
            ratio, times = _point("cpu_relation", f"regular_permuted_i32_n{n}_degree16_t16",
                                  "tiga.llvm.parallel_relation", baseline)
            rows.append((f"N={label} · {baseline}", ratio, f"{ratio:.2f}× · {times}", GF))
    _ratio_figure(OUT / "cpu-relations", "CPU relation: size changes the result",
                  "FP32/i32 · degree 16 · 16 threads · both wins and losses",
                  rows, x_max=max(r[1] for r in rows) * 1.15)


def figure_primitive_parity() -> None:
    manifest = json.loads(Path("benchmarks/evidence_manifest.json").read_text())
    operations = {"knn_graph", "sparse_attention", "visualization_heatmap",
                  "dense_attention", "linear_attention", "dense_matmul_calibration"}
    rows = []
    for panel in manifest["report_panels"]:
        if panel["operation"] not in operations:
            continue
        ratio, times = _point(panel["operation"], panel["case"], panel["providers"][0],
                              panel["baseline"], **panel.get("filters", {}))
        rows.append((panel["title"], ratio, f"{ratio:.3f}× · {times}", GF))
    _ratio_figure(OUT / "primitive-parity", "Specialized primitive comparisons",
                  "Fixed manifest cases · exact peer and timing scope in the evidence table",
                  rows, x_max=max(r[1] for r in rows) * 1.2)


def _read(path):
    path = Path(path)
    raw = path.read_bytes()
    SOURCES.add(str(path.relative_to(ROOT)))
    return json.loads(raw)


def _case_json(operation: str, case_hint: str, filename="roofline.json") -> dict:
    directory = ROOT / "roofline" / operation
    exact = directory / case_hint / filename
    if exact.is_file():
        return _read(exact)
    matches = [p / filename for p in sorted(directory.iterdir())
               if case_hint in p.name and (p / filename).is_file()]
    if len(matches) != 1:
        raise ValueError(f"ambiguous/missing case: {operation}/{case_hint}/{filename}")
    return _read(matches[0])


def figure_sparse_relations() -> None:
    slices = [
        ("Regular local · F16", "regular_local_cuda_i64_n131072_degree16_f1_16_64", 16),
        ("Regular local · F64", "regular_local_cuda_i64_n131072_degree16_f1_16_64", 64),
        ("Ragged random · F16", "irregular_random_cuda_i32_n131072_degree16_f16_64", 16),
        ("Ragged random · F64", "irregular_random_cuda_i32_n131072_degree16_f16_64", 64),
        ("Power-law random · F1", "powerlaw_random_cuda_i32_n131072_degree16_f1", 1),
        ("Log-normal random · F1", "lognormal_random_cuda_i32_n131072_degree16_f1", 1),
        ("Exponential random · F1", "exponential_random_cuda_i32_n131072_degree16_f1", 1),
    ]
    rows = []
    for label, case, features in slices:
        ratio, times = _point("weighted_aggregation", case,
                             "tiga.prepared_auto", "torch.sparse.mm",
                             features=features, cache="hot")
        rows.append((label, ratio, f"{ratio:.2f}× · {times}", GF))
    _ratio_figure(OUT / "sparse-relations", "Sparse relations: one fixed candidate",
                  "N=131072 · hot · prepared_auto / cuSPARSE · F is feature width",
                  rows, x_max=max(r[1] for r in rows) * 1.12)


def figure_provider_comparison() -> None:
    case = "irregular_random_cuda_i32_n131072_degree16_f16_64"
    payload = _case_json("weighted_aggregation", case)
    selected = [r for r in payload["results"] if r.get("cache") == "hot" and r.get("features") == 16]
    providers = [r["provider"] for r in selected]
    if len(providers) != len(set(providers)):
        raise ValueError("ambiguous provider comparison")
    labels = {
        "tiga.prepared_auto": "Tiga (prepared)*",
        "tiga.auto": "Tiga (regular call)",
        "torch.sparse.mm": "Torch sparse matmul",
        "torch.compile.index_add": "Torch compiled scatter",
        "triton.csr": "Handwritten Triton",
        "pyg.message_passing": "PyG gather-scatter",
        "torch.index_add": "Torch scatter",
    }
    if set(providers) != set(labels) | {"tiga.reference"}:
        raise ValueError("unexpected providers in historical comparison")
    # The reference is a correctness oracle, not an optimized execution path;
    # keep its timing in the complete table directly below the chart instead.
    rows = sorted((r for r in selected if r["provider"] in labels),
                  key=lambda r: r["milliseconds"])
    values = [float(r["milliseconds"]) for r in rows]
    if not all(np.isfinite(v) and v > 0 for v in values):
        raise ValueError("invalid comparison latency")
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    fig.subplots_adjust(left=0.29, right=0.97, top=0.79, bottom=0.24)
    bars = ax.barh(np.arange(len(rows)), values, height=0.58,
                   color=[GF if r["provider"].startswith("tiga.") else SLATE
                          for r in rows], zorder=3)
    ax.set_yticks(np.arange(len(rows)), [labels[r["provider"]] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.20)
    ax.set_xlabel("Execution time (ms) — shorter is faster")
    for bar, value in zip(bars, values):
        ax.text(value + max(values) * 0.02,
                bar.get_y() + bar.get_height() / 2, f"{value:.3f}",
                va="center", fontsize=9, color=INK)
    _finish_axes(ax, horizontal=True)
    fig.text(0.03, 0.94, "Weighted graph aggregation: execution time",
             fontsize=11, fontweight="bold", color=INK)
    fig.text(0.03, 0.875, "Historical · RTX 5070 Ti · 131,072 nodes · 16 features · hot cache",
             fontsize=8.5, color=INK)
    fig.text(0.03, 0.075, "* Prepared: fixed input bindings; preparation is outside timing.",
             fontsize=8.5, color=INK)
    fig.text(0.03, 0.025, "Forward only. No first compilation, graph build, transfer or backward.",
             fontsize=8.5, color=INK)
    _save(fig, OUT / "spmm-provider-compare")


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
    case = _read(ROOT / "roofline" / "radius_edge_mlp" / "n262144" / "results.json")
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
        payload = _read(ROOT / "roofline" / "radius_edge_mlp" / directory
             / "results.json")
        scales.append(
            (name, payload["memory"]["eager_message_activation_bytes"] / 2**30))
    rows = [(
        f"eager (PyTorch / Tiga) · {name}", gb, f"{gb:.2f} GiB", BASE)
        for name, gb in scales]
    rows.append(("fused tile (Tiga + oracle) · every scale",
                 0.0, "0 at every scale", GF))
    bars = ax.barh(
        np.arange(len(rows)), [row[1] for row in rows], height=0.55,
        color=[row[3] for row in rows], zorder=3)
    ax.invert_yaxis()
    ax.set_yticks(np.arange(len(rows)), [row[0] for row in rows])
    ax.set_xlim(0, max(gb for _name, gb in scales) * 1.3)
    ax.set_xlabel("O(E) message activations (GiB)")
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
    payload = _read(ROOT / "roofline" / "edge_nn_backward" / "cuda_n131072_degree32"
         / "roofline.json")
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
    payload = _read(ROOT / "roofline" / "gat_attention" / "cuda_n131072_degree32"
         / "roofline.json")
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
    payload = _read(ROOT / "distributed" / "automatic_cpu_overlap" / "results.json")
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
        "one sample trace; median reported separately\n"
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
        "Two CPU processes: controlled-link overlap",
        f"stdlib transport · {payload['entities']:,} entities · "
        f"degree {payload['degree']} · F={payload['features']} · "
        f"boundary {payload['boundary_fraction']:.0%} · 2 processes · "
        f"end-to-end {payload['median_end_to_end_ms']['tiga.auto']:.2f} ms "
        f"vs serialized "
        f"{payload['median_end_to_end_ms']['tiga.serialized']:.2f} ms "
        f"({payload['speedup_vs_serialized']:.3f}×)")
    _save(fig, OUT / "distributed-overlap")


def main() -> None:
    import argparse
    global ROOT, OUT, INCLUDES, CHART_SOURCES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--includes", type=Path, default=Path("docs/includes"))
    args = parser.parse_args()
    ROOT, OUT, INCLUDES = args.root, args.out, args.includes
    OUT.mkdir(parents=True, exist_ok=True)
    CHART_SOURCES = {}
    _style()
    SOURCES.clear()
    figure_transformations()
    CHART_SOURCES["transformations"] = sorted(SOURCES)
    SOURCES.clear()
    figure_cpu_relations()
    CHART_SOURCES["cpu-relations"] = sorted(SOURCES)
    SOURCES.clear()
    figure_primitive_parity()
    CHART_SOURCES["primitive-parity"] = sorted(SOURCES)
    SOURCES.clear()
    figure_sparse_relations()
    CHART_SOURCES["sparse-relations"] = sorted(SOURCES)
    SOURCES.clear()
    figure_edge_nn()
    CHART_SOURCES["edge-nn"] = sorted(SOURCES)
    CHART_SOURCES["edge-nn-message-passing"] = sorted(SOURCES)
    CHART_SOURCES["edge-nn-memory"] = sorted(SOURCES)
    del CHART_SOURCES["edge-nn"]
    SOURCES.clear()
    figure_edge_nn_backward()
    CHART_SOURCES["edge-nn-backward"] = sorted(SOURCES)
    SOURCES.clear()
    figure_gat_attention()
    CHART_SOURCES["gat-attention"] = sorted(SOURCES)
    SOURCES.clear()
    figure_dynamic_boundaries()
    CHART_SOURCES["dynamic-boundaries"] = sorted(SOURCES)
    SOURCES.clear()
    figure_attention()
    CHART_SOURCES["attention"] = sorted(SOURCES)
    SOURCES.clear()
    figure_distributed_overlap()
    CHART_SOURCES["distributed-overlap"] = sorted(SOURCES)
    SOURCES.clear()
    figure_provider_comparison()
    CHART_SOURCES["spmm-provider-compare"] = sorted(SOURCES)
    full_matrix.main(root=ROOT / "roofline", out_dir=INCLUDES)
    import hashlib
    index = {"schema": "tiga.chart-inputs.v1", "charts": CHART_SOURCES,
             "files": [{"path": p, "sha256": hashlib.sha256((ROOT / p).read_bytes()).hexdigest()}
                       for p in sorted(set().union(*map(set, CHART_SOURCES.values())))]}
    (OUT / "chart-inputs.json").write_text(json.dumps(index, indent=2) + "\n")


if __name__ == "__main__":
    main()
