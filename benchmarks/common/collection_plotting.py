"""Operation-level roofline summaries built from registered case JSON.

Case directories remain the reproducible evidence boundary.  This module adds
the human-facing view: cases with identical semantics and different input
sizes are connected, while condition changes are separated by line style.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from benchmarks.common import docs_style
from benchmarks.common.plotting import (
    _number_marker,
    _save,
    _style,
    provider_color,
    scatter_numbered,
)

CONDITION_CONFIG_KEYS = (
    "causal", "kv_heads", "dtype", "periodic", "dimensions", "device",
)


def _condition(payload: dict, item: dict) -> tuple[tuple[str, object], ...]:
    values: list[tuple[str, object]] = []
    for key in ("topology", "locality", "cache", "index_dtype"):
        value = item.get(key)
        if value is not None:
            values.append((key, value))
    config = payload.get("config") or {}
    for key in CONDITION_CONFIG_KEYS:
        if key in config:
            values.append((key, config[key]))
    return tuple(values)


def _condition_label(condition: tuple[tuple[str, object], ...]) -> str:
    useful = [
        f"{key}={value}" for key, value in condition
        if key not in {"cache", "device"} or value not in {"hot", "cuda"}
    ]
    return ", ".join(useful) if useful else "default"


def _size_label(item: dict) -> str:
    parts = []
    nodes = item.get("nodes")
    edges = item.get("edges")
    features = item.get("features")
    if nodes is not None:
        parts.append(f"N={int(nodes):g}")
    if nodes and edges:
        parts.append(f"deg≈{float(edges) / float(nodes):g}")
    if features is not None and int(features) != 1:
        parts.append(f"F={int(features)}")
    return " · ".join(parts)


def _intensity(item: dict) -> float:
    return float(item["arithmetic_intensity_flop_per_byte"])


def plot_operation_summary(payloads: list[dict], output: Path) -> Path:
    """Render all cases into condition-faceted SVG plus a PNG fallback."""
    if not payloads:
        raise ValueError("at least one roofline payload is required")
    operations = {payload["operation"] for payload in payloads}
    if len(operations) != 1:
        raise ValueError(f"payloads span multiple operations: {sorted(operations)}")
    operation = operations.pop()
    all_results = [item for payload in payloads for item in payload["results"]]
    providers = sorted({item["provider"] for item in all_results})
    numbers = {provider: index + 1 for index, provider in enumerate(providers)}

    facets: dict[
        tuple[str, tuple[tuple[str, object], ...]],
        list[tuple[dict, dict]],
    ] = defaultdict(list)
    for payload in payloads:
        workload = payload.get("workload") or operation
        for item in payload["results"]:
            facets[(workload, _condition(payload, item))].append((payload, item))
    ordered_facets = sorted(facets.items(), key=lambda entry: entry[0])

    import textwrap
    _style()
    images = []
    for page_start in range(0, len(ordered_facets), 3):
        page_facets = ordered_facets[page_start:page_start + 3]
        fig, axes = plt.subplots(
            len(page_facets), 2, squeeze=False,
            figsize=(16.5, max(5.6, 4.5 * len(page_facets))),
            constrained_layout=True,
        )
        for row, ((workload, condition), records) in enumerate(page_facets):
            facet_payloads = list({id(payload): payload for payload, _item in records}.values())
            roofs = [payload["roof"] for payload in facet_payloads]
            dram = statistics.median(float(roof["dram_bandwidth_gbs"]) for roof in roofs)
            l2 = statistics.median(float(roof["l2_bandwidth_gbs"]) for roof in roofs)
            compute = statistics.median(float(
                roof.get("compute_gflops", roof["fp32_gflops"])) for roof in roofs)
            intensities = [_intensity(item) for _payload, item in records]
            achieved = [float(item["achieved_gflops"]) for _payload, item in records]
            xmin, xmax = min(intensities) / 1.6, max(intensities) * 1.6
            if math.isclose(xmin, xmax):
                xmin, xmax = xmin / 1.3, xmax * 1.3
            xs = np.logspace(math.log10(xmin), math.log10(xmax), 300)

            series: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
            for payload, item in records:
                series[item["provider"]].append((payload, item))

            for column, ax in enumerate(axes[row]):
                ax.plot(xs, np.minimum(compute, dram * xs), "--", color="#64748b",
                        linewidth=1.8, label="median DRAM roof")
                ax.plot(xs, np.minimum(compute, l2 * xs), "-", color="#111827",
                        linewidth=1.8, label="median L2 roof")
                if column == 0:
                    ax.axhline(compute, color="#9333ea", linestyle=":", linewidth=1.6,
                               label="median compute ceiling")
                numbered_points = []
                size_anchors: dict[tuple[float, str], float] = {}
                for provider, points in sorted(series.items()):
                    ordered = sorted(
                        points,
                        key=lambda pair: (
                            _intensity(pair[1]), pair[1].get("nodes", 0),
                            pair[1].get("edges", 0), pair[1].get("features", 0)),
                    )
                    xvalues = [_intensity(item) for _payload, item in ordered]
                    yvalues = [float(item["achieved_gflops"])
                               for _payload, item in ordered]
                    color = provider_color(provider)
                    if len(ordered) > 1:
                        ax.plot(xvalues, yvalues, "-", color=color,
                                linewidth=1.5, alpha=0.72)
                    numbered_points.extend(
                        (xvalue, yvalue, provider)
                        for xvalue, yvalue in zip(xvalues, yvalues))
                    if column == 1 and len(ordered) > 1:
                        for _payload, item in ordered:
                            label = _size_label(item)
                            if label:
                                key = (_intensity(item), label)
                                size_anchors[key] = max(
                                    size_anchors.get(key, 0.0),
                                    float(item["achieved_gflops"]))
                scatter_numbered(ax, numbered_points, numbers)
                if column == 1:
                    for point_index, ((xvalue, label), yvalue) in enumerate(
                            sorted(size_anchors.items())):
                        ax.annotate(
                            label, (xvalue, yvalue),
                            xytext=(5, 7 if point_index % 2 == 0 else -11),
                            textcoords="offset points", fontsize=8,
                            color="#475569")
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlim(xmin, xmax)
                ax.grid(True, which="both")
                ax.set_xlabel("Semantic arithmetic intensity (FLOP / common byte)")
                ax.set_ylabel("Achieved performance (GFLOP/s)")

            axes[row, 0].set_ylim(
                10 ** math.floor(math.log10(min(achieved) / 1.5)), compute * 1.5)
            condition_label = _condition_label(condition)
            axes[row, 0].set_title(
                textwrap.fill(condition_label, 65) + "\nFull roofline")
            axes[row, 0].legend(fontsize=8, loc="upper left", framealpha=0.92)
            axes[row, 1].set_ylim(min(achieved) / 1.35, max(achieved) * 1.35)
            axes[row, 1].set_title(
                textwrap.fill(condition_label, 65) + "\nInput-size trend")

            provider_handles = []
            for provider in sorted(series):
                provider_handles.append(Line2D(
                    [0], [0], color=provider_color(provider),
                    linestyle="-", linewidth=1.5,
                    marker=_number_marker(numbers[provider]), markersize=9,
                    label=f"{numbers[provider]} = {provider}",
                ))
            axes[row, 1].legend(
                handles=provider_handles, fontsize=8, loc="best", framealpha=0.93,
                title="stable provider marker/color; lines vary input size")

        fig.suptitle(
            f"Tiga {operation.replace('_', ' ')} · registered-case summary",
            fontsize=15, fontweight="bold")
        output.mkdir(parents=True, exist_ok=True)
        stem = "summary" if page_start == 0 else f"summary-{page_start // 3 + 1}"
        path = output / f"{stem}.png"
        _save(fig, path.with_suffix(""))
        images.append(f"![Conditions {page_start + 1}–{page_start + len(page_facets)}]({stem}.svg)")
    (output / "SUMMARY.md").write_text(
        f"# {operation.replace('_', ' ').title()} summary\n\n"
        "Cases with the same semantics and different input sizes are connected. "
        "Condition changes use separate panels; numeric markers identify "
        "providers without changing measured coordinates.\n\n"
        "All conditions are retained across pages; each page shows at most three.\n\n" + "\n\n".join(images) + "\n",
        encoding="utf-8",
    )
    return output / "summary.png"


def load_payloads(paths: list[Path]) -> list[dict]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def plot_manifest_dashboard(manifest: dict, output: Path) -> Path:
    """Render the roofline manifest as a human-readable coverage dashboard."""
    operations = manifest["operations"]
    names = list(operations)
    counts = [len(operations[name].get("cases", [])) for name in names]
    colors = [
        "#059669" if "performance-ready" in operations[name].get("status", "")
        else "#2563eb" if operations[name].get("cases") else "#94a3b8"
        for name in names
    ]
    _style()
    fig, ax = plt.subplots(
        figsize=(12, max(6, 0.48 * len(names))), constrained_layout=True)
    positions = np.arange(len(names))
    bars = ax.barh(positions, counts, color=colors, edgecolor="white")
    ax.set_yticks(positions, [name.replace("_", " ") for name in names])
    ax.invert_yaxis()
    ax.set_xlabel("Registered reproducible cases")
    ax.set_title("Tiga benchmark evidence coverage")
    ax.grid(True, axis="x")
    for bar, name, count in zip(bars, names, counts):
        ax.text(
            max(0.05, count + 0.05), bar.get_y() + bar.get_height() / 2,
            f"{count} · {operations[name].get('status', 'unspecified')}",
            va="center", fontsize=7.5)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "dashboard.png"
    _save(fig, path.with_suffix(""))
    lines = [
        "# Tiga benchmark dashboard", "",
        "![Registered evidence coverage](dashboard.svg)", "",
        ("Operation summaries connect only cases with matching semantics; each "
         "case directory remains the source-of-truth evidence boundary."), "",
    ]
    for name in names:
        if len(operations[name].get("cases", [])) >= 2:
            lines.append(f"- [{name} summary]({name}/SUMMARY.md)")
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _matches_filters(item: dict, filters: dict) -> bool:
    return all(item.get(key) == value for key, value in filters.items())


def _short_provider(provider: str) -> str:
    replacements = (
        ("tiga.", "Tiga · "),
        ("torch.", "PyTorch · "),
        ("pyg.", "PyG · "),
        ("triton.", "Triton · "),
        ("handwritten.", "Handwritten · "),
        ("flash_sparse_attn.", "FSA · "),
        ("fla.", "FLA · "),
        ("scipy.", "SciPy · "),
    )
    for prefix, label in replacements:
        if provider.startswith(prefix):
            provider = label + provider[len(prefix):]
            break
    return provider.replace("_", " ")


def _report_method_color(provider: str, *, primary: bool = False) -> str:
    """House-palette series color for the docs-facing report charts.

    The Tiga primary path is the indigo hero series, further
    Tiga alternatives are soft indigo, and every matched peer is
    muted slate; provider identity is carried by the y-tick labels.
    """
    return provider_color(provider)


def _matched_panel_records(
    manifest: dict, root: Path, key: str,
) -> list[tuple[dict, dict[str, dict], float]]:
    panels = manifest.get(key, [])
    if not panels:
        raise ValueError(f"manifest has no {key}")
    records = []
    for panel in panels:
        path = (root / panel["operation"] / panel["case"]
                / panel.get("artifact", "roofline.json"))
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        selected = {
            item["provider"]: item for item in payload["results"]
            if item["provider"] in panel["providers"]
            and _matches_filters(item, panel.get("filters", {}))
        }
        missing = set(panel["providers"]) - set(selected)
        if missing:
            raise ValueError(
                f"{panel['title']} is missing providers {sorted(missing)}")
        baseline = panel["baseline"]
        if baseline not in selected:
            raise ValueError(f"{panel['title']} baseline {baseline!r} is absent")
        records.append(
            (panel, selected, float(selected[baseline]["milliseconds"])))
    return records


def _showcase_badge(provider: str, primary: bool) -> str:
    if primary:
        return "GF"
    if provider.startswith("tiga."):
        return "AUTO"
    if provider.startswith("torch.compile"):
        return "TC"
    if provider.startswith("torch"):
        return "PT"
    if provider.startswith("pyg"):
        return "PyG"
    if provider.startswith(("triton", "handwritten")):
        return "TR"
    if provider.startswith("warp"):
        return "WP"
    if provider.startswith("flash_sparse_attn."):
        return "FSA"
    return "PEER"


def plot_release_showcase(manifest: dict, root: Path, output: Path) -> Path:
    """Render a wide release-style benchmark overview for the README.

    This is a presentation layer over exact registered buckets, not a composite
    score. Every mini-chart normalizes only to its predeclared matched baseline.
    The portrait report remains the exhaustive view.
    """
    records = _matched_panel_records(manifest, root, "showcase_panels")
    if len(records) != 6:
        raise ValueError("the README showcase requires exactly six panels")

    docs_style.apply()
    fig = plt.figure(figsize=(18.0, 10.6))
    grid = fig.add_gridspec(
        2, 3, left=0.055, right=0.965, bottom=0.12, top=0.705,
        wspace=0.34, hspace=0.56,
    )
    fig.text(
        0.045, 0.94, "Compiler Performance Evaluation",
        ha="left", va="top", fontsize=31, fontweight="normal",
        color=docs_style.TEXT,
    )
    fig.text(
        0.045, 0.875,
        "6 matched workloads · fixed semantics and baselines · median latency · higher is better",
        ha="left", va="top", fontsize=15.5, color=docs_style.TEXT,
    )
    fig.add_artist(Line2D(
        [0.045, 0.955], [0.825, 0.825], transform=fig.transFigure,
        color=docs_style.SLATE, alpha=0.55, linewidth=2.0,
    ))
    fig.legend(
        handles=(
            Patch(facecolor=docs_style.GF, label="Tiga compiled path"),
            Patch(facecolor=docs_style.GF_SOFT,
                  label="Tiga auto alternative"),
            Patch(facecolor="#c3ccd9", label="Matched peer"),
        ),
        loc="upper left", bbox_to_anchor=(0.043, 0.808), ncol=3,
        frameon=False, fontsize=13.5, handlelength=0.8, handleheight=0.8,
        columnspacing=2.0, handletextpad=0.45,
    )
    fig.text(
        0.925, 0.92, "G›", ha="center", va="center", color="white",
        fontsize=20, fontweight="bold",
        bbox={"boxstyle": "round,pad=0.42,rounding_size=0.2",
              "facecolor": "#0b0d10", "edgecolor": "none"},
    )

    peer_colors = ("#c3ccd9", "#d9dfe8", "#e9edf2")
    for index, (panel, selected, baseline_ms) in enumerate(records):
        ax = fig.add_subplot(grid[index // 3, index % 3])
        providers = panel["providers"]
        relative = np.asarray([
            baseline_ms / float(selected[name]["milliseconds"])
            for name in providers
        ])
        colors = [docs_style.GF] + [
            docs_style.GF_SOFT if name.startswith("tiga.")
            else peer_colors[min(peer_index, len(peer_colors) - 1)]
            for peer_index, name in enumerate(providers[1:])
        ]
        positions = np.arange(len(providers))
        bars = ax.bar(
            positions, relative, width=0.74, color=colors,
            edgecolor="none", zorder=3,
        )
        ceiling = max(1.18, float(relative.max()) * 1.22)
        ax.set_ylim(0, ceiling)
        ax.set_xlim(-0.55, len(providers) - 0.45)
        ax.axhline(0, color=docs_style.SLATE, linewidth=1.2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for item_index, (bar, provider, ratio) in enumerate(
            zip(bars, providers, relative, strict=True)
        ):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ceiling * 0.025,
                f"{ratio:.2f}×", ha="center", va="bottom",
                fontsize=15.5,
                fontweight="bold" if item_index == 0 else "normal",
                color=docs_style.TEXT,
            )
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(bar.get_height() * 0.72, ceiling * 0.34),
                _showcase_badge(provider, item_index == 0),
                ha="center", va="center", fontsize=9.5, fontweight="bold",
                color="white" if item_index < 2 else docs_style.TEXT,
                bbox={
                    "boxstyle": "round,pad=0.34,rounding_size=0.18",
                    "facecolor": "#0b0d10" if item_index < 2 else "none",
                    "edgecolor": "none" if item_index < 2 else docs_style.SLATE,
                    "linewidth": 0.8,
                },
            )
        ax.text(
            0.5, -0.13, panel["title"], transform=ax.transAxes,
            ha="center", va="center", fontsize=12.5, fontweight="bold",
            color=docs_style.TEXT,
            bbox={"boxstyle": "round,pad=0.38,rounding_size=0.9",
                  "facecolor": "none", "edgecolor": docs_style.SLATE,
                  "linewidth": 1.1},
        )
        detail = panel.get("detail")
        if detail:
            ax.text(
                0.5, -0.25, detail, transform=ax.transAxes,
                ha="center", va="top", fontsize=8.2, color=docs_style.TEXT,
            )

    fig.text(
        0.955, 0.025,
        "No aggregate score · each group uses its own declared 1.00× baseline",
        ha="right", va="bottom", fontsize=9.5, color=docs_style.SLATE,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    _save(fig, output.with_suffix(""))
    return output.with_suffix(".png")


def plot_compiler_report(manifest: dict, root: Path, output: Path) -> Path:
    """Render an auditable, portrait all-in-one matched-case report.

    Every panel is one exact workload bucket.  Its baseline is fixed in the
    manifest, so the chart never chooses a favorable peer after reading the
    measurements.  Latency is converted to relative throughput solely to make
    otherwise incomparable units share a visual grammar.
    """
    records = _matched_panel_records(manifest, root, "report_panels")

    docs_style.apply()
    fig, axes = plt.subplots(
        len(records) + 1, 1, squeeze=False,
        figsize=(10.8, 2.45 * len(records) + 2.2),
        gridspec_kw={"height_ratios": [0.42] + [1.0] * len(records)},
        constrained_layout=True,
    )
    header = axes[0, 0]
    header.axis("off")
    device = manifest.get("report_device", "registered hardware")
    header.text(
        0.5, 0.74, "Tiga compiler performance report",
        ha="center", va="center", fontsize=18, fontweight="bold",
        color=docs_style.CARD_INK,
        transform=header.transAxes,
    )
    header.text(
        0.5, 0.22,
        f"{device} · matched semantics · median hot latency · higher is better",
        ha="center", va="center", fontsize=9.2, color=docs_style.TEXT,
        transform=header.transAxes,
    )
    for index, (panel, selected, baseline_ms) in enumerate(records):
        ax = axes[index + 1, 0]
        providers = panel["providers"]
        relative = [
            baseline_ms / float(selected[name]["milliseconds"])
            for name in providers
        ]
        positions = np.arange(len(providers))
        bars = ax.barh(
            positions, relative,
            color=[_report_method_color(name, primary=position == 0)
                   for position, name in enumerate(providers)],
            height=0.62,
        )
        ax.set_yticks(positions, [_short_provider(name) for name in providers])
        ax.invert_yaxis()
        ax.axvline(1.0, color=docs_style.SLATE, linestyle="--", linewidth=1.15)
        ax.set_xlim(0, max(1.18, max(relative) * 1.23))
        ax.set_xlabel(f"Relative throughput · {_short_provider(panel['baseline'])} = 1.00×")
        ax.set_title(panel["title"], loc="left", fontsize=11.2, pad=7)
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")
        for bar, name, ratio in zip(bars, providers, relative):
            latency = float(selected[name]["milliseconds"])
            ax.text(
                bar.get_width() + max(relative) * 0.018,
                bar.get_y() + bar.get_height() / 2,
                f"{ratio:.2f}×  ·  {latency:.4g} ms",
                va="center", fontsize=8.1, color=docs_style.TEXT,
            )
        detail = panel.get("detail")
        if detail:
            ax.text(
                1.0, 1.03, detail, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7.5, color=docs_style.TEXT,
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    _save(fig, output.with_suffix(""))
    return output.with_suffix(".png")


def write_interactive_compiler_report(
    manifest: dict,
    root: Path,
    output: Path,
    *,
    fallback: str = "../compiler-performance-report.svg",
) -> Path:
    """Write a Plotly report from the same manifest cases as the static SVG.

    Plotly is loaded by the browser, so regenerating documentation adds no
    Python dependency. If JavaScript or the CDN is unavailable, the committed
    SVG report remains the visible fallback.
    """
    records = _matched_panel_records(manifest, root, "report_panels")
    panels = []
    for panel, selected, baseline_ms in records:
        providers = panel["providers"]
        panels.append({
            "title": panel["title"],
            "detail": panel.get("detail", ""),
            "baseline": _short_provider(panel["baseline"]),
            "providers": [_short_provider(name) for name in providers],
            "relative": [
                baseline_ms / float(selected[name]["milliseconds"])
                for name in providers
            ],
            "milliseconds": [
                float(selected[name]["milliseconds"]) for name in providers
            ],
            "colors": [_report_method_color(name, primary=position == 0)
                       for position, name in enumerate(providers)],
        })
    payload = json.dumps(panels, ensure_ascii=True).replace("<", "\\u003c")
    document = f"""<!doctype html>
<html lang=\"en\"><head><meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>Tiga interactive compiler performance report</title>
<style>
:root{{--paper:#f8fafc;--plot:#f8fafc;--ink:#172033;--muted:#64748b;--grid:#dbe3ef;--menu:#fff;--line:#cbd5e1}}
:root[data-theme=dark]{{--paper:#101a2b;--plot:#101a2b;--ink:#edf3fc;--muted:#9cabc0;--grid:#26354b;--menu:#17243a;--line:#3a4b64}}
html,body{{height:100%;margin:0;background:var(--paper);font-family:Inter,ui-sans-serif,system-ui,sans-serif}}
#chart{{width:100%;height:100%;min-height:600px}}
#fallback{{display:none;width:100%;height:100%;align-items:flex-start;justify-content:center;background:#f8fafc}}
#fallback img{{display:block;width:100%;height:auto}}
@media(max-width:640px){{#chart{{min-height:700px}}}}
</style></head><body>
<div id=\"chart\" role=\"img\" aria-label=\"Interactive Tiga compiler performance report\"></div>
<div id=\"fallback\"><img src=\"{fallback}\" alt=\"Static Tiga compiler performance report\"></div>
<noscript><style>#chart{{display:none}}#fallback{{display:flex}}</style></noscript>
<script id=\"gf-report-data\" type=\"application/json\">{payload}</script>
<script>
function notify(type,value){{if(window.parent!==window)window.parent.postMessage({{source:'tiga-report',type:type,value:value}},window.location.origin)}}
function notifyHeight(){{notify('height',Math.max(document.documentElement.scrollHeight,window.innerHeight))}}
function showFallback(){{document.getElementById('chart').style.display='none';document.getElementById('fallback').style.display='flex';notify('fallback',true);notifyHeight()}}
</script>
<script src=\"https://cdn.plot.ly/plotly-3.1.0.min.js\" onerror=\"showFallback()\"></script>
<script>
if(!window.Plotly){{showFallback()}}else{{
const panels=JSON.parse(document.getElementById('gf-report-data').textContent);
let activePanel=0;
let dark=false;
const traces=panels.map((panel,index)=>({{type:'bar',orientation:'h',visible:index===0,
 y:panel.providers,x:panel.relative,marker:{{color:panel.colors,line:{{color:'#fff',width:1}}}},
 text:panel.relative.map(value=>value.toFixed(3)+'×'),textposition:'outside',cliponaxis:false,
 customdata:panel.milliseconds.map(value=>[value]),
 hovertemplate:'<b>%{{y}}</b><br>Relative throughput: %{{x:.4f}}×<br>Median latency: %{{customdata[0]:.5g}} ms<extra></extra>'}}));
function colors(){{return dark?{{paper:'#101a2b',ink:'#edf3fc',muted:'#9cabc0',grid:'#26354b',menu:'#17243a',line:'#3a4b64'}}:{{paper:'#f8fafc',ink:'#172033',muted:'#64748b',grid:'#dbe3ef',menu:'#fff',line:'#cbd5e1'}}}}
function layoutFor(index){{const panel=panels[index];const c=colors();const compact=window.innerWidth<640;const ceiling=Math.max(1.16,...panel.relative)*1.17;return {{
 title:{{text:'<b>'+panel.title+'</b><br><span style=\"font-size:12px;color:'+c.muted+'\">'+panel.detail+'</span>',x:.03,xanchor:'left',font:{{size:compact?18:23,color:c.ink}}}},
 margin:{{l:compact?32:180,r:compact?36:90,t:compact?142:150,b:compact?92:75}},paper_bgcolor:c.paper,plot_bgcolor:c.paper,font:{{color:c.ink}},
 xaxis:{{title:'Relative throughput · '+panel.baseline+' = 1.00×',range:[0,ceiling],gridcolor:c.grid,zeroline:false,titlefont:{{size:compact?10:12}}}},
 yaxis:{{autorange:'reversed',automargin:true,tickfont:{{size:compact?10:13}}}},
 shapes:[{{type:'line',x0:1,x1:1,y0:-.6,y1:panel.providers.length-.4,line:{{color:c.muted,width:1.4,dash:'dash'}}}}]}}}}
const buttons=panels.map((panel,index)=>({{label:panel.title,method:'update',args:[{{visible:panels.map((_,candidate)=>candidate===index)}}],execute:true}}));
const base={{autosize:true,showlegend:false,updatemenus:[{{type:'dropdown',active:0,x:.03,y:1.13,xanchor:'left',yanchor:'top',buttons:buttons}}],
 annotations:[{{text:'Fixed semantics and predeclared baseline · no aggregate score',xref:'paper',yref:'paper',x:0,y:-.18,showarrow:false,xanchor:'left'}}]}};
function renderLayout(){{const c=colors();const layout=Object.assign({{}},base,layoutFor(activePanel));layout.updatemenus=base.updatemenus.map(menu=>Object.assign({{}},menu,{{active:activePanel,bgcolor:c.menu,bordercolor:c.line,font:{{size:12,color:c.ink}}}}));layout.annotations=base.annotations.map(item=>Object.assign({{}},item,{{font:{{size:11,color:c.muted}}}}));return layout}}
Plotly.newPlot('chart',traces,renderLayout(),{{responsive:true,displaylogo:false,modeBarButtonsToRemove:['lasso2d','select2d']}}).then(()=>{{
 const chart=document.getElementById('chart');chart.on('plotly_buttonclicked',event=>{{activePanel=panels.findIndex(panel=>panel.title===event.button.label);Plotly.relayout(chart,renderLayout());setTimeout(notifyHeight,0)}});notifyHeight();
}}).catch(showFallback);
window.addEventListener('message',event=>{{if(event.origin!==window.location.origin||event.data?.source!=='tiga-docs'||event.data.type!=='theme')return;dark=event.data.value==='dark';document.documentElement.dataset.theme=dark?'dark':'light';Plotly.relayout('chart',renderLayout()).then(notifyHeight)}});
window.addEventListener('resize',()=>{{Plotly.relayout('chart',renderLayout()).then(notifyHeight)}});
}}
</script></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return output
