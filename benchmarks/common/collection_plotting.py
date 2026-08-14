"""Operation-level roofline summaries built from registered case JSON.

Case directories remain the reproducible evidence boundary.  This module adds
the human-facing view: cases with identical semantics and different input
sizes are connected, while condition changes are separated by line style.
"""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from benchmarks.common.plotting import (
    _number_marker, _save, _style, provider_color, scatter_numbered,
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
    """Render all cases for one operation into a condition-faceted PNG."""
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

    _style()
    fig, axes = plt.subplots(
        len(ordered_facets), 2, squeeze=False,
        figsize=(16.5, max(5.6, 4.5 * len(ordered_facets))),
        constrained_layout=True,
    )
    for row, ((workload, condition), records) in enumerate(ordered_facets):
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
                        textcoords="offset points", fontsize=6.5,
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
            f"{workload} · {condition_label} · full roofline")
        axes[row, 0].legend(fontsize=7, loc="upper left", framealpha=0.92)
        axes[row, 1].set_ylim(min(achieved) / 1.35, max(achieved) * 1.35)
        axes[row, 1].set_title(
            f"{workload} · {condition_label} · input-size trend")

        provider_handles = []
        for provider in sorted(series):
            provider_handles.append(Line2D(
                [0], [0], color=provider_color(provider),
                linestyle="-", linewidth=1.5,
                marker=_number_marker(numbers[provider]), markersize=9,
                label=f"{numbers[provider]} = {provider}",
            ))
        axes[row, 1].legend(
            handles=provider_handles, fontsize=7, loc="best", framealpha=0.93,
            title="stable provider marker/color; lines vary input size")

    fig.suptitle(
        f"GraphForge {operation.replace('_', ' ')} · registered-case summary",
        fontsize=15, fontweight="bold")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "summary.png"
    _save(fig, path.with_suffix(""))
    (output / "SUMMARY.md").write_text(
        f"# {operation.replace('_', ' ').title()} summary\n\n"
        "Cases with the same semantics and different input sizes are connected. "
        "Condition changes use separate line styles; numeric markers identify "
        "providers without changing measured coordinates.\n\n"
        "![Registered-case summary](summary.png)\n",
        encoding="utf-8",
    )
    return path


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
    ax.set_title("GraphForge benchmark evidence coverage")
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
        "# GraphForge benchmark dashboard", "",
        "![Registered evidence coverage](dashboard.png)", "",
        "Operation summaries connect only cases with matching semantics; each "
        "case directory remains the source-of-truth evidence boundary.", "",
    ]
    for name in names:
        if len(operations[name].get("cases", [])) >= 2:
            lines.append(f"- [{name} summary]({name}/SUMMARY.md)")
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
