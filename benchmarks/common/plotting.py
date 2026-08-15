"""Render benchmark JSON as publication-ready PNG and SVG figures."""

from __future__ import annotations

import argparse
import colorsys
import hashlib
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

PALETTE = (
    "#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c",
    "#0891b2", "#4f46e5", "#be123c", "#65a30d",
)

PROVIDER_FAMILY_HUES = {
    "graphforge": 0.60,
    "torch": 0.01,
    "triton": 0.76,
    "handwritten": 0.82,
    "cuda": 0.08,
    "fla": 0.42,
    "flash_sparse_attn": 0.50,
    "dgl": 0.27,
    "pyg": 0.92,
    "scipy": 0.57,
}

PROVIDER_COLORS = {
    "graphforge.auto": "#2563eb",
    "graphforge.prepared_auto": "#059669",
    "graphforge.reference": "#64748b",
    "torch.sparse.mm": "#dc2626",
    "triton.csr": "#7c3aed",
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _style():
    plt.rcParams.update({
        "figure.facecolor": "#f8fafc",
        "axes.facecolor": "#ffffff",
        "axes.edgecolor": "#94a3b8",
        "axes.labelcolor": "#0f172a",
        "axes.titlecolor": "#0f172a",
        "text.color": "#0f172a",
        "xtick.color": "#334155",
        "ytick.color": "#334155",
        "grid.color": "#cbd5e1",
        "grid.alpha": 0.45,
        "font.size": 10,
        "axes.titleweight": "bold",
    })


def provider_color(provider: str) -> str:
    """Return one corpus-wide stable color for an exact provider name.

    The mapping depends only on the name, never on which other providers are
    present in a particular figure. Providers in the same ecosystem retain a
    recognizable hue while a stable name hash changes lightness/saturation.
    """
    if provider in PROVIDER_COLORS:
        return PROVIDER_COLORS[provider]
    family = provider.split(".", 1)[0].split("-", 1)[0]
    base_hue = PROVIDER_FAMILY_HUES.get(family)
    digest = hashlib.blake2b(provider.encode("utf-8"), digest_size=8).digest()
    integer = int.from_bytes(digest, "big")
    if base_hue is None:
        base_hue = (integer & 0xFFFF) / 0x10000
    hue_offset = (((integer >> 16) & 0xFF) / 255.0 - 0.5) * 0.055
    saturation = 0.62 + ((integer >> 24) & 0xFF) / 255.0 * 0.18
    value = 0.70 + ((integer >> 32) & 0xFF) / 255.0 * 0.18
    rgb = colorsys.hsv_to_rgb((base_hue + hue_offset) % 1.0, saturation, value)
    return "#" + "".join(f"{round(channel * 255):02x}" for channel in rgb)


def _colors(results):
    providers = sorted({item["provider"] for item in results})
    return {name: provider_color(name) for name in providers}


def _provider_numbers(results):
    """Stable, human-readable provider IDs for coincident plot points."""
    providers = sorted({item["provider"] for item in results})
    return {name: index + 1 for index, name in enumerate(providers)}


def _number_marker(number: int) -> str:
    # MathText markers keep the measured coordinate exact while making two
    # providers at the same semantic x visually distinguishable.
    return f"${number}$"


def scatter_numbered(
    ax,
    points: list[tuple[float, float, str]],
    numbers: dict[str, int],
    colors: dict[str, str] | None = None,
) -> None:
    """Draw provider numbers without hiding nearly coincident measurements.

    A collision group keeps one hollow anchor at the measured coordinate. Its
    provider badges are displaced only in display space and connected back to
    that anchor, so the chart remains numerically honest.
    """
    colors = colors or {provider: provider_color(provider)
                        for _x, _y, provider in points}
    groups: dict[tuple[float, float], list[tuple[float, float, str]]] = defaultdict(list)
    for x, y, provider in points:
        # Same semantic x must be exact. A 0.01 log-y bucket treats values
        # within roughly 2.3% as visually coincident at publication scale.
        key = (round(math.log10(x), 8), round(math.log10(y), 2))
        groups[key].append((x, y, provider))
    for group in groups.values():
        if len(group) == 1:
            x, y, provider = group[0]
            ax.scatter(
                x, y, s=150, marker=_number_marker(numbers[provider]),
                color=colors[provider], linewidth=1.1, zorder=7)
            continue
        anchor_x = statistics.fmean(item[0] for item in group)
        anchor_y = statistics.geometric_mean(item[1] for item in group)
        ax.scatter(
            anchor_x, anchor_y, s=48, marker="o", facecolors="none",
            edgecolors="#64748b", linewidth=1.0, zorder=5)
        radius = 17.0
        for index, (_x, _y, provider) in enumerate(
                sorted(group, key=lambda item: numbers[item[2]])):
            angle = 2 * math.pi * index / len(group) + math.pi / 2
            offset = (radius * math.cos(angle), radius * math.sin(angle))
            ax.annotate(
                str(numbers[provider]), (anchor_x, anchor_y), xytext=offset,
                textcoords="offset points", ha="center", va="center",
                fontsize=9, fontweight="bold", color=colors[provider],
                bbox={"boxstyle": "circle,pad=0.18", "facecolor": "white",
                      "edgecolor": colors[provider], "linewidth": 1.0},
                arrowprops={"arrowstyle": "-", "color": "#94a3b8",
                            "linewidth": 0.7},
                zorder=8)


def _intensity(item, roof):
    value = item.get("arithmetic_intensity_flop_per_byte")
    if value is not None:
        return float(value)
    bandwidth = (roof["l2_bandwidth_gbs"] if item["memory_roof"] == "L2"
                 else roof["dram_bandwidth_gbs"])
    return float(item["optimistic_roof_gflops"]) / bandwidth


def plot_roofline(payload: dict, output: Path):
    operation = payload.get("operation")
    if operation == "radius_distance_aggregation":
        return plot_radius_aggregation_roofline(payload, output)
    if operation == "radius_graph_build":
        return plot_radius_build_operational_roofline(payload, output)
    if operation == "dense_attention":
        return plot_dense_attention_roofline(payload, output)
    if operation == "message_passing_backward":
        return plot_backward_roofline(payload, output)
    if operation == "knn_graph":
        return plot_knn_roofline(payload, output)
    roof = payload["roof"]
    results = payload["results"]
    colors = _colors(results)
    intensities = [_intensity(item, roof) for item in results]
    fp32 = float(roof.get("compute_gflops", roof["fp32_gflops"]))
    compute_label = roof.get("compute_ceiling_label", "FP32 ceiling")
    knees = [
        fp32 / float(roof["dram_bandwidth_gbs"]),
        fp32 / float(roof["l2_bandwidth_gbs"]),
    ]
    xmin = 10 ** math.floor(math.log10(min(intensities) / 2))
    xmax = 10 ** math.ceil(math.log10(max(max(intensities) * 2, *knees)))
    xs = np.logspace(math.log10(xmin), math.log10(xmax), 400)

    fig, ax = plt.subplots(figsize=(11.5, 7.2), constrained_layout=True)
    ceilings = (
        ("DRAM roof", float(roof["dram_bandwidth_gbs"]), "--", "#64748b"),
        ("L2 roof", float(roof["l2_bandwidth_gbs"]), "-", "#111827"),
    )
    for label, bandwidth, style, color in ceilings:
        ys = np.minimum(fp32, bandwidth * xs)
        ax.plot(xs, ys, style, color=color, linewidth=2.0, label=label)
    ax.axhline(fp32, color="#9333ea", linestyle=":", linewidth=1.8,
               label=f"{compute_label} ({fp32 / 1000:.1f} TFLOP/s)")

    markers = {"hot": "o", "cold": "s"}
    for item in results:
        x = _intensity(item, roof)
        y = float(item["achieved_gflops"])
        ax.scatter(
            x, y, s=62, marker=markers.get(item["cache"], "D"),
            color=colors[item["provider"]], edgecolor="white", linewidth=0.8,
            zorder=5,
        )
        ax.annotate(
            f"F{item['features']}", (x, y), xytext=(4, 4),
            textcoords="offset points", fontsize=7, color="#475569")

    provider_handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=7,
               markerfacecolor=colors[name], markeredgecolor="white", label=name)
        for name in colors
    ]
    cache_handles = [
        Line2D([0], [0], marker=marker, linestyle="", color="#334155",
               markersize=7, label=f"{cache} working set")
        for cache, marker in markers.items()
        if any(item["cache"] == cache for item in results)
    ]
    ceiling_handles, ceiling_labels = ax.get_legend_handles_labels()
    legend = ax.legend(
        ceiling_handles + provider_handles + cache_handles,
        ceiling_labels + list(colors) + [item.get_label() for item in cache_handles],
        loc="lower right", fontsize=8, framealpha=0.92, ncol=2)
    legend.get_frame().set_edgecolor("#cbd5e1")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(xmin, xmax)
    positive_y = [float(item["achieved_gflops"]) for item in results]
    ax.set_ylim(10 ** math.floor(math.log10(min(positive_y) / 2)), fp32 * 1.7)
    ax.grid(True, which="both")
    ax.set_xlabel("Arithmetic intensity (FLOP / ideal-cache byte)")
    ax.set_ylabel("Achieved performance (GFLOP/s)")
    workload = payload.get("workload", "weighted aggregation").replace("_", " ")
    title_prefix = payload.get("title_prefix", "GraphForge")
    ax.set_title(f"{title_prefix} {workload} hierarchical roofline")
    ax.text(
        0.01, 0.99,
        f"DRAM {roof['dram_bandwidth_gbs']:.0f} GB/s  ·  "
        f"L2 {roof['l2_bandwidth_gbs']:.0f} GB/s",
        transform=ax.transAxes, va="top", fontsize=9, color="#475569")
    _save(fig, output / "roofline")


def plot_knn_roofline(payload: dict, output: Path):
    """Render exact-kNN at its true roofline coordinate plus a useful zoom.

    Exact builders share the same semantic work and byte model, so their x
    coordinates intentionally coincide. Numeric provider markers expose that
    overlap without jittering or otherwise falsifying the measurement.
    """
    roof = payload["roof"]
    results = payload["results"]
    colors = _colors(results)
    numbers = _provider_numbers(results)
    intensities = [_intensity(item, roof) for item in results]
    achieved = [float(item["achieved_gflops"]) for item in results]
    fp32 = float(roof.get("compute_gflops", roof["fp32_gflops"]))
    xmin = min(intensities) / 1.7
    xmax = max(intensities) * 1.7
    xs = np.logspace(math.log10(xmin), math.log10(xmax), 300)

    fig, axes = plt.subplots(
        1, 2, figsize=(14.2, 6.2), constrained_layout=True, sharex=True,
        gridspec_kw={"width_ratios": (1.05, 1.0)})
    ceilings = (
        ("DRAM roof", float(roof["dram_bandwidth_gbs"]), "--", "#64748b"),
        ("L2 roof", float(roof["l2_bandwidth_gbs"]), "-", "#111827"),
    )
    for ax in axes:
        for label, bandwidth, style, color in ceilings:
            ax.plot(xs, np.minimum(fp32, bandwidth * xs), style,
                    color=color, linewidth=2.0, label=label)
        scatter_numbered(
            ax,
            [(_intensity(item, roof), float(item["achieved_gflops"]),
              item["provider"]) for item in results],
            numbers, colors)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(xmin, xmax)
        ax.grid(True, which="both")
        ax.set_xlabel("Semantic arithmetic intensity (useful FLOP / common byte)")

    axes[0].axhline(
        fp32, color="#9333ea", linestyle=":", linewidth=1.8,
        label=f"FP32 ceiling ({fp32 / 1000:.1f} TFLOP/s)")
    axes[0].set_ylim(
        10 ** math.floor(math.log10(min(achieved) / 1.5)), fp32 * 1.5)
    axes[0].set_ylabel("Useful semantic throughput (GFLOP/s)")
    axes[0].set_title("Full hierarchical roofline")
    axes[0].legend(fontsize=8, loc="upper left", framealpha=0.92)

    axes[1].set_ylim(min(achieved) / 1.25, max(achieved) * 1.25)
    axes[1].set_ylabel("Useful semantic throughput (GFLOP/s)")
    axes[1].set_title("Achieved-region zoom · identical semantic x")
    ordered = sorted(results, key=lambda item: numbers[item["provider"]])
    provider_handles = [
        Line2D(
            [0], [0], linestyle="", marker=_number_marker(numbers[item["provider"]]),
            markersize=10, color=colors[item["provider"]],
            label=(f"{numbers[item['provider']]} = {item['provider']} · "
                   f"{float(item['milliseconds']):.4f} ms"),
        )
        for item in ordered
    ]
    axes[1].legend(
        handles=provider_handles, loc="lower right", fontsize=8,
        framealpha=0.94, title="numeric marker = provider")

    config = payload.get("config", {})
    detail = (
        f"N={int(config['nodes']):,}, D={config['dimensions']}, k={config['k']}"
        if {"nodes", "dimensions", "k"} <= config.keys() else ""
    )
    workload = payload.get("workload", "exact k-nearest-neighbor build")
    fig.suptitle(
        f"GraphForge {workload}\n{detail}", fontsize=14, fontweight="bold")
    _save(fig, output / "roofline")


def plot_backward_roofline(payload: dict, output: Path):
    """Keep the common semantic x exact while exposing close backward peers.

    The copy-calibrated ceilings are diagnostic proxies. Indexed read kernels
    can exceed the copy proxy because their load/store mix and persistent-cache
    behavior differ; provider-specific physical bytes stay in the JSON.
    """
    roof = payload["roof"]
    results = payload["results"]
    colors = _colors(results)
    intensity = float(results[0]["arithmetic_intensity_flop_per_byte"])
    fp32 = float(roof["fp32_gflops"])
    xs = np.logspace(math.log10(intensity / 2),
                     math.log10(intensity * 2), 300)
    values = [float(item["achieved_gflops"]) for item in results]
    markers = ("o", "s", "D", "^", "v", "P", "X")
    fig, axes = plt.subplots(
        1, 2, figsize=(13.8, 5.9), constrained_layout=True, sharex=True,
        gridspec_kw={"width_ratios": (1.05, 1.0)})
    ceilings = (
        ("DRAM roof", float(roof["dram_bandwidth_gbs"]), "--", "#64748b"),
        ("L2 roof", float(roof["l2_bandwidth_gbs"]), "-", "#111827"),
    )
    for ax in axes:
        for label, bandwidth, style, color in ceilings:
            ax.plot(xs, np.minimum(fp32, bandwidth * xs), style,
                    color=color, linewidth=2, label=label)
        for index, item in enumerate(results):
            marker = markers[index % len(markers)]
            ax.scatter(
                intensity, float(item["achieved_gflops"]), marker=marker,
                s=82, color=colors[item["provider"]], edgecolor="white",
                linewidth=0.9, zorder=5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(intensity / 2, intensity * 2)
        ax.grid(True, which="both")
        ax.set_xlabel("Common backward intensity (FLOP / ideal-cache byte)")
    axes[0].axhline(fp32, color="#9333ea", linestyle=":", linewidth=1.8,
                    label=f"FP32 ceiling ({fp32 / 1000:.1f} TFLOP/s)")
    axes[0].set_ylim(
        10 ** math.floor(math.log10(min(values) / 1.5)), fp32 * 1.35)
    axes[0].set_ylabel("Backward throughput (GFLOP/s)")
    axes[0].set_title("Full hierarchical roofline")
    axes[0].legend(fontsize=8, loc="lower left", framealpha=0.92)
    fast = [value for item, value in zip(results, values)
            if item["provider"] not in {
                "torch.autograd", "torch.explicit_gather"}]
    axes[1].set_ylim(min(fast) * 0.94, max(fast) * 1.06)
    axes[1].set_ylabel("Backward throughput (GFLOP/s)")
    axes[1].set_title("Generated / handwritten / sparse zoom")
    handles = [
        Line2D(
            [0], [0], linestyle="", marker=marker, markersize=7,
            markerfacecolor=colors[item["provider"]], markeredgecolor="white",
            label=f"{item['provider']} · {item['milliseconds']:.4f} ms")
        for index, item in enumerate(results)
        for marker in (markers[index % len(markers)],)
    ]
    axes[1].legend(handles=handles, loc="lower left", fontsize=7.5,
                   framealpha=0.94,
                   title="same semantic x · physical bytes in JSON")
    axes[0].text(
        0.99, 0.02,
        "copy-calibrated roofs are proxies for indexed traffic",
        transform=axes[0].transAxes, ha="right", va="bottom",
        fontsize=7.5, color="#64748b")
    gradient = payload.get("backward_contract", {}).get("gradient", "backward")
    kind = "edge-weight gradient" if str(gradient).startswith("dweight") \
        else "source gradient"
    fig.suptitle(
        f"MessagePassing {kind} backward · compiler VJP vs peers",
        fontsize=13.5, fontweight="bold")
    _save(fig, output / "roofline")


def plot_dense_attention_roofline(payload: dict, output: Path):
    """Show close SOTA points without changing their common semantic x."""
    roof = payload["roof"]
    results = payload["results"]
    colors = _colors(results)
    intensity = float(results[0]["arithmetic_intensity_flop_per_byte"])
    fp16 = float(roof["compute_gflops"])
    xmin, xmax = intensity / 2.0, intensity * 2.0
    xs = np.logspace(math.log10(xmin), math.log10(xmax), 300)
    values = [float(item["achieved_gflops"]) for item in results]
    markers = ("o", "s", "D", "^")
    fig, axes = plt.subplots(
        1, 2, figsize=(13.8, 5.9), constrained_layout=True, sharex=True,
        gridspec_kw={"width_ratios": (1.05, 1.0)})
    ceilings = (
        ("DRAM roof", float(roof["dram_bandwidth_gbs"]), "--", "#64748b"),
        ("L2 roof", float(roof["l2_bandwidth_gbs"]), "-", "#111827"),
    )
    for ax in axes:
        for label, bandwidth, style, color in ceilings:
            ax.plot(xs, np.minimum(fp16, bandwidth * xs), style,
                    color=color, linewidth=2, label=label)
        for marker, item in zip(markers, results):
            ax.scatter(
                intensity, float(item["achieved_gflops"]), marker=marker,
                s=78, color=colors[item["provider"]], edgecolor="white",
                linewidth=0.9, zorder=5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(xmin, xmax)
        ax.grid(True, which="both")
        ax.set_xlabel("Common semantic intensity (useful FLOP / common byte)")
    axes[0].axhline(
        fp16, color="#9333ea", linestyle=":", linewidth=1.8,
        label=f"FP16 tensor-core ceiling ({fp16 / 1000:.1f} TFLOP/s)")
    axes[0].set_ylim(
        10 ** math.floor(math.log10(min(values) / 1.5)), fp16 * 1.35)
    axes[0].set_ylabel("Useful semantic throughput (GFLOP/s)")
    axes[0].set_title("Full hierarchical roofline")
    axes[0].legend(fontsize=8, loc="lower left", framealpha=0.92)
    sota_values = [value for item, value in zip(results, values)
                   if item["provider"] != "torch.sdpa.math"]
    axes[1].set_ylim(min(sota_values) * 0.94, max(sota_values) * 1.06)
    axes[1].set_ylabel("Useful semantic throughput (GFLOP/s)")
    axes[1].set_title("SOTA-region zoom (unchanged x)")
    handles = [
        Line2D(
            [0], [0], linestyle="", marker=marker, markersize=7,
            markerfacecolor=colors[item["provider"]], markeredgecolor="white",
            label=(f"{_short_provider(item['provider'])} · "
                   f"{item['milliseconds']:.4f} ms · "
                   f"{item['achieved_gflops'] / 1000:.2f} TFLOP/s"))
        for marker, item in zip(markers, results)
    ]
    axes[1].legend(handles=handles, loc="lower left", fontsize=7.5,
                   framealpha=0.94, title="same workload and semantic x")
    config = payload["config"]
    fig.suptitle(
        "Dense Cartesian streaming reduction · compiler output vs SOTA\n"
        f"B={config['batch']}, H={config['heads']}, "
        f"N={config['sequence']}, D={config['width']}, {config['dtype']}",
        fontsize=13.5, fontweight="bold")
    _save(fig, output / "roofline")


def _short_provider(name: str) -> str:
    return (name.replace("graphforge.", "gf.")
            .replace("precomputed_distance", "precomputed")
            .replace("generated_", "gen."))


def plot_radius_aggregation_roofline(payload: dict, output: Path):
    """Plot one provider-independent semantic x coordinate with a zoom."""
    roof = payload["roof"]
    results = payload["results"]
    colors = _colors(results)
    intensities = [float(item["arithmetic_intensity_flop_per_byte"])
                   for item in results]
    xmin = min(intensities) / 1.5
    xmax = max(intensities) * 1.5
    xs = np.logspace(math.log10(xmin), math.log10(xmax), 300)
    fp32 = float(roof["fp32_gflops"])

    fig, axes = plt.subplots(
        1, 2, figsize=(14.2, 6.2), constrained_layout=True, sharex=True,
        gridspec_kw={"width_ratios": (1.05, 1.0)})
    ceilings = (
        ("DRAM semantic roof", float(roof["dram_bandwidth_gbs"]),
         "--", "#64748b"),
        ("L2 semantic roof", float(roof["l2_bandwidth_gbs"]),
         "-", "#111827"),
    )
    markers = {
        "consumer-only": "o",
        "relation-reuse": "s",
        "fresh-build+consume": "D",
        "logical-rebind": "P",
        "topology-rebuild+consume": "X",
        # Backward compatibility for older artifacts.
        "build+consume": "D",
    }
    for ax in axes:
        for label, bandwidth, style, color in ceilings:
            ax.plot(xs, np.minimum(fp32, bandwidth * xs), style,
                    color=color, linewidth=2, label=label)
        for item in results:
            x = float(item["arithmetic_intensity_flop_per_byte"])
            y = float(item["achieved_gflops"])
            ax.scatter(
                x, y, s=70, marker=markers.get(item["cache"], "v"),
                color=colors[item["provider"]], edgecolor="white",
                linewidth=0.8, zorder=5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(xmin, xmax)
        ax.grid(True, which="both")
        ax.set_xlabel("Semantic arithmetic intensity (useful FLOP / common byte)")
    axes[0].axhline(fp32, color="#9333ea", linestyle=":", linewidth=1.8,
                    label=f"FP32 ceiling ({fp32 / 1000:.1f} TFLOP/s)")
    values = [float(item["achieved_gflops"]) for item in results]
    axes[0].set_ylim(
        10 ** math.floor(math.log10(min(values) / 1.5)), fp32 * 1.5)
    axes[0].set_ylabel("Useful semantic throughput (GFLOP/s)")
    axes[0].set_title("Full semantic roofline")
    axes[0].legend(fontsize=8, loc="upper left", framealpha=0.92)

    roof_at_workload = min(
        fp32, float(roof["dram_bandwidth_gbs"]) * intensities[0])
    axes[1].set_ylim(
        10 ** math.floor(math.log10(min(values) / 1.5)),
        10 ** math.ceil(math.log10(max(values) * 1.8)))
    axes[1].set_title("Achieved-region zoom (same x axis)")
    axes[1].set_ylabel("Useful semantic throughput (GFLOP/s)")
    ordered = sorted(results, key=lambda item: float(item["achieved_gflops"]),
                     reverse=True)
    provider_handles = [
        Line2D(
            [0], [0], linestyle="", marker=markers.get(item["cache"], "v"),
            markersize=7, markerfacecolor=colors[item["provider"]],
            markeredgecolor="white",
            label=(f"{_short_provider(item['provider'])} · "
                   f"{item['milliseconds']:.4f} ms · "
                   f"{100 * float(item['achieved_gflops']) / roof_at_workload:.2f}% roof"),
        )
        for item in ordered
    ]
    axes[1].legend(
        handles=provider_handles, loc="lower right", fontsize=7.5,
        framealpha=0.94, title="same semantic x · provider / latency / DRAM roof")

    config = payload["config"]
    fig.suptitle(
        "Radius distance aggregation · provider-independent semantic roofline\n"
        f"N={config['particles']:,}, D={config['dimensions']}, "
        f"accepted={config['accepted_edges']:,}, "
        f"candidates={config['candidate_pairs']:,}",
        fontsize=14, fontweight="bold")
    _save(fig, output / "roofline")


def plot_radius_build_operational_roofline(payload: dict, output: Path):
    """Keep semantic x common; show candidate amplification as diagnostics."""
    roof = payload["roof"]
    results = payload["results"]
    colors = _colors(results)
    intensities = [float(item["arithmetic_intensity_flop_per_byte"])
                   for item in results]
    xmin = min(intensities) / 1.5
    xmax = max(intensities) * 1.5
    xs = np.logspace(math.log10(xmin), math.log10(xmax), 300)

    fp32 = float(roof["fp32_gflops"])
    fig, axes = plt.subplots(
        1, 2, figsize=(13.8, 5.9), constrained_layout=True, sharex=True,
        gridspec_kw={"width_ratios": (1.15, 1.0)})
    ceilings = (
        ("DRAM semantic roof", float(roof["dram_bandwidth_gbs"]),
         "--", "#64748b"),
        ("L2 semantic roof", float(roof["l2_bandwidth_gbs"]),
         "-", "#111827"),
    )
    achieved = [float(item["achieved_gflops"]) for item in results]
    for ax in axes:
        for label, bandwidth, style, color in ceilings:
            ax.plot(xs, np.minimum(fp32, bandwidth * xs), style,
                    color=color, linewidth=2, label=label)
        for item in results:
            ax.scatter(
                float(item["arithmetic_intensity_flop_per_byte"]),
                float(item["achieved_gflops"]), s=75,
                color=colors[item["provider"]], edgecolor="white",
                linewidth=0.8, zorder=5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(xmin, xmax)
        ax.grid(True, which="both")
        ax.set_xlabel("Semantic arithmetic intensity (useful FLOP / common byte)")
    axes[0].set_ylim(
        10 ** math.floor(math.log10(min(achieved) / 1.5)), fp32 * 1.5)
    axes[0].set_ylabel("Useful semantic throughput (GFLOP/s)")
    axes[0].set_title("Full semantic roofline")
    axes[0].legend(fontsize=8, loc="upper left", framealpha=0.92)
    axes[1].set_ylim(
        10 ** math.floor(math.log10(min(achieved) / 1.5)),
        10 ** math.ceil(math.log10(max(achieved) * 2)))
    axes[1].set_ylabel("Useful semantic throughput (GFLOP/s)")
    axes[1].set_title("Achieved-region zoom (same x axis)")
    semantic_roof = min(
        fp32, float(roof["dram_bandwidth_gbs"]) * intensities[0])
    for index, item in enumerate(results):
        x = float(item["arithmetic_intensity_flop_per_byte"])
        y = float(item["achieved_gflops"])
        physical = (
            f"{item['candidate_to_edge']:.2f}× candidates/edge"
            if item.get("materialized", True)
            else f"compact directory · {item['implementation_modeled_bytes'] / 2**20:.2f} MiB")
        axes[1].annotate(
            f"{_short_provider(item['provider'])}\n"
            f"{item['milliseconds']:.3f} ms · {physical}\n"
            f"{100 * y / semantic_roof:.2f}% DRAM semantic roof",
            (x, y), xytext=(8 if index % 2 == 0 else -120, 8),
            textcoords="offset points", fontsize=8,
            arrowprops={"arrowstyle": "-", "color": "#94a3b8", "lw": 0.6})

    config = payload["config"]
    fig.suptitle(
        f"Radius graph build · N={config['particles']:,}, "
        f"D={config['dimensions']}, target degree={config['target_degree']:g}",
        fontsize=14, fontweight="bold")
    _save(fig, output / "roofline")


def plot_latency(
    payload: dict,
    output: Path,
    *,
    stem: str = "provider_latency",
):
    """Render a vendor-style small-multiple provider leaderboard.

    Each cache/feature bucket is one kernel panel. Provider identity is the
    hue and is stable across the whole benchmark corpus. Bars report matched
    throughput speedup over torch.sparse.mm when available, otherwise over the
    fastest measured non-GraphForge provider. Absolute latency remains on each
    bar so the normalized view cannot hide the timing scale.
    """
    results = payload["results"]
    colors = _colors(results)
    caches = sorted({item["cache"] for item in results},
                    key=lambda name: (name != "hot", name))
    features = sorted({int(item.get("features", 1)) for item in results})
    panels = [
        (cache, feature, [
            item for item in results
            if item["cache"] == cache and int(item.get("features", 1)) == feature
        ])
        for cache in caches for feature in features
    ]
    panels = [panel for panel in panels if panel[2]]
    columns = 2 if len(panels) == 4 else min(3, len(panels))
    rows = math.ceil(len(panels) / columns)
    fig, axes = plt.subplots(
        rows, columns, figsize=(4.2 * columns, 3.8 * rows),
        squeeze=False, constrained_layout=True, sharey=True)
    speedups_by_panel = []
    for _cache, _feature, selected in panels:
        baseline = next(
            (item for item in selected
             if item["provider"] == "torch.sparse.mm"),
            None,
        )
        if baseline is None:
            peers = [
                item for item in selected
                if not item["provider"].startswith("graphforge.")
            ]
            baseline = min(
                peers or selected,
                key=lambda item: item["milliseconds"],
            )
        baseline_ms = float(baseline["milliseconds"])
        ordered = sorted(
            selected,
            key=lambda item: (
                not item["provider"].startswith("graphforge."),
                item["provider"],
            ),
        )
        speedups_by_panel.append((baseline["provider"], ordered, [
            baseline_ms / float(item["milliseconds"]) for item in ordered
        ]))
    ymax = max(max(speedups) for _baseline, _items, speedups in speedups_by_panel)

    for panel_index, ((cache, feature, _selected), comparison) in enumerate(
            zip(panels, speedups_by_panel)):
        ax = axes.flat[panel_index]
        baseline_name, ordered, speedups = comparison
        positions = np.arange(len(ordered))
        bars = ax.bar(
            positions,
            speedups,
            color=[colors[item["provider"]] for item in ordered],
            edgecolor="white",
            linewidth=0.9,
        )
        ax.axhline(1.0, color="#475569", linestyle="--", linewidth=1.1)
        ax.set_xticks(positions, [str(index + 1) for index in positions])
        ax.set_ylim(0, ymax * 1.22)
        ax.grid(True, axis="y")
        ax.set_title(f"F={feature} · {cache}")
        ax.set_xlabel("method ID")
        if panel_index % columns == 0:
            ax.set_ylabel(f"Speedup vs {baseline_name}\n(higher is better)")
        for bar, speedup, item in zip(bars, speedups, ordered):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ymax * 0.025,
                f"{speedup:.2f}×\n{float(item['milliseconds']):.3f} ms",
                ha="center", va="bottom", fontsize=7,
            )
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)

    providers = sorted(colors)
    handles = [
        Line2D(
            [0], [0], marker="s", linestyle="", markersize=8,
            markerfacecolor=colors[provider], markeredgecolor="white",
            label=f"{index + 1} = {provider}",
        )
        for index, provider in enumerate(providers)
    ]
    # Panel-local method IDs follow the global provider order after filtering.
    # Re-label ticks so their numeric IDs remain stable even when an optional
    # provider is absent from one panel.
    provider_ids = {
        provider: index + 1 for index, provider in enumerate(providers)
    }
    for panel_index, (_baseline, ordered, _speedups) in enumerate(speedups_by_panel):
        axes.flat[panel_index].set_xticklabels([
            str(provider_ids[item["provider"]]) for item in ordered
        ])
    workload = payload.get("workload", "weighted aggregation").replace("_", " ")
    fig.suptitle(
        f"{workload} · provider leaderboard",
        fontsize=14, fontweight="bold")
    fig.legend(
        handles=handles, loc="outside lower center", ncol=min(3, len(handles)),
        fontsize=8, framealpha=0.92, title="method hue is stable across panels")
    _save(fig, output / stem)


def plot_jit(payload: dict, output: Path):
    cold = payload["cold"]
    disk = payload["disk_cache_hit"]
    labels = [
        "Cold provider\ncompile", "Cold first\nlaunch/setup",
        "Cold ready\nto result", "Disk cache\nlookup",
        "Disk first\nlaunch/setup", "Disk ready\nto result", "Warm\nkernel",
    ]
    values = [
        cold["provider_compile_or_cache_ms"], cold["first_launch_ms"],
        cold["ready_to_result_ms"], disk["provider_compile_or_cache_ms"],
        disk["first_launch_ms"], disk["ready_to_result_ms"],
        cold["warm_kernel_median_ms"],
    ]
    colors = ["#dc2626"] * 3 + ["#ea580c"] * 3 + ["#059669"]
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.0)
    ax.set_yscale("log")
    ax.set_ylabel("Latency (ms, log scale)")
    ax.grid(True, axis="y", which="both")
    ax.set_ylim(top=max(values) * 2.2)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, value * 1.16,
            f"{value:.3f} ms", ha="center", va="bottom", fontsize=8)
    fig.suptitle(
        "Triton JIT lifecycle: compile/cache versus warm execution",
        y=0.98, fontsize=14, fontweight="bold")
    fig.text(
        0.5, 0.925,
        f"{payload['device']} · Triton {payload['triton_version']} · "
        f"N={payload['nodes']:,}, degree={payload['degree']}, "
        f"F={payload['features']}",
        ha="center", va="center", fontsize=9, color="#475569")
    fig.subplots_adjust(top=0.86, bottom=0.2, left=0.08, right=0.98)
    _save(fig, output / "jit_latency")


def plot_radius_build(payload: dict, output: Path):
    results = payload["results"]
    labels = [item["provider"].replace("graphforge.", "gf.") for item in results]
    colors = [PALETTE[index % len(PALETTE)] for index in range(len(results))]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), constrained_layout=True)

    latency = [float(item["milliseconds"]) for item in results]
    bars = axes[0].bar(labels, latency, color=colors, edgecolor="white")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Build latency (ms, log scale)")
    axes[0].set_title("End-to-end radius build")
    axes[0].grid(True, axis="y", which="both")
    for bar, value in zip(bars, latency):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2, value * 1.08,
            f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    x = np.arange(len(results))
    width = 0.36
    candidates = [int(item["candidate_pairs"]) for item in results]
    accepted = [int(item["accepted_edges"]) for item in results]
    axes[1].bar(
        x - width / 2, candidates, width, label="candidate pairs",
        color="#64748b")
    axes[1].bar(
        x + width / 2, accepted, width, label="accepted edges",
        color="#059669")
    axes[1].set_xticks(x, labels)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Pairs / edges (log scale)")
    axes[1].set_title("Broad-phase amplification")
    axes[1].grid(True, axis="y", which="both")
    axes[1].legend(fontsize=8)

    config = payload["config"]
    fig.suptitle(
        f"Dynamic RadiusGraph · N={config['particles']:,}, "
        f"D={config['dimensions']}, target degree={config['target_degree']:g}",
        fontsize=14, fontweight="bold")
    _save(fig, output / "radius_build")


def plot_radius_pipeline(payload: dict, output: Path):
    results = payload["results"]
    phases = ("build-only", "consume-only", "build+consume")
    grouped = {
        phase: [item for item in results if item["phase"] == phase]
        for phase in phases
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), constrained_layout=True)

    x = np.arange(len(phases))
    width = 0.34
    candidate_values = []
    peer_values = []
    candidate_labels = []
    peer_labels = []
    candidate_providers = {"graphforge.auto", "graphforge.dynamic.auto"}
    for phase in phases:
        items = grouped[phase]
        graphforge = next(
            (item for item in items if item["provider"] in candidate_providers),
            None,
        )
        if phase == "build-only":
            graphforge = next(iter(items), None)
        peers = [item for item in items if item is not graphforge]
        peer = min(
            peers, key=lambda item: float(item["milliseconds"]), default=None)
        # build-only has no external peer yet; retain the measured GF bar.
        candidate_values.append(
            float(graphforge["milliseconds"]) if graphforge is not None else np.nan)
        peer_values.append(
            float(peer["milliseconds"]) if peer is not None else np.nan)
        candidate_labels.append(graphforge["provider"] if graphforge else "")
        peer_labels.append(peer["provider"] if peer else "")

    axes[0].bar(
        x - width / 2, candidate_values, width,
        label="GraphForge", color="#2563eb", edgecolor="white")
    axes[0].bar(
        x + width / 2, peer_values, width,
        label="fastest matched peer", color="#dc2626", edgecolor="white")
    axes[0].set_xticks(x, phases)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Synchronized wall latency (ms, log scale)")
    axes[0].set_title("Radius pipeline latency")
    axes[0].grid(True, axis="y", which="both")
    axes[0].legend(fontsize=8)
    for center, values, shift in (
        (x, candidate_values, -width / 2), (x, peer_values, width / 2)):
        for location, value in zip(center, values):
            if not np.isnan(value):
                axes[0].text(
                    location + shift, value * 1.08, f"{value:.3f}",
                    ha="center", va="bottom", fontsize=8)

    gates = payload.get("sota_gates", [])
    gate_names = [gate["cache"] for gate in gates]
    speedups = [float(gate["speedup_vs_sota"]) for gate in gates]
    lows = [float(gate["speedup_ci_low"]) for gate in gates]
    highs = [float(gate["speedup_ci_high"]) for gate in gates]
    gate_x = np.arange(len(gates))
    colors = ["#059669" if gate["passed"] else "#dc2626" for gate in gates]
    axes[1].axhline(1.0, color="#111827", linestyle="--", linewidth=1.2)
    axes[1].bar(gate_x, speedups, color=colors, edgecolor="white")
    axes[1].errorbar(
        gate_x,
        speedups,
        yerr=[
            [value - low for value, low in zip(speedups, lows)],
            [high - value for value, high in zip(speedups, highs)],
        ],
        fmt="none", ecolor="#334155", capsize=4,
    )
    axes[1].set_xticks(gate_x, gate_names)
    axes[1].set_ylabel("Speedup versus matched peer")
    axes[1].set_title("Median speedup and bootstrap 95% CI")
    axes[1].grid(True, axis="y")
    for location, value in zip(gate_x, speedups):
        axes[1].text(
            location, value + 0.025, f"{value:.3f}×",
            ha="center", va="bottom", fontsize=9)

    config = payload["config"]
    graph = payload["graph"]
    fig.suptitle(
        f"RadiusGraph build + fused consume · N={config['particles']:,}, "
        f"edges={graph['accepted_edges']:,}, D={config['dimensions']}",
        fontsize=14, fontweight="bold")
    _save(fig, output / "radius_pipeline")


def _save(fig, stem: Path):
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    svg = stem.with_suffix(".svg")
    # Stable element IDs and omitted wall-clock metadata keep tracked SVG
    # assets byte-for-byte reproducible across equivalent benchmark renders.
    with matplotlib.rc_context({"svg.hashsalt": "graphforge"}):
        fig.savefig(svg, bbox_inches="tight", metadata={"Date": None})
    # Matplotlib emits trailing spaces in multiline SVG path data.  Normalize
    # generated assets so publication updates remain reviewable and pass the
    # repository whitespace contract.
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n",
        encoding="utf-8",
    )
    plt.close(fig)


def write_report(
    roofline: dict,
    jit: dict | None,
    radius: dict | None,
    radius_pipeline: dict | None,
    output: Path,
):
    lines = [
        "# GraphForge benchmark visualization",
        "",
        "Generated from machine-readable benchmark JSON.",
        "",
        "## Hierarchical roofline",
        "",
        "![Hierarchical roofline](roofline.png)",
        "",
        "## Provider latency",
        "",
        "![Provider latency](provider_latency.png)",
    ]
    if jit is not None:
        lines.extend([
            "",
            "## JIT lifecycle",
            "",
            "![JIT lifecycle](jit_latency.png)",
        ])
    if radius is not None:
        lines.extend([
            "",
            "## Dynamic radius build",
            "",
            "![Dynamic radius build](radius_build.png)",
        ])
    if radius_pipeline is not None:
        lines.extend([
            "",
            "## Dynamic radius build + consume",
            "",
            "![Dynamic radius build and fused consume](radius_pipeline.png)",
            "",
            radius_pipeline["baseline_scope"],
        ])
        radius_gates = radius_pipeline.get("sota_gates", [])
        if radius_gates:
            lines.extend([
                "",
                "| Phase | Candidate | Fastest peer | Speedup | 95% CI | Result |",
                "|---|---|---|---:|---:|---|",
            ])
            for gate in radius_gates:
                lines.append(
                    f"| {gate['cache']} | {gate['candidate']} | "
                    f"{gate['baseline']} | {gate['speedup_vs_sota']:.3f}x | "
                    f"[{gate['speedup_ci_low']:.3f}, "
                    f"{gate['speedup_ci_high']:.3f}] | "
                    f"{'PASS' if gate['passed'] else 'FAIL'} |")
    gates = roofline.get("sota_gates", [])
    if gates:
        lines.extend([
            "",
            "## SOTA gates",
            "",
            "| Candidate | Cache | Index | F | Baseline | Speedup | 95% CI | Result |",
            "|---|---:|---:|---:|---|---:|---:|---|",
        ])
        for gate in gates:
            ci = (f"[{gate['speedup_ci_low']:.3f}, "
                  f"{gate['speedup_ci_high']:.3f}]")
            lines.append(
                f"| {gate['candidate']} | {gate['cache']} | "
                f"{gate.get('index_dtype', 'unspecified')} | {gate['features']} | "
                f"{gate['baseline']} | {gate['speedup_vs_sota']:.3f}x | {ci} | "
                f"{'PASS' if gate['passed'] else 'FAIL'} |")
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roofline", type=Path, required=True)
    parser.add_argument("--jit", type=Path)
    parser.add_argument("--radius", type=Path)
    parser.add_argument("--radius-pipeline", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    args = parser.parse_args()
    _style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    roofline = _load(args.roofline)
    jit = _load(args.jit) if args.jit else None
    radius = _load(args.radius) if args.radius else None
    radius_pipeline = _load(args.radius_pipeline) if args.radius_pipeline else None
    plot_roofline(roofline, args.output_dir)
    plot_latency(roofline, args.output_dir)
    if jit is not None:
        plot_jit(jit, args.output_dir)
    if radius is not None:
        plot_radius_build(radius, args.output_dir)
    if radius_pipeline is not None:
        plot_radius_pipeline(radius_pipeline, args.output_dir)
    write_report(roofline, jit, radius, radius_pipeline, args.output_dir)
    for path in sorted(args.output_dir.iterdir()):
        if path.suffix in {".png", ".md"}:
            print(path)


if __name__ == "__main__":
    main()
