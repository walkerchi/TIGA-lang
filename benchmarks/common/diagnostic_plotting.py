"""Human-facing plots for non-roofline benchmark JSON artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from benchmarks.common.plotting import (
    _save, _style, plot_radius_pipeline, provider_color,
)


def _bars(ax, labels, values, colors, *, ylabel, log=False):
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8)
    if log and values and min(values) > 0:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", which="both")
    ax.tick_params(axis="x", rotation=18)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value,
                f"{value:.3g}", ha="center", va="bottom", fontsize=8)


def _persistent_worker(payload: dict, output: Path) -> None:
    phases = ["cold", "warm_worker", "post_crash_disk_hit"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, field, title in (
        (axes[0], "worker_compile_ms", "Vendor compile"),
        (axes[1], "ready_ms", "Ready to result"),
    ):
        values = [float(payload[phase][field]) for phase in phases]
        _bars(ax, ["cold", "warm", "post-crash"], values,
              ["#dc2626", "#059669", "#ea580c"], ylabel="Latency (ms)", log=True)
        ax.set_title(title)
    fig.suptitle("Persistent compile-worker lifecycle", fontsize=14, fontweight="bold")
    _save(fig, output)


def _tensor_fusion(payload: dict, output: Path) -> None:
    methods = [
        ("tiga.kernel", payload["tiga"].get("kernel_median_ms")
         or payload["tiga"].get("kernel", {}).get("median_ms")),
        ("tiga.end_to_end", payload["tiga"].get("warm_median_ms")
         or payload["tiga"].get("python_e2e", {}).get("median_ms")),
        ("torch.eager", payload.get("torch_eager", {}).get("warm_median_ms")
         or payload.get("torch_eager", {}).get("median_ms")),
        ("torch.compile", payload.get("torch_compile", {}).get("warm_median_ms")
         or payload.get("torch_compile", {}).get("median_ms")),
    ]
    methods = [(name, float(value)) for name, value in methods if value is not None]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    for ax, kernel_only in zip(axes, (False, True)):
        selected = [(name, value) for name, value in methods
                    if (name == "tiga.kernel") == kernel_only]
        _bars(ax, [name for name, _value in selected],
              [value for _name, value in selected],
              [provider_color(name) for name, _value in selected],
              ylabel="Latency (ms)", log=True)
        ax.set_title("Kernel-only · diagnostic" if kernel_only else "Warm end-to-end · matched scope")
    shape = " × ".join(str(value) for value in payload.get("shape", []))
    fig.suptitle(f"Tensor fusion · {payload.get('dtype', '')} · {shape}")
    _save(fig, output)


def _halo(payload: dict, output: Path) -> None:
    labels = ["pack", "transport", "unpack", "total"]
    values = [
        float(payload["median_rank_pack_ms"]),
        float(payload["median_rank_transport_ms"]),
        float(payload["median_rank_unpack_ms"]),
        float(payload["median_rank_total_ms"]),
    ]
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    _bars(ax, labels, values, ["#2563eb", "#7c3aed", "#059669", "#111827"],
          ylabel="Median rank latency (ms)", log=True)
    ax.set_title(
        f"{payload['world_size']}-rank halo exchange · "
        f"{payload['aggregate_payload_GBps']:.3f} GB/s · gate {payload['gate']}")
    _save(fig, output)


def _nccl_loopback(payload: dict, output: Path) -> None:
    cases = payload["cases"]
    sizes = [float(case["bytes"]) / (1 << 20) for case in cases]
    latencies = [float(case["median_ms"]) for case in cases]
    bandwidth = [float(case["diagnostic_payload_GBps"]) for case in cases]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].plot(sizes, latencies, marker="o", color=provider_color("tiga.nccl"))
    axes[1].plot(sizes, bandwidth, marker="o", color=provider_color("tiga.nccl"))
    for ax in axes:
        ax.set_xscale("symlog", linthresh=0.01)
        ax.grid(True, which="both")
        ax.set_xlabel("Payload (MiB)")
    axes[0].set_ylabel("Median latency (ms)")
    axes[1].set_ylabel("Diagnostic payload (GB/s)")
    fig.suptitle("NCCL rank-local device-pointer conformance (not communication)",
                 fontsize=14, fontweight="bold")
    _save(fig, output)


def _memory_hierarchy(payload: dict, output: Path) -> None:
    labels = ["pinned→HBM", "HBM→pinned", "RAM→NVMe", "NVMe→RAM"]
    latency = [float(payload[key]) for key in (
        "h2d_ms", "d2h_ms", "nvme_spill_ms", "nvme_restore_ms")]
    bandwidth = [float(payload[key]) for key in (
        "h2d_GBps", "d2h_GBps", "nvme_spill_GBps", "nvme_restore_GBps")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    colors = ["#2563eb", "#0891b2", "#ea580c", "#059669"]
    _bars(axes[0], labels, latency, colors, ylabel="Latency (ms)", log=True)
    _bars(axes[1], labels, bandwidth, colors, ylabel="Bandwidth (GB/s)", log=True)
    fig.suptitle(f"Hierarchy transfer · gate {payload['gate']}",
                 fontsize=14, fontweight="bold")
    _save(fig, output)


def _capacity(payload: dict, output: Path) -> None:
    profiles = payload["profiles"]
    labels = [f"{item['case']}\n{item['profile']}" for item in profiles]
    working = [float(item["minimum_working_set"]) / (1 << 30) for item in profiles]
    tier_colors = {
        "hbm-resident": "#2563eb",
        "ram-resident/hbm-streamed": "#059669",
        "ssd-resident/ram+hbm-streamed": "#ea580c",
        "distributed-or-more-storage": "#dc2626",
    }
    colors = [tier_colors[item["tier"]] for item in profiles]
    height = max(6, 0.45 * len(profiles))
    fig, ax = plt.subplots(figsize=(12, height), constrained_layout=True)
    positions = np.arange(len(labels))
    ax.barh(positions, working, color=colors)
    ax.set_yticks(positions, labels, fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("Minimum working set (GiB, log scale)")
    ax.grid(True, axis="x", which="both")
    for name, bytes_value, color in (
        ("HBM", payload["capacity_bytes"]["hbm"], "#2563eb"),
        ("RAM", payload["capacity_bytes"]["ram"], "#059669"),
        ("SSD free", payload["capacity_bytes"]["ssd_free"], "#ea580c"),
    ):
        ax.axvline(float(bytes_value) * float(payload["headroom"]) / (1 << 30),
                   color=color, linestyle="--", linewidth=1.4, label=name)
    ax.legend(title=f"usable capacity ({float(payload['headroom']):.0%})")
    ax.set_title("Large-graph storage-tier capacity plan")
    _save(fig, output)


def _provider_conformance(payload: dict, output: Path) -> None:
    targets = list(payload["targets"])
    stages = ["plugin", "correctness", "artifact", "compile", "roofline", "performance"]
    matrix = np.zeros((len(targets), len(stages)))
    for row, target in enumerate(targets):
        ready = payload["targets"][target]["status"] == "READY_FOR_HARDWARE"
        matrix[row, :] = 1.0 if ready else 0.0
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    ax.imshow(matrix, cmap=matplotlib.colors.ListedColormap(["#fee2e2", "#dcfce7"]),
              vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(stages)), stages, rotation=20)
    ax.set_yticks(range(len(targets)), targets)
    for row, target in enumerate(targets):
        status = payload["targets"][target]["status"]
        for column in range(len(stages)):
            ax.text(column, row, "READY" if matrix[row, column] else "PENDING",
                    ha="center", va="center", fontsize=7, fontweight="bold")
        ax.text(len(stages) - 0.5, row, "", alpha=0)
    ax.set_title("External provider conformance matrix")
    _save(fig, output)


def _radius_matrix(payload: dict, output: Path) -> None:
    cases = payload["cases"]
    gates = list(payload["matrix_axes"]["lifecycle"])
    rows = [case["case"] for case in cases]
    values = np.full((len(rows), len(gates)), np.nan)
    for row, case in enumerate(cases):
        by_gate = {gate["cache"]: gate for gate in case["gates"]}
        for column, gate in enumerate(gates):
            if gate in by_gate:
                values[row, column] = float(by_gate[gate]["speedup_vs_sota"])
    fig, ax = plt.subplots(figsize=(12, max(5, 0.6 * len(rows))), constrained_layout=True)
    image = ax.imshow(values, cmap="RdYlGn", vmin=0.8,
                      vmax=max(1.2, float(np.nanmax(values))), aspect="auto")
    ax.set_xticks(range(len(gates)), gates, rotation=18)
    ax.set_yticks(range(len(rows)), rows)
    for row in range(len(rows)):
        for column in range(len(gates)):
            value = values[row, column]
            if not math.isnan(value):
                ax.text(column, row, f"{value:.2f}×", ha="center", va="center",
                        fontsize=8, fontweight="bold")
    fig.colorbar(image, ax=ax, label="Speedup vs matched peer")
    ax.set_title("RadiusGraph lifecycle matrix")
    _save(fig, output)


def _generic(payload: dict, output: Path, title: str) -> None:
    """Last-resort visual index for a new diagnostic schema."""
    scalars = [
        (key, float(value))
        for key, value in payload.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(float(value))
    ][:16]
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.42 * max(1, len(scalars)))),
                           constrained_layout=True)
    if scalars:
        labels = [key for key, _value in scalars]
        values = [value for _key, value in scalars]
        ax.table(cellText=[[name, f"{value:g}"] for name, value in zip(labels, values)],
                 colLabels=["Recorded field", "Value (original unit)"],
                 loc="center", cellLoc="left")
        ax.set_axis_off()
    else:
        keys = "\n".join(sorted(payload)[:24])
        ax.text(0.03, 0.97, "Structured fields:\n" + keys,
                transform=ax.transAxes, va="top", family="monospace")
        ax.set_axis_off()
    ax.set_title(f"{title} · diagnostic artifact overview")
    _save(fig, output)


def plot_json(path: Path) -> Path | None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    output = path.with_suffix("")
    _style()
    schema = payload.get("schema")
    operation = payload.get("operation")
    if operation == "persistent_vendor_compile_worker":
        _persistent_worker(payload, output)
    elif operation == "broadcast_mul_add_axis_sum":
        _tensor_fusion(payload, output)
    elif schema == "tiga.distributed-halo.v1":
        _halo(payload, output)
    elif schema == "tiga.nccl-device-conformance.v1":
        _nccl_loopback(payload, output)
    elif schema == "tiga.memory-hierarchy.v1":
        _memory_hierarchy(payload, output)
    elif schema == "tiga.provider-conformance.v1":
        _provider_conformance(payload, output)
    elif payload.get("schema_version") == 1 and "profiles" in payload:
        _capacity(payload, output)
    elif operation == "radius_distance_aggregation" and "matrix_axes" in payload:
        _radius_matrix(payload, output)
    elif {"config", "graph", "results", "sota_gates"} <= payload.keys():
        plot_radius_pipeline(payload, path.parent)
        generated = path.parent / "radius_pipeline.png"
        target = path.with_suffix(".png")
        generated.replace(target)
        return target
    else:
        _generic(payload, output, path.stem)
    return path.with_suffix(".png")
