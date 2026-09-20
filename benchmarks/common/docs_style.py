"""Shared chart style for every docs-facing benchmark figure.

The documentation site is mkdocs-material with an indigo accent and an
automatic light/slate palette.  Figures are embedded bare on the page
(``docs/benchmark-results.md``) or on the light ``.gf-figure`` card
(homepage showcase, report fallback), so every figure ships with a
light background and dark slate ink that remains readable independently of
the surrounding page theme.

Usage::

    from benchmarks.common import docs_style

    docs_style.apply()
    ...build figure...
    docs_style.headline(fig, "title", "subtitle")
    docs_style.finish_axes(ax, horizontal=True)
    docs_style.save(fig, Path("docs/assets/results/stem"))

Only the visual language lives here; what each chart measures is defined
by the calling generator.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- house palette -----------------------------------------------------------
GF = "#4F46E5"       # indigo-600: Tiga hero series
GF_SOFT = "#A5B4FC"  # indigo-300: Tiga alternative / oracle-class peer
SLATE = "#94A3B8"    # slate-400: matched peers, parity reference lines
BASE = "#CBD5E1"     # slate-200: eager/baseline series

# Explicit white surfaces keep chart contrast independent of the page theme.
TEXT = "#334155"     # dark text on an explicit light chart surface
GRID = "#94A3B8"     # rendered at low alpha, subtle on both surfaces
GRID_ALPHA = 0.30

# Light-card ink: only for figures that always sit on the light ``.gf-figure``
# card (homepage showcase, report fallback inside its #f8fafc iframe page).
CARD_INK = "#0F172A"    # slate-900: card headlines
CARD_TEXT = "#475569"   # slate-600: card value labels

PROVIDER_COLORS = {
    "tiga": GF,
    "tiga-compiled": GF,
    "oracle": GF_SOFT,
    "peer": SLATE,
    "baseline": BASE,
    "slate": SLATE,
}

FONT_STACK = [
    "Arial", "Helvetica Neue", "Helvetica", "Liberation Sans", "DejaVu Sans",
]


def apply() -> None:
    """Install the shared rcParams; call once before building any figure."""
    plt.rcParams.update({
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#ffffff",
        "savefig.facecolor": "#ffffff",
        "savefig.transparent": False,
        "font.family": "sans-serif",
        "font.sans-serif": FONT_STACK,
        "font.size": 9.5,
        "text.color": TEXT,
        "axes.edgecolor": SLATE,
        "axes.labelcolor": TEXT,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.alpha": GRID_ALPHA,
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "svg.fonttype": "none",
        "svg.hashsalt": "tiga-docs",
    })


def finish_axes(ax, *, horizontal: bool = False) -> None:
    """Strip chart junk: only a quiet baseline spine and one grid direction."""
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(SLATE)
    ax.spines["bottom"].set_alpha(0.6)
    if horizontal:
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", zorder=0)
    else:
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", zorder=0)
    ax.tick_params(length=0)


def subtitle(fig, text: str, y: float = 0.915) -> None:
    fig.text(0.065, y, text, color=TEXT, fontsize=8.5)


def headline(fig, title: str, subtitle_text: str) -> None:
    """Two-line left-aligned figure header that never collides."""
    fig.text(0.065, 0.955, title, color=TEXT, fontsize=11,
             fontweight="bold")
    fig.text(0.065, 0.885, subtitle_text, color=TEXT, fontsize=8.5)


def save(fig, stem: Path) -> None:
    """Write light-surface SVG + PNG with reproducible SVG bytes."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.12, metadata={"Date": None})
    fig.savefig(
        stem.with_suffix(".png"), bbox_inches="tight", pad_inches=0.06, dpi=220)
    plt.close(fig)
    svg = stem.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    print(f"wrote {stem.with_suffix('.svg')}")


def label_bars(ax, bars, fmt="{:.2f}×", *, horizontal=True, color=TEXT):
    for bar in bars:
        if horizontal:
            width = bar.get_width()
            ax.text(
                width + ax.get_xlim()[1] * 0.008,
                bar.get_y() + bar.get_height() / 2,
                fmt.format(width), va="center", ha="left",
                fontsize=8.5, color=color)
        else:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + ax.get_ylim()[1] * 0.01,
                fmt.format(height), va="bottom", ha="center",
                fontsize=8.5, color=color)
