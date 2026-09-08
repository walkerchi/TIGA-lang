"""Regenerate the four attention-matrix figures for docs/examples/attention.md.

The attention page explains each mask variant as an attention matrix: row i is
the query (destination), column j the key (source), and a filled cell is an
edge j→i in the relation.  All four figures share one grid grammar so the
differences — full, causal, varlen block-diagonal, tile-pruned — are the only
thing that changes.

Run from the repository root:

    python docs/scripts/attention_matrix_figures.py
"""

from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "assets" / "examples"

CARD = "#f8fafc"
FILL = "#4f46e5"        # admitted / existing edge
EXISTS = "#c7d2fe"      # edge exists but not admitted (tile pruning)
EMPTY = "#ffffff"       # no such edge
EMPTY_STROKE = "#e2e8f0"
PRUNED = "#e2e8f0"
SLATE = "#64748b"

STYLE = """
      text{font-family:Arial,sans-serif}
      .title{font-size:20px;font-weight:700;fill:#0f172a}
      .body{font-size:12.5px;fill:#334155}
      .mono{font-family:Consolas,monospace;font-size:12px;fill:#312e81}
      .axis{font-size:11.5px;fill:#64748b}
      .cap{font-size:11px;fill:#64748b}
"""


def _cell(x: int, y: int, s: int, fill: str, stroke: str = "#ffffff") -> str:
    return (f'<rect x="{x}" y="{y}" width="{s}" height="{s}" rx="3" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.5" />')


def _cross(x: int, y: int, s: int) -> str:
    pad = s * 0.32
    return (f'<path d="M{x + pad} {y + pad} L{x + s - pad} {y + s - pad} '
            f'M{x + s - pad} {y + pad} L{x + pad} {y + s - pad}" '
            f'stroke="{SLATE}" stroke-width="2" stroke-linecap="round" '
            f'fill="none" />')


def _figure(name: str, title: str, desc: str, body: str, width: int,
            height: int) -> None:
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
        f'aria-labelledby="title desc">\n'
        f'  <title id="title">{title}</title>\n'
        f'  <desc id="desc">{desc}</desc>\n'
        f'  <defs><style>{STYLE}</style></defs>\n'
        f'  <rect width="{width}" height="{height}" rx="18" fill="{CARD}" />\n'
        f'  <text x="32" y="44" class="title">{title}</text>\n'
        f'{body}\n</svg>\n'
    )
    (OUT / name).write_text(svg, encoding="utf-8")
    print(f"wrote {OUT / name}")


def _matrix(n: int, s: int, x0: int, y0: int, filled,
            axis_note: str = "filled cell = edge j→i") -> str:
    parts = []
    for i in range(n):
        for j in range(n):
            parts.append(_cell(x0 + j * s, y0 + i * s, s,
                               FILL if filled(i, j) else EMPTY,
                               "#ffffff" if filled(i, j) else EMPTY_STROKE))
    parts.append(
        f'<text x="{x0 + n * s // 2}" y="{y0 - 16}" text-anchor="middle" '
        f'class="axis">j · key / value (src)</text>')
    parts.append(
        f'<text x="{x0 - 18}" y="{y0 + n * s // 2}" text-anchor="middle" '
        f'class="axis" transform="rotate(-90 {x0 - 18} {y0 + n * s // 2})">'
        f'i · query (dst)</text>')
    parts.append(
        f'<text x="{x0 + n * s // 2}" y="{y0 + n * s + 32}" '
        f'text-anchor="middle" class="cap">{axis_note}</text>')
    return "\n  ".join(parts)


def _notes(x: int, y: int, lines: list[tuple[str, str]]) -> str:
    parts = []
    for offset, (cls, text) in enumerate(lines):
        parts.append(f'<text x="{x}" y="{y + offset * 24}" class="{cls}">'
                     f'{text}</text>')
    return "\n  ".join(parts)


def full_attention() -> None:
    n, s, x0, y0 = 8, 28, 150, 112
    body = _matrix(n, s, x0, y0, lambda i, j: True)
    body += "\n  " + _notes(x0 + n * s + 56, y0 + 70, [
        ("body", "cost = the full N×N score matrix"),
        ("body", "nothing to mask, nothing to skip"),
    ])
    _figure(
        "full-attention.svg",
        "Full attention: every cell exists",
        "An 8 by 8 attention matrix with every cell filled: row i is the "
        "query, column j the key, and a filled cell is an edge j to i. In "
        "full attention all N squared edges exist.",
        body, 680, 420)


def causal_attention() -> None:
    n, s, x0, y0 = 8, 28, 150, 112
    body = _matrix(n, s, x0, y0, lambda i, j: j <= i)
    body += "\n  " + _notes(x0 + n * s + 56, y0 + 58, [
        ("body", "j ≤ i keeps N(N+1)/2 edges"),
        ("body", "the upper triangle is not masked —"),
        ("body", "it does not exist"),
    ])
    _figure(
        "causal-dense-relation.svg",
        "Causal: the upper triangle does not exist",
        "The same attention matrix with only the lower triangle filled, j at "
        "most i. Cells above the diagonal are absent edges, not masked-out "
        "scores.",
        body, 680, 420)


def varlen_attention() -> None:
    n, s, x0, y0 = 5, 34, 150, 128
    bounds = [(0, 3), (3, 5)]

    def filled(i: int, j: int) -> bool:
        return j <= i and any(a <= i < b and a <= j < b for a, b in bounds)

    parts = [_matrix(n, s, x0, y0, filled)]
    for seq, (a, b) in enumerate(bounds):
        xa, xb = x0 + a * s + 3, x0 + b * s - 3
        parts.append(
            f'<path d="M{xa} {y0 - 44} L{xa} {y0 - 36} L{xb} {y0 - 36} '
            f'L{xb} {y0 - 44}" fill="none" stroke="{SLATE}" '
            f'stroke-width="1.6" />')
        parts.append(
            f'<text x="{(xa + xb) // 2}" y="{y0 - 52}" text-anchor="middle" '
            f'class="axis">seq {seq}</text>')
    for k in range(n):
        parts.append(
            f'<text x="{x0 + k * s + s // 2}" y="{y0 - 66}" '
            f'text-anchor="middle" class="cap">{k}</text>')
    parts.append(_notes(x0 + n * s + 56, y0 + 34, [
        ("mono", "cu_seqlens = [0, 3, 5]"),
        ("body", "9 edges = 6 + 3"),
        ("body", "cross-sequence cells"),
        ("body", "have no edge"),
    ]))
    _figure(
        "varlen-causal-attention.svg",
        "Varlen: edges stop at sequence boundaries",
        "A 5 by 5 attention matrix for cu_seqlens 0, 3, 5: two block-diagonal "
        "causal triangles. Positions 0 to 2 attend within sequence 0, "
        "positions 3 to 4 within sequence 1, and every cross-sequence cell "
        "is absent.",
        "\n  ".join(parts), 680, 420)


def tile_pruned_attention() -> None:
    n, s, x0, y0 = 8, 28, 150, 112
    tile = 2

    def admitted(ti: int, tj: int) -> bool:
        return abs(ti - tj) <= 1

    parts = []
    for i in range(n):
        for j in range(n):
            x, y = x0 + j * s, y0 + i * s
            ti, tj = i // tile, j // tile
            if admitted(ti, tj):
                parts.append(_cell(x, y, s, FILL))
            else:
                parts.append(_cell(x, y, s, EXISTS))
    for t in range(1, n // tile):
        parts.append(
            f'<path d="M{x0 + t * tile * s} {y0} L{x0 + t * tile * s} '
            f'{y0 + n * s} M{x0} {y0 + t * tile * s} L{x0 + n * s} '
            f'{y0 + t * tile * s}" stroke="#f8fafc" stroke-width="3.5" '
            f'fill="none" />')
    for ti in range(n // tile):
        for tj in range(n // tile):
            if not admitted(ti, tj):
                parts.append(_cross(x0 + tj * tile * s, y0 + ti * tile * s,
                                    tile * s))
    parts.append(
        f'<text x="{x0 + n * s // 2}" y="{y0 - 16}" text-anchor="middle" '
        f'class="axis">j · key / value (src)</text>')
    parts.append(
        f'<text x="{x0 - 18}" y="{y0 + n * s // 2}" text-anchor="middle" '
        f'class="axis" transform="rotate(-90 {x0 - 18} {y0 + n * s // 2})">'
        f'i · query (dst)</text>')
    parts.append(
        f'<text x="{x0 + n * s // 2}" y="{y0 + n * s + 32}" '
        f'text-anchor="middle" class="cap">every cell is an edge; '
        f'tiles are admitted or skipped whole</text>')
    lx = x0 + n * s + 56
    parts.append(
        f'<rect x="{lx}" y="{y0 + 40}" width="16" height="16" rx="3" '
        f'fill="{FILL}" />'
        f'<text x="{lx + 24}" y="{y0 + 53}" class="body">admitted tile</text>')
    parts.append(
        f'<rect x="{lx}" y="{y0 + 68}" width="16" height="16" rx="3" '
        f'fill="{PRUNED}" stroke="{EMPTY_STROKE}" stroke-width="1.5" />'
        + _cross(lx, y0 + 68, 16)
        + f'<text x="{lx + 24}" y="{y0 + 81}" class="body">pruned — '
          f'never read</text>')
    parts.append(_notes(lx, y0 + 128, [
        ("body", "a tile enters the softmax only"),
        ("body", "if its best score is within τ"),
        ("body", "of the running row max"),
    ]))
    _figure(
        "tile-pruned-attention.svg",
        "Tile pruning: whole tiles skipped, never read",
        "An 8 by 8 attention matrix grouped into 2 by 2 tiles. Tiles near "
        "the diagonal are admitted in solid indigo; far tiles are crossed "
        "out and their scores and values are never read.",
        "\n  ".join(parts), 680, 420)


if __name__ == "__main__":
    full_attention()
    causal_attention()
    varlen_attention()
    tile_pruned_attention()
