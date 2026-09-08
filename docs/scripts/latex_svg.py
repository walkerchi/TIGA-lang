"""Splice LaTeX-rendered formulas into the hand-written docs SVG diagrams.

Replaces selected <text> elements (pseudo-math like ``out[i] = Σ_j α[i,j]·v[j]``)
with real typeset math rendered by matplotlib's mathtext (Computer Modern) as
vector paths — no JavaScript, no external references, so the SVGs still work
when loaded through <img>.

Usage:

    python docs/scripts/latex_svg.py docs/assets/examples/full-attention.svg \
        table=docs/scripts/latex_table.py

The table module defines ``LATEX``: {exact text content: LaTeX source}.  Text
elements whose content is not in the table are left untouched.  The original
text's ``x``/``y`` (baseline) and ``text-anchor`` are honored; the fill color
of the original class is reused.
"""

from __future__ import annotations

import io
import sys
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.rcParams["mathtext.fontset"] = "cm"  # Computer Modern, LaTeX look

from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import MathTextParser, math_to_image

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)

_PROP = FontProperties(size=13.0)
_PARSER = MathTextParser("path")


def render_formula(latex: str, prefix: str) -> tuple[str, float, float, float]:
    """Return (inner-svg-fragment, width, height, baseline_y_from_top)."""
    buf = io.BytesIO()
    math_to_image(f"${latex}$", buf, prop=_PROP, dpi=72, format="svg")
    _width, height, depth, _glyphs, _rects = _PARSER.parse(
        f"${latex}$", dpi=72, prop=_PROP)
    mini = ET.fromstring(buf.getvalue().decode())
    width = float(mini.get("width").removesuffix("pt"))
    # Prefix every glyph id / href so multiple formulas in one host file
    # cannot collide; drop per-glyph fills so the host's fill applies.
    for element in mini.iter():
        tag = element.tag.rsplit("}", 1)[-1]
        if tag == "path" and "id" in element.attrib:
            element.attrib["id"] = f"{prefix}-{element.attrib['id']}"
        if tag == "use":
            for key in (f"{{{XLINK_NS}}}href", "href"):
                if key in element.attrib:
                    value = element.attrib[key]
                    element.attrib[key] = (
                        f"#{prefix}-{value[1:]}" if value.startswith("#")
                        else value)
        if "style" in element.attrib:
            del element.attrib["style"]
        if "fill" in element.attrib:
            del element.attrib["fill"]
    parts: list[str] = []
    # keep only the math group — the figure/background patches would
    # otherwise inherit the host fill and paint a solid bar
    for group in mini.iter(f"{{{SVG_NS}}}g"):
        if (group.get("id") or "").startswith("text_"):
            parts.append(ET.tostring(group, encoding="unicode"))
    return "".join(parts), width, float(height), float(height) - float(depth)


def latexize(path: str, table: dict[str, str]) -> int:
    tree = ET.parse(path)
    root = tree.getroot()
    # map class -> fill color so the math keeps the original ink
    style_text = "".join(
        node.text or "" for node in root.iter(f"{{{SVG_NS}}}style"))
    class_fill = {}
    import re as _re
    for match in _re.finditer(r"\.(\w+)\{[^}]*?fill:(#[0-9a-fA-F]+)", style_text):
        class_fill[match.group(1)] = match.group(2)

    replaced = 0
    for parent in root.iter():
        for child in list(parent):
            if child.tag != f"{{{SVG_NS}}}text":
                continue
            content = (child.text or "").strip()
            if content not in table:
                continue
            latex = table[content]
            fragment, width, _height, baseline = render_formula(
                latex, prefix=f"m{replaced}")
            x = float(child.get("x", "0"))
            y = float(child.get("y", "0"))
            if child.get("text-anchor") == "middle":
                x -= width / 2.0
            elif child.get("text-anchor") == "end":
                x -= width
            fill = class_fill.get(child.get("class", ""), "#334155")
            group = ET.fromstring(
                f'<g xmlns="{SVG_NS}" xmlns:xlink="{XLINK_NS}" '
                f'transform="translate({x:.2f} {y - baseline:.2f})" '
                f'fill="{fill}">{fragment}</g>')
            index = list(parent).index(child)
            parent.remove(child)
            parent.insert(index, group)
            replaced += 1
    tree.write(path, encoding="unicode", xml_declaration=False)
    return replaced


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    table: dict[str, str] = {}
    for arg in sys.argv[1:]:
        if arg.startswith("table="):
            namespace: dict[str, object] = {}
            exec(open(arg.partition("=")[2]).read(), namespace)  # noqa: S102
            table.update(namespace["LATEX"])
    for arg in sys.argv[1:]:
        if not arg.startswith("table="):
            count = latexize(arg, table)
            print(f"{arg}: {count} formula(s) typeset")


if __name__ == "__main__":
    main()
