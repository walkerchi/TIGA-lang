"""Render the canonical vector mark into docs, README and paper assets.

Run with ``uv run --no-project --with cairosvg python tools/render_brand_assets.py``.
Pass --paper ../tiga-lang-paper to refresh the report's vector logo too.
"""
import argparse
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import cairosvg
from cairosvg.surface import PDFSurface, cairo

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"


class PaperPDFSurface(PDFSurface):
    """Match the report's PDF 1.5 target without rasterizing the logo."""

    def _create_surface(self, width, height):
        surface, width, height = super()._create_surface(width, height)
        surface.restrict_to_version(cairo.PDF_VERSION_1_5)
        return surface, width, height


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", type=Path)
    args = parser.parse_args()
    source = (ASSETS / "tiga-mark.svg").read_text()
    root = ET.fromstring(source)
    ns = "{http://www.w3.org/2000/svg}"
    body = "\n".join(ET.tostring(child, encoding="unicode") for child in root
                     if child.tag not in {ns + "title", ns + "desc"})
    logo = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="380" height="128" '
        'viewBox="0 0 380 128" role="img" aria-labelledby="title desc">\n'
        '<title id="title">Tiga</title>\n'
        '<desc id="desc">Tiga: graph message-passing JIT. Two angular wings and a long lower chevron surround a geometric G.</desc>\n'
        + body + '\n<text x="147" y="91" font-family="DejaVu Sans, Arial, sans-serif" '
        'font-size="82" font-weight="700" letter-spacing="-3" fill="#746194">Tiga</text>\n</svg>\n'
    )
    (ASSETS / "tiga-logo.svg").write_text(logo)
    for name in ("tiga-mark", "tiga-logo"):
        cairosvg.svg2png(url=str(ASSETS / f"{name}.svg"),
                        write_to=str(ASSETS / f"{name}.png"), scale=4)
        for extension in ("svg", "png"):
            shutil.copy2(ASSETS / f"{name}.{extension}",
                         ROOT / "docs/assets" / f"{name}.{extension}")
    if args.paper:
        figures = args.paper / "figures"
        figures.mkdir(parents=True, exist_ok=True)
        (figures / "tiga-logo.svg").write_text(logo)
        PaperPDFSurface.convert(bytestring=logo.encode(), write_to=str(figures / "tiga-logo.pdf"))


if __name__ == "__main__":
    main()
