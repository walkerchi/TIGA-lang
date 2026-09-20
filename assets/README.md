# Tiga identity assets

The canonical vector source is [tiga-mark.svg](tiga-mark.svg). The flat,
single-color TG emblem places a geometric G between two short angular wings
and a long lower chevron, forming an open triangle. The emblem uses vector paths;
the G does not depend on an installed font. There are no gradients, shadows
or decorative layers.

- [tiga-logo.svg](tiga-logo.svg) / [tiga-logo.png](tiga-logo.png): emblem and `Tiga` wordmark.
- [tiga-mark.svg](tiga-mark.svg) / [tiga-mark.png](tiga-mark.png): standalone emblem.

Regenerate PNGs, the wordmark and docs copies from the canonical mark:

```bash
uv run --no-project --with cairosvg python tools/render_brand_assets.py
```

With the sibling paper project present, append `--paper ../tiga-lang-paper` to
also refresh its vector PDF and SVG. PNGs have transparent backgrounds and
4x resolution. Edit the root asset, not the generated docs or paper copies.
