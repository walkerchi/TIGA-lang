"""Regenerate non-roofline human plots from existing machine JSON."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.common.diagnostic_plotting import plot_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--root", type=Path, default=Path("output"))
    args = parser.parse_args()
    paths = args.paths or sorted(args.root.rglob("*.json"))
    for path in paths:
        if path.name in {"roofline.json", "MANIFEST.json"}:
            continue
        generated = plot_json(path)
        if generated is not None:
            print(generated)


if __name__ == "__main__":
    main()
