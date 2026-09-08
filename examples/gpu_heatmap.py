"""Compile a scalar field into RGB pixels and optionally save a PNG."""

from __future__ import annotations

import argparse
import math

import tiga as gf


# --8<-- [start:core]
def scalar_field_to_rgb(size: int, device: str):
    values = [
        [0.5 + 0.5 * math.sin(x / 17.0) * math.cos(y / 23.0)
         for x in range(size)]
        for y in range(size)
    ]
    field = gf.tensor(values, dtype=gf.float32, device=device)  # (H, W)
    return gf.visualize.heatmap(field).realize()  # pixels (H*W, 3), viridis
# --8<-- [end:core]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--output", default="output/examples/heatmap.png")
    args = parser.parse_args()

    raster = scalar_field_to_rgb(args.size, args.device)
    raster.save(args.output)

    # Inspect: raster.mlir(), raster.execution, raster.pixels.expression()


if __name__ == "__main__":
    main()
