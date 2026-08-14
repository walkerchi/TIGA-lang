"""Compile a scalar field into RGB pixels and optionally save a PNG."""

from __future__ import annotations

import argparse
import math

import graphforge as gf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--output", default="output/examples/heatmap.png")
    args = parser.parse_args()

    values = [
        [0.5 + 0.5 * math.sin(x / 17.0) * math.cos(y / 23.0)
         for x in range(args.size)]
        for y in range(args.size)
    ]
    field = gf.tensor(values, dtype=gf.float32, device=args.device)
    raster = gf.visualize.heatmap(field).realize()
    destination = raster.save(args.output)
    print(f"saved {destination} through {raster.execution['backend']}")
    print(raster.mlir().splitlines()[1])


if __name__ == "__main__":
    main()
