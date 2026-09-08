"""Render an anisotropic Gaussian scene two ways: splats and volume.

Shapes: positions (N, 3), colors (N, 3), scales (N, 3), rotations (N, 4),
density grid (X, Y, Z), raster pixels (H*W, 3). Outputs land in
output/examples/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import tiga as gf


def torus_gaussians(count: int, seed: int = 0):
    """Colored Gaussians on a torus, stretched along the tangent."""
    rng = np.random.default_rng(seed)
    angle = rng.uniform(0.0, 2.0 * np.pi, count)  # (N,)
    tube_angle = rng.uniform(0.0, 2.0 * np.pi, count)
    tube_radius = 0.28 * np.sqrt(rng.uniform(0.0, 1.0, count))
    ring = 1.0 + tube_radius * np.cos(tube_angle)
    positions = np.stack([ring * np.cos(angle), ring * np.sin(angle),
                          tube_radius * np.sin(tube_angle)], axis=1)  # (N, 3)
    # tangent frame: (tangent, radial, vertical) — Gaussians stretch along it
    tangent = np.stack([-np.sin(angle), np.cos(angle),
                        np.zeros(count)], axis=1)
    radial = np.stack([np.cos(angle), np.sin(angle), np.zeros(count)], axis=1)
    vertical = np.broadcast_to([0.0, 0.0, 1.0], (count, 3)).copy()
    frames = np.stack([tangent, radial, vertical], axis=2)  # (N, 3, 3)
    # quaternion of each frame (columns are the rotated basis vectors)
    w = np.sqrt(1.0 + frames[:, 0, 0] + frames[:, 1, 1]
                + frames[:, 2, 2]) / 2.0
    rotations = np.stack([
        w,
        (frames[:, 2, 1] - frames[:, 1, 2]) / (4.0 * w),
        (frames[:, 0, 2] - frames[:, 2, 0]) / (4.0 * w),
        (frames[:, 1, 0] - frames[:, 0, 1]) / (4.0 * w),
    ], axis=1)  # (N, 4), (w, x, y, z)
    scales = np.stack([np.full(count, 0.09),
                       np.full(count, 0.035),
                       np.full(count, 0.035)], axis=1)  # (N, 3)
    hue = angle / (2.0 * np.pi)
    colors = np.stack([  # (N, 3), a blue → orange sweep around the ring
        0.5 + 0.5 * np.cos(2.0 * np.pi * hue),
        0.4 + 0.3 * np.sin(2.0 * np.pi * hue),
        0.5 - 0.5 * np.cos(2.0 * np.pi * hue),
    ], axis=1)
    opacities = np.full(count, 0.85)  # (N,)
    return positions, colors, scales, rotations, opacities


def splat_density(positions: np.ndarray, opacities: np.ndarray,
                  voxel: int = 48):
    """Accumulate isotropic Gaussian density into a (X, Y, Z) grid."""
    grid = np.zeros((voxel, voxel, voxel))
    half = 1.0  # the grid spans [-1, 1]^3
    for (x, y, z), opacity in zip(positions, opacities):
        center = (np.array([x, y, z]) + half) / (2.0 * half) * (voxel - 1)
        radius = max(int(3.0 * 0.09 / (2.0 * half) * (voxel - 1)), 1)
        lo = np.maximum((center - radius).astype(int), 0)
        hi = np.minimum((center + radius).astype(int) + 1, voxel)
        axes = [np.arange(lo[d], hi[d]) for d in range(3)]
        if any(len(axis) == 0 for axis in axes):
            continue
        gx, gy, gz = np.meshgrid(*axes, indexing="ij")
        distance2 = ((gx - center[0]) ** 2 + (gy - center[1]) ** 2
                     + (gz - center[2]) ** 2)
        sigma_v = 0.09 / (2.0 * half) * (voxel - 1)
        grid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] += (
            opacity * np.exp(-0.5 * distance2 / sigma_v**2))
    return grid  # (X, Y, Z)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=900)
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--output", default="output/examples")
    args = parser.parse_args()

    positions, colors, scales, rotations, opacities = torus_gaussians(
        args.count)
    output = Path(args.output)
    # --8<-- [start:core]
    camera = gf.visualize.Camera.from_angles(  # fixed elevation/azimuth view
        20, -60, positions=positions)

    gf.visualize.splats(                          # EWA Gaussian splatting
        positions, colors, scales,
        rotations=rotations, opacities=opacities, camera=camera,
        width=args.size, height=args.size,
    ).save(output / "gaussians_splats.png")

    density = splat_density(positions, opacities)  # (X, Y, Z) voxel grid
    gf.visualize.volume(                          # ray-marched density
        density, camera=camera, width=args.size, height=args.size,
        cmap="inferno", scale=3.0,
    ).save(output / "gaussians_volume.png")
    # --8<-- [end:core]

    frames = (                                    # 12-frame azimuth orbit
        gf.visualize.splats(
            positions, colors, scales,
            rotations=rotations, opacities=opacities,
            camera=gf.visualize.Camera.from_angles(
                20, azimuth, positions=positions),
            width=256, height=256)
        for azimuth in range(0, 360, 30)
    )
    gf.visualize.save_video(frames, output / "gaussians_orbit.gif", fps=6)

    # Inspect: camera.world_to_ndc(positions), raster.to_numpy()


if __name__ == "__main__":
    main()
