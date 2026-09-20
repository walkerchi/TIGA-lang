"""Render an anisotropic Gaussian scene two ways: splats and volume.

Shapes: positions (N, 3), colors (N, 3), scales (N, 3), rotations (N, 4),
density grid (X, Y, Z), raster pixels (H*W, 3). Outputs land in
output/examples/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import tiga as tg


def torus_gaussians(count: int, seed: int = 0):
    """Colored Gaussians on a torus, stretched along the tangent."""
    if count < 1:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(seed)
    # Stratified angles avoid gaps while retaining a seeded, irregular cloud.
    angle = (np.arange(count) + rng.random(count)) * (2.0 * np.pi / count)
    tube_angle = (np.arange(count) * np.pi * (3.0 - np.sqrt(5.0)) + rng.uniform(0, 2*np.pi)) % (2*np.pi)
    tube_radius = 0.14
    ring = 0.62 + tube_radius * np.cos(tube_angle)
    positions = np.stack([ring * np.cos(angle), ring * np.sin(angle),
                          tube_radius * np.sin(tube_angle)], axis=1)  # (N, 3)
    # A proper z rotation maps local x onto the torus tangent. Using
    # (tangent, outward radial, vertical) as columns would be a reflection.
    half_angle = (angle + np.pi / 2.0) / 2.0
    rotations = np.stack([np.cos(half_angle), np.zeros(count),
                          np.zeros(count), np.sin(half_angle)], axis=1)
    scales = np.broadcast_to([0.035, 0.018, 0.018], (count, 3)).copy()
    # One restrained cool-to-warm sweep; position, not density, sets this RGB.
    blend = (0.5 + 0.5 * np.cos(angle - 0.5))[:, None]
    colors = (1.0 - blend) * np.array([0.20, 0.55, 0.70]) + blend * np.array([0.93, 0.66, 0.38])
    normals = np.stack([np.cos(angle) * np.cos(tube_angle),
                        np.sin(angle) * np.cos(tube_angle), np.sin(tube_angle)], axis=1)
    light = np.array([0.3, -0.4, 0.85])
    diffuse = np.maximum(normals @ (light / np.linalg.norm(light)), 0)[:, None]
    # Bake a fixed light into RGB; the renderer itself does not shade normals.
    colors = np.clip(colors * (0.35 + 0.65 * diffuse) + 0.12 * diffuse**10, 0, 1)
    opacities = np.full(count, 0.75)
    return positions, colors, scales, rotations, opacities


def splat_density(positions: np.ndarray, opacities: np.ndarray,
                  scales: np.ndarray, voxel: int = 64):
    """Sample this scene's tangent-aligned Gaussians on the unit cube.

    The volume retains the anisotropic density, not the per-Gaussian RGB.
    Each contribution is truncated at a three-sigma bounding box.
    """
    if voxel < 2:
        raise ValueError("voxel must be at least 2")
    grid = np.zeros((voxel, voxel, voxel))
    for position, opacity, sigma in zip(positions, opacities, scales, strict=True):
        center = (position + 1.0) * 0.5 * (voxel - 1)
        radius = 3.0 * max(sigma) * 0.5 * (voxel - 1)
        lo = np.maximum(np.floor(center - radius).astype(int), 0)
        hi = np.minimum(np.ceil(center + radius).astype(int) + 1, voxel)
        axes = [np.arange(lo[d], hi[d]) for d in range(3)]
        if any(len(axis) == 0 for axis in axes):
            continue
        gx, gy, gz = np.meshgrid(*axes, indexing="ij")
        dx, dy, dz = [(axis - c) * 2.0 / (voxel - 1)
                      for axis, c in zip((gx, gy, gz), center)]
        angle = np.arctan2(position[1], position[0])
        along = -np.sin(angle) * dx + np.cos(angle) * dy
        across = np.cos(angle) * dx + np.sin(angle) * dy
        distance2 = (along / sigma[0])**2 + (across / sigma[1])**2 + (dz / sigma[2])**2
        grid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] += (
            opacity * np.exp(-0.5 * distance2))
    return grid  # (X, Y, Z)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=2200)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--frames", type=int, default=36)
    parser.add_argument("--video-format", choices=("gif", "mp4"), default="gif")
    parser.add_argument("--output", default="output/examples")
    args = parser.parse_args()
    if min(args.count, args.size, args.frames) < 1:
        parser.error("count, size and frames must be positive")

    positions, colors, scales, rotations, opacities = torus_gaussians(
        args.count)
    output = Path(args.output)
    background = (0.035, 0.048, 0.08)
    # --8<-- [start:core]
    camera = tg.visualize.Camera.from_angles(38, -55, distance=2.6)

    tg.visualize.gaussians(
        positions, colors, scales,
        rotations=rotations, opacities=opacities, camera=camera,
        width=args.size, height=args.size, background=background,
    ).save(output / "gaussians_splats.png")
    # --8<-- [end:core]

    # --8<-- [start:volume]
    density = splat_density(positions, opacities, scales)
    tg.visualize.volume(
        density, camera=camera, width=args.size, height=args.size,
        cmap=[(0.10, 0.18, 0.30), (0.20, 0.62, 0.72), (0.96, 0.83, 0.60)],
        scale=4.0, background=background,
    ).save(output / "gaussians_volume.png")
    # --8<-- [end:volume]

    frames = (
        tg.visualize.gaussians(
            positions, colors, scales,
            rotations=rotations, opacities=opacities,
            camera=tg.visualize.Camera.from_angles(
                38, azimuth, distance=2.6),
            width=384, height=384, background=background)
        for azimuth in np.linspace(-55, 305, args.frames, endpoint=False)
    )
    tg.visualize.save_video(frames, output / f"gaussians_orbit.{args.video_format}", fps=12)

    # Inspect: camera.world_to_ndc(positions), raster.to_numpy()


if __name__ == "__main__":
    main()
