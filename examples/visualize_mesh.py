"""Diffuse a bump over an icosphere simulation mesh and render it.

Shapes: positions (N, 3), faces (M, 3), field u (N,), raster pixels
(H*W, 3). Outputs land in output/examples/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import tiga as gf


class Diffusion(gf.MessagePassing):
    reducer = gf.mean()

    def edge(self, src, dst, edge):
        return src.u

    def node(self, dst, flux, rate):
        return dst.u + rate * (flux - dst.u)


def mesh_graph(faces: np.ndarray, num_vertices: int) -> gf.Graph:
    """Undirected vertex-adjacency CSR from the unique edges of ``faces``."""
    pairs = set()
    for a, b, c in faces.tolist():
        for src, dst in ((a, b), (b, c), (c, a)):
            pairs.add((src, dst))
            pairs.add((dst, src))
    by_dst = [[] for _ in range(num_vertices)]
    for src, dst in pairs:
        by_dst[dst].append(src)
    row_ptr = np.zeros(num_vertices + 1, dtype=np.int64)
    col_idx = []
    for dst, sources in enumerate(by_dst):
        col_idx.extend(sorted(sources))
        row_ptr[dst + 1] = len(col_idx)
    return gf.Graph.from_csr(
        gf.tensor(row_ptr.tolist(), dtype=gf.int64),  # (N+1,)
        gf.tensor(col_idx, dtype=gf.int64),           # (E,)
        num_src=num_vertices,
    )


def diffuse(graph: gf.Graph, u0: np.ndarray, steps: int, rate: float):
    kernel = Diffusion()
    u = gf.tensor(u0.tolist())  # (N,)
    for _ in range(steps):
        u = kernel(graph=graph, ndata={"u": u}, rate=rate)  # (N,)
        u = gf.tensor(u.to_numpy().tolist())
    return u.to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="examples/assets/icosphere.obj")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--output", default="output/examples")
    args = parser.parse_args()

    positions, faces = gf.visualize.load_obj(args.model)  # (N, 3), (M, 3)
    graph = mesh_graph(faces, len(positions))
    direction = np.array([1.0, 0.0, 0.35])  # off-axis so the orbit reads
    top = positions[np.argmax(positions @ direction)]  # bump vertex (3,)
    u0 = np.exp(-(((positions - top) ** 2).sum(axis=1)) / 0.15)  # (N,)
    u = diffuse(graph, u0, steps=args.steps, rate=0.6)
    u_render = 0.12 + u  # ambient floor keeps the far side off the background
    vmax = float(u_render.max())  # fixed range: colors stay comparable

    output = Path(args.output)
    # --8<-- [start:core]
    camera = gf.visualize.Camera.from_angles(
        25, -60, positions=positions)  # fixed elevation/azimuth view

    gf.visualize.mesh(                            # fill the given triangles
        positions, faces, values=u_render, camera=camera,
        width=args.size, height=args.size, vmin=0.0, vmax=vmax,
    ).save(output / "mesh_flat.png")
    gf.visualize.mesh(                            # same mesh as wireframe
        positions, faces, values=u_render, camera=camera,
        width=args.size, height=args.size, wireframe=True,
        vmin=0.0, vmax=vmax,
    ).save(output / "mesh_wireframe.png")
    gf.visualize.particles(                       # vertices splatted as disks
        positions, values=u_render, camera=camera,
        width=args.size, height=args.size, point_radius=2.0,
        vmin=0.0, vmax=vmax,
    ).save(output / "mesh_particles.png")

    frames = (                                    # 12-frame azimuth orbit
        gf.visualize.mesh(
            positions, faces, values=u_render,
            camera=gf.visualize.Camera.from_angles(
                25, azimuth, positions=positions),
            width=256, height=256, vmin=0.0, vmax=vmax)
        for azimuth in range(0, 360, 30)
    )
    gf.visualize.save_video(frames, output / "mesh_orbit.gif", fps=6)

    gf.visualize.export_ply(                      # Blender-ready binary dump
        output / "mesh.ply", positions, faces=faces, values=u_render)
    gf.visualize.export_obj(                      # geometry-only, load_obj-symmetric
        output / "mesh.obj", positions, faces)
    # --8<-- [end:core]

    # Inspect: camera.world_to_ndc(positions), raster.mlir()


if __name__ == "__main__":
    main()
