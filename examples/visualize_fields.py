"""Steady heat conduction on a disc: FEM-style mesh relaxation, rendered.

Shapes: positions (N, 2), temperature T (N,), boundary mask (N,),
raster pixels (H*W, 3). Outputs land in output/examples/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import tiga as gf


class HeatRelax(gf.MessagePassing):
    """One Jacobi sweep for ∇²T = 0; Dirichlet nodes stay fixed."""

    reducer = gf.mean()

    def edge(self, src, dst, edge):
        return src.T

    def node(self, dst, flux):
        # boundary=1.0 keeps the prescribed temperature; interior relaxes
        return dst.boundary * dst.T + (1.0 - dst.boundary) * flux


def disc_points(rings: int, spokes: int, seed: int = 0):
    """Jittered polar grid on the unit disc: (N, 2) positions."""
    rng = np.random.default_rng(seed)
    radius = np.linspace(0.02, 1.0, rings)  # (R,)
    angle = np.linspace(0.0, 2.0 * np.pi, spokes, endpoint=False)  # (S,)
    r, a = np.meshgrid(radius, angle, indexing="ij")
    r = r + rng.uniform(-0.3, 0.3, r.shape) * (radius[1] - radius[0])
    a = a + rng.uniform(-0.3, 0.3, a.shape) * (angle[1] - angle[0])
    return np.stack([r.ravel() * np.cos(a.ravel()),
                     r.ravel() * np.sin(a.ravel())], axis=1)  # (N, 2)


def mesh_graph(faces: np.ndarray, num_vertices: int) -> gf.Graph:
    """Undirected vertex-adjacency CSR from the unique edges of ``faces``."""
    by_dst = [set() for _ in range(num_vertices)]
    for a, b, c in faces.tolist():
        for src, dst in ((a, b), (b, c), (c, a), (b, a), (c, b), (a, c)):
            by_dst[dst].add(src)
    row_ptr = np.zeros(num_vertices + 1, dtype=np.int64)
    col_idx: list[int] = []
    for dst, sources in enumerate(by_dst):
        col_idx.extend(sorted(sources))
        row_ptr[dst + 1] = len(col_idx)
    return gf.Graph.from_csr(
        gf.tensor(row_ptr.tolist(), dtype=gf.int64),  # (N+1,)
        gf.tensor(col_idx, dtype=gf.int64),           # (E,)
        num_src=num_vertices,
    )


def relax(positions: np.ndarray, boundary: np.ndarray, steps: int):
    from matplotlib.tri import Triangulation

    triangles = Triangulation(positions[:, 0], positions[:, 1]).triangles
    graph = mesh_graph(triangles, len(positions))
    kernel = HeatRelax()
    radius = np.hypot(positions[:, 0], positions[:, 1])  # (N,)
    temperature = np.where(radius <= 0.2, 1.0, 0.0)  # hot core, cold rim
    snapshots = [temperature]
    T = gf.tensor(temperature.tolist())       # (N,)
    fixed = gf.tensor(boundary.tolist())      # (N,)
    for _ in range(steps):
        T = kernel(graph=graph, ndata={"T": T, "boundary": fixed})  # (N,)
        snapshots.append(T.to_numpy())
        T = gf.tensor(snapshots[-1].tolist())
    return snapshots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rings", type=int, default=28)
    parser.add_argument("--spokes", type=int, default=48)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--output", default="output/examples")
    args = parser.parse_args()

    positions = disc_points(args.rings, args.spokes)  # (N, 2)
    radius = np.hypot(positions[:, 0], positions[:, 1])
    boundary = ((radius <= 0.2) | (radius >= 0.95)).astype(np.float64)  # (N,)
    snapshots = relax(positions, boundary, steps=args.steps)
    output = Path(args.output)
    # --8<-- [start:core]
    gf.visualize.particles(                       # one disk per mesh node
        positions, values=snapshots[-1],
        width=args.size, height=args.size, point_radius=2.0,
        cmap="inferno",
    ).save(output / "fields_particles.png")

    frames = (                                    # generator: streamed, not buffered
        gf.visualize.particles(
            positions, values=T,
            width=256, height=256, point_radius=2.0,
            vmin=0.0, vmax=1.0, cmap="inferno")
        for T in snapshots[:: max(1, len(snapshots) // 24)]
    )
    gf.visualize.save_video(frames, output / "fields_diffusion.gif", fps=8)
    # --8<-- [end:core]

    # --8<-- [start:delaunay]
    gf.visualize.delaunay(                        # positions → faces: the FEM mesh
        positions, values=snapshots[-1],
        width=args.size, height=args.size,
        vmin=0.0, vmax=1.0, cmap="inferno",
    ).save(output / "fields_delaunay.png")
    # --8<-- [end:delaunay]

    # Inspect: raster.mlir(), gf.visualize.delaunay(...).execution


if __name__ == "__main__":
    main()
