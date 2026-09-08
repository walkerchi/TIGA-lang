"""Emission-absorption volume ray marching of a density grid.

The grid ``density[i, j, k]`` spans the unit cube [-1, 1]³ with i/j/k along
x/y/z. One ray per pixel is intersected with the cube, marched in ``steps``
samples, and composited front-to-back: each sample emits the colormap color
of its (trilinearly interpolated) density and absorbs with
``alpha = 1 - exp(-density * scale * dt)``. Host NumPy, chunked per march
step so memory stays linear in the ray count.
"""

from __future__ import annotations

import math

from . import Raster, _apply_cmap_host
from .camera import Camera
from .splats import _wrap_rgb


def _grid_to_numpy(density):
    import numpy as np

    from ..tensor import Tensor

    if isinstance(density, Tensor):
        array = np.asarray(density.to_numpy(), dtype=np.float64)
    elif type(density).__module__.startswith("torch"):
        array = np.asarray(density.detach().cpu().numpy(), dtype=np.float64)
    else:
        array = np.asarray(density, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError("density must have shape (X, Y, Z)")
    if any(extent < 2 for extent in array.shape):
        raise ValueError("density needs at least 2 voxels per axis")
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def volume(
    density,
    *,
    camera="auto",
    width: int = 512,
    height: int = 512,
    steps: int = 128,
    cmap="viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    scale: float = 8.0,
    background=(0.0, 0.0, 0.0),
) -> Raster:
    """Ray-march a density grid into an RGB image.

    ``density`` (X, Y, Z) maps onto the unit cube centered at the origin.
    ``scale`` converts density into extinction (higher → more opaque);
    ``vmin``/``vmax`` normalize the density before the colormap lookup and
    default to the data range. ``camera`` accepts ``"auto"`` (frames the
    unit cube), a ``Camera``, or an ``(elevation, azimuth)`` tuple.
    """
    import numpy as np

    grid = _grid_to_numpy(density)
    if not steps > 0:
        raise ValueError("steps must be positive")
    if not float(scale) >= 0.0:
        raise ValueError("scale must be non-negative")
    background_rgb = np.clip(
        np.asarray(background, dtype=np.float64).reshape(3), 0.0, 1.0)
    if vmin is None:
        vmin = float(grid.min())
    if vmax is None:
        vmax = float(grid.max())
    if not vmax > vmin:
        vmax = vmin + 1.0

    corners = np.array(
        [[sx, sy, sz] for sx in (-1.0, 1.0)
         for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)])
    if isinstance(camera, Camera):
        cam = camera
    elif isinstance(camera, str) and camera == "auto":
        cam = Camera.auto(corners)
    elif isinstance(camera, tuple) and len(camera) == 2:
        cam = Camera.from_angles(*camera, positions=corners)
    else:
        raise TypeError(
            'camera must be "auto", a Camera, or an (elevation, azimuth) tuple')

    # Ray directions: invert the projection convention of _ndc_to_pixels.
    position, right, true_up, forward = cam._view_basis()
    focal = 1.0 / math.tan(math.radians(cam.fov) / 2.0)
    aspect = width / height
    px, py = np.meshgrid(np.arange(width), np.arange(height))
    ndc_x = ((px.ravel() + 0.5) / (width - 1) - 0.5) * 2.0 * aspect
    ndc_y = (0.5 - (py.ravel() + 0.5) / (height - 1)) * 2.0
    directions = (
        ndc_x[:, None] * (right / focal)
        + ndc_y[:, None] * (true_up / focal)
        + forward)

    # Slab intersection with the unit cube.
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / directions
    t_near = np.full(len(directions), -np.inf)
    t_far = np.full(len(directions), np.inf)
    for axis in range(3):
        t1 = (-1.0 - position[axis]) * inv[:, axis]
        t2 = (1.0 - position[axis]) * inv[:, axis]
        t_near = np.maximum(t_near, np.minimum(t1, t2))
        t_far = np.minimum(t_far, np.maximum(t1, t2))
    t_near = np.maximum(t_near, 0.0)

    image = np.zeros((len(directions), 3), dtype=np.float64)
    transmittance = np.ones(len(directions), dtype=np.float64)
    dims = np.asarray(grid.shape, dtype=np.float64) - 1.0  # (3,)
    flat = grid.ravel()
    strides = np.array([grid.shape[1] * grid.shape[2], grid.shape[2], 1])
    for step in range(steps):
        t = t_near + (t_far - t_near) * (step + 0.5) / steps
        live = t_far > t_near
        if not live.any():
            break
        points = position + t[:, None] * directions
        coords = (points + 1.0) * 0.5 * dims  # grid space, (R, 3)
        base = np.clip(np.floor(coords), 0.0, dims - 1.0)
        frac = coords - base
        corner = base.astype(np.int64)
        # Trilinear interpolation: 8 corner gathers, weighted by frac.
        sample = np.zeros(len(points))
        for bit in range(8):
            offset = np.array(
                [(bit >> 2) & 1, (bit >> 1) & 1, bit & 1], dtype=np.int64)
            weight = np.prod(
                np.where(offset == 1, frac, 1.0 - frac), axis=1)
            flat_index = ((corner + offset) * strides).sum(axis=1)
            sample += weight * flat[flat_index]
        sample = np.where(live, sample, 0.0)
        dt = np.where(live, (t_far - t_near) / steps, 0.0)
        alpha = 1.0 - np.exp(-np.maximum(sample, 0.0) * float(scale) * dt)
        emission = _apply_cmap_host(
            np.clip((sample - vmin) / (vmax - vmin), 0.0, 1.0), cmap)
        image += (transmittance * alpha)[:, None] * emission
        transmittance *= 1.0 - alpha

    image += transmittance[:, None] * background_rgb
    image = image.reshape(height, width, 3)
    return _wrap_rgb(np.clip(image, 0.0, 1.0), height, width)
