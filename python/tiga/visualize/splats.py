"""EWA volume splatting of anisotropic 3-D Gaussians (3DGS-style).

Each Gaussian carries a 3-D covariance Σ = R·diag(s²)·Rᵀ (quaternion +
per-axis scales). Rendering follows Zwicker et al.: Σ is projected to a
2-D screen covariance through the perspective Jacobian evaluated at the
Gaussian center, the Gaussian is evaluated inside its ``cutoff``-sigma
ellipse, and colors are alpha-composited near-to-far with front-to-back
transmittance. All of it is host NumPy — one bounding-box block per
Gaussian, no per-pixel Python work.
"""

from __future__ import annotations

import math

from ..tensor import float32, tensor
from . import Raster
from .camera import Camera
from .geometry import _ndc_to_pixels, _positions_to_xyz, _resolve_camera


def _quaternion_to_matrix(quaternions):
    """Normalize (N, 4) ``(w, x, y, z)`` quaternions to (N, 3, 3) rotations."""
    import numpy as np

    quat = np.asarray(quaternions, dtype=np.float64)
    if quat.ndim != 2 or quat.shape[1] != 4:
        raise ValueError("rotations must have shape (N, 4)")
    norms = np.linalg.norm(quat, axis=1)
    if (norms <= 0.0).any():
        raise ValueError("rotation quaternions must be nonzero")
    w, x, y, z = (quat / norms[:, None]).T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
                  2 * (x * z + y * w)], axis=-1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z),
                  2 * (y * z - x * w)], axis=-1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w),
                  1 - 2 * (x * x + y * y)], axis=-1),
    ], axis=1)  # (N, 3, 3)


def _wrap_rgb(image, height: int, width: int) -> Raster:
    pixels = tensor(
        image.reshape(height * width, 3).astype("float32").tolist(),
        dtype=float32)
    return Raster(pixels=pixels, height=height, width=width)


def gaussians(
    positions,
    colors,
    scales,
    *,
    rotations=None,
    opacities=None,
    camera="auto",
    width: int = 512,
    height: int = 512,
    cutoff: float = 3.0,
    background=(0.0, 0.0, 0.0),
) -> Raster:
    """Alpha-composite anisotropic 3-D Gaussians into an image.

    ``positions`` (N, 3) are Gaussian centers; ``colors`` (N, 3) RGB in
    [0, 1]; ``scales`` (N, 3) per-axis standard deviations; ``rotations``
    (N, 4) ``(w, x, y, z)`` quaternions (default: isotropic); ``opacities``
    (N,) in [0, 1] (default: 1). Gaussians behind the camera, with
    zero-sized projections, or fainter than 1/255 inside their cutoff
    ellipse are skipped. Pixels never touched by a Gaussian show
    ``background``.
    """
    import numpy as np

    xyz, _ = _positions_to_xyz(positions)
    count = len(xyz)
    rgb = np.asarray(colors, dtype=np.float64)
    if rgb.shape != (count, 3):
        raise ValueError("colors must have shape (N, 3)")
    rgb = np.clip(rgb, 0.0, 1.0)
    sigma = np.asarray(scales, dtype=np.float64)
    if sigma.ndim == 1 and sigma.shape[0] == count:
        sigma = np.repeat(sigma[:, None], 3, axis=1)  # isotropic shorthand
    if sigma.shape != (count, 3) or (sigma <= 0.0).any():
        raise ValueError("scales must be positive with shape (N, 3) or (N,)")
    if rotations is None:
        orientations = np.broadcast_to(np.eye(3), (count, 3, 3)).copy()
    else:
        orientations = _quaternion_to_matrix(rotations)
        if orientations.shape[0] != count:
            raise ValueError("rotations must match the number of positions")
    if opacities is None:
        alpha0 = np.ones(count)
    else:
        alpha0 = np.asarray(opacities, dtype=np.float64).reshape(-1)
        if alpha0.shape[0] != count:
            raise ValueError("opacities must have shape (N,)")
        alpha0 = np.clip(alpha0, 0.0, 1.0)
    if not cutoff > 0.0:
        raise ValueError("cutoff must be positive")
    background_rgb = np.clip(
        np.asarray(background, dtype=np.float64).reshape(3), 0.0, 1.0)

    cam = _resolve_camera(camera, xyz)
    position, right, true_up, forward = cam._view_basis()
    view = cam.world_to_view(xyz)  # (N, 3): right, up, forward coordinates
    depth = view[:, 2]
    alive = depth > 1e-6

    # World covariance → view covariance: Σ_v = (B R) diag(s²) (B R)ᵀ with
    # B the world→view rotation (rows: right, up, forward).
    basis = np.stack([right, true_up, forward])  # (3, 3)
    transform = basis[None] @ orientations  # (N, 3, 3)
    scaled = transform * (sigma**2)[:, None, :]
    covariance = scaled @ transform.transpose(0, 2, 1)  # (N, 3, 3)

    # EWA projection through the perspective Jacobian at each center.
    focal = 1.0 / math.tan(math.radians(cam.fov) / 2.0)
    x, y, w = view[:, 0], view[:, 1], np.maximum(depth, 1e-6)
    jacobian = np.zeros((count, 2, 3))
    jacobian[:, 0, 0] = focal / w
    jacobian[:, 1, 1] = focal / w
    jacobian[:, 0, 2] = -focal * x / w**2
    jacobian[:, 1, 2] = -focal * y / w**2
    cov2d = jacobian @ covariance @ jacobian.transpose(0, 2, 1)  # (N, 2, 2)
    # Convert from NDC units to pixel units (fov spans the image height).
    aspect = width / height
    scale_x = 0.5 * (width - 1) / aspect
    scale_y = 0.5 * (height - 1)
    cov2d[:, 0, 0] *= scale_x**2
    cov2d[:, 0, 1] *= scale_x * scale_y
    cov2d[:, 1, 0] *= scale_x * scale_y
    cov2d[:, 1, 1] *= scale_y**2
    cov2d += np.eye(2) * 1e-6  # keeps the inverse finite for thin splats

    a, b, d = cov2d[:, 0, 0], cov2d[:, 0, 1], cov2d[:, 1, 1]
    determinant = a * d - b * b
    alive &= determinant > 1e-12
    trace = a + d
    root = np.sqrt(np.maximum((a - d) ** 2 / 4.0 + b * b, 0.0))
    lambda_max = trace / 2.0 + root
    radius = cutoff * np.sqrt(np.maximum(lambda_max, 0.0))
    alive &= radius >= 0.3  # sub-pixel splats are invisible

    # NDC of the center, then pixel coordinates (fov spans image height).
    safe_w = np.maximum(depth, 1e-6)
    ndc = np.column_stack([focal * x / safe_w, focal * y / safe_w])
    px, py = _ndc_to_pixels(ndc, width, height)

    inv_det = np.where(alive, 1.0 / np.maximum(determinant, 1e-12), 0.0)
    o_xx, o_xy, o_yy = d * inv_det, -b * inv_det, a * inv_det

    image = np.zeros((height, width, 3), dtype=np.float64)
    transmittance = np.ones((height, width), dtype=np.float64)
    order = np.argsort(depth, kind="stable")  # near → far: C += T·α·c, T *= 1-α
    for index in order:
        if not alive[index] or alpha0[index] <= 0.0:
            continue
        cx, cy, extent = px[index], py[index], radius[index]
        x0, x1 = max(int(cx - extent), 0), min(int(cx + extent) + 1, width)
        y0, y1 = max(int(cy - extent), 0), min(int(cy + extent) + 1, height)
        if x0 >= x1 or y0 >= y1:
            continue
        grid_x, grid_y = np.meshgrid(
            np.arange(x0, x1) - cx, np.arange(y0, y1) - cy)
        power = -0.5 * (
            o_xx[index] * grid_x**2
            + 2.0 * o_xy[index] * grid_x * grid_y
            + o_yy[index] * grid_y**2)
        alpha = alpha0[index] * np.exp(np.clip(power, None, 0.0))
        alpha = np.minimum(alpha, 0.99)
        alpha[alpha < 1.0 / 255.0] = 0.0
        block = (slice(y0, y1), slice(x0, x1))
        contribution = (
            transmittance[block] * alpha)[:, :, None] * rgb[index]
        image[block] += contribution
        transmittance[block] *= 1.0 - alpha

    image += transmittance[:, :, None] * background_rgb
    return _wrap_rgb(np.clip(image, 0.0, 1.0), height, width)


# Compatibility for pre-release callers; new code uses visualize.gaussians.
splats = gaussians
