"""Perspective camera fitting and projection, implemented in host NumPy.

The camera is a pure host-side helper: it turns point clouds into a viewing
transform and projects world points to normalized device coordinates (NDC).
No compiler operation is involved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_TAN_HALF_DEFAULT_FOV = math.tan(math.radians(45.0) / 2.0)


def _positions_to_xyz(positions) -> "tuple[object, bool]":
    """Normalize gf.Tensor / torch.Tensor / NumPy input to an (N, 3) array."""
    import numpy as np

    from ..tensor import Tensor

    if isinstance(positions, Tensor):
        array = np.asarray(positions.to_numpy(), dtype=np.float64)
    elif type(positions).__module__.startswith("torch"):
        array = np.asarray(
            positions.detach().cpu().numpy(), dtype=np.float64)
    else:
        array = np.asarray(positions, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] not in (2, 3):
        raise ValueError("positions must have shape (N, 2) or (N, 3)")
    planar = array.shape[1] == 2
    if planar:
        array = np.column_stack([array, np.zeros(len(array))])
    return array, planar


def _fallback_up(direction: "object") -> "tuple[float, float, float]":
    """Pick an up vector that is not parallel to the view direction."""
    import numpy as np

    up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(direction, up))) > 0.99:
        up = np.array([0.0, 1.0, 0.0])
    return tuple(float(component) for component in up)


@dataclass(frozen=True)
class Camera:
    """A perspective pinhole camera described by position, target and fov."""

    position: tuple[float, float, float]
    target: tuple[float, float, float]
    fov: float = 45.0  # vertical field of view, degrees
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)

    @classmethod
    def auto(cls, positions, *, fov: float = 45.0, margin: float = 1.2) -> "Camera":
        """Fit a camera that frames ``positions`` with a bounding sphere.

        Planar (N, 2) input is viewed top-down along +z. For 3-D input the
        view direction follows the smallest-variance principal axis (the most
        informative viewpoint); an isotropic cloud falls back to the
        isometric (1, 1, 1) direction.
        """
        import numpy as np

        xyz, planar = _positions_to_xyz(positions)
        centroid = xyz.mean(axis=0)
        centered = xyz - centroid
        radius = float(np.linalg.norm(centered, axis=1).max(initial=0.0))
        radius = max(radius, 1e-12)
        distance = radius * float(margin) / math.tan(math.radians(fov) / 2.0)
        if planar:
            direction = np.array([0.0, 0.0, 1.0])
        else:
            covariance = centered.T @ centered / max(len(xyz) - 1, 1)
            eigvals, eigvecs = np.linalg.eigh(covariance)
            if eigvals[0] >= 0.95 * eigvals[2]:
                direction = np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0)
            else:
                direction = eigvecs[:, 0]
        position = centroid + direction * distance
        return cls(
            position=tuple(float(component) for component in position),
            target=tuple(float(component) for component in centroid),
            fov=float(fov),
            up=_fallback_up(direction),
        )

    @classmethod
    def from_angles(
        cls,
        elevation: float = 30.0,
        azimuth: float = -60.0,
        *,
        positions=None,
        distance: float | None = None,
        fov: float = 45.0,
    ) -> "Camera":
        """Place the camera on a sphere around the target.

        ``elevation`` and ``azimuth`` are spherical angles in degrees. When
        ``positions`` is given, the target and distance are fitted to the
        point cloud; otherwise the target is the origin and the distance
        defaults to 3.
        """
        import numpy as np

        if positions is not None:
            xyz, _ = _positions_to_xyz(positions)
            target = xyz.mean(axis=0)
            if distance is None:
                centered = xyz - target
                radius = float(
                    np.linalg.norm(centered, axis=1).max(initial=0.0))
                radius = max(radius, 1e-12)
                distance = 1.2 * radius / math.tan(math.radians(fov) / 2.0)
        else:
            target = np.zeros(3)
            if distance is None:
                distance = 3.0
        elev = math.radians(elevation)
        azim = math.radians(azimuth)
        offset = np.array([
            math.cos(elev) * math.cos(azim),
            math.cos(elev) * math.sin(azim),
            math.sin(elev),
        ])
        position = target + offset * float(distance)
        return cls(
            position=tuple(float(component) for component in position),
            target=tuple(float(component) for component in target),
            fov=float(fov),
            up=_fallback_up(offset),
        )

    def _view_basis(self) -> "tuple[object, object, object, object]":
        """Return (position, right, true_up, forward) as float64 arrays."""
        import numpy as np

        position = np.asarray(self.position, dtype=np.float64)
        target = np.asarray(self.target, dtype=np.float64)
        up = np.asarray(self.up, dtype=np.float64)
        forward = target - position
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, up)
        norm = np.linalg.norm(right)
        if norm < 1e-12:
            raise ValueError("camera up vector is parallel to the view direction")
        right /= norm
        true_up = np.cross(right, forward)
        return position, right, true_up, forward

    def world_to_view(self, points_xyz):
        """Transform world points into camera space.

        Returns an (N, 3) array whose columns are the right/up/forward
        coordinates; a positive forward component means in front of the
        camera.
        """
        import numpy as np

        points = np.asarray(points_xyz, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points_xyz must have shape (N, 3)")
        position, right, true_up, forward = self._view_basis()
        relative = points - position
        return np.column_stack(
            [relative @ right, relative @ true_up, relative @ forward])

    def world_to_ndc(self, points_xyz):
        """Project world points to NDC.

        Returns ``(ndc, depth, visible)``: ``ndc`` is an (N, 2) array with
        clip.xy / clip.w (aspect ratio is applied by the caller), ``depth``
        is the forward distance, and ``visible`` marks points in front of
        the camera (clip.w > 0).
        """
        import math as _math

        import numpy as np

        view = self.world_to_view(points_xyz)
        x, y, w = view[:, 0], view[:, 1], view[:, 2]
        focal = 1.0 / _math.tan(_math.radians(self.fov) / 2.0)
        safe_w = np.where(w > 0.0, w, 1.0)
        ndc = np.column_stack([focal * x / safe_w, focal * y / safe_w])
        return ndc, w, w > 0.0
