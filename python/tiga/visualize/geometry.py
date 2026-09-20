"""Host-side geometry rasterization feeding the Tensor color expression.

Point splatting and Delaunay triangle filling run on the host (NumPy /
Matplotlib / Pillow) and produce a scalar field; that field is wrapped as a
tg.Tensor and colored through the ordinary ``heatmap`` expression, so the
color transform stays device-side Tensor IR.
"""

from __future__ import annotations

from ..tensor import float32, tensor
from . import Color, ColorMap, Raster, heatmap
from .camera import Camera, _positions_to_xyz


def _resolve_camera(camera, positions) -> Camera:
    if isinstance(camera, Camera):
        return camera
    if isinstance(camera, str) and camera == "auto":
        return Camera.auto(positions)
    if isinstance(camera, tuple) and len(camera) == 2:
        return Camera.from_angles(*camera, positions=positions)
    raise TypeError(
        'camera must be "auto", a Camera, or an (elevation, azimuth) tuple')


def _values_to_numpy(values, count: int):
    import numpy as np

    from ..tensor import Tensor

    if isinstance(values, Tensor):
        array = np.asarray(values.to_numpy(), dtype=np.float64)
    elif type(values).__module__.startswith("torch"):
        array = np.asarray(values.detach().cpu().numpy(), dtype=np.float64)
    else:
        array = np.asarray(values, dtype=np.float64)
    array = array.reshape(-1)
    if array.shape[0] != count:
        raise ValueError(
            f"values must contain one scalar per position ({count})")
    return array


def _ndc_to_pixels(ndc, width: int, height: int):
    """Map square-NDC points to pixel centers; fov spans the image height."""
    aspect = width / height
    px = (ndc[:, 0] / aspect * 0.5 + 0.5) * (width - 1)
    py = (0.5 - ndc[:, 1] * 0.5) * (height - 1)
    return px, py


def _triangle_fill(values, triangle):
    """Mean of a triangle's finite vertex values; None when all non-finite."""
    import numpy as np

    finite = values[triangle]
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if finite.size else None


def _rasterize_triangles(corners_list, fills, width: int, height: int,
                         *, wireframe: bool):
    """Fill triangles into a scalar field via Pillow, in list order.

    Callers order ``corners_list`` so overlapping triangles composite
    correctly (e.g. painter's algorithm: far to near).
    """
    import numpy as np
    from PIL import Image, ImageDraw

    image = Image.new("F", (width, height), 0.0)
    draw = ImageDraw.Draw(image)
    for corners, fill in zip(corners_list, fills):
        if fill is None:
            continue
        if wireframe:
            draw.line(corners + [corners[0]], fill=fill, width=1)
        else:
            draw.polygon(corners, fill=fill)
    return np.asarray(image, dtype=np.float64)  # (H, W)


def _field_to_raster(field, *, vmin, vmax, low, high, cmap) -> Raster:
    import numpy as np

    field = np.nan_to_num(
        field, nan=0.0, posinf=0.0, neginf=0.0)  # non-finite renders as "no data"
    if vmin is None:
        vmin = float(field.min())
    if vmax is None:
        vmax = float(field.max())
    vmin, vmax = float(vmin), float(vmax)
    if not np.isfinite(vmin):
        vmin = 0.0
    if not np.isfinite(vmax) or not vmax > vmin:
        vmax = vmin + 1.0
    field_tensor = tensor(field.astype("float32").tolist(), dtype=float32)
    return heatmap(
        field_tensor, vmin=vmin, vmax=vmax, low=low, high=high, cmap=cmap)


def particles(
    positions,
    values=None,
    *,
    camera="auto",
    width: int = 512,
    height: int = 512,
    point_radius: float = 2.0,
    vmin: float | None = None,
    vmax: float | None = None,
    low: Color | None = None,
    high: Color | None = None,
    cmap: ColorMap | None = None,
) -> Raster:
    """Splat points as disks into a scalar field, then color it as a heatmap.

    ``values=None`` accumulates a density field (1.0 per point); otherwise
    each splat adds its scalar value. Points behind the camera and non-finite
    values (e.g. degree-0 mean reductions) are dropped. ``cmap`` selects a
    multi-stop colormap (see ``heatmap``), overriding ``low``/``high``.
    """
    import numpy as np

    xyz, _ = _positions_to_xyz(positions)
    cam = _resolve_camera(camera, positions)
    samples = (
        _values_to_numpy(values, len(xyz))
        if values is not None
        else np.ones(len(xyz))
    )
    ndc, _, visible = cam.world_to_ndc(xyz)
    visible = visible & np.isfinite(samples)
    px, py = _ndc_to_pixels(ndc, width, height)
    field = np.zeros((height, width), dtype=np.float64)
    radius = float(point_radius)
    for index in np.nonzero(visible)[0]:
        cx, cy = px[index], py[index]
        x0 = max(int(np.floor(cx - radius)), 0)
        x1 = min(int(np.ceil(cx + radius)), width - 1)
        y0 = max(int(np.floor(cy - radius)), 0)
        y1 = min(int(np.ceil(cy + radius)), height - 1)
        if x0 > x1 or y0 > y1:
            continue
        grid_y, grid_x = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
        disk = (grid_x - cx) ** 2 + (grid_y - cy) ** 2 <= radius**2
        field[y0 : y1 + 1, x0 : x1 + 1][disk] += samples[index]
    return _field_to_raster(
        field, vmin=vmin, vmax=vmax, low=low, high=high, cmap=cmap)


def delaunay(
    positions,
    values=None,
    *,
    camera="auto",
    width: int = 512,
    height: int = 512,
    vmin: float | None = None,
    vmax: float | None = None,
    low: Color | None = None,
    high: Color | None = None,
    cmap: ColorMap | None = None,
    wireframe: bool = False,
) -> Raster:
    """Triangulate positions and fill each triangle with its mean value.

    Planar (N, 2) input is triangulated directly; 3-D input is projected
    through the camera first. Triangles are rasterized host-side with Pillow
    into a scalar field that is colored through ``heatmap``. Triangles whose
    vertices are all non-finite are skipped; partially non-finite triangles
    average their finite vertices. ``cmap`` selects a multi-stop colormap,
    overriding ``low``/``high``.
    """
    try:
        from matplotlib.tri import Triangulation
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "delaunay() requires Matplotlib for triangulation; install matplotlib"
        ) from error
    try:
        from PIL import Image, ImageDraw
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "delaunay() requires tiga-lang[visualization]"
        ) from error
    import numpy as np

    xyz, planar = _positions_to_xyz(positions)
    cam = _resolve_camera(camera, positions)
    samples = (
        _values_to_numpy(values, len(xyz))
        if values is not None
        else np.ones(len(xyz))
    )
    ndc, _, visible = cam.world_to_ndc(xyz)
    if planar:
        keep = np.ones(len(xyz), dtype=bool)
        tri_xy = xyz[:, :2]
    else:
        keep = visible
        tri_xy = ndc[keep]
    px, py = _ndc_to_pixels(ndc[keep], width, height)
    kept_values = samples[keep]
    triangulation = Triangulation(tri_xy[:, 0], tri_xy[:, 1])
    corners_list = [
        list(zip(px[triangle], py[triangle]))
        for triangle in triangulation.triangles
    ]
    fills = [
        _triangle_fill(kept_values, triangle)
        for triangle in triangulation.triangles
    ]
    field = _rasterize_triangles(
        corners_list, fills, width, height, wireframe=wireframe)
    return _field_to_raster(
        field, vmin=vmin, vmax=vmax, low=low, high=high, cmap=cmap)


def mesh(
    positions,
    faces,
    values=None,
    *,
    camera="auto",
    width: int = 512,
    height: int = 512,
    wireframe: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    low: Color | None = None,
    high: Color | None = None,
    cmap: ColorMap | None = None,
) -> Raster:
    """Render a given triangle mesh — unlike ``delaunay``, which computes the
    triangulation from points, ``mesh`` takes explicit ``faces`` (e.g. a
    simulation mesh or a 3-D model loaded with ``load_obj``).

    Triangles are painted far-to-near by mean vertex depth (painter's
    algorithm) so nearer triangles occlude farther ones. Each triangle is
    filled with the mean of its finite vertex values (``values=None`` uses a
    uniform value; a triangle whose vertices are all non-finite is skipped),
    and the rasterized scalar field is colored through ``heatmap``.
    ``cmap`` selects a multi-stop colormap, overriding ``low``/``high``.
    ``wireframe=True`` draws triangle edges instead of filling. As in
    ``delaunay``, planar (N, 2) positions render directly while 3-D input is
    projected through the camera, and triangles with a vertex behind the
    camera are dropped.
    """
    try:
        from PIL import Image as _pillow_check  # noqa: F401
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "mesh() requires tiga-lang[visualization]"
        ) from error
    import numpy as np

    xyz, planar = _positions_to_xyz(positions)
    faces = np.asarray(faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape (M, 3)")
    if faces.size and (faces.min() < 0 or faces.max() >= len(xyz)):
        raise ValueError(
            f"faces index out of range for {len(xyz)} positions")
    cam = _resolve_camera(camera, positions)
    samples = (
        _values_to_numpy(values, len(xyz))
        if values is not None
        else np.ones(len(xyz))
    )
    ndc, depth, visible = cam.world_to_ndc(xyz)
    keep = np.ones(len(xyz), dtype=bool) if planar else visible
    faces = faces[keep[faces].all(axis=1)]
    px, py = _ndc_to_pixels(ndc, width, height)
    corners_list = [list(zip(px[face], py[face])) for face in faces]
    fills = [_triangle_fill(samples, face) for face in faces]
    # painter's algorithm: mean depth far-to-near so near triangles win
    order = np.argsort(-depth[faces].mean(axis=1), kind="stable")
    corners_list = [corners_list[index] for index in order]
    fills = [fills[index] for index in order]
    field = _rasterize_triangles(
        corners_list, fills, width, height, wireframe=wireframe)
    return _field_to_raster(
        field, vmin=vmin, vmax=vmax, low=low, high=high, cmap=cmap)


def load_obj(path):
    """Parse a minimal Wavefront OBJ file into ``(positions, faces)``.

    Only ``v x y z`` vertex rows and ``f`` face rows are read; face tokens may
    carry texture/normal suffixes (``f a/b/c``), only the vertex index is
    used. OBJ indices are 1-based (negative indices count back from the last
    vertex), and faces with more than three vertices are fan-triangulated.
    Returns ``positions`` as an (N, 3) float64 NumPy array and ``faces`` as an
    (M, 3) integer NumPy array. An empty file, a file without vertices or
    faces, or an out-of-range index raises ``ValueError``.
    """
    import numpy as np

    vertices = []
    triangles = []
    with open(path, "r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            parts = line.split()
            if not parts or parts[0].startswith("#"):
                continue
            if parts[0] == "v":
                if len(parts) < 4:
                    raise ValueError(
                        f"{path}:{lineno}: 'v' row needs three coordinates")
                try:
                    vertices.append(
                        [float(parts[1]), float(parts[2]), float(parts[3])])
                except ValueError as error:
                    raise ValueError(
                        f"{path}:{lineno}: invalid vertex coordinate"
                    ) from error
            elif parts[0] == "f":
                indices = []
                for token in parts[1:]:
                    try:
                        index = int(token.split("/")[0])
                    except ValueError as error:
                        raise ValueError(
                            f"{path}:{lineno}: invalid face index {token!r}"
                        ) from error
                    resolved = index - 1 if index > 0 else len(vertices) + index
                    if index == 0 or not 0 <= resolved < len(vertices):
                        raise ValueError(
                            f"{path}:{lineno}: face index {index} out of "
                            f"range for {len(vertices)} vertices")
                    indices.append(resolved)
                if len(indices) < 3:
                    raise ValueError(
                        f"{path}:{lineno}: face needs at least three vertices")
                for k in range(1, len(indices) - 1):  # fan triangulation
                    triangles.append([indices[0], indices[k], indices[k + 1]])
    if not vertices:
        raise ValueError(f"{path} contains no 'v' vertex rows")
    if not triangles:
        raise ValueError(f"{path} contains no 'f' face rows")
    positions = np.asarray(vertices, dtype=np.float64)
    return positions, np.asarray(triangles, dtype=np.int64)
