"""Blender-interop geometry export: PLY, OBJ, and (optionally) VDB.

Export is a host-side one-shot dump: positions and values arrive through the
same single-copy helpers used by rendering (``_positions_to_xyz`` /
``_values_to_numpy``), then the payload is serialized as one contiguous
buffer — no per-vertex Python loops. ``pyopenvdb`` is imported only when
``export_vdb`` is requested.
"""

from __future__ import annotations

from pathlib import Path

from . import Color, ColorMap, _apply_cmap_host, _color
from .camera import _positions_to_xyz
from .geometry import _values_to_numpy


def _ramp_to_rgb(values, low, high, cmap):
    """Normalize values to [0, 1] by data range and apply a colormap."""
    import numpy as np

    field = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    vmin, vmax = float(field.min(initial=0.0)), float(field.max(initial=0.0))
    if not np.isfinite(vmax) or not vmax > vmin:
        vmax = vmin + 1.0
    normalized = np.clip((field - vmin) / (vmax - vmin), 0.0, 1.0)
    if cmap is None and (low is not None or high is not None):
        endpoints = _COLORMAP_ENDPOINTS
        lower = np.asarray(
            _color(low, "low") if low is not None else endpoints[0],
            dtype=np.float32)
        upper = np.asarray(
            _color(high, "high") if high is not None else endpoints[1],
            dtype=np.float32)
        rgb = lower + normalized[:, None] * (upper - lower)  # (N, 3) float32
    else:
        rgb = _apply_cmap_host(
            normalized, cmap if cmap is not None else "viridis")
    return np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


_COLORMAP_ENDPOINTS = ((0.267, 0.0049, 0.3294), (0.9932, 0.9062, 0.1439))


def _validate_faces(faces, count: int):
    import numpy as np

    array = np.ascontiguousarray(faces, dtype=np.int32)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("faces must have shape (M, 3)")
    if array.size and (array.min() < 0 or array.max() >= count):
        raise ValueError(f"faces index out of range for {count} positions")
    return array


def export_ply(
    path,
    positions,
    *,
    faces=None,
    values=None,
    low: Color | None = None,
    high: Color | None = None,
    cmap: ColorMap | None = None,
) -> Path:
    """Dump positions (and optional faces/colors) as binary_little_endian PLY.

    ``positions`` accepts gf.Tensor / torch / NumPy with shape (N, 2) or
    (N, 3) — planar input gets ``z = 0``. Vertices carry ``x y z`` float32;
    when ``values`` is given, each vertex additionally carries
    ``red green blue`` uint8 (colored with ``cmap``, defaulting to the
    ``viridis`` colormap like ``heatmap``; ``low``/``high`` select a plain
    two-color ramp) plus ``scalar_value`` float32 holding the raw field so
    Blender-side shading can recolor from scratch. ``faces`` (M, 3) is written
    as a ``vertex_indices`` list element. The whole payload is assembled in
    one structured NumPy buffer and written with a single ``tofile``.
    """
    import numpy as np

    xyz, _ = _positions_to_xyz(positions)
    xyz = np.ascontiguousarray(xyz, dtype=np.float32)
    count = len(xyz)

    properties = ["property float x", "property float y", "property float z"]
    if values is None:
        dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
        vertex_block = np.empty(count, dtype=dtype)
        vertex_block["x"], vertex_block["y"], vertex_block["z"] = (
            xyz[:, 0], xyz[:, 1], xyz[:, 2])
    else:
        samples = _values_to_numpy(values, count).astype(np.float32)
        rgb = _ramp_to_rgb(samples, low, high, cmap)
        properties += [
            "property uchar red", "property uchar green", "property uchar blue",
            "property float scalar_value",
        ]
        dtype = np.dtype([
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("red", "<u1"), ("green", "<u1"), ("blue", "<u1"),
            ("scalar_value", "<f4"),
        ])
        vertex_block = np.empty(count, dtype=dtype)
        vertex_block["x"], vertex_block["y"], vertex_block["z"] = (
            xyz[:, 0], xyz[:, 1], xyz[:, 2])
        vertex_block["red"], vertex_block["green"], vertex_block["blue"] = (
            rgb[:, 0], rgb[:, 1], rgb[:, 2])
        vertex_block["scalar_value"] = samples

    lines = [
        "ply",
        "format binary_little_endian 1.0",
        "comment exported by tiga",
        f"element vertex {count}",
        *properties,
    ]
    face_block = None
    if faces is not None:
        triangles = _validate_faces(faces, count)
        lines += [
            f"element face {len(triangles)}",
            "property list uchar int vertex_indices",
        ]
        face_block = np.empty(
            len(triangles), dtype=[("count", "<u1"), ("indices", "<i4", 3)])
        face_block["count"] = 3
        face_block["indices"] = triangles
    lines.append("end_header")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "wb") as handle:
        handle.write(("\n".join(lines) + "\n").encode("ascii"))
        vertex_block.tofile(handle)
        if face_block is not None:
            face_block.tofile(handle)
    return destination


_PLY_SCALARS = {
    "char": "<i1", "int8": "<i1", "uchar": "<u1", "uint8": "<u1",
    "short": "<i2", "int16": "<i2", "ushort": "<u2", "uint16": "<u2",
    "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4",
    "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
}


def load_ply(path):
    """Parse a PLY file into ``(positions, faces, extras)``.

    ``positions`` is an (N, 3) float64 array built from the vertex ``x y z``
    properties. ``faces`` is an (M, K) int64 array from the first ``list``
    property of the face element (usually ``vertex_indices``), or ``None``
    when the file has no faces. ``extras`` maps every remaining vertex
    property name to its (N,) float64 array — this is where 3DGS payloads
    (``f_dc_*``, ``opacity``, ``scale_*``, ``rot_*``) arrive. ASCII and
    binary_little_endian formats are supported.
    """
    import numpy as np

    source = Path(path)
    with open(source, "rb") as handle:
        first = handle.readline().strip()
        if first != b"ply":
            raise ValueError(f"{source} is not a PLY file")
        fmt = None
        elements: list[dict[str, object]] = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError("PLY header is missing end_header")
            tokens = line.split()
            if not tokens:
                continue
            keyword = tokens[0].decode("ascii")
            if keyword == "format":
                fmt = tokens[1].decode("ascii")
                if fmt not in ("ascii", "binary_little_endian"):
                    raise ValueError(f"unsupported PLY format {fmt!r}")
            elif keyword == "element":
                elements.append(
                    {"name": tokens[1].decode("ascii"),
                     "count": int(tokens[2]), "properties": []})
            elif keyword == "property":
                if not elements:
                    raise ValueError("PLY property declared before element")
                properties = elements[-1]["properties"]
                if tokens[1] == b"list":
                    properties.append(
                        ("list", tokens[2].decode("ascii"),
                         tokens[3].decode("ascii"),
                         tokens[4].decode("ascii")))
                else:
                    properties.append(
                        ("scalar", tokens[1].decode("ascii"),
                         tokens[2].decode("ascii")))
            elif keyword == "end_header":
                break
        if fmt is None:
            raise ValueError("PLY header is missing the format line")
        body = handle.read().decode("ascii") if fmt == "ascii" else None
        tokens_iter = iter(body.split()) if body is not None else None

        parsed: dict[str, dict[str, object]] = {}
        for element in elements:
            name, count = element["name"], element["count"]
            scalar_props = [
                prop for prop in element["properties"] if prop[0] == "scalar"]
            list_props = [
                prop for prop in element["properties"] if prop[0] == "list"]
            columns: dict[str, object] = {}
            if fmt == "binary_little_endian":
                try:
                    if scalar_props:
                        dtype = np.dtype([
                            (prop[2], _PLY_SCALARS[prop[1]])
                            for prop in scalar_props])
                        block = np.fromfile(handle, dtype=dtype, count=count)
                        for prop in scalar_props:
                            columns[prop[2]] = block[prop[2]].astype(
                                np.float64)
                    for _, count_type, item_type, prop_name in list_props:
                        item_dtype = _PLY_SCALARS[item_type]
                        rows = []
                        for _ in range(count):
                            items = int(np.fromfile(
                                handle, dtype=_PLY_SCALARS[count_type],
                                count=1)[0])
                            rows.append(np.fromfile(
                                handle, dtype=item_dtype, count=items))
                        columns[prop_name] = rows
                except KeyError as error:
                    raise ValueError(
                        f"unsupported PLY property type {error}") from None
            if fmt == "ascii":
                rows_scalar: list[list[float]] = []
                list_columns: dict[str, list] = {
                    prop[3]: [] for prop in list_props}
                for _ in range(count):
                    rows_scalar.append(
                        [float(next(tokens_iter)) for _ in scalar_props])
                    for _, _, _, prop_name in list_props:
                        items = int(next(tokens_iter))
                        list_columns[prop_name].append(
                            [int(next(tokens_iter)) for _ in range(items)])
                for index, prop in enumerate(scalar_props):
                    columns[prop[2]] = np.array(
                        [row[index] for row in rows_scalar],
                        dtype=np.float64)
                columns.update(list_columns)
            parsed[name] = columns

    if "vertex" not in parsed:
        raise ValueError("PLY file has no vertex element")
    vertices = parsed["vertex"]
    if not all(axis in vertices for axis in ("x", "y", "z")):
        raise ValueError("PLY vertices must carry x, y, z properties")
    positions = np.column_stack(
        [np.asarray(vertices[axis], dtype=np.float64)
         for axis in ("x", "y", "z")])
    extras = {
        key: np.asarray(value, dtype=np.float64)
        for key, value in vertices.items()
        if key not in ("x", "y", "z") and not isinstance(value, list)
    }
    faces = None
    if "face" in parsed:
        face_columns = parsed["face"]
        lists = [value for value in face_columns.values()
                 if isinstance(value, list)]
        if lists:
            rows = lists[0]
            widths = {len(row) for row in rows}
            if len(widths) != 1:
                raise ValueError("PLY faces must have a uniform vertex count")
            faces = np.asarray(rows, dtype=np.int64)
    return positions, faces, extras


def export_obj(path, positions, faces) -> Path:
    """Write a text Wavefront OBJ — the exact inverse of ``load_obj``.

    ``v x y z`` rows are dumped at full float64 round-trip precision and
    ``f`` rows use 1-based indices, so ``load_obj(export_obj(...))`` restores
    the input coordinates and faces identically. Both blocks go through
    ``np.savetxt`` — no per-row Python formatting loop.
    """
    import numpy as np

    xyz, _ = _positions_to_xyz(positions)
    triangles = _validate_faces(faces, len(xyz))
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(destination, xyz, fmt="v %.17g %.17g %.17g",
               header="exported by tiga")
    with open(destination, "a", encoding="ascii") as handle:
        np.savetxt(handle, triangles + 1, fmt="f %d %d %d")
    return destination


def export_vdb(path, positions, values, *, voxel_size: float = 0.05) -> Path:
    """Rasterize points plus a scalar field into an OpenVDB dense grid.

    OpenVDB is an optional dependency — the standard PLY export covers the
    Blender interchange path with zero extra installs, so ``pyopenvdb`` is
    only required when a volumetric VDB is explicitly wanted. Without it the
    function raises ``ModuleNotFoundError`` rather than silently degrading.
    """
    try:
        import pyopenvdb as openvdb
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "export_vdb() requires the optional pyopenvdb dependency "
            "(OpenVDB); PLY export needs no dependency"
        ) from error
    import numpy as np

    xyz, _ = _positions_to_xyz(positions)
    samples = _values_to_numpy(values, len(xyz))
    indices = np.floor(xyz / float(voxel_size)).astype(np.int64)

    grid = openvdb.FloatGrid()
    grid.name = "value"
    accessor = grid.getAccessor()
    for (i, j, k), sample in zip(indices.tolist(), samples.tolist()):
        accessor.setValueOn((i, j, k), float(sample))
    grid.transform = openvdb.createLinearTransform(float(voxel_size))

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    openvdb.write(str(destination), grids=[grid])
    return destination
