"""GPU-native raster preparation built from ordinary Tiga Tensor IR.

Rendering is intentionally a parallel product layer: it composes public
Tensor expressions and does not add workload-named operations to the compiler.
Image encoding/display dependencies are imported only when requested.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ..tensor import Tensor, tensor


Color = Sequence[float]
ColorMap = str | Sequence[Color] | Sequence[tuple[float, Color]]
_PALETTES: dict[tuple[object, ...], tuple[Tensor, ...]] = {}

# Eight-stop samples of the Matplotlib colormaps of the same name.  The stops
# are interpolated piecewise-linearly inside one fused Tensor kernel, so a
# named colormap costs a handful of extra elementwise operations per pixel and
# adds no runtime dependency.
_COLORMAPS: dict[str, tuple[tuple[float, float, float], ...]] = {
    "viridis": (
        (0.267, 0.0049, 0.3294), (0.2752, 0.1949, 0.496),
        (0.2124, 0.3597, 0.5517), (0.1534, 0.497, 0.5577),
        (0.1223, 0.6332, 0.5304), (0.2889, 0.7584, 0.4284),
        (0.6266, 0.8546, 0.2234), (0.9932, 0.9062, 0.1439),
    ),
    "magma": (
        (0.0015, 0.0005, 0.0139), (0.1351, 0.0684, 0.315),
        (0.3721, 0.0928, 0.4991), (0.5945, 0.1757, 0.5012),
        (0.8289, 0.2622, 0.4306), (0.9734, 0.4615, 0.362),
        (0.9973, 0.7335, 0.5052), (0.9871, 0.9914, 0.7495),
    ),
    "plasma": (
        (0.0504, 0.0298, 0.528), (0.3251, 0.0069, 0.6395),
        (0.5462, 0.039, 0.647), (0.7234, 0.1962, 0.539),
        (0.8598, 0.3606, 0.4069), (0.9555, 0.5331, 0.2855),
        (0.9945, 0.7409, 0.1663), (0.94, 0.9752, 0.1313),
    ),
    "inferno": (
        (0.0015, 0.0005, 0.0139), (0.1558, 0.0446, 0.3253),
        (0.3977, 0.0833, 0.4332), (0.6217, 0.1642, 0.3888),
        (0.8323, 0.2839, 0.2574), (0.9613, 0.4887, 0.0843),
        (0.9812, 0.7591, 0.1569), (0.9884, 0.9984, 0.6449),
    ),
    "jet": (
        (0.0, 0.0, 0.5), (0.0, 0.0647, 1.0),
        (0.0, 0.6451, 1.0), (0.2498, 1.0, 0.7179),
        (0.7179, 1.0, 0.2498), (1.0, 0.7269, 0.0),
        (1.0, 0.1895, 0.0), (0.5, 0.0, 0.0),
    ),
    "coolwarm": (
        (0.2298, 0.2987, 0.7537), (0.4044, 0.5346, 0.932),
        (0.6032, 0.7315, 0.9996), (0.7867, 0.8448, 0.9398),
        (0.9307, 0.8189, 0.7591), (0.9673, 0.6575, 0.5382),
        (0.8846, 0.41, 0.3225), (0.7057, 0.0156, 0.1502),
    ),
    "gray": (
        (0.0, 0.0, 0.0), (0.1412, 0.1412, 0.1412),
        (0.2863, 0.2863, 0.2863), (0.4275, 0.4275, 0.4275),
        (0.5725, 0.5725, 0.5725), (0.7137, 0.7137, 0.7137),
        (0.8588, 0.8588, 0.8588), (1.0, 1.0, 1.0),
    ),
}


def _color(value: Color, name: str) -> tuple[float, float, float]:
    result = tuple(float(component) for component in value)
    if len(result) != 3 or any(not 0.0 <= component <= 1.0 for component in result):
        raise ValueError(f"{name} must contain three components in [0, 1]")
    return result


def _resolve_cmap(
    cmap: ColorMap,
) -> tuple[tuple[float, ...], tuple[tuple[float, float, float], ...]]:
    """Normalize a colormap spec into (positions, colors)."""
    if isinstance(cmap, str):
        try:
            colors = _COLORMAPS[cmap]
        except KeyError:
            names = ", ".join(sorted(_COLORMAPS))
            raise ValueError(
                f"unknown colormap {cmap!r}; available: {names}"
            ) from None
        count = len(colors)
        return (
            tuple(index / (count - 1) for index in range(count)),
            tuple(colors),
        )
    entries = list(cmap)
    if len(entries) < 2:
        raise ValueError("a colormap needs at least two color stops")
    positioned = all(
        isinstance(entry, (list, tuple)) and len(entry) == 2
        and isinstance(entry[0], (int, float))
        and isinstance(entry[1], (list, tuple))
        for entry in entries
    )
    plain = all(
        isinstance(entry, (list, tuple)) and len(entry) == 3
        and all(isinstance(component, (int, float)) for component in entry)
        for entry in entries
    )
    if positioned and not plain:
        positions = tuple(float(entry[0]) for entry in entries)
        colors = tuple(
            _color(entry[1], f"colormap stop {index}")
            for index, entry in enumerate(entries)
        )
        if any(right <= left for left, right in zip(positions, positions[1:])):
            raise ValueError("colormap positions must be strictly increasing")
        if positions[0] < 0.0 or positions[-1] > 1.0:
            raise ValueError("colormap positions must lie in [0, 1]")
        return positions, colors
    if plain:
        colors = tuple(
            _color(entry, f"colormap stop {index}")
            for index, entry in enumerate(entries)
        )
        count = len(colors)
        return tuple(index / (count - 1) for index in range(count)), colors
    raise ValueError(
        "cmap must be a colormap name, a list of RGB colors, or a list of "
        "(position, RGB) stops"
    )


@dataclass(frozen=True)
class Raster:
    """A lazy RGB raster whose pixels remain a Tiga Tensor."""

    pixels: Tensor
    height: int
    width: int

    def realize(self) -> "Raster":
        self.pixels.realize()
        return self

    def prepare(self):
        """Return a hot callable that redraws into the same pixel storage."""
        submit = self.pixels.prepare()

        def redraw() -> "Raster":
            submit()
            return self

        return redraw

    @property
    def execution(self):
        return self.pixels.execution

    def mlir(self, *, verify: bool = False) -> str:
        return self.pixels.mlir(verify=verify)

    def generated_code(self, kind: str | None = None):
        return self.pixels.generated_code(kind)

    def to_numpy(self, *, clip: bool = True):
        """Return a host ``[height,width,3]`` float array."""
        import numpy as np

        result = self.pixels.to_numpy().reshape(self.height, self.width, 3)
        return np.clip(result, 0.0, 1.0) if clip else result

    def save(self, path: str | Path) -> Path:
        """Encode the raster as an image; Pillow is an optional dependency."""
        try:
            from PIL import Image
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "Raster.save() requires tiga-lang[visualization]"
            ) from error
        import numpy as np

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        encoded = np.rint(self.to_numpy() * 255.0).astype(np.uint8)
        Image.fromarray(encoded, mode="RGB").save(destination)
        return destination

    def show(self, **imshow_options) -> None:
        """Display through optional Matplotlib without changing compilation."""
        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "Raster.show() requires Matplotlib"
            ) from error
        plt.imshow(self.to_numpy(), **imshow_options)
        plt.axis("off")
        plt.show()


def _palette_tensors(
    key_colors: tuple[tuple[float, float, float], ...],
    *,
    dtype,
    device,
) -> tuple[Tensor, ...]:
    palette_key = (str(device), dtype.name, key_colors)
    palette = _PALETTES.get(palette_key)
    if palette is None:
        palette = tuple(
            tensor([color], dtype=dtype, device=device) for color in key_colors
        )
        _PALETTES[palette_key] = palette
    return palette


def _relu(value: Tensor) -> Tensor:
    """max(value, 0) via sqrt(x²) = |x|, staying inside basic Tensor IR."""
    return (value + (value * value).sqrt()) * 0.5


def heatmap(
    values: Tensor,
    *,
    vmin: float = 0.0,
    vmax: float = 1.0,
    low: Color | None = None,
    high: Color | None = None,
    cmap: ColorMap | None = None,
) -> Raster:
    """Compile a colormapped heatmap into one fused Tensor kernel.

    ``values`` must be a contiguous rank-two floating Tensor. By default the
    built-in ``viridis`` colormap is used. ``cmap`` selects another
    multi-stop colormap — a name from ``tg.visualize.colormaps()``, a list
    of evenly spaced RGB colors, or a list of ``(position, RGB)`` stops with
    strictly increasing positions in [0, 1]. Passing ``low``/``high``
    instead defines a plain two-color ramp (a missing end falls back to the
    matching viridis endpoint).

    With a two-color ramp, values outside ``[vmin,vmax]`` remain visible in
    the lazy Tensor and are clipped only by host image export, keeping the
    device expression differentiable. Multi-stop colormaps (including the
    default) clamp to the end colors, like Matplotlib.
    """
    if not isinstance(values, Tensor) or values.ndim != 2:
        raise TypeError("heatmap values must be a rank-two tiga.Tensor")
    if values.dtype.kind != "float":
        raise TypeError("heatmap values must use a floating-point dtype")
    if not values.is_contiguous:
        raise ValueError("heatmap currently requires contiguous values")
    if not float(vmax) > float(vmin):
        raise ValueError("vmax must be greater than vmin")
    if cmap is None and (low is not None or high is not None):
        endpoints = _COLORMAPS["viridis"]
        positions, colors = (0.0, 1.0), (
            _color(low, "low") if low is not None else endpoints[0],
            _color(high, "high") if high is not None else endpoints[-1],
        )
    else:
        positions, colors = _resolve_cmap(cmap if cmap is not None else "viridis")
    height, width = values.shape
    rows = height * width
    scalar = values.reshape(rows, 1)
    palette = _palette_tensors(colors, dtype=values.dtype, device=values.device)
    normalized = (scalar - float(vmin)) / float(vmax - vmin)
    if len(colors) == 2:
        lo, hi = palette
        pixels = (
            lo.broadcast_to((rows, 3))
            + normalized.broadcast_to((rows, 3))
            * (hi - lo).broadcast_to((rows, 3))
        )
        return Raster(pixels=pixels, height=height, width=width)

    # Multi-stop piecewise-linear interpolation expressed with tent basis
    # functions built from sqrt(x²); the whole colormap stays one fused kernel.
    first, last = positions[0], positions[-1]
    clamped = first + _relu(normalized - first)
    clamped = last - _relu(-(clamped - last))
    weights: list[Tensor] = [
        _relu((positions[1] - clamped) * (1.0 / (positions[1] - first)))
    ]
    for index in range(1, len(positions) - 1):
        left, center, right = positions[index - 1 : index + 2]
        ascending = (clamped - left) * (1.0 / (center - left))
        descending = (right - clamped) * (1.0 / (right - center))
        difference = ascending - descending
        weights.append(
            _relu((ascending + descending) * 0.5 - (difference * difference).sqrt() * 0.5)
        )
    weights.append(
        _relu((clamped - positions[-2]) * (1.0 / (last - positions[-2])))
    )
    pixels = weights[0].broadcast_to((rows, 3)) * palette[0].broadcast_to((rows, 3))
    for weight, color in zip(weights[1:], palette[1:]):
        pixels = pixels + weight.broadcast_to((rows, 3)) * color.broadcast_to((rows, 3))
    return Raster(pixels=pixels, height=height, width=width)


def colormaps() -> tuple[str, ...]:
    """Names of the built-in heatmap colormaps."""
    return tuple(sorted(_COLORMAPS))


def _apply_cmap_host(values, cmap):
    """Map a host float array to RGB in [0, 1] through a colormap spec.

    This is the host-side mirror of the fused tent-basis kernel in
    ``heatmap`` — used by renderers whose geometry work already lives on
    the host (volume ray marching).
    """
    import numpy as np

    positions, colors = _resolve_cmap(cmap)
    samples = np.clip(np.asarray(values, dtype=np.float64),
                      positions[0], positions[-1])
    stops = np.asarray(colors, dtype=np.float64)  # (K, 3)
    channels = [
        np.interp(samples, positions, stops[:, channel])
        for channel in range(3)
    ]
    return np.stack(channels, axis=-1)


from .camera import Camera
from .exchange import export_obj, export_ply, export_vdb, load_ply
from .geometry import delaunay, load_obj, mesh, particles
from .splats import gaussians, splats
from .video import save_video
from .volume import volume

__all__ = [
    "Camera",
    "Raster",
    "colormaps",
    "delaunay",
    "export_obj",
    "export_ply",
    "export_vdb",
    "gaussians",
    "heatmap",
    "load_obj",
    "load_ply",
    "mesh",
    "particles",
    "save_video",
    "splats",
    "volume",
]
