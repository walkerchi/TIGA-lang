"""GPU-native raster preparation built from ordinary GraphForge Tensor IR.

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
_PALETTES: dict[tuple[object, ...], tuple[Tensor, Tensor]] = {}


def _color(value: Color, name: str) -> tuple[float, float, float]:
    result = tuple(float(component) for component in value)
    if len(result) != 3 or any(not 0.0 <= component <= 1.0 for component in result):
        raise ValueError(f"{name} must contain three components in [0, 1]")
    return result


@dataclass(frozen=True)
class Raster:
    """A lazy RGB raster whose pixels remain a GraphForge Tensor."""

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
                "Raster.save() requires graphforge-compiler[visualization]"
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


def heatmap(
    values: Tensor,
    *,
    vmin: float = 0.0,
    vmax: float = 1.0,
    low: Color = (0.267, 0.005, 0.329),
    high: Color = (0.993, 0.906, 0.144),
) -> Raster:
    """Compile a two-color heatmap into one fused Tensor kernel.

    ``values`` must be a contiguous rank-two floating Tensor. Values outside
    ``[vmin,vmax]`` remain visible in the lazy Tensor and are clipped only by
    host image export, keeping the device expression differentiable.
    """
    if not isinstance(values, Tensor) or values.ndim != 2:
        raise TypeError("heatmap values must be a rank-two graphforge.Tensor")
    if values.dtype.kind != "float":
        raise TypeError("heatmap values must use a floating-point dtype")
    if not values.is_contiguous:
        raise ValueError("heatmap currently requires contiguous values")
    if not float(vmax) > float(vmin):
        raise ValueError("vmax must be greater than vmin")
    lower = _color(low, "low")
    upper = _color(high, "high")
    height, width = values.shape
    rows = height * width
    scalar = values.reshape(rows, 1)
    palette_key = (str(values.device), values.dtype.name, lower, upper)
    palette = _PALETTES.get(palette_key)
    if palette is None:
        palette = (
            tensor([lower], dtype=values.dtype, device=values.device),
            tensor([upper], dtype=values.dtype, device=values.device),
        )
        _PALETTES[palette_key] = palette
    lo, hi = palette
    normalized = (scalar - float(vmin)) / float(vmax - vmin)
    pixels = (
        lo.broadcast_to((rows, 3))
        + normalized.broadcast_to((rows, 3))
        * (hi - lo).broadcast_to((rows, 3))
    )
    return Raster(pixels=pixels, height=height, width=width)


__all__ = ["Raster", "heatmap"]
