"""Streaming video export for Raster/NumPy frame sequences."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from . import Raster


def _encode_frame(frame) -> "object":
    """Convert one Raster or (H, W, 3) float [0, 1] array to uint8 RGB."""
    import numpy as np

    if isinstance(frame, Raster):
        array = frame.to_numpy()
    else:
        array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("video frames must have shape (H, W, 3)")
    if array.dtype == np.uint8:
        return array
    array = np.asarray(array, dtype=np.float64)
    return np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)


def _save_gif(frames, destination: Path, fps: float) -> Path:
    try:
        from PIL import Image
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "save_video() requires tiga-lang[visualization]"
        ) from error
    images = [Image.fromarray(_encode_frame(frame), mode="RGB")
              for frame in frames]
    if not images:
        raise ValueError("save_video() requires at least one frame")
    shape = images[0].size
    if any(image.size != shape for image in images):
        raise ValueError("all video frames must share the same shape")
    images[0].save(
        destination,
        save_all=True,
        append_images=images[1:],
        duration=int(round(1000.0 / fps)),
        loop=0,
    )
    return destination


def _save_mp4(frames, destination: Path, fps: float) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "save_video() to .mp4 requires an ffmpeg executable on PATH")
    command = [
        ffmpeg, "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", None,  # placeholder, filled from the first frame
        "-r", f"{fps:g}",
        "-i", "-",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        str(destination),
    ]
    process = None
    expected = None
    try:
        for frame in frames:
            encoded = _encode_frame(frame)
            if expected is None:
                expected = encoded.shape
                height, width = encoded.shape[:2]
                command[command.index(None)] = f"{width}x{height}"
                process = subprocess.Popen(
                    command, stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            elif encoded.shape != expected:
                raise ValueError(
                    "all video frames must share the same shape")
            process.stdin.write(encoded.tobytes())
        if process is None:
            raise ValueError("save_video() requires at least one frame")
        _, stderr = process.communicate()
    except BaseException:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        raise
    if process.returncode != 0:
        raise RuntimeError(
            f"ffmpeg exited with code {process.returncode}: "
            f"{stderr.decode(errors='replace').strip()}")
    return destination


def save_video(frames, path: str | Path, *, fps: float = 30) -> Path:
    """Encode an iterable of frames (Raster or float (H, W, 3)) as a video.

    ``.gif`` is written through Pillow; ``.mp4`` streams raw RGB frames into
    an ffmpeg subprocess, so generators are consumed without buffering.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = destination.suffix.lower()
    if suffix == ".gif":
        return _save_gif(frames, destination, fps)
    if suffix == ".mp4":
        return _save_mp4(frames, destination, fps)
    raise ValueError(
        f"unsupported video format {suffix!r}; supported formats: .gif, .mp4")
