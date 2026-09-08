"""Chainable tensor spill: RAM <-> disk with a Tiga-managed lifecycle.

``tensor.disk()`` writes the payload to the spill store and releases the
in-memory buffer; any later read (``tolist``/``to_numpy``/execution) lazily
reloads it.  Anonymous spills live in a per-process temporary directory and
are released when the owning tensor is collected or the process exits.
``tensor.disk(name=...)`` instead writes to a stable directory
(``TIGA_SPILL_DIR`` or ``~/.cache/tiga/spill``) so another
process can attach the same payload with ``gf.from_disk(name)``.
"""

from __future__ import annotations

import atexit
import json
import os
import shutil
import struct
import tempfile
import uuid
from pathlib import Path

_MAGIC = b"GFSPILL1\n"

_anonymous_dir: Path | None = None

_STRUCT_CODES = {
    "float16": "e", "float32": "f", "float64": "d",
    "int32": "i", "int64": "q", "bool": "?",
}


def _anonymous_store() -> Path:
    global _anonymous_dir
    if _anonymous_dir is None:
        _anonymous_dir = Path(tempfile.mkdtemp(prefix="tiga-spill-"))
        atexit.register(_cleanup_anonymous)
    return _anonymous_dir


def _named_store() -> Path:
    override = os.environ.get("TIGA_SPILL_DIR")
    path = (Path(override) if override else
            Path.home() / ".cache" / "tiga" / "spill")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cleanup_anonymous() -> None:
    if _anonymous_dir is not None:
        shutil.rmtree(_anonymous_dir, ignore_errors=True)


def _pack_values(tensor, values: list[object]) -> bytes:
    name = tensor.dtype.name
    if name in _STRUCT_CODES:
        return struct.pack(f"={len(values)}{_STRUCT_CODES[name]}", *values)
    component = "f" if name == "complex64" else "d"
    flat: list[float] = []
    for value in values:
        flat.extend((value.real, value.imag))
    return struct.pack(f"={len(flat)}{component}", *flat)


def _payload_bytes(tensor) -> bytes:
    buffer = tensor._buffer
    if (buffer is not None and tensor.is_contiguous and tensor.offset == 0
            and not getattr(buffer, "_graphforge_torch_buffer", False)):
        return buffer.read(bytes=tensor.nbytes)
    return _pack_values(tensor, tensor._read_flat())


def spill_tensor(tensor, *, name: str | None) -> None:
    """Move ``tensor``'s payload into the spill store and free its buffer."""
    if getattr(tensor, "_spill", None) is not None:
        raise RuntimeError("tensor is already spilled; reload it before respilling")
    if name is not None and (not name or "/" in name or name.startswith(".")):
        raise ValueError("spill name must be a plain file-name-safe identifier")
    tensor.realize()  # force the payload into a physical buffer
    payload = _payload_bytes(tensor)
    header = {
        "shape": list(tensor.shape),
        "dtype": tensor.dtype.name,
        "version": tensor.version,
    }
    encoded = json.dumps(header).encode()
    if name is None:
        path = _anonymous_store() / f"{uuid.uuid4().hex}.gfspill"
    else:
        path = _named_store() / f"{name}.gfspill"
    temporary = path.with_suffix(".gfspill.tmp")
    temporary.write_bytes(_MAGIC + struct.pack("=I", len(encoded)) + encoded + payload)
    temporary.replace(path)  # atomic publish for cross-process readers
    tensor._spill = {"path": path, "named": name is not None}
    tensor._buffer = None
    tensor._expr = None
    tensor.ready_event = None


def read_header(path: Path) -> dict:
    raw = path.read_bytes()
    if raw[: len(_MAGIC)] != _MAGIC:
        raise ValueError(f"{path} is not a Tiga spill file")
    (length,) = struct.unpack("=I", raw[len(_MAGIC): len(_MAGIC) + 4])
    return json.loads(raw[len(_MAGIC) + 4: len(_MAGIC) + 4 + length])


def reload_tensor(tensor) -> None:
    """Load a spilled payload back into a fresh native CPU buffer."""
    from ..runtime import Buffer, Device, Event
    from .core import _contiguous_strides

    info = tensor._spill
    raw = info["path"].read_bytes()
    (length,) = struct.unpack("=I", raw[len(_MAGIC): len(_MAGIC) + 4])
    payload = raw[len(_MAGIC) + 4 + length:]
    if len(payload) != tensor.nbytes:
        raise RuntimeError(
            f"spill payload is {len(payload)} bytes, expected {tensor.nbytes}")
    buffer = Buffer(len(payload), device="cpu", _pooled=True)
    buffer.write(payload)
    tensor._buffer = buffer
    # the payload is stored in logical (contiguous) order even for views
    tensor.strides = _contiguous_strides(tensor.shape)
    tensor.offset = 0
    tensor.device = Device.parse("cpu")
    tensor.ready_event = Event(tensor.device, _pooled=True)
    tensor._expr = None
    tensor._spill = None


def open_spill(name: str):
    """Attach a named on-disk payload as a lazily-loaded Tensor shell."""
    from .core import _DTYPES as _DTYPES_BY_NAME
    from .core import Tensor  # no cycle at call time

    path = _named_store() / f"{name}.gfspill"
    if not path.exists():
        raise FileNotFoundError(
            f"no spill named {name!r} in {_named_store()}")
    header = read_header(path)
    tensor = Tensor(
        tuple(header["shape"]),
        dtype=_DTYPES_BY_NAME[header["dtype"]],
        device="cpu",
        version=header["version"],
    )
    tensor._buffer = None  # drop the shell allocation; the payload is on disk
    tensor._spill = {"path": path, "named": True}
    return tensor


__all__ = ["open_spill", "reload_tensor", "spill_tensor"]
