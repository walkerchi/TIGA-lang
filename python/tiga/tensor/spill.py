"""Bounded temporary spill and atomic Tensor value snapshots.

Residency preserves logical device and autograd. Open snapshots retain their
inode so replacement cannot change an attached value. Persistence stores values,
not the computation graph.
"""
from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile
import uuid

from ..runtime.storage_io import CHUNK_BYTES, file_to_buffer

_MAGIC = b"GFSPILL1\n"
_anonymous_dir = None


def _anonymous_store():
    from ..runtime.memory import current_execution
    active = current_execution()
    if active is not None and active.spill_dir is not None:
        active.spill_dir.mkdir(parents=True, exist_ok=True)
        return active.spill_dir
    global _anonymous_dir
    if _anonymous_dir is None:
        _anonymous_dir = Path(tempfile.mkdtemp(prefix="tiga-spill-"))
        atexit.register(shutil.rmtree, _anonymous_dir, ignore_errors=True)
    return _anonymous_dir


def _named_store():
    path = Path(os.environ.get("TIGA_SPILL_DIR", Path.home() / ".cache/tiga/spill"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _validate_name(name):
    if not isinstance(name, str) or not name or name.startswith(".") or any(c in name for c in ("/", "\\", "\x00")):
        raise ValueError("spill name must be a plain file-name-safe identifier")


def _chunks(tensor):
    from ..runtime import Buffer
    from .core import _logical_offsets
    tensor.realize()
    if tensor.ready_event is not None:
        tensor.ready_event.wait()
    source = tensor._buffer
    if getattr(source, "_tiga_torch_buffer", False):
        source = Buffer.wrap_address(source.address, source.nbytes, device=source.device, owner=source)
    width = tensor.dtype.itemsize
    if tensor.is_contiguous:
        for start in range(0, tensor.nbytes, CHUNK_BYTES):
            yield source.read(offset=tensor.offset * width + start,
                              bytes=min(CHUNK_BYTES, tensor.nbytes - start))
    else:
        chunk = bytearray()
        for offset in _logical_offsets(tensor.shape, tensor.strides):
            chunk.extend(source.read(offset=(tensor.offset + offset) * width, bytes=width))
            if len(chunk) >= CHUNK_BYTES:
                yield chunk
                chunk = bytearray()
        if chunk:
            yield chunk


def save_tensor(tensor, path, *, overwrite=False):
    """Save a value snapshot without evicting or discarding gradient history."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and target.exists():
        raise FileExistsError(target)
    header = json.dumps({"shape": list(tensor.shape), "dtype": tensor.dtype.name,
                         "version": tensor.version}).encode()
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(_MAGIC + struct.pack("=I", len(header)) + header)
            for chunk in _chunks(tensor):
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, target)
        else:
            os.link(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target


def _header(stream):
    if stream.read(len(_MAGIC)) != _MAGIC:
        raise ValueError("not a Tiga Tensor snapshot")
    length_bytes = stream.read(4)
    if len(length_bytes) != 4:
        raise ValueError("truncated Tensor snapshot header")
    length = struct.unpack("=I", length_bytes)[0]
    if length > 65536:
        raise ValueError("Tensor snapshot header is too large")
    try:
        result = json.loads(stream.read(length))
        from .core import _DTYPES, _numel
        shape = result["shape"]
        if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError("invalid snapshot shape")
        if type(result["version"]) is not int or result["version"] < 0:
            raise ValueError("invalid snapshot version")
        size = _numel(tuple(shape)) * _DTYPES[result["dtype"]].itemsize
        if os.fstat(stream.fileno()).st_size != stream.tell() + size:
            raise ValueError("Tensor snapshot payload length mismatch")
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid Tensor snapshot metadata") from exc
    return result, stream.tell()


def read_header(path):
    with Path(path).open("rb") as stream:
        return _header(stream)[0]


def _attachment(path, *, named, reservation=None):
    stream = Path(path).open("rb")
    try:
        header, offset = _header(stream)
    except BaseException:
        stream.close()
        raise
    return header, {"path": Path(path), "named": named, "stream": stream,
                    "offset": offset, "reservation": reservation}


def release_spill(info):
    info["stream"].close()
    if not info["named"]:
        info["path"].unlink(missing_ok=True)
    if info.get("reservation") is not None:
        info["reservation"].close()


def spill_tensor(tensor, *, name=None):
    if tensor._spill is not None:
        raise RuntimeError("tensor is already spilled; reload it before respilling")
    if name is not None:
        _validate_name(name)
    from ..runtime.memory import reserve
    reservation = reserve("nvme", tensor.nbytes) if name is None else None
    path = (_anonymous_store() / f"{uuid.uuid4().hex}.gfspill" if name is None
            else _named_store() / f"{name}.gfspill")
    written = False
    try:
        save_tensor(tensor, path)
        written = True
        _, info = _attachment(path, named=name is not None, reservation=reservation)
    except BaseException:
        if reservation is not None:
            reservation.close()
        if written and name is None:
            path.unlink(missing_ok=True)
        raise
    tensor._spill = info
    tensor._buffer = None
    tensor.ready_event = None
    tensor._prepared_launch = None


def reload_tensor(tensor):
    from ..runtime import Buffer, Event, DeviceType
    from .core import _contiguous_strides
    info = tensor._spill
    buffer = Buffer(tensor.nbytes, device=tensor.device, _pooled=True)
    try:
        info["stream"].seek(info["offset"])
        file_to_buffer(info["stream"], buffer, tensor.nbytes)
    except BaseException:
        buffer.close()
        raise
    tensor._buffer = buffer
    tensor.strides = _contiguous_strides(tensor.shape)
    tensor.offset = 0
    tensor.ready_event = Event(tensor.device, _pooled=True) if tensor.device.type == DeviceType.CPU else None
    tensor._spill = None
    release_spill(info)
    from ..runtime.memory import current_execution
    active = current_execution()
    if active is not None:
        active._restores += 1


def load_tensor(path, *, device="cpu"):
    from .core import Tensor, _DTYPES
    header, info = _attachment(path, named=True)
    try:
        tensor = Tensor(tuple(header["shape"]), dtype=_DTYPES[header["dtype"]],
                        device=device, version=header["version"], _allocate=False)
    except BaseException:
        release_spill(info)
        raise
    tensor._spill = info
    return tensor


def open_spill(name):
    _validate_name(name)
    return load_tensor(_named_store() / f"{name}.gfspill")


def copy_tensor(tensor, device):
    from ..runtime import Buffer
    from .core import Tensor, _Expr
    buffer = Buffer(tensor.nbytes, device=device)
    try:
        if tensor._spill is not None:
            info = tensor._spill
            info["stream"].seek(info["offset"])
            file_to_buffer(info["stream"], buffer, tensor.nbytes)
        else:
            offset = 0
            for chunk in _chunks(tensor):
                buffer.write(chunk, offset=offset)
                offset += len(chunk)
    except BaseException:
        buffer.close()
        raise
    return Tensor(tensor.shape, dtype=tensor.dtype, device=device, buffer=buffer,
                  requires_grad=tensor.requires_grad, version=tensor.version,
                  expression=_Expr("device_copy", (tensor,)))
