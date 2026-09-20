"""Versioned, framework-free persistent storage for paged CSR relations."""

from __future__ import annotations

import ctypes
import json
import math
import mmap
import os
from pathlib import Path
import shutil
import struct
import tempfile
from typing import Iterator, Mapping, Sequence

from ..tensor import DType, Tensor, int32, int64, tensor
from ..tensor.core import _DTYPES as _DTYPES_BY_NAME


_FORMAT = "tiga.gfg.csr.v1"
# v2 adds an optional ``fields`` manifest section: each node/edge field is one
# fixed-row-width binary payload under ``fields/`` next to the CSR indices.
_FORMAT_V2 = "tiga.gfg.csr.v2"
_FORMATS = {_FORMAT, _FORMAT_V2}
_STRUCT = {"int32": struct.Struct("<i"), "int64": struct.Struct("<q")}

_FIELD_ROLES = ("src", "dst", "edge")
_FIELD_STRUCT_CODES = {
    "float16": "e", "float32": "f", "float64": "d",
    "int32": "i", "int64": "q", "bool": "?",
}
_FIELD_COMPLEX_CODES = {"complex64": "f", "complex128": "d"}

try:  # NumPy is optional; it accelerates random gathers and grad scatter-adds.
    import numpy as _np
except ImportError:  # pragma: no cover - exercised on numpy-free installs
    _np = None


def _field_codec(dtype: DType) -> tuple[str, int]:
    """Return the little-endian struct code and components per logical value."""
    if dtype.name in _FIELD_STRUCT_CODES:
        return _FIELD_STRUCT_CODES[dtype.name], 1
    if dtype.name in _FIELD_COMPLEX_CODES:
        return _FIELD_COMPLEX_CODES[dtype.name], 2
    raise ValueError(f"unsupported persistent field dtype {dtype.name!r}")


def _as_components(values: Sequence[object], components: int) -> list[object]:
    if components == 1:
        return list(values)
    flat: list[float] = []
    for value in values:
        pair = complex(value)  # type: ignore[arg-type]
        flat.extend((pair.real, pair.imag))
    return flat


def _from_components(values: Sequence[object], components: int) -> list[object]:
    if components == 1:
        return list(values)
    return [
        complex(values[index], values[index + 1])
        for index in range(0, len(values), 2)
    ]


def _pread_into(fd: int, address: int, nbytes: int, offset: int) -> None:
    """Positional read straight into a caller-owned address (zero-copy IO)."""
    if not nbytes:
        return
    libc = ctypes.CDLL(None, use_errno=True)
    done = 0
    while done < nbytes:
        got = libc.pread(
            ctypes.c_int(fd),
            ctypes.c_void_p(address + done),
            ctypes.c_size_t(nbytes - done),
            ctypes.c_longlong(offset + done),
        )
        if got < 0:
            raise OSError(ctypes.get_errno(), "persistent field pread failed")
        if got == 0:
            raise RuntimeError("short persistent field read")
        done += got


def _buffer_view(address: int, nbytes: int):
    """A writable buffer-protocol view over a caller-owned address range."""
    return (ctypes.c_char * max(nbytes, 1)).from_address(address)


class PagedField:
    """mmap-backed fixed-row field payload with bounded random row reads.

    Contiguous row ranges stream through positional ``pread`` directly into
    the paged executor's reused staging slabs — page-cache pages are never
    mapped into the process, so streamed volume does not accumulate in RSS.
    Random source gathers go through the read-only mmap (NumPy ``take`` when
    available, else a row-wise copy), so the OS page cache — not the process
    heap — owns residency for irregular power-law source access.  Decoding to
    Python values only happens for an explicit full-field materialization.
    """

    def __init__(self, path: Path, shape: tuple[int, ...], dtype: DType) -> None:
        self.path = path
        self.shape = shape
        self.dtype = dtype
        self.rows = shape[0]
        self.row_width = math.prod(shape[1:])
        self._code, self._components = _field_codec(dtype)
        width = self.row_width * self._components
        self._row_struct = struct.Struct(f"<{width}{self._code}")
        self.row_bytes = self._row_struct.size
        self._fd = os.open(path, os.O_RDONLY)
        self._mm: mmap.mmap | None = None
        self._np = None
        if self.rows and self.row_bytes:
            self._mm = mmap.mmap(self._fd, 0, access=mmap.ACCESS_READ)
            if _np is not None:
                self._np = _np.frombuffer(self._mm, dtype=dtype.name).reshape(shape)

    def hint(self, begin: int, end: int) -> None:
        """Ask the OS to readahead one contiguous row range (prefetch aid)."""
        if begin == end or not hasattr(os, "posix_fadvise"):
            return
        os.posix_fadvise(
            self._fd,
            begin * self.row_bytes,
            (end - begin) * self.row_bytes,
            os.POSIX_FADV_WILLNEED,
        )

    def read_into(self, begin: int, end: int, address: int) -> None:
        """pread one contiguous row range into ``address`` (caller-owned)."""
        if not 0 <= begin <= end <= self.rows:
            raise ValueError("paged field row range is outside the field")
        _pread_into(
            self._fd, address, (end - begin) * self.row_bytes,
            begin * self.row_bytes)

    def gather_into(self, indices: Sequence[int], address: int) -> None:
        """Gather random rows in index order into ``address`` (caller-owned)."""
        if not indices:
            return
        if self._mm is None:
            raise IndexError("gather from an empty persistent field")
        if self._np is not None:
            destination = _np.frombuffer(
                _buffer_view(address, len(indices) * self.row_bytes),
                dtype=self.dtype.name,
            ).reshape(len(indices), *self.shape[1:])
            _np.take(self._np, list(indices), axis=0, out=destination)
            return
        destination = memoryview(
            _buffer_view(address, len(indices) * self.row_bytes)).cast("B")
        mm = self._mm
        row_bytes = self.row_bytes
        rows = self.rows
        cursor = 0
        for index in indices:
            if index < 0 or index >= rows:
                raise IndexError("persistent field gather index is out of range")
            destination[cursor:cursor + row_bytes] = \
                mm[index * row_bytes:(index + 1) * row_bytes]
            cursor += row_bytes

    def read_rows(self, begin: int, end: int) -> list[object]:
        """Full materialization path: decode one row range to Python values."""
        if not 0 <= begin <= end <= self.rows:
            raise ValueError("paged field row range is outside the field")
        if begin == end:
            return []
        payload = os.pread(
            self._fd, (end - begin) * self.row_bytes, begin * self.row_bytes)
        if len(payload) != (end - begin) * self.row_bytes:
            raise RuntimeError("short persistent field read")
        count = len(payload) // (self.dtype.itemsize // self._components)
        flat = struct.unpack(f"<{count}{self._code}", payload)
        return _from_components(flat, self._components)

    def close(self) -> None:
        self._np = None  # dropping the view releases its hold on the mmap
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class PagedGradFile:
    """Zero-initialized disk row file for scatter-accumulated gradients.

    Source adjoints from different destination pages overlap on shared source
    rows, so they accumulate with read-modify-write row updates through a
    writable mapping (a NumPy memmap when available, else a pure-Python mmap
    row loop).  The file is created sparse (truncated, therefore zeroed) and
    only ever written by the main thread.
    """

    def __init__(self, path: Path, shape: tuple[int, ...], dtype: DType) -> None:
        self.path = path
        self.shape = shape
        self.dtype = dtype
        self.rows = shape[0]
        self.row_width = math.prod(shape[1:])
        self._code, self._components = _field_codec(dtype)
        width = self.row_width * self._components
        self._row_struct = struct.Struct(f"<{width}{self._code}")
        self.row_bytes = self._row_struct.size
        self._fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)
        nbytes = self.rows * self.row_bytes
        os.ftruncate(self._fd, nbytes)  # sparse file: reads back as zeros
        self._mm: mmap.mmap | None = None
        self._np = None
        if nbytes:
            self._mm = mmap.mmap(self._fd, nbytes, access=mmap.ACCESS_WRITE)
            if _np is not None:
                self._np = _np.frombuffer(self._mm, dtype=dtype.name).reshape(shape)

    def add_rows(self, indices: Sequence[int], payload: bytes) -> None:
        """Accumulate raw little-endian rows onto rows selected by ``indices``."""
        if not indices:
            return
        if len(payload) != len(indices) * self.row_bytes:
            raise ValueError("gradient row payload does not match its indices")
        if self._np is not None:
            delta = _np.frombuffer(payload, dtype=self.dtype.name).reshape(
                len(indices), *self.shape[1:])
            _np.add.at(self._np, list(indices), delta)
            return
        if self._mm is None:
            raise RuntimeError("gradient accumulation into an empty grad file")
        unpack = self._row_struct.unpack_from
        pack = self._row_struct.pack_into
        mm = self._mm
        row_bytes = self.row_bytes
        rows = self.rows
        cursor = 0
        for index in indices:
            if index < 0 or index >= rows:
                raise IndexError("gradient scatter index is out of range")
            offset = index * row_bytes
            current = unpack(mm, offset)
            delta = unpack(payload, cursor)
            cursor += row_bytes
            pack(mm, offset, *(left + right for left, right in zip(current, delta)))

    def close(self) -> None:
        self._np = None
        if self._mm is not None:
            self._mm.flush()
            self._mm.close()
            self._mm = None
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _flatten(values) -> list[int]:
    result = values.tolist()
    return [int(value) for value in result]


def _write_indices(path: Path, values: Sequence[int], dtype: DType) -> None:
    item = _STRUCT[dtype.name]
    with path.open("wb") as stream:
        chunk = bytearray()
        for value in values:
            chunk += item.pack(int(value))
            if len(chunk) >= 8 * 1024 * 1024:
                stream.write(chunk)
                chunk.clear()
        if chunk:
            stream.write(chunk)


def _field_payload_bytes(value: Tensor) -> bytes:
    buffer = value._buffer
    if (buffer is not None and value.is_contiguous and value.offset == 0
            and not getattr(buffer, "_tiga_torch_buffer", False)):
        return buffer.read(bytes=value.nbytes)
    code, components = _field_codec(value.dtype)
    flat = _as_components(value._read_flat(), components)
    return struct.pack(f"<{len(flat)}{code}", *flat)


def _validate_field_bindings(graph, fields: Mapping[str, Mapping[str, Tensor]]) -> None:
    if not isinstance(fields, Mapping):
        raise TypeError("fields must map 'src'/'dst'/'edge' to field mappings")
    expected = {
        "src": graph.schema.num_src,
        "dst": graph.schema.num_dst,
        "edge": graph.num_edges,
    }
    for role, mapping in fields.items():
        if role not in _FIELD_ROLES:
            raise ValueError(f"unknown field role {role!r}")
        if not isinstance(mapping, Mapping):
            raise TypeError(f"fields[{role!r}] must be a mapping of Tensors")
        for name, value in mapping.items():
            if (not isinstance(name, str) or not name or "/" in name
                    or "\\" in name or name.startswith(".")):
                raise ValueError(f"invalid persistent field name {name!r}")
            if not isinstance(value, Tensor):
                raise TypeError(f"fields[{role!r}][{name!r}] must be a tiga.Tensor")
            if value.ndim == 0 or value.shape[0] != expected[role]:
                raise ValueError(
                    f"fields[{role!r}][{name!r}] leading dimension must be "
                    f"{expected[role]}, got {value.shape}")
            _field_codec(value.dtype)  # reject unsupported dtypes early


def _write_fields(
    root: Path, fields: Mapping[str, Mapping[str, Tensor]]
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    if not fields:
        return entries
    (root / "fields").mkdir(exist_ok=True)
    for role in _FIELD_ROLES:
        for name, value in (fields.get(role) or {}).items():
            value.realize()  # spill shells and disk-backed fields reload here
            relative = f"fields/{role}.{name}.bin"
            payload = _field_payload_bytes(value)
            if len(payload) != value.nbytes:
                raise RuntimeError(f"field {role}.{name} payload size mismatch")
            (root / relative).write_bytes(payload)
            entries.append({
                "role": role,
                "name": name,
                "shape": list(value.shape),
                "dtype": value.dtype.name,
                "file": relative,
            })
    return entries


class PagedCSRStore:
    """Small manifest handle over CSR indices resident on a filesystem.

    Reads are explicit destination-row ranges. No method used by distributed
    execution constructs the global ``col_idx`` array in RAM.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        root = Path(path).expanduser().resolve()
        manifest_path = root / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise FileNotFoundError(f"not a Tiga graph: {root}") from error
        if manifest.get("format") not in _FORMATS:
            raise ValueError(f"unsupported Tiga graph format in {manifest_path}")
        required = {"num_src", "num_dst", "num_edges", "index_dtype"}
        if not required <= manifest.keys():
            raise ValueError("Tiga graph manifest is incomplete")
        if manifest["index_dtype"] not in _STRUCT:
            raise ValueError("persistent CSR index_dtype must be int32 or int64")
        self.path = root
        self.num_src = int(manifest["num_src"])
        self.num_dst = int(manifest["num_dst"])
        self.num_edges = int(manifest["num_edges"])
        self.index_dtype = int32 if manifest["index_dtype"] == "int32" else int64
        self.sorted_by_dst = bool(manifest.get("sorted_by_dst", True))
        self._item = _STRUCT[self.index_dtype.name]
        self._row_path = root / "row_ptr.bin"
        self._column_path = root / "col_idx.bin"
        expected_rows = (self.num_dst + 1) * self._item.size
        expected_columns = self.num_edges * self._item.size
        if self.num_src < 0 or self.num_dst < 0 or self.num_edges < 0:
            raise ValueError("persistent CSR extents must be non-negative")
        if self._row_path.stat().st_size != expected_rows:
            raise ValueError("persistent row_ptr size disagrees with manifest")
        if self._column_path.stat().st_size != expected_columns:
            raise ValueError("persistent col_idx size disagrees with manifest")
        endpoints = [
            self._read_one(self._row_path, 0),
            self._read_one(self._row_path, self.num_dst),
        ]
        if endpoints != [0, self.num_edges]:
            raise ValueError("persistent CSR row pointer endpoints are invalid")
        self._field_specs: dict[tuple[str, str], tuple[tuple[int, ...], DType, Path]] = {}
        self._field_readers: dict[tuple[str, str], PagedField] = {}
        expected_leading = {
            "src": self.num_src, "dst": self.num_dst, "edge": self.num_edges,
        }
        for entry in manifest.get("fields", []):
            role = entry.get("role")
            name = entry.get("name")
            if role not in _FIELD_ROLES or not isinstance(name, str) or not name:
                raise ValueError("persistent field entry has an invalid role/name")
            if (role, name) in self._field_specs:
                raise ValueError(f"duplicate persistent field {role}.{name}")
            dtype = _DTYPES_BY_NAME.get(entry.get("dtype"))
            if dtype is None:
                raise ValueError(f"persistent field {role}.{name} has unknown dtype")
            try:
                _field_codec(dtype)
            except ValueError:
                raise ValueError(
                    f"persistent field {role}.{name} dtype {dtype.name} "
                    "is not storable") from None
            shape = tuple(int(extent) for extent in entry.get("shape", ()))
            if not shape or any(extent < 0 for extent in shape):
                raise ValueError(f"persistent field {role}.{name} shape is invalid")
            if shape[0] != expected_leading[role]:
                raise ValueError(
                    f"persistent field {role}.{name} leading dimension must be "
                    f"{expected_leading[role]}")
            file = Path(entry.get("file", ""))
            if file.is_absolute() or ".." in file.parts:
                raise ValueError(f"persistent field {role}.{name} path is unsafe")
            field_path = root / file
            expected_bytes = math.prod(shape) * dtype.itemsize
            if field_path.stat().st_size != expected_bytes:
                raise ValueError(
                    f"persistent field {role}.{name} size disagrees with manifest")
            self._field_specs[(role, name)] = (shape, dtype, field_path)

    @classmethod
    def create(
        cls,
        graph,
        path: str | os.PathLike[str],
        *,
        fields: Mapping[str, Mapping[str, Tensor]] | None = None,
    ) -> "PagedCSRStore":
        destination = Path(path).expanduser().resolve()
        if destination.exists():
            raise FileExistsError(
                f"refusing to overwrite existing Tiga graph: {destination}")
        if graph.schema.lifecycle != "static" or graph.schema.realization not in {
            "materialized_csr", "paged_csr"
        }:
            raise TypeError("only a static CSR Graph can be saved as .gfg")
        if fields:
            _validate_field_bindings(graph, fields)
        source_store = getattr(graph, "_paged_store", None)
        if source_store is not None:
            parent = destination.parent
            parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(
                tempfile.mkdtemp(prefix=f".{destination.name}.", dir=parent))
            try:
                shutil.copytree(source_store.path, temporary, dirs_exist_ok=True)
                if fields:
                    entries = _write_fields(temporary, fields)
                    manifest_path = temporary / "manifest.json"
                    manifest = json.loads(
                        manifest_path.read_text(encoding="utf-8"))
                    merged = {
                        (item["role"], item["name"]): item
                        for item in manifest.get("fields", [])
                    }
                    for item in entries:
                        merged[(item["role"], item["name"])] = item
                    manifest["format"] = _FORMAT_V2
                    manifest["fields"] = list(merged.values())
                    manifest_path.write_text(
                        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                os.replace(temporary, destination)
            except BaseException:
                shutil.rmtree(temporary, ignore_errors=True)
                raise
            return cls(destination)
        row_ptr, col_idx = graph.resolve_csr()
        rows, columns = _flatten(row_ptr), _flatten(col_idx)
        if not rows or rows[0] != 0 or rows[-1] != len(columns):
            raise ValueError("invalid CSR row pointer endpoints")
        if any(left > right for left, right in zip(rows, rows[1:])):
            raise ValueError("row_ptr must be monotonic")
        if any(value < 0 or value >= graph.schema.num_src for value in columns):
            raise ValueError("col_idx contains an out-of-range source index")
        parent = destination.parent
        parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=parent))
        try:
            _write_indices(temporary / "row_ptr.bin", rows, graph.schema.index_dtype)
            _write_indices(temporary / "col_idx.bin", columns, graph.schema.index_dtype)
            entries = _write_fields(temporary, fields or {})
            manifest = {
                "format": _FORMAT_V2 if entries else _FORMAT,
                "num_src": graph.schema.num_src,
                "num_dst": graph.schema.num_dst,
                "num_edges": len(columns),
                "index_dtype": graph.schema.index_dtype.name,
                "sorted_by_dst": graph.schema.sorted_by_dst,
            }
            if entries:
                manifest["fields"] = entries
            (temporary / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, destination)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return cls(destination)

    def _read_one(self, path: Path, index: int) -> int:
        with path.open("rb", buffering=0) as stream:
            stream.seek(index * self._item.size)
            payload = stream.read(self._item.size)
        if len(payload) != self._item.size:
            raise RuntimeError("short persistent graph read")
        return self._item.unpack(payload)[0]

    def _read_range(self, path: Path, begin: int, end: int) -> list[int]:
        if begin == end:
            return []
        with path.open("rb", buffering=0) as stream:
            stream.seek(begin * self._item.size)
            payload = stream.read((end - begin) * self._item.size)
        if len(payload) != (end - begin) * self._item.size:
            raise RuntimeError("short persistent graph read")
        return [value[0] for value in self._item.iter_unpack(payload)]

    def read_rows(self, begin: int, end: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Read and normalize one contiguous destination partition."""
        if not 0 <= begin <= end <= self.num_dst:
            raise ValueError("paged CSR row range is outside the graph")
        rows = self._read_range(self._row_path, begin, end + 1)
        edge_begin, edge_end = rows[0], rows[-1]
        columns = self._read_range(self._column_path, edge_begin, edge_end)
        if any(left > right for left, right in zip(rows, rows[1:])):
            raise ValueError("persistent row_ptr must be monotonic")
        if any(value < 0 or value >= self.num_src for value in columns):
            raise ValueError("persistent col_idx contains an out-of-range source index")
        return (
            tuple(value - edge_begin for value in rows),
            tuple(columns),
        )

    def read_page(
        self, begin: int, end: int
    ) -> tuple[tuple[int, ...], tuple[int, ...], int]:
        """One destination partition plus its global edge offset."""
        if not 0 <= begin <= end <= self.num_dst:
            raise ValueError("paged CSR row range is outside the graph")
        rows = self._read_range(self._row_path, begin, end + 1)
        edge_begin, edge_end = rows[0], rows[-1]
        columns = self._read_range(self._column_path, edge_begin, edge_end)
        if any(left > right for left, right in zip(rows, rows[1:])):
            raise ValueError("persistent row_ptr must be monotonic")
        if any(value < 0 or value >= self.num_src for value in columns):
            raise ValueError("persistent col_idx contains an out-of-range source index")
        return (
            tuple(value - edge_begin for value in rows),
            tuple(columns),
            edge_begin,
        )

    def field_names(self, role: str) -> tuple[str, ...]:
        """Persisted field names for one role, in manifest order."""
        if role not in _FIELD_ROLES:
            raise ValueError("field role must be 'src', 'dst' or 'edge'")
        return tuple(
            name for field_role, name in self._field_specs if field_role == role)

    def field_shape_dtype(self, role: str, name: str) -> tuple[tuple[int, ...], DType]:
        try:
            shape, dtype, _path = self._field_specs[(role, name)]
        except KeyError:
            raise KeyError(f"no persistent field {role}.{name}") from None
        return shape, dtype

    def field_reader(self, role: str, name: str) -> PagedField:
        """A cached mmap row reader over one persistent field payload."""
        key = (role, name)
        reader = self._field_readers.get(key)
        if reader is None:
            try:
                shape, dtype, path = self._field_specs[key]
            except KeyError:
                raise KeyError(f"no persistent field {role}.{name}") from None
            reader = PagedField(path, shape, dtype)
            self._field_readers[key] = reader
        return reader

    def close(self) -> None:
        for reader in self._field_readers.values():
            reader.close()
        self._field_readers.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def iter_pages(
        self, *, rows_per_page: int
    ) -> Iterator[tuple[int, int, tuple[int, ...], tuple[int, ...]]]:
        if rows_per_page <= 0:
            raise ValueError("rows_per_page must be positive")
        for begin in range(0, self.num_dst, rows_per_page):
            end = min(self.num_dst, begin + rows_per_page)
            rows, columns = self.read_rows(begin, end)
            yield begin, end, rows, columns

    def materialize(self, *, device="cpu") -> tuple[Tensor, Tensor]:
        rows, columns = self.read_rows(0, self.num_dst)
        return (
            tensor(rows, dtype=self.index_dtype, device=device),
            tensor(columns, dtype=self.index_dtype, device=device),
        )


def save_graph(
    graph,
    path: str | os.PathLike[str],
    *,
    fields: Mapping[str, Mapping[str, Tensor]] | None = None,
) -> None:
    PagedCSRStore.create(graph, path, fields=fields)


__all__ = ["PagedCSRStore", "PagedField", "PagedGradFile", "save_graph"]
