"""Versioned, framework-free persistent storage for paged CSR relations."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import struct
import tempfile
from typing import Iterator, Sequence

from ..tensor import DType, Tensor, int32, int64, tensor


_FORMAT = "graphforge.gfg.csr.v1"
_STRUCT = {"int32": struct.Struct("<i"), "int64": struct.Struct("<q")}


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
            raise FileNotFoundError(f"not a GraphForge graph: {root}") from error
        if manifest.get("format") != _FORMAT:
            raise ValueError(f"unsupported GraphForge graph format in {manifest_path}")
        required = {"num_src", "num_dst", "num_edges", "index_dtype"}
        if not required <= manifest.keys():
            raise ValueError("GraphForge graph manifest is incomplete")
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

    @classmethod
    def create(cls, graph, path: str | os.PathLike[str]) -> "PagedCSRStore":
        destination = Path(path).expanduser().resolve()
        if destination.exists():
            raise FileExistsError(
                f"refusing to overwrite existing GraphForge graph: {destination}")
        if graph.schema.lifecycle != "static" or graph.schema.realization not in {
            "materialized_csr", "paged_csr"
        }:
            raise TypeError("only a static CSR Graph can be saved as .gfg")
        source_store = getattr(graph, "_paged_store", None)
        if source_store is not None:
            parent = destination.parent
            parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(
                tempfile.mkdtemp(prefix=f".{destination.name}.", dir=parent))
            try:
                shutil.copytree(source_store.path, temporary, dirs_exist_ok=True)
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
            manifest = {
                "format": _FORMAT,
                "num_src": graph.schema.num_src,
                "num_dst": graph.schema.num_dst,
                "num_edges": len(columns),
                "index_dtype": graph.schema.index_dtype.name,
                "sorted_by_dst": graph.schema.sorted_by_dst,
            }
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


def save_graph(graph, path: str | os.PathLike[str]) -> None:
    PagedCSRStore.create(graph, path)


__all__ = ["PagedCSRStore", "save_graph"]
