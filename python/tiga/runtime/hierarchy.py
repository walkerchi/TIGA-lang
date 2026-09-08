"""Execution of compiler-planned hierarchical storage operations.

This module is intentionally below the public tensor/graph API.  It consumes
typed compiler resource requirements and preserves logical versions while
moving physical instances between device, pinned host, RAM and NVMe tiers.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import tempfile
from threading import Lock
from typing import Mapping

from .plan import ResourceRequirement


class MemoryTier(str, Enum):
    REGISTER = "register"
    SHARED = "shared"
    DEVICE = "device"
    HOST_PINNED = "host-pinned"
    RAM = "ram"
    NVME = "nvme"
    REMOTE = "remote"

    @classmethod
    def parse(cls, value: str) -> "MemoryTier":
        aliases = {
            "rmem": cls.REGISTER,
            "smem": cls.SHARED,
            "hbm": cls.DEVICE,
            "gmem": cls.DEVICE,
            "pinned_ram": cls.HOST_PINNED,
            "host_pinned": cls.HOST_PINNED,
            "host": cls.RAM,
            "ssd": cls.NVME,
            "distributed": cls.REMOTE,
        }
        try:
            return aliases.get(value.lower(), cls(value.lower()))
        except ValueError as error:
            raise ValueError(f"unknown memory tier {value!r}") from error


class HierarchyCompletion:
    """Completion retaining resources needed by an asynchronous transfer."""

    def __init__(self, completion, *, retained=()) -> None:
        self._completion = completion
        self._retained = tuple(retained)

    @property
    def ready(self) -> bool:
        if isinstance(self._completion, Future):
            return self._completion.done()
        return bool(self._completion.ready)

    def wait(self) -> None:
        if isinstance(self._completion, Future):
            self._completion.result()
        else:
            self._completion.wait()


@dataclass
class PhysicalInstance:
    logical_name: str
    version: int
    tier: MemoryTier
    capacity_bytes: int
    device: str
    layout: str = "contiguous"
    buffer: object | None = None
    path: Path | None = None
    completion: HierarchyCompletion | None = None
    _runtime: "HierarchyRuntime | None" = field(default=None, repr=False)
    _instance_id: int = field(default=-1, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    @property
    def ready(self) -> bool:
        return self.completion is None or self.completion.ready

    def wait(self) -> None:
        if self.completion is not None:
            self.completion.wait()

    def close(self) -> None:
        if self._closed:
            return
        self.wait()
        if self.buffer is not None:
            self.buffer.close()
        if self.path is not None:
            self.path.unlink(missing_ok=True)
        self._closed = True
        if self._runtime is not None:
            self._runtime._released(self)

    def __enter__(self) -> "PhysicalInstance":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


class HierarchyRuntime:
    """Capacity-accounted physical storage runtime for compiler plans."""

    def __init__(
        self,
        *,
        budgets: Mapping[MemoryTier | str, int] | None = None,
        nvme_directory: str | Path | None = None,
        io_workers: int = 2,
    ) -> None:
        self.budgets = {
            (key if isinstance(key, MemoryTier) else MemoryTier.parse(key)): int(value)
            for key, value in (budgets or {}).items()
        }
        if any(value < 0 for value in self.budgets.values()):
            raise ValueError("memory budgets must be non-negative")
        self.live_bytes = {tier: 0 for tier in MemoryTier}
        self.peak_live_bytes = {tier: 0 for tier in MemoryTier}
        self._instances: dict[int, PhysicalInstance] = {}
        self._next_instance_id = 0
        self._lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=io_workers)
        self._owned_directory = nvme_directory is None
        self.nvme_directory = Path(
            tempfile.mkdtemp(prefix="tiga-spill-")
            if nvme_directory is None else nvme_directory
        )
        self.nvme_directory.mkdir(parents=True, exist_ok=True)

    def allocate_requirement(self, requirement: ResourceRequirement) -> PhysicalInstance:
        return self.allocate(
            requirement.name,
            requirement.snapshot_version,
            tier=MemoryTier.parse(requirement.memory_space),
            capacity_bytes=requirement.capacity_bytes,
            device=requirement.device,
            layout=requirement.layout,
        )

    def allocate(
        self,
        logical_name: str,
        version: int,
        *,
        tier: MemoryTier | str,
        capacity_bytes: int,
        device: str = "cpu:0",
        layout: str = "contiguous",
    ) -> PhysicalInstance:
        from . import Buffer

        resolved = tier if isinstance(tier, MemoryTier) else MemoryTier.parse(tier)
        if resolved in {MemoryTier.REGISTER, MemoryTier.SHARED, MemoryTier.REMOTE}:
            raise ValueError(f"{resolved.value} is kernel/transport managed, not allocatable")
        if not logical_name or version < 0 or capacity_bytes < 0:
            raise ValueError("invalid logical instance name/version/capacity")
        with self._lock:
            live = self.live_bytes[resolved] + capacity_bytes
            budget = self.budgets.get(resolved)
            if budget is not None and live > budget:
                raise MemoryError(
                    f"{resolved.value} capacity exceeded: {live} > {budget} bytes"
                )
            self.live_bytes[resolved] = live
            self.peak_live_bytes[resolved] = max(self.peak_live_bytes[resolved], live)
            instance_id = self._next_instance_id
            self._next_instance_id += 1
        buffer = None
        path = None
        try:
            if resolved == MemoryTier.DEVICE:
                buffer = Buffer(capacity_bytes, device=device)
            elif resolved == MemoryTier.HOST_PINNED:
                buffer = Buffer.pinned_host(capacity_bytes)
            elif resolved == MemoryTier.RAM:
                buffer = Buffer(capacity_bytes, device="cpu")
            elif resolved == MemoryTier.NVME:
                path = self.nvme_directory / f"{logical_name}.v{version}.{instance_id}.bin"
                with path.open("wb") as stream:
                    stream.truncate(capacity_bytes)
            instance = PhysicalInstance(
                logical_name, version, resolved, capacity_bytes, device, layout,
                buffer=buffer, path=path, _runtime=self,
                _instance_id=instance_id,
            )
            with self._lock:
                self._instances[instance_id] = instance
            return instance
        except BaseException:
            with self._lock:
                self.live_bytes[resolved] -= capacity_bytes
            raise

    def transfer(
        self,
        source: PhysicalInstance,
        destination: PhysicalInstance,
        *,
        stream=None,
    ) -> HierarchyCompletion:
        from . import Stream

        if source.version != destination.version or source.logical_name != destination.logical_name:
            raise ValueError("transfer must preserve logical identity and version")
        if source.capacity_bytes != destination.capacity_bytes:
            raise ValueError("transfer instances must have equal capacity")
        source.wait()
        # Preserve write-after-write ordering when a physical instance is
        # reused by a later compiler-planned transfer.
        destination.wait()
        if source.buffer is not None and destination.buffer is not None:
            owns_stream = stream is None
            if stream is None:
                uses_cuda = source.tier == MemoryTier.DEVICE or destination.tier == MemoryTier.DEVICE
                stream = Stream(source.device if source.tier == MemoryTier.DEVICE else destination.device) if uses_cuda else Stream("cpu")
            event = source.buffer.copy_to(destination.buffer, stream=stream)
            completion = HierarchyCompletion(event, retained=(source, destination, stream) if owns_stream else (source, destination))
        else:
            future = self._executor.submit(self._file_transfer, source, destination)
            completion = HierarchyCompletion(future, retained=(source, destination))
        destination.completion = completion
        return completion

    @staticmethod
    def _file_transfer(source: PhysicalInstance, destination: PhysicalInstance) -> None:
        if source.path is not None:
            payload = source.path.read_bytes()
        elif source.buffer is not None:
            payload = source.buffer.read(bytes=source.capacity_bytes)
        else:
            raise RuntimeError("source instance has no physical storage")
        if len(payload) != destination.capacity_bytes:
            raise RuntimeError("short hierarchical transfer")
        if destination.path is not None:
            destination.path.write_bytes(payload)
        elif destination.buffer is not None:
            destination.buffer.write(payload)
        else:
            raise RuntimeError("destination instance has no physical storage")

    def latest(self, logical_name: str, *, tier: MemoryTier | str | None = None) -> PhysicalInstance:
        resolved = None if tier is None else (
            tier if isinstance(tier, MemoryTier) else MemoryTier.parse(tier)
        )
        candidates = [
            instance for instance in self._instances.values()
            if instance.logical_name == logical_name and not instance._closed and
            (resolved is None or instance.tier == resolved)
        ]
        if not candidates:
            raise KeyError(logical_name)
        return max(candidates, key=lambda instance: instance.version)

    def _released(self, instance: PhysicalInstance) -> None:
        with self._lock:
            self._instances.pop(instance._instance_id, None)
            self.live_bytes[instance.tier] -= instance.capacity_bytes

    def close(self) -> None:
        for instance in list(self._instances.values()):
            instance.close()
        self._executor.shutdown(wait=True)
        if self._owned_directory:
            self.nvme_directory.rmdir()

    def __enter__(self) -> "HierarchyRuntime":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


__all__ = [
    "HierarchyCompletion", "HierarchyRuntime", "MemoryTier", "PhysicalInstance"
]
