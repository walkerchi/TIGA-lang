"""Torch-independent GraphForge execution runtime.

The native core owns CPU/CUDA buffers, streams, events and CUDA Driver modules.
Higher-level hierarchy and distributed planners consume this ABI without a
Torch dependency.
"""

from __future__ import annotations

import atexit
import ctypes
import os
import sys
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Sequence

from .bundle import (
    AccessMode,
    ArgumentBinding,
    BundleSubmission,
    Completion,
    ExecutableBundle,
    ImmediateCompletion,
    KernelInvocation,
    PreparedBundle,
    PreparedInvocation,
    PreparingProvider,
    ResourceAccess,
    SubmissionProvider,
    SynchronousProvider,
)
from .plan import ExecutableBundlePlan, PlannedInvocation, ResourceRequirement


class RuntimeStatus(IntEnum):
    OK = 0
    INVALID_ARGUMENT = 1
    UNSUPPORTED = 2
    OUT_OF_MEMORY = 3
    INTERNAL = 4


class DeviceType(IntEnum):
    CPU = 0
    CUDA = 1
    HIP = 2
    METAL = 3
    PPU = 4


_DEVICE_NAMES = {
    "cpu": DeviceType.CPU,
    "cuda": DeviceType.CUDA,
    "hip": DeviceType.HIP,
    "metal": DeviceType.METAL,
    "mps": DeviceType.METAL,
    "ppu": DeviceType.PPU,
}


@dataclass(frozen=True)
class Device:
    type: DeviceType
    ordinal: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.type, DeviceType):
            object.__setattr__(self, "type", DeviceType(self.type))
        if not isinstance(self.ordinal, int) or self.ordinal < 0:
            raise ValueError("device ordinal must be a non-negative integer")

    @classmethod
    def parse(cls, value: Device | str) -> Device:
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError("device must be a graphforge.runtime.Device or string")
        name, separator, ordinal = value.lower().partition(":")
        try:
            device_type = _DEVICE_NAMES[name]
        except KeyError as error:
            raise ValueError(f"unknown device type {name!r}") from error
        if separator:
            try:
                parsed_ordinal = int(ordinal)
            except ValueError as error:
                raise ValueError(f"invalid device ordinal {ordinal!r}") from error
        else:
            parsed_ordinal = 0
        return cls(device_type, parsed_ordinal)

    def __str__(self) -> str:
        name = self.type.name.lower()
        return f"{name}:{self.ordinal}"


class _CDevice(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("ordinal", ctypes.c_int32)]


def _shared_library_names() -> tuple[str, ...]:
    if sys.platform == "darwin":
        return ("libgraphforge_runtime.dylib",)
    if os.name == "nt":
        return ("graphforge_runtime.dll", "libgraphforge_runtime.dll")
    return ("libgraphforge_runtime.so",)


def _library_candidates() -> list[Path]:
    override = os.environ.get("GRAPHFORGE_RUNTIME_LIBRARY")
    candidates = [Path(override)] if override else []
    package = Path(__file__).resolve().parents[1]
    repository = Path(__file__).resolve().parents[3]
    for name in _shared_library_names():
        candidates.append(package / "lib" / name)
        development = list(repository.glob(f"build/*/lib/Runtime/{name}"))
        development.sort(
            key=lambda path: path.stat().st_mtime if path.exists() else 0,
            reverse=True,
        )
        candidates.extend(development)
    return candidates


_LIBRARY: ctypes.CDLL | None = None
_BUFFER_POOL: dict[tuple[Device, int, int], list[ctypes.c_void_p]] = {}
_EVENT_POOL: dict[Device, list[ctypes.c_void_p]] = {}
_POOL_LIMIT = 16


def _drain_pools() -> None:
    library = _LIBRARY
    if library is None:
        return
    for handles in _BUFFER_POOL.values():
        for handle in handles:
            library.gfrt_buffer_release(handle)
    for handles in _EVENT_POOL.values():
        for handle in handles:
            library.gfrt_event_release(handle)
    _BUFFER_POOL.clear()
    _EVENT_POOL.clear()


atexit.register(_drain_pools)


def _library() -> ctypes.CDLL:
    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY
    failures: list[str] = []
    for candidate in _library_candidates():
        if not candidate.is_file():
            continue
        try:
            _LIBRARY = ctypes.CDLL(str(candidate))
            _configure(_LIBRARY)
            return _LIBRARY
        except OSError as error:
            failures.append(f"{candidate}: {error}")
    detail = "\n".join(failures) if failures else "no candidate library exists"
    raise RuntimeError(
        "GraphForge native runtime is unavailable; build GraphForgeRuntime or "
        "install a native wheel. Search result:\n" + detail
    )


def _configure(library: ctypes.CDLL) -> None:
    library.gfrt_status_string.argtypes = [ctypes.c_int]
    library.gfrt_status_string.restype = ctypes.c_char_p
    library.gfrt_runtime_version.argtypes = []
    library.gfrt_runtime_version.restype = ctypes.c_char_p

    library.gfrt_buffer_allocate.argtypes = [
        _CDevice, ctypes.c_size_t, ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.gfrt_buffer_allocate.restype = ctypes.c_int
    library.gfrt_buffer_allocate_pinned.argtypes = [
        ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)
    ]
    library.gfrt_buffer_allocate_pinned.restype = ctypes.c_int
    library.gfrt_buffer_wrap_external.argtypes = [
        _CDevice, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
    ]
    library.gfrt_buffer_wrap_external.restype = ctypes.c_int
    library.gfrt_buffer_release.argtypes = [ctypes.c_void_p]
    library.gfrt_buffer_release.restype = None
    library.gfrt_buffer_size.argtypes = [ctypes.c_void_p]
    library.gfrt_buffer_size.restype = ctypes.c_size_t
    library.gfrt_buffer_data.argtypes = [ctypes.c_void_p]
    library.gfrt_buffer_data.restype = ctypes.c_void_p
    library.gfrt_buffer_address.argtypes = [ctypes.c_void_p]
    library.gfrt_buffer_address.restype = ctypes.c_size_t
    library.gfrt_buffer_write.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t
    ]
    library.gfrt_buffer_write.restype = ctypes.c_int
    library.gfrt_buffer_read.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t
    ]
    library.gfrt_buffer_read.restype = ctypes.c_int
    library.gfrt_buffer_copy_async.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
        ctypes.c_size_t, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
    ]
    library.gfrt_buffer_copy_async.restype = ctypes.c_int

    library.gfrt_stream_create.argtypes = [
        _CDevice, ctypes.POINTER(ctypes.c_void_p),
    ]
    library.gfrt_stream_create.restype = ctypes.c_int
    library.gfrt_stream_wrap_external.argtypes = [
        _CDevice, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)
    ]
    library.gfrt_stream_wrap_external.restype = ctypes.c_int
    library.gfrt_stream_release.argtypes = [ctypes.c_void_p]
    library.gfrt_stream_release.restype = None
    library.gfrt_stream_address.argtypes = [ctypes.c_void_p]
    library.gfrt_stream_address.restype = ctypes.c_size_t
    library.gfrt_stream_synchronize.argtypes = [ctypes.c_void_p]
    library.gfrt_stream_synchronize.restype = ctypes.c_int
    library.gfrt_stream_wait_event.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    library.gfrt_stream_wait_event.restype = ctypes.c_int

    library.gfrt_event_create_completed.argtypes = [
        _CDevice, ctypes.POINTER(ctypes.c_void_p),
    ]
    library.gfrt_event_create_completed.restype = ctypes.c_int
    library.gfrt_event_release.argtypes = [ctypes.c_void_p]
    library.gfrt_event_release.restype = None
    library.gfrt_event_is_ready.argtypes = [ctypes.c_void_p]
    library.gfrt_event_is_ready.restype = ctypes.c_int
    library.gfrt_event_wait.argtypes = [ctypes.c_void_p]
    library.gfrt_event_wait.restype = ctypes.c_int
    library.gfrt_event_record.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
    ]
    library.gfrt_event_record.restype = ctypes.c_int

    library.gfrt_module_load.argtypes = [
        _CDevice, ctypes.c_void_p, ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.gfrt_module_load.restype = ctypes.c_int
    library.gfrt_module_release.argtypes = [ctypes.c_void_p]
    library.gfrt_module_release.restype = None
    library.gfrt_module_get_kernel.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)
    ]
    library.gfrt_module_get_kernel.restype = ctypes.c_int
    library.gfrt_kernel_release.argtypes = [ctypes.c_void_p]
    library.gfrt_kernel_release.restype = None
    library.gfrt_kernel_launch.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
        ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
        ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.gfrt_kernel_launch.restype = ctypes.c_int
    library.gfrt_kernel_launch_async.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
        ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
        ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t,
    ]
    library.gfrt_kernel_launch_async.restype = ctypes.c_int


def _c_device(device: Device) -> _CDevice:
    return _CDevice(int(device.type), device.ordinal)


def _check(status: int, operation: str) -> None:
    if status == RuntimeStatus.OK:
        return
    library = _library()
    message = library.gfrt_status_string(status).decode()
    raise RuntimeError(f"{operation} failed: {message}")


def version() -> str:
    return _library().gfrt_runtime_version().decode()


def cuda_compute_capability(
    device: Device | str = "cuda:0",
) -> tuple[int, int, int]:
    """Query CUDA architecture without importing a tensor framework."""
    resolved = Device.parse(device)
    if resolved.type != DeviceType.CUDA:
        raise ValueError("CUDA capability requires a CUDA device")
    try:
        driver = ctypes.CDLL("libcuda.so.1")
    except OSError as error:
        raise RuntimeError("CUDA Driver library is unavailable") from error
    driver.cuInit.argtypes = [ctypes.c_uint]
    driver.cuInit.restype = ctypes.c_int
    driver.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    driver.cuDeviceGet.restype = ctypes.c_int
    driver.cuDeviceComputeCapability.argtypes = [
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int
    ]
    driver.cuDeviceComputeCapability.restype = ctypes.c_int
    driver.cuDeviceGetAttribute.argtypes = [
        ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int
    ]
    driver.cuDeviceGetAttribute.restype = ctypes.c_int
    native_device = ctypes.c_int()
    major, minor, warp_size = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
    statuses = (
        driver.cuInit(0),
        driver.cuDeviceGet(ctypes.byref(native_device), resolved.ordinal),
        driver.cuDeviceComputeCapability(
            ctypes.byref(major), ctypes.byref(minor), native_device.value),
        # CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10.
        driver.cuDeviceGetAttribute(
            ctypes.byref(warp_size), 10, native_device.value),
    )
    if any(status != 0 for status in statuses):
        raise RuntimeError(f"CUDA capability query failed with statuses {statuses}")
    return major.value, minor.value, warp_size.value


class Buffer:
    """Reference-counted physical allocation owned by the native runtime."""

    def __init__(
        self,
        bytes: int,
        *,
        device: Device | str = "cpu",
        alignment: int = 64,
        _pooled: bool = False,
        pinned: bool = False,
    ) -> None:
        if not isinstance(bytes, int) or bytes < 0:
            raise ValueError("buffer size must be a non-negative integer")
        self.device = Device.parse(device)
        if pinned and self.device.type != DeviceType.CPU:
            raise ValueError("pinned buffers are host allocations")
        self.pinned = bool(pinned)
        self._pool_key = (
            (self.device, bytes, alignment)
            if _pooled and self.device.type == DeviceType.CPU and not pinned
            else None
        )
        available = _BUFFER_POOL.get(self._pool_key, []) if self._pool_key else []
        if available:
            self._handle = available.pop()
            return
        handle = ctypes.c_void_p()
        if pinned:
            status = _library().gfrt_buffer_allocate_pinned(
                bytes, ctypes.byref(handle)
            )
        else:
            status = _library().gfrt_buffer_allocate(
                _c_device(self.device), bytes, alignment, ctypes.byref(handle)
            )
        _check(status, "pinned buffer allocation" if pinned else "buffer allocation")
        self._handle = handle

    @classmethod
    def pinned_host(cls, bytes: int) -> "Buffer":
        """Allocate page-locked host memory suitable for asynchronous DMA."""
        return cls(bytes, device="cpu", pinned=True)

    @classmethod
    def wrap_address(
        cls,
        address: int,
        bytes: int,
        *,
        device: Device | str,
        owner: object | None = None,
    ) -> "Buffer":
        """Create a non-owning runtime view of provider/framework storage.

        ``owner`` is retained by the Python wrapper until the native view is
        released. The C runtime receives no deleter, so GraphForge can use the
        pointer in stream-ordered copies and communication without taking
        allocation ownership.
        """
        resolved = Device.parse(device)
        if not isinstance(address, int) or address < 0:
            raise ValueError("external buffer address must be non-negative")
        if not isinstance(bytes, int) or bytes < 0:
            raise ValueError("external buffer size must be non-negative")
        if bytes and address == 0:
            raise ValueError("non-empty external buffer requires an address")
        handle = ctypes.c_void_p()
        _check(
            _library().gfrt_buffer_wrap_external(
                _c_device(resolved), ctypes.c_void_p(address), bytes,
                None, None, ctypes.byref(handle),
            ),
            "external buffer wrap",
        )
        result = object.__new__(cls)
        result.device = resolved
        result.pinned = False
        result._pool_key = None
        result._handle = handle
        result._external_owner = owner
        return result

    @property
    def nbytes(self) -> int:
        self._require_open()
        return int(_library().gfrt_buffer_size(self._handle))

    @property
    def address(self) -> int:
        self._require_open()
        address = _library().gfrt_buffer_address(self._handle)
        if not address and self.nbytes:
            raise RuntimeError(f"allocation address is unavailable for {self.device}")
        return int(address)

    @property
    def host_address(self) -> int:
        """Return a CPU-dereferenceable address, rejecting device memory."""
        self._require_open()
        address = _library().gfrt_buffer_data(self._handle)
        if not address and self.nbytes:
            raise RuntimeError(f"host address is unavailable for {self.device}")
        return int(address or 0)

    def write(self, data: bytes | bytearray | memoryview, *, offset: int = 0) -> None:
        encoded = bytes(data)
        if not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be non-negative")
        storage = ctypes.create_string_buffer(encoded)
        _check(
            _library().gfrt_buffer_write(
                self._handle, offset, ctypes.cast(storage, ctypes.c_void_p),
                len(encoded)
            ),
            "buffer write",
        )

    def read(self, *, offset: int = 0, bytes: int | None = None) -> bytes:
        if not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be non-negative")
        count = self.nbytes - offset if bytes is None else bytes
        if not isinstance(count, int) or count < 0:
            raise ValueError("bytes must be non-negative")
        storage = ctypes.create_string_buffer(count)
        _check(
            _library().gfrt_buffer_read(
                self._handle, offset, ctypes.cast(storage, ctypes.c_void_p), count
            ),
            "buffer read",
        )
        return storage.raw

    def copy_to(
        self,
        destination: "Buffer",
        *,
        stream: "Stream",
        source_offset: int = 0,
        destination_offset: int = 0,
        bytes: int | None = None,
    ) -> "Event":
        """Enqueue an exact buffer transfer and return its completion event."""
        self._require_open()
        destination._require_open()
        stream._require_open()
        count = self.nbytes - source_offset if bytes is None else bytes
        if any(not isinstance(value, int) or value < 0 for value in (
            source_offset, destination_offset, count
        )):
            raise ValueError("copy offsets and bytes must be non-negative integers")
        handle = ctypes.c_void_p()
        _check(
            _library().gfrt_buffer_copy_async(
                self._handle, source_offset, destination._handle,
                destination_offset, count, stream._handle, ctypes.byref(handle)
            ),
            "asynchronous buffer copy",
        )
        return Event._from_handle(stream.device, handle)

    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle:
            key = getattr(self, "_pool_key", None)
            available = _BUFFER_POOL.setdefault(key, []) if key else []
            if key and len(available) < _POOL_LIMIT:
                available.append(handle)
            else:
                _library().gfrt_buffer_release(handle)
            self._handle = None
            self._external_owner = None

    def _require_open(self) -> None:
        if not getattr(self, "_handle", None):
            raise RuntimeError("buffer has been closed")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class Stream:
    def __init__(self, device: Device | str = "cpu") -> None:
        self.device = Device.parse(device)
        handle = ctypes.c_void_p()
        _check(
            _library().gfrt_stream_create(_c_device(self.device), ctypes.byref(handle)),
            "stream creation",
        )
        self._handle = handle

    @classmethod
    def wrap_address(
        cls, address: int, *, device: Device | str = "cuda"
    ) -> "Stream":
        resolved = Device.parse(device)
        if resolved.type != DeviceType.CUDA or not isinstance(address, int):
            raise ValueError("external stream requires a CUDA address")
        handle = ctypes.c_void_p()
        _check(
            _library().gfrt_stream_wrap_external(
                _c_device(resolved), address, ctypes.byref(handle)
            ),
            "external stream wrap",
        )
        result = object.__new__(cls)
        result.device = resolved
        result._handle = handle
        return result

    def synchronize(self) -> None:
        self._require_open()
        _check(_library().gfrt_stream_synchronize(self._handle), "stream sync")

    @property
    def address(self) -> int:
        """Native provider stream handle for library/plugin interoperation."""
        self._require_open()
        return int(_library().gfrt_stream_address(self._handle))

    def wait_event(self, event: "Event") -> None:
        self._require_open()
        event._require_open()
        _check(
            _library().gfrt_stream_wait_event(self._handle, event._handle),
            "stream wait event",
        )

    def record_event(self) -> "Event":
        self._require_open()
        handle = ctypes.c_void_p()
        _check(
            _library().gfrt_event_record(self._handle, ctypes.byref(handle)),
            "event record",
        )
        return Event._from_handle(self.device, handle)

    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle:
            _library().gfrt_stream_release(handle)
            self._handle = None

    def _require_open(self) -> None:
        if not getattr(self, "_handle", None):
            raise RuntimeError("stream has been closed")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class Event:
    def __init__(
        self, device: Device | str = "cpu", *, _pooled: bool = False
    ) -> None:
        self.device = Device.parse(device)
        self._pool_key = (
            self.device
            if _pooled and self.device.type == DeviceType.CPU
            else None
        )
        available = _EVENT_POOL.get(self._pool_key, []) if self._pool_key else []
        if available:
            self._handle = available.pop()
            return
        handle = ctypes.c_void_p()
        _check(
            _library().gfrt_event_create_completed(
                _c_device(self.device), ctypes.byref(handle)
            ),
            "event creation",
        )
        self._handle = handle

    @classmethod
    def _from_handle(cls, device: Device, handle: ctypes.c_void_p) -> "Event":
        result = object.__new__(cls)
        result.device = device
        result._pool_key = None
        result._handle = handle
        return result

    @property
    def ready(self) -> bool:
        self._require_open()
        return bool(_library().gfrt_event_is_ready(self._handle))

    def wait(self) -> None:
        self._require_open()
        _check(_library().gfrt_event_wait(self._handle), "event wait")

    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle:
            key = getattr(self, "_pool_key", None)
            available = _EVENT_POOL.setdefault(key, []) if key else []
            if key and len(available) < _POOL_LIMIT:
                available.append(handle)
            else:
                _library().gfrt_event_release(handle)
            self._handle = None

    def _require_open(self) -> None:
        if not getattr(self, "_handle", None):
            raise RuntimeError("event has been closed")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class Module:
    """Runtime-owned CUDA Driver module loaded from PTX or a binary image."""

    def __init__(self, image: str | bytes, *, device: Device | str = "cuda"):
        self.device = Device.parse(device)
        if self.device.type != DeviceType.CUDA:
            raise ValueError("Module currently supports CUDA Driver images")
        encoded = image.encode() if isinstance(image, str) else bytes(image)
        if not encoded:
            raise ValueError("module image must not be empty")
        storage = ctypes.create_string_buffer(encoded)
        handle = ctypes.c_void_p()
        _check(
            _library().gfrt_module_load(
                _c_device(self.device), ctypes.cast(storage, ctypes.c_void_p),
                len(encoded), ctypes.byref(handle)
            ),
            "module load",
        )
        self._handle = handle

    def kernel(self, name: str) -> "RuntimeKernel":
        self._require_open()
        if not isinstance(name, str) or not name:
            raise ValueError("kernel name must not be empty")
        handle = ctypes.c_void_p()
        _check(
            _library().gfrt_module_get_kernel(
                self._handle, name.encode(), ctypes.byref(handle)
            ),
            "kernel lookup",
        )
        return RuntimeKernel(self.device, handle)

    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle:
            _library().gfrt_module_release(handle)
            self._handle = None

    def _require_open(self) -> None:
        if not getattr(self, "_handle", None):
            raise RuntimeError("module has been closed")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class RuntimeKernel:
    """CUDA kernel handle retaining its runtime-owned module."""

    def __init__(self, device: Device, handle: ctypes.c_void_p):
        self.device = device
        self._handle = handle

    def launch(
        self,
        stream: Stream,
        *,
        grid: Sequence[int],
        block: Sequence[int],
        arguments: Sequence[object],
        shared_bytes: int = 0,
    ) -> Event:
        self._require_open()
        stream._require_open()
        if stream.device != self.device:
            raise ValueError("kernel and stream must use the same device")
        resolved_grid = _launch_shape("grid", grid)
        resolved_block = _launch_shape("block", block)
        if not isinstance(shared_bytes, int) or shared_bytes < 0:
            raise ValueError("shared_bytes must be non-negative")
        storage: list[ctypes._SimpleCData] = []
        for argument in arguments:
            if isinstance(argument, Buffer):
                if argument.device != self.device:
                    raise ValueError("buffer argument is on a different device")
                storage.append(ctypes.c_uint64(argument.address))
            elif isinstance(argument, ctypes._SimpleCData):
                storage.append(argument)
            elif callable(getattr(argument, "data_ptr", None)):
                storage.append(ctypes.c_uint64(int(argument.data_ptr())))
            else:
                raise TypeError(
                    "kernel arguments must be Buffer, a data_ptr object, or "
                    "explicit ctypes scalars"
                )
        pointers = (ctypes.c_void_p * len(storage))(
            *(ctypes.cast(ctypes.byref(value), ctypes.c_void_p) for value in storage)
        )
        event = ctypes.c_void_p()
        _check(
            _library().gfrt_kernel_launch(
                self._handle, stream._handle,
                *resolved_grid, *resolved_block, shared_bytes,
                pointers, len(storage), ctypes.byref(event),
            ),
            "kernel launch",
        )
        return Event._from_handle(self.device, event)

    def prepare_async(
        self,
        stream_provider,
        *,
        grid: Sequence[int],
        block: Sequence[int],
        arguments: Sequence[object],
        shared_bytes: int = 0,
    ):
        """Pre-bind a stable ABI for ordered submission to an external stream.

        The caller owns ordering/completion through that stream (for example,
        Torch records the timing/end event after the launch), so this path
        deliberately avoids allocating a redundant GraphForge event.
        """
        self._require_open()
        resolved_grid = _launch_shape("grid", grid)
        resolved_block = _launch_shape("block", block)
        if not isinstance(shared_bytes, int) or shared_bytes < 0:
            raise ValueError("shared_bytes must be non-negative")
        storage: list[ctypes._SimpleCData] = []
        for argument in arguments:
            if isinstance(argument, Buffer):
                if argument.device != self.device:
                    raise ValueError("buffer argument is on a different device")
                storage.append(ctypes.c_uint64(argument.address))
            elif isinstance(argument, ctypes._SimpleCData):
                storage.append(argument)
            elif callable(getattr(argument, "data_ptr", None)):
                storage.append(ctypes.c_uint64(int(argument.data_ptr())))
            else:
                raise TypeError(
                    "kernel arguments must be Buffer, a data_ptr object, or "
                    "explicit ctypes scalars")
        pointers = (ctypes.c_void_p * len(storage))(
            *(ctypes.cast(ctypes.byref(value), ctypes.c_void_p)
              for value in storage))
        library_launch = _library().gfrt_kernel_launch_async
        kernel_handle = self._handle
        argument_count = len(storage)

        def launch():
            # ctypes pointer arrays do not own the scalar cells they address.
            # Retain the pre-bound cells explicitly for the closure lifetime.
            _bound_storage = storage
            stream = stream_provider()
            stream._require_open()
            if stream.device != self.device:
                raise ValueError("kernel and stream must use the same device")
            _check(
                library_launch(
                    kernel_handle, stream._handle,
                    *resolved_grid, *resolved_block, shared_bytes,
                    pointers, argument_count),
                "asynchronous kernel launch")
            # Keep storage/pointers captured for the duration of submission.
            del _bound_storage
            return None

        return launch

    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle:
            _library().gfrt_kernel_release(handle)
            self._handle = None

    def _require_open(self) -> None:
        if not getattr(self, "_handle", None):
            raise RuntimeError("kernel has been closed")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _launch_shape(name: str, value: Sequence[int]) -> tuple[int, int, int]:
    result = tuple(value)
    if not 1 <= len(result) <= 3 or any(
        not isinstance(extent, int) or extent <= 0 for extent in result
    ):
        raise ValueError(f"{name} must contain one to three positive integers")
    return (*result, *(1 for _ in range(3 - len(result))))


from .hierarchy import (  # noqa: E402 -- runtime primitives must exist first
    HierarchyCompletion,
    HierarchyRuntime,
    MemoryTier,
    PhysicalInstance,
)


__all__ = [
    "AccessMode",
    "ArgumentBinding",
    "Buffer",
    "BundleSubmission",
    "Completion",
    "cuda_compute_capability",
    "Device",
    "DeviceType",
    "Event",
    "ExecutableBundle",
    "ExecutableBundlePlan",
    "ImmediateCompletion",
    "HierarchyCompletion",
    "HierarchyRuntime",
    "KernelInvocation",
    "Module",
    "MemoryTier",
    "PreparedBundle",
    "PlannedInvocation",
    "PreparedInvocation",
    "PhysicalInstance",
    "PreparingProvider",
    "ResourceAccess",
    "ResourceRequirement",
    "RuntimeStatus",
    "RuntimeKernel",
    "Stream",
    "SubmissionProvider",
    "SynchronousProvider",
    "version",
]
