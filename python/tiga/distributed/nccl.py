"""Optional Torch-free NCCL GPU-direct neighbor transport.

NCCL is discovered only when the provider is activated.  Communicator IDs are
deployment data: rank zero calls :func:`unique_id`, distributes the returned
128 bytes through its launcher/control plane, and every rank constructs the
same provider below the unchanged Graph/MessagePassing API.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from dataclasses import dataclass
from importlib import metadata
import os
from pathlib import Path
from typing import Sequence

from ..runtime import Device, DeviceType, Stream, cuda_compute_capability
from .registry import TransportCapabilities
from .transport import DeviceBufferSlice


_UNIQUE_ID_BYTES = 128
_NCCL_UINT8 = 1


class _UniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_char * _UNIQUE_ID_BYTES)]


def _candidate_libraries(explicit: str | os.PathLike[str] | None):
    if explicit is not None:
        yield Path(explicit)
    configured = os.environ.get("TIGA_NCCL_LIBRARY")
    if configured:
        yield Path(configured)
    located = ctypes.util.find_library("nccl")
    if located:
        yield Path(located)
    for distribution_name in ("nvidia-nccl-cu13", "nvidia-nccl-cu12"):
        try:
            distribution = metadata.distribution(distribution_name)
        except metadata.PackageNotFoundError:
            continue
        for relative in (
            "nvidia/nccl/lib/libnccl.so.2",
            "nvidia/nccl/lib/libnccl.so",
        ):
            candidate = Path(distribution.locate_file(relative))
            if candidate.is_file():
                yield candidate


class _NCCL:
    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        failures: list[str] = []
        library = None
        for candidate in _candidate_libraries(path):
            try:
                library = ctypes.CDLL(str(candidate))
                self.path = str(candidate)
                break
            except OSError as error:
                failures.append(f"{candidate}: {error}")
        if library is None:
            detail = "; ".join(failures) or "no library candidate was found"
            raise ModuleNotFoundError(
                "NCCL transport requires libnccl.so.2; set "
                f"TIGA_NCCL_LIBRARY ({detail})"
            )
        self.library = library
        self._bind()

    def _bind(self) -> None:
        lib = self.library
        lib.ncclGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        lib.ncclGetVersion.restype = ctypes.c_int
        lib.ncclGetUniqueId.argtypes = [ctypes.POINTER(_UniqueId)]
        lib.ncclGetUniqueId.restype = ctypes.c_int
        lib.ncclCommInitRank.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, _UniqueId,
            ctypes.c_int,
        ]
        lib.ncclCommInitRank.restype = ctypes.c_int
        lib.ncclCommDestroy.argtypes = [ctypes.c_void_p]
        lib.ncclCommDestroy.restype = ctypes.c_int
        lib.ncclGetErrorString.argtypes = [ctypes.c_int]
        lib.ncclGetErrorString.restype = ctypes.c_char_p
        operation = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p,
        ]
        lib.ncclSend.argtypes = operation
        lib.ncclSend.restype = ctypes.c_int
        lib.ncclRecv.argtypes = operation
        lib.ncclRecv.restype = ctypes.c_int
        lib.ncclGroupStart.argtypes = []
        lib.ncclGroupStart.restype = ctypes.c_int
        lib.ncclGroupEnd.argtypes = []
        lib.ncclGroupEnd.restype = ctypes.c_int

    def check(self, status: int, operation: str) -> None:
        if status == 0:
            return
        message = self.library.ncclGetErrorString(status)
        detail = message.decode(errors="replace") if message else "unknown error"
        raise RuntimeError(f"{operation} failed: NCCL {status}: {detail}")

    @property
    def version(self) -> int:
        value = ctypes.c_int()
        self.check(self.library.ncclGetVersion(ctypes.byref(value)), "ncclGetVersion")
        return int(value.value)

    def unique_id(self) -> bytes:
        value = _UniqueId()
        self.check(self.library.ncclGetUniqueId(ctypes.byref(value)), "ncclGetUniqueId")
        return ctypes.string_at(ctypes.byref(value), _UNIQUE_ID_BYTES)


def unique_id(*, library: str | os.PathLike[str] | None = None) -> bytes:
    """Generate the exact communicator token that a launcher must broadcast."""
    return _NCCL(library).unique_id()


class NCCLTransport:
    """NCCL communicator implementing Tiga's device-buffer extension."""

    # Point-to-point operations are enqueued on the caller-provided CUDA
    # stream and return an Event.  The distributed executor may therefore run
    # an independent interior kernel before waiting for the halo stream.
    prefer_compute_overlap = True

    def __init__(
        self,
        rank: int,
        world_size: int,
        communicator_id: bytes | bytearray | memoryview | None = None,
        *,
        device: Device | str = "cuda:0",
        library: str | os.PathLike[str] | None = None,
    ) -> None:
        if not isinstance(world_size, int) or world_size <= 0:
            raise ValueError("NCCL world_size must be positive")
        if not isinstance(rank, int) or not 0 <= rank < world_size:
            raise ValueError("NCCL rank is outside the communicator")
        self.rank = rank
        self.world_size = world_size
        self.device = Device.parse(device)
        if self.device.type != DeviceType.CUDA:
            raise ValueError("NCCL transport requires a CUDA device")
        self._api = _NCCL(library)
        capability = cuda_compute_capability(self.device)
        # NCCL 2.21 is known to block inside communicator initialization on
        # compute-capability 12 devices instead of returning an actionable
        # compatibility error.  Fail before entering vendor code.  This is a
        # conservative provider policy, not a claim about every NCCL release.
        if capability[0] >= 12 and self._api.version < 22800:
            raise RuntimeError(
                f"Tiga requires NCCL >= 2.28 on {self.device} "
                f"(compute capability {capability[0]}.{capability[1]}); "
                f"loaded {self._api.version // 10000}."
                f"{(self._api.version // 100) % 100}."
                f"{self._api.version % 100} from {self._api.path}"
            )
        if communicator_id is None:
            if world_size != 1:
                raise ValueError(
                    "multi-rank NCCL requires a launcher-distributed communicator_id"
                )
            communicator_id = self._api.unique_id()
        encoded = bytes(communicator_id)
        if len(encoded) != _UNIQUE_ID_BYTES:
            raise ValueError("NCCL communicator_id must contain exactly 128 bytes")
        native_id = _UniqueId()
        ctypes.memmove(ctypes.byref(native_id), encoded, _UNIQUE_ID_BYTES)

        # Creating a runtime stream makes the selected primary context current
        # without importing Torch or depending on a framework's device guard.
        self._bootstrap_stream = Stream(self.device)
        self._communicator = ctypes.c_void_p()
        self._api.check(
            self._api.library.ncclCommInitRank(
                ctypes.byref(self._communicator), world_size, native_id, rank
            ),
            "ncclCommInitRank",
        )
        self._closed = False

    @property
    def version(self) -> int:
        return self._api.version

    @property
    def library_path(self) -> str:
        return self._api.path

    def send(self, peer: int, payload: object) -> None:
        del peer, payload
        raise NotImplementedError(
            "NCCL is device-direct; use compiler-bound device buffers"
        )

    def receive(self, peer: int) -> object:
        del peer
        raise NotImplementedError(
            "NCCL is device-direct; use compiler-bound device buffers"
        )

    def exchange_device(
        self,
        sends: Sequence[tuple[int, DeviceBufferSlice]],
        receives: Sequence[tuple[int, DeviceBufferSlice]],
        *,
        stream: Stream,
    ):
        if self._closed:
            raise RuntimeError("NCCL communicator is closed")
        if not isinstance(stream, Stream) or stream.device != self.device:
            raise ValueError("NCCL exchange stream must match its CUDA device")
        operations = (*sends, *receives)
        for peer, region in operations:
            if not isinstance(peer, int) or not 0 <= peer < self.world_size:
                raise ValueError("NCCL peer is outside the communicator")
            if not isinstance(region, DeviceBufferSlice):
                raise TypeError("NCCL messages require DeviceBufferSlice values")
            if getattr(region.buffer, "device", None) != self.device:
                raise ValueError("NCCL message buffer is on a different device")

        if self.world_size == 1:
            # A rank cannot be its own NCCL peer.  Some NCCL versions appear
            # to copy eager self-messages but silently leave rendezvous-sized
            # destinations untouched.  Preserve transport semantics with a
            # stream-ordered D2D copy; the two-device gate below is the only
            # place where Tiga claims an NCCL point-to-point data path.
            if len(sends) != len(receives):
                raise ValueError("single-rank exchange requires paired buffers")
            completions = []
            for (_send_peer, source), (_receive_peer, destination) in zip(
                sends, receives, strict=True
            ):
                if source.byte_count != destination.byte_count:
                    raise ValueError("single-rank exchange sizes must match")
                copy_to = getattr(source.buffer, "copy_to", None)
                if not callable(copy_to):
                    raise TypeError(
                        "single-rank device buffer must support stream copy_to"
                    )
                completions.append(copy_to(
                    destination.buffer, stream=stream,
                    source_offset=source.offset,
                    destination_offset=destination.offset,
                    bytes=source.byte_count,
                ))
            return completions[-1] if completions else stream.record_event()

        lib = self._api.library
        self._api.check(lib.ncclGroupStart(), "ncclGroupStart")
        try:
            # Receives first makes symmetric neighbor exchanges deterministic
            # while group semantics enqueue one operation.
            for peer, region in receives:
                self._api.check(
                    lib.ncclRecv(
                        region.address, region.byte_count, _NCCL_UINT8, peer,
                        self._communicator, stream.address,
                    ),
                    "ncclRecv",
                )
            for peer, region in sends:
                self._api.check(
                    lib.ncclSend(
                        region.address, region.byte_count, _NCCL_UINT8, peer,
                        self._communicator, stream.address,
                    ),
                    "ncclSend",
                )
        except BaseException:
            # Balance NCCL's thread-local group depth before propagating the
            # first enqueue error. ncclGroupEnd may itself report the same
            # asynchronous failure, which must not hide the original one.
            lib.ncclGroupEnd()
            raise
        self._api.check(lib.ncclGroupEnd(), "ncclGroupEnd")
        return stream.record_event()

    def close(self) -> None:
        if getattr(self, "_closed", True):
            return
        self._api.check(
            self._api.library.ncclCommDestroy(self._communicator),
            "ncclCommDestroy",
        )
        self._bootstrap_stream.close()
        self._closed = True

    def __enter__(self) -> "NCCLTransport":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


@dataclass(frozen=True)
class NCCLTransportProvider:
    capabilities: TransportCapabilities = TransportCapabilities(
        name="nccl",
        backends=("nccl", "cuda"),
        device_direct=True,
        asynchronous=True,
        object_messages=False,
    )

    def create(self, **options) -> NCCLTransport:
        rank = options.pop("rank", None)
        world_size = options.pop("world_size", None)
        communicator_id = options.pop("communicator_id", None)
        device = options.pop("device", "cuda:0")
        library = options.pop("library", None)
        if options:
            raise TypeError(f"unknown NCCL transport options: {sorted(options)}")
        if rank is None or world_size is None:
            raise TypeError("NCCL transport requires rank and world_size")
        return NCCLTransport(
            rank, world_size, communicator_id,
            device=device, library=library,
        )


__all__ = ["NCCLTransport", "NCCLTransportProvider", "unique_id"]
