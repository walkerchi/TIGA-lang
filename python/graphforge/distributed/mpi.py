"""Optional MPI neighbor transport; mpi4py is imported only on activation."""

from __future__ import annotations

from dataclasses import dataclass

from .registry import TransportCapabilities


class MPITransport:
    """GraphForge neighbor ABI over an mpi4py-compatible communicator."""

    def __init__(self, communicator, *, tag: int = 0) -> None:
        self._communicator = communicator
        self.rank = int(communicator.Get_rank())
        self.world_size = int(communicator.Get_size())
        self._tag = int(tag)
        if self.world_size <= 0 or not 0 <= self.rank < self.world_size:
            raise ValueError("MPI communicator returned an invalid topology")
        if self._tag < 0:
            raise ValueError("MPI tag must be non-negative")

    def send(self, peer: int, payload: object) -> None:
        self._communicator.send(payload, dest=peer, tag=self._tag)

    def receive(self, peer: int) -> object:
        return self._communicator.recv(source=peer, tag=self._tag)

    def send_bytes(self, peer: int, payload: bytes) -> None:
        self._communicator.Send(memoryview(payload), dest=peer, tag=self._tag)

    def receive_bytes(self, peer: int, size: int) -> bytes:
        storage = bytearray(size)
        self._communicator.Recv(memoryview(storage), source=peer, tag=self._tag)
        return bytes(storage)


@dataclass(frozen=True)
class MPITransportProvider:
    capabilities: TransportCapabilities = TransportCapabilities(
        name="mpi4py",
        backends=("mpi",),
        device_direct=False,
        asynchronous=True,
        object_messages=True,
    )

    def create(self, **options) -> MPITransport:
        communicator = options.pop("communicator", None)
        tag = options.pop("tag", 0)
        if options:
            raise TypeError(f"unknown MPI transport options: {sorted(options)}")
        if communicator is None:
            try:
                from mpi4py import MPI
            except (ImportError, RuntimeError) as error:
                raise ModuleNotFoundError(
                    "MPI transport requires the optional 'mpi4py' package"
                ) from error
            communicator = MPI.COMM_WORLD
        return MPITransport(communicator, tag=tag)


__all__ = ["MPITransport", "MPITransportProvider"]
