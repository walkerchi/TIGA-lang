"""Stdlib TCP neighbor transport with an explicit rank/address rendezvous.

This is the dependency-free multi-host option: one rank hosts a rendezvous
listener, every other rank joins through it, and the world then brings up a
full mesh of length-prefixed TCP connections with a deterministic rank
handshake.  Hostnames are resolved once, at connect time.  MPI remains the
battle-tested deployment path; this transport mirrors the exact
:class:`~.transport.PipeTransport` byte contract over plain sockets.
"""

from __future__ import annotations

import json
import pickle
import socket
import struct
import threading
import time
from dataclasses import dataclass

from .registry import TransportCapabilities

_REGISTER = struct.Struct("!4sIIH")  # magic, rank, world_size, listen port
_HANDSHAKE = struct.Struct("!4sII")  # magic, rank, expected peer
_HEADER = struct.Struct("!Q")
_REGISTER_MAGIC = b"GFR1"
_HANDSHAKE_MAGIC = b"GFT1"
_STATUS_OK = b"\x00"
_STATUS_ERROR = b"\x01"


def _listener(host: str, port: int) -> socket.socket:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host, port))
    listener.listen()
    return listener


def _dial(address: tuple[str, int], timeout: float) -> socket.socket:
    """Connect, retrying a refused peer until the timeout budget is spent."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            return socket.create_connection(address, timeout=timeout)
        except ConnectionRefusedError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)


def _recv_exactly(connection: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = connection.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("TCP peer closed the connection mid-frame")
        chunks.extend(chunk)
    return bytes(chunks)


def _send_frame(connection: socket.socket, payload: bytes) -> None:
    connection.sendall(_HEADER.pack(len(payload)) + payload)


def _recv_frame(connection: socket.socket) -> bytes:
    (size,) = _HEADER.unpack(_recv_exactly(connection, _HEADER.size))
    return _recv_exactly(connection, size)


def _send_error(connection: socket.socket, message: str) -> None:
    try:
        _send_frame(connection, _STATUS_ERROR + message.encode("utf-8"))
    except OSError:
        pass
    connection.close()


def _recv_status(connection: socket.socket) -> bytes:
    payload = _recv_frame(connection)
    if payload[:1] == _STATUS_ERROR:
        raise RuntimeError(payload[1:].decode("utf-8", errors="replace"))
    return payload[1:]


class TCPTransport:
    """Full-mesh stdlib TCP transport with a validated rank handshake.

    ``TCPTransport.host`` brings up the rendezvous listener side;
    ``TCPTransport.join`` dials it from every other rank.  Both return only
    after every rank pair has exchanged and validated rank ids, so the
    resulting transport offers the same send/receive contract as
    :class:`~.transport.PipeTransport`: object payloads are pickled, and the
    fixed-byte path exchanges length-prefixed frames of a known size.  The
    join side retries a refused rendezvous connection until ``timeout``, so
    ranks may start in any order.
    """

    def __init__(
        self, rank: int, world_size: int,
        connections: dict[int, socket.socket], *, port: int | None = None,
    ) -> None:
        if not 0 <= rank < world_size:
            raise ValueError("invalid transport rank")
        if any(peer == rank or not 0 <= peer < world_size for peer in connections):
            raise ValueError("invalid peer connection map")
        self.rank = rank
        self.world_size = world_size
        self._connections = dict(connections)
        self._send_locks = {peer: threading.Lock() for peer in connections}
        self._port = port
        # TCP deployments may span hosts; the automatic executor can hide a
        # meaningful fraction of their halo latency behind interior work.
        self.prefer_compute_overlap = True

    @property
    def port(self) -> int | None:
        """Actual rendezvous port on the host side (``None`` after join)."""
        return self._port

    @classmethod
    def host(
        cls, rank: int, world_size: int, *, host: str = "", port: int,
        timeout: float = 30.0,
    ) -> TCPTransport:
        """Bring up the rendezvous and wait for every other rank to join."""
        if not 0 <= rank < world_size:
            raise ValueError("invalid transport rank")
        rendezvous = _listener(host, port)
        mesh = _listener(host, 0)
        try:
            actual_port = rendezvous.getsockname()[1]
            plan, pending = cls._accept_registrations(
                rendezvous, rank, world_size, mesh.getsockname()[1],
                advertised_host=host, timeout=timeout)
            try:
                body = json.dumps(
                    sorted((peer, *address) for peer, address in plan.items())
                ).encode("utf-8")
                for connection in pending:
                    _send_frame(connection, _STATUS_OK + body)
                    connection.close()
            except BaseException:
                for connection in pending:
                    connection.close()
                raise
            connections = cls._build_mesh(
                mesh, plan, rank, world_size, rendezvous_host=host,
                timeout=timeout)
        except BaseException:
            rendezvous.close()
            mesh.close()
            raise
        rendezvous.close()
        return cls(rank, world_size, connections, port=actual_port)

    @classmethod
    def join(
        cls, rank: int, world_size: int, *, host: str, port: int,
        timeout: float = 30.0,
    ) -> TCPTransport:
        """Join the rendezvous hosted by another rank and build the mesh."""
        if not 0 <= rank < world_size:
            raise ValueError("invalid transport rank")
        listener = _listener("", 0)
        try:
            try:
                rendezvous = _dial((host, port), timeout)
            except OSError as error:
                raise ConnectionError(
                    f"rank {rank}: could not reach the rendezvous at "
                    f"{host}:{port}: {error}"
                ) from error
            rendezvous.settimeout(timeout)
            try:
                rendezvous.sendall(_REGISTER.pack(
                    _REGISTER_MAGIC, rank, world_size,
                    listener.getsockname()[1]))
                try:
                    body = _recv_status(rendezvous)
                except TimeoutError as error:
                    raise TimeoutError(
                        f"rank {rank}: timed out after {timeout}s waiting for "
                        f"the rendezvous at {host}:{port}"
                    ) from error
            finally:
                rendezvous.close()
            plan = {
                int(peer): (str(address), int(peer_port))
                for peer, address, peer_port in json.loads(body.decode("utf-8"))
            }
            if set(plan) != set(range(world_size)) or (
                plan[rank][1] != listener.getsockname()[1]
            ):
                raise RuntimeError("rendezvous returned an inconsistent rank plan")
            connections = cls._build_mesh(
                listener, plan, rank, world_size, rendezvous_host=host,
                timeout=timeout)
        except BaseException:
            listener.close()
            raise
        return cls(rank, world_size, connections)

    @staticmethod
    def _accept_registrations(
        listener: socket.socket, rank: int, world_size: int, mesh_port: int,
        *, advertised_host: str, timeout: float,
    ) -> tuple[dict[int, tuple[str, int]], list[socket.socket]]:
        listener.settimeout(timeout)
        advertised = "" if advertised_host in ("", "0.0.0.0") else advertised_host
        plan: dict[int, tuple[str, int]] = {rank: (advertised, mesh_port)}
        pending: list[socket.socket] = []
        try:
            while len(plan) < world_size:
                try:
                    connection, address = listener.accept()
                except TimeoutError as error:
                    missing = world_size - len(plan)
                    raise TimeoutError(
                        f"rank {rank}: timed out after {timeout}s waiting for "
                        f"{missing} rank(s) to join the rendezvous"
                    ) from error
                connection.settimeout(timeout)
                try:
                    magic, peer, peer_world, peer_port = _REGISTER.unpack(
                        _recv_exactly(connection, _REGISTER.size))
                    if magic != _REGISTER_MAGIC:
                        raise ValueError("bad rendezvous registration magic")
                    if peer_world != world_size:
                        raise ValueError(
                            f"peer announced world_size {peer_world}, "
                            f"expected {world_size}")
                    if not 0 <= peer < world_size or peer == rank:
                        raise ValueError(f"peer announced invalid rank {peer}")
                    if peer in plan:
                        raise ValueError(f"rank {peer} is already registered")
                except (TimeoutError, ValueError, ConnectionError) as error:
                    _send_error(connection, f"rendezvous rejected: {error}")
                    continue
                plan[peer] = (address[0], peer_port)
                pending.append(connection)
        except BaseException:
            for connection in pending:
                connection.close()
            raise
        return plan, pending

    @staticmethod
    def _build_mesh(
        listener: socket.socket, plan: dict[int, tuple[str, int]],
        rank: int, world_size: int, *, rendezvous_host: str, timeout: float,
    ) -> dict[int, socket.socket]:
        """Dial every lower rank and accept every higher rank, in any order."""
        connections: dict[int, socket.socket] = {}
        listener.settimeout(timeout)
        try:
            for peer in range(rank):
                host, peer_port = plan[peer]
                address = (host or rendezvous_host, peer_port)
                try:
                    connection = _dial(address, timeout)
                except OSError as error:
                    raise ConnectionError(
                        f"rank {rank}: could not connect to rank {peer} at "
                        f"{address[0]}:{address[1]}: {error}"
                    ) from error
                connection.settimeout(timeout)
                connection.sendall(_HANDSHAKE.pack(_HANDSHAKE_MAGIC, rank, peer))
                try:
                    _recv_status(connection)
                except TimeoutError as error:
                    connection.close()
                    raise TimeoutError(
                        f"rank {rank}: timed out after {timeout}s in the mesh "
                        f"handshake with rank {peer}"
                    ) from error
                connections[peer] = connection
            while len(connections) < world_size - 1:
                try:
                    connection, _address = listener.accept()
                except TimeoutError as error:
                    raise TimeoutError(
                        f"rank {rank}: timed out after {timeout}s waiting for "
                        f"mesh connections from higher ranks"
                    ) from error
                connection.settimeout(timeout)
                try:
                    magic, peer, expected = _HANDSHAKE.unpack(
                        _recv_exactly(connection, _HANDSHAKE.size))
                    if magic != _HANDSHAKE_MAGIC or expected != rank:
                        raise ValueError(
                            f"peer {peer} expected rank {expected}, not {rank}")
                    if not rank < peer < world_size or peer in connections:
                        raise ValueError(f"unexpected mesh peer rank {peer}")
                except (TimeoutError, ValueError, ConnectionError) as error:
                    _send_error(
                        connection,
                        f"rank {rank} rejected the mesh handshake: {error}")
                    continue
                _send_frame(connection, _STATUS_OK)
                connections[peer] = connection
        except BaseException:
            listener.close()
            for connection in connections.values():
                connection.close()
            raise
        listener.close()
        for connection in connections.values():
            connection.settimeout(None)
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return connections

    def _connection(self, peer: int) -> socket.socket:
        try:
            return self._connections[peer]
        except KeyError as error:
            raise ValueError(f"rank {peer} is not connected") from error

    def send(self, peer: int, payload: object) -> None:
        self.send_bytes(peer, pickle.dumps(payload))

    def receive(self, peer: int) -> object:
        return pickle.loads(_recv_frame(self._connection(peer)))

    def send_bytes(self, peer: int, payload: bytes) -> None:
        connection = self._connection(peer)
        with self._send_locks[peer]:
            _send_frame(connection, bytes(payload))

    def receive_bytes(self, peer: int, size: int) -> bytes:
        connection = self._connection(peer)
        (framed,) = _HEADER.unpack(_recv_exactly(connection, _HEADER.size))
        if framed != size:
            raise RuntimeError(
                f"TCP frame from rank {peer} has {framed} bytes, expected {size}")
        return _recv_exactly(connection, size)

    def close(self) -> None:
        for connection in self._connections.values():
            connection.close()


@dataclass(frozen=True)
class TCPTransportProvider:
    capabilities: TransportCapabilities = TransportCapabilities(
        name="tcp",
        backends=("tcp",),
        device_direct=False,
        asynchronous=True,
        object_messages=True,
    )

    def create(self, **options) -> TCPTransport:
        try:
            rank = options.pop("rank")
            world_size = options.pop("world_size")
            port = options.pop("port")
        except KeyError as error:
            raise TypeError(f"missing TCP transport option: {error}") from error
        host = options.pop("host", None)
        bind = options.pop("bind", "")
        timeout = options.pop("timeout", 30.0)
        if options:
            raise TypeError(f"unknown TCP transport options: {sorted(options)}")
        if host is None:
            return TCPTransport.host(
                rank, world_size, host=bind, port=port, timeout=timeout)
        return TCPTransport.join(
            rank, world_size, host=host, port=port, timeout=timeout)


__all__ = ["TCPTransport", "TCPTransportProvider"]
