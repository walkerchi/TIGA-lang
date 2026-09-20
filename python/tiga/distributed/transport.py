"""Torch-free transport and exact owner/ghost halo exchange primitives."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import ContextVar
from dataclasses import dataclass
from multiprocessing.connection import Connection
import time
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

from .model import HaloMap, derive_halo_map


_ACTIVE_RUNTIME: ContextVar["DistributedRuntime | None"] = ContextVar(
    "tiga_distributed_runtime", default=None)


@runtime_checkable
class NeighborTransport(Protocol):
    rank: int
    world_size: int

    def send(self, peer: int, payload: object) -> None: ...
    def receive(self, peer: int) -> object: ...


@runtime_checkable
class FixedByteTransport(Protocol):
    """Optional zero-pickle path for compiler-known halo payload sizes."""

    def send_bytes(self, peer: int, payload: bytes) -> None: ...
    def receive_bytes(self, peer: int, size: int) -> bytes: ...


@dataclass(frozen=True)
class DeviceBufferSlice:
    """One provider-visible byte range in device memory.

    The transport ABI deliberately carries a buffer object plus an offset,
    rather than accepting an unowned integer pointer.  This keeps the
    allocation alive until the provider completion is recorded and lets the
    runtime validate range/device compatibility before entering NCCL/RCCL.
    """

    buffer: Any
    offset: int
    byte_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.offset, int) or self.offset < 0:
            raise ValueError("device buffer offset must be non-negative")
        if not isinstance(self.byte_count, int) or self.byte_count < 0:
            raise ValueError("device buffer byte_count must be non-negative")
        size = getattr(self.buffer, "nbytes", None)
        address = getattr(self.buffer, "address", None)
        if not isinstance(size, int) or not isinstance(address, int):
            raise TypeError("device buffer must expose integer nbytes/address")
        if self.offset + self.byte_count > size:
            raise ValueError("device buffer slice exceeds its allocation")

    @property
    def address(self) -> int:
        return int(self.buffer.address) + self.offset


@runtime_checkable
class DeviceBufferTransport(Protocol):
    """Stream-ordered GPU-direct point-to-point transport extension."""

    rank: int
    world_size: int

    def exchange_device(
        self,
        sends: Sequence[tuple[int, DeviceBufferSlice]],
        receives: Sequence[tuple[int, DeviceBufferSlice]],
        *,
        stream: Any,
    ) -> Any: ...


class PipeTransport:
    """Real multi-process transport over stdlib duplex Connections.

    It is a conformance/runtime provider, not a claim to replace NCCL/MPI.
    Vendor transports implement the same small neighbor contract.
    """

    def __init__(
        self, rank: int, world_size: int,
        connections: Mapping[int, Connection],
    ) -> None:
        if not 0 <= rank < world_size:
            raise ValueError("invalid transport rank")
        if any(peer == rank or not 0 <= peer < world_size for peer in connections):
            raise ValueError("invalid peer connection map")
        self.rank = rank
        self.world_size = world_size
        self._connections = dict(connections)
        # Same-host stdlib pipes are a semantic provider. Their communication
        # is usually too short to repay an extra interior/boundary launch.
        self.prefer_compute_overlap = False

    def send(self, peer: int, payload: object) -> None:
        try:
            connection = self._connections[peer]
        except KeyError as error:
            raise ValueError(f"rank {peer} is not connected") from error
        connection.send(payload)

    def receive(self, peer: int) -> object:
        try:
            connection = self._connections[peer]
        except KeyError as error:
            raise ValueError(f"rank {peer} is not connected") from error
        return connection.recv()


@dataclass(frozen=True)
class HaloBuffer:
    ids: tuple[int, ...]
    data: bytes
    element_bytes: int

    def __post_init__(self) -> None:
        if self.element_bytes <= 0 or len(self.data) != len(self.ids) * self.element_bytes:
            raise ValueError("halo data does not match ids/element size")

    def value_bytes(self, entity: int) -> bytes:
        try:
            offset = self.ids.index(entity) * self.element_bytes
        except ValueError as error:
            raise KeyError(entity) from error
        return self.data[offset:offset + self.element_bytes]


@dataclass(frozen=True)
class PackedHalo:
    messages: tuple[tuple[int, tuple[int, ...], bytes], ...]
    element_bytes: int


@dataclass(frozen=True)
class ReceivedHalo:
    values: tuple[tuple[int, bytes], ...]
    element_bytes: int
    packed_ids: tuple[int, ...] = ()
    packed_data: bytes = b""

    def __post_init__(self) -> None:
        if self.packed_ids and len(self.packed_data) != (
            len(self.packed_ids) * self.element_bytes
        ):
            raise ValueError("packed received halo size is inconsistent")


@dataclass
class HaloSlot:
    """Compiler-owned communication resource bound to a bundle resource."""

    value: PackedHalo | ReceivedHalo | None = None


@dataclass
class ShardedField:
    """Rank-local owned bytes plus compiler-installed ghost storage."""

    halo: HaloMap
    owned_data: bytes
    element_bytes: int
    ghosts: HaloBuffer | None = None

    def __post_init__(self) -> None:
        self.owned_data = bytes(self.owned_data)
        if self.element_bytes <= 0 or len(self.owned_data) != (
            self.halo.owned_entities * self.element_bytes
        ):
            raise ValueError("sharded field does not match halo ownership")

    def install_ghosts(self, ghosts: HaloBuffer) -> None:
        if ghosts.ids != self.halo.ghost_ids or ghosts.element_bytes != self.element_bytes:
            raise ValueError("ghost buffer does not match sharded field")
        self.ghosts = ghosts


def pack_halo(
    halo: HaloMap,
    owned_data: bytes | bytearray | memoryview,
    *,
    element_bytes: int,
) -> PackedHalo:
    """Pack compiler-derived send lists without communicating."""
    if element_bytes <= 0:
        raise ValueError("element_bytes must be positive")
    owned = bytes(owned_data)
    expected = halo.owned_entities * element_bytes
    if len(owned) != expected:
        raise ValueError(f"owned_data has {len(owned)} bytes, expected {expected}")
    messages = []
    for peer, ids in halo.send_to:
        locals_ = tuple(entity - halo.owned_begin for entity in ids)
        if any(not 0 <= local < halo.owned_entities for local in locals_):
            raise ValueError("send map references a non-owned entity")
        if locals_ and locals_ == tuple(range(locals_[0], locals_[0] + len(locals_))):
            begin = locals_[0] * element_bytes
            payload = owned[begin:begin + len(locals_) * element_bytes]
        else:
            buffer = bytearray(len(locals_) * element_bytes)
            for index, local in enumerate(locals_):
                begin = local * element_bytes
                output = index * element_bytes
                buffer[output:output + element_bytes] = owned[
                    begin:begin + element_bytes]
            payload = bytes(buffer)
        messages.append((peer, ids, payload))
    return PackedHalo(tuple(messages), element_bytes)


def exchange_packed(
    halo: HaloMap,
    packed: PackedHalo,
    *,
    transport: NeighborTransport,
) -> ReceivedHalo:
    """Exchange packed neighbor messages with independent send progress."""
    if transport.rank != halo.rank or transport.world_size != halo.world_size:
        raise ValueError("transport and halo topology disagree")
    expected_sends = tuple(peer for peer, _ids in halo.send_to)
    if tuple(peer for peer, _ids, _payload in packed.messages) != expected_sends:
        raise ValueError("packed sends disagree with compiler halo map")
    fixed_bytes = isinstance(transport, FixedByteTransport)
    with ThreadPoolExecutor(max_workers=max(1, len(packed.messages))) as executor:
        sends = [
            executor.submit(
                transport.send_bytes if fixed_bytes else transport.send,
                peer, payload if fixed_bytes else (ids, payload),
            )
            for peer, ids, payload in packed.messages
        ]
        received: dict[int, bytes] = {}
        fixed_chunks: list[tuple[tuple[int, ...], bytes]] = []
        for peer, expected_ids in halo.receive_from:
            if fixed_bytes:
                ids = expected_ids
                payload = transport.receive_bytes(
                    peer, len(ids) * packed.element_bytes)
            else:
                message = transport.receive(peer)
                if not isinstance(message, tuple) or len(message) != 2:
                    raise RuntimeError("malformed halo transport message")
                ids, payload = tuple(message[0]), bytes(message[1])
            if ids != expected_ids or len(payload) != len(ids) * packed.element_bytes:
                raise RuntimeError("halo peer payload disagrees with compiler map")
            if fixed_bytes:
                fixed_chunks.append((ids, payload))
            else:
                for index, entity in enumerate(ids):
                    begin = index * packed.element_bytes
                    received[entity] = payload[begin:begin + packed.element_bytes]
        for future in sends:
            future.result()
    if fixed_bytes:
        received_ids = tuple(entity for ids, _payload in fixed_chunks for entity in ids)
        if received_ids == halo.ghost_ids:
            data = b"".join(payload for _ids, payload in fixed_chunks)
        else:
            locations = {
                entity: (payload, index * packed.element_bytes)
                for ids, payload in fixed_chunks
                for index, entity in enumerate(ids)
            }
            if set(locations) != set(halo.ghost_ids):
                raise RuntimeError("halo exchange did not produce every ghost")
            data_buffer = bytearray(len(halo.ghost_ids) * packed.element_bytes)
            for index, entity in enumerate(halo.ghost_ids):
                payload, begin = locations[entity]
                output = index * packed.element_bytes
                data_buffer[output:output + packed.element_bytes] = payload[
                    begin:begin + packed.element_bytes]
            data = bytes(data_buffer)
        return ReceivedHalo(
            (), packed.element_bytes, packed_ids=halo.ghost_ids, packed_data=data)
    if set(received) != set(halo.ghost_ids):
        raise RuntimeError("halo exchange did not produce every ghost")
    return ReceivedHalo(
        tuple((entity, received[entity]) for entity in halo.ghost_ids),
        packed.element_bytes,
    )


def unpack_halo(halo: HaloMap, received: ReceivedHalo) -> HaloBuffer:
    """Install receive values in the deterministic compiler ghost order."""
    if received.packed_ids:
        if received.packed_ids != halo.ghost_ids:
            raise RuntimeError("packed halo order disagrees with compiler map")
        return HaloBuffer(
            received.packed_ids, received.packed_data, received.element_bytes)
    ids = tuple(entity for entity, _value in received.values)
    if ids != halo.ghost_ids:
        raise RuntimeError("received halo order disagrees with compiler map")
    return HaloBuffer(
        ids,
        b"".join(value for _entity, value in received.values),
        received.element_bytes,
    )


def collective_halo_maps(
    row_ptr: Sequence[int],
    col_idx: Sequence[int],
    *,
    num_entities: int,
    world_size: int,
) -> tuple[HaloMap, ...]:
    """Create mutually consistent receive/send maps for all ranks."""
    receive_only = tuple(
        derive_halo_map(
            row_ptr, col_idx, num_entities=num_entities,
            world_size=world_size, rank=rank,
        )
        for rank in range(world_size)
    )
    maps = []
    for owner in range(world_size):
        peer_requests: list[tuple[int, ...]] = []
        for peer in range(world_size):
            requests = dict(receive_only[peer].receive_from).get(owner, ())
            peer_requests.append(requests)
        maps.append(
            derive_halo_map(
                row_ptr, col_idx, num_entities=num_entities,
                world_size=world_size, rank=owner,
                peer_requests=peer_requests,
            )
        )
    return tuple(maps)


def collective_paged_halo_maps(graph, *, world_size: int) -> tuple[HaloMap, ...]:
    """Derive exact halo maps while retaining at most one CSR shard in RAM."""
    from .model import destination_owner, owned_range

    receive_only: list[HaloMap] = []
    for rank in range(world_size):
        begin, end = owned_range(graph.schema.num_dst, world_size, rank)
        _rows, columns = graph.paged_rows(begin, end)
        ghosts = tuple(sorted({
            source for source in columns if not begin <= source < end
        }))
        receives = tuple(
            (owner, tuple(
                source for source in ghosts
                if destination_owner(source, graph.schema.num_dst, world_size) == owner
            ))
            for owner in range(world_size)
        )
        receive_only.append(HaloMap(
            rank, world_size, begin, end, ghosts,
            tuple(item for item in receives if item[1]), (),
        ))
    result = []
    for owner, halo in enumerate(receive_only):
        sends = tuple(
            (peer, dict(receive_only[peer].receive_from).get(owner, ()))
            for peer in range(world_size)
        )
        result.append(HaloMap(
            halo.rank, halo.world_size, halo.owned_begin, halo.owned_end,
            halo.ghost_ids, halo.receive_from,
            tuple(item for item in sends if item[1]),
        ))
    return tuple(result)


def exchange_halo(
    halo: HaloMap,
    owned_data: bytes | bytearray | memoryview,
    *,
    element_bytes: int,
    transport: NeighborTransport,
) -> HaloBuffer:
    """Pack, exchange and unpack one compiler-derived neighbor halo.

    ``owned_data`` is destination-owner local storage ordered from
    ``halo.owned_begin``.  No framework tensor type is involved.
    """
    packed = pack_halo(halo, owned_data, element_bytes=element_bytes)
    received = exchange_packed(halo, packed, transport=transport)
    return unpack_halo(halo, received)


def reverse_halo_values(
    halo: HaloMap,
    combined_values: Sequence[object],
    *,
    row_width: int,
    transport: NeighborTransport,
) -> list[object]:
    """Apply the adjoint of the forward ``owned || ghosts`` installation."""
    if row_width <= 0:
        raise ValueError("reverse halo row_width must be positive")
    expected_rows = halo.owned_entities + len(halo.ghost_ids)
    if len(combined_values) != expected_rows * row_width:
        raise ValueError("reverse halo cotangent shape disagrees with halo map")
    if transport.rank != halo.rank or transport.world_size != halo.world_size:
        raise ValueError("transport and reverse halo topology disagree")
    owned = list(combined_values[:halo.owned_entities * row_width])
    ghost_values = combined_values[halo.owned_entities * row_width:]
    ghost_rows = {
        entity: tuple(
            ghost_values[index * row_width:(index + 1) * row_width]
        )
        for index, entity in enumerate(halo.ghost_ids)
    }
    messages = tuple(
        (peer, ids, tuple(ghost_rows[entity] for entity in ids))
        for peer, ids in halo.receive_from
    )
    with ThreadPoolExecutor(max_workers=max(1, len(messages))) as executor:
        sends = [
            executor.submit(transport.send, peer, (ids, rows))
            for peer, ids, rows in messages
        ]
        for peer, expected_ids in halo.send_to:
            message = transport.receive(peer)
            if not isinstance(message, tuple) or len(message) != 2:
                raise RuntimeError("malformed reverse halo transport message")
            ids, rows = tuple(message[0]), tuple(message[1])
            if ids != expected_ids or len(rows) != len(ids):
                raise RuntimeError("reverse halo payload disagrees with compiler map")
            for entity, row in zip(ids, rows, strict=True):
                if len(row) != row_width:
                    raise RuntimeError("reverse halo row width is malformed")
                local = entity - halo.owned_begin
                if not 0 <= local < halo.owned_entities:
                    raise RuntimeError("reverse halo targets a non-owned entity")
                begin = local * row_width
                for column, value in enumerate(row):
                    owned[begin + column] += value
        for future in sends:
            future.result()
    return owned


def reverse_halo_device(cotangent, halo: HaloMap, transport: DeviceBufferTransport):
    """Build the adjoint halo using stream-ordered device buffers only.

    Ghost cotangents are returned to their owners. Received contributions are
    reduced into owned rows with ordinary Tiga Tensor primitives, so the
    consumer remains provider-independent and no device value is decoded by
    Python.
    """
    from ..runtime import Buffer, Device, DeviceType, Stream
    from ..tensor import Tensor, int64, tensor

    if not isinstance(cotangent, Tensor) or cotangent.device.type != DeviceType.CUDA:
        raise TypeError("device reverse halo requires a CUDA tg.Tensor")
    if not isinstance(transport, DeviceBufferTransport):
        raise TypeError("device reverse halo requires DeviceBufferTransport")
    if transport.rank != halo.rank or transport.world_size != halo.world_size:
        raise ValueError("transport and reverse halo topology disagree")
    expected_rows = halo.owned_entities + len(halo.ghost_ids)
    if cotangent.ndim == 0 or cotangent.shape[0] != expected_rows:
        raise ValueError("reverse halo cotangent shape disagrees with halo map")
    if not cotangent.is_contiguous:
        raise NotImplementedError("device reverse halo requires contiguous cotangents")
    cotangent.realize()
    physical = cotangent._buffer
    temporary_source = False
    if not isinstance(physical, Buffer):
        physical = Buffer.wrap_address(
            physical.address, physical.nbytes, device=cotangent.device,
            owner=physical,
        )
        temporary_source = True
    row_bytes = cotangent.dtype.itemsize * cotangent.numel // expected_rows
    ghost_position = {
        entity: halo.owned_entities + index
        for index, entity in enumerate(halo.ghost_ids)
    }
    stream = Stream(cotangent.device)
    packed_tensors: list[Tensor] = []
    packed_buffers: list[Buffer] = []
    received_buffer: Buffer | None = None
    try:
        sends: list[tuple[int, DeviceBufferSlice]] = []
        for peer, ids in halo.receive_from:
            positions = tuple(ghost_position[entity] for entity in ids)
            if not positions:
                continue
            if positions == tuple(range(positions[0], positions[0] + len(positions))):
                region = DeviceBufferSlice(
                    physical,
                    (cotangent.offset + positions[0] * cotangent.strides[0])
                    * cotangent.dtype.itemsize,
                    len(positions) * row_bytes,
                )
            else:
                indices = tensor(positions, dtype=int64, device=cotangent.device)
                packed = cotangent.gather(indices)
                packed.realize()
                packed_tensors.append(packed)
                packed_physical = packed._buffer
                if not isinstance(packed_physical, Buffer):
                    packed_physical = Buffer.wrap_address(
                        packed_physical.address, packed_physical.nbytes,
                        device=cotangent.device, owner=packed_physical,
                    )
                    packed_buffers.append(packed_physical)
                region = DeviceBufferSlice(
                    packed_physical,
                    packed.offset * packed.dtype.itemsize,
                    packed.nbytes,
                )
            sends.append((peer, region))

        receive_rows = sum(len(ids) for _peer, ids in halo.send_to)
        receives: list[tuple[int, DeviceBufferSlice]] = []
        if receive_rows:
            received_buffer = Buffer(receive_rows * row_bytes, device=cotangent.device)
            offset = 0
            for peer, ids in halo.send_to:
                byte_count = len(ids) * row_bytes
                if byte_count:
                    receives.append((
                        peer, DeviceBufferSlice(received_buffer, offset, byte_count)
                    ))
                    offset += byte_count
        transport.exchange_device(tuple(sends), tuple(receives), stream=stream)
        stream.synchronize()
    finally:
        stream.close()
        for buffer in packed_buffers:
            buffer.close()
        if temporary_source:
            physical.close()

    owned_indices = tensor(
        tuple(range(halo.owned_entities)),
        dtype=int64, device=cotangent.device,
    )
    owned = cotangent.gather(owned_indices)
    owned.realize()
    owned = Tensor(
        owned.shape, dtype=owned.dtype, device=owned.device,
        buffer=owned._buffer, offset=owned.offset, strides=owned.strides,
        requires_grad=False, version=owned.version,
        ready_event=owned.ready_event,
    )
    if received_buffer is None:
        return owned
    received = Tensor(
        (sum(len(ids) for _peer, ids in halo.send_to), *cotangent.shape[1:]),
        dtype=cotangent.dtype, device=cotangent.device,
        buffer=received_buffer, requires_grad=False,
    )
    destinations = tensor(
        [
            entity - halo.owned_begin
            for _peer, ids in halo.send_to for entity in ids
        ],
        dtype=int64, device=cotangent.device,
    )
    contribution = received.segment_sum(destinations, halo.owned_entities)
    contribution.realize()
    contribution = Tensor(
        contribution.shape, dtype=contribution.dtype,
        device=contribution.device, buffer=contribution._buffer,
        offset=contribution.offset, strides=contribution.strides,
        requires_grad=False, version=contribution.version,
        ready_event=contribution.ready_event,
    )
    return owned + contribution


class HaloPackExecutable:
    def __init__(self, halo: HaloMap, element_bytes: int) -> None:
        self.halo, self.element_bytes = halo, element_bytes

    def launch(self, *, source: ShardedField, send: HaloSlot) -> None:
        send.value = pack_halo(
            self.halo, source.owned_data, element_bytes=self.element_bytes)


class HaloTransportExecutable:
    def __init__(self, halo: HaloMap, transport: NeighborTransport) -> None:
        self.halo, self.transport = halo, transport

    def launch(self, *, send: HaloSlot, receive: HaloSlot) -> None:
        if not isinstance(send.value, PackedHalo):
            raise RuntimeError("halo exchange launched before pack")
        receive.value = exchange_packed(
            self.halo, send.value, transport=self.transport)


class HaloUnpackExecutable:
    def __init__(self, halo: HaloMap) -> None:
        self.halo = halo

    def launch(self, *, receive: HaloSlot, destination: ShardedField) -> None:
        if not isinstance(receive.value, ReceivedHalo):
            raise RuntimeError("halo unpack launched before exchange")
        destination.install_ghosts(unpack_halo(self.halo, receive.value))


class DistributedTaskResolver:
    """Bind typed compiler halo symbols to transport executables."""

    def __init__(
        self,
        halo: HaloMap,
        transport: NeighborTransport,
        *,
        element_bytes: int,
        fallback: Callable[[Any], Any] | None = None,
    ) -> None:
        self.halo = halo
        self.transport = transport
        self.element_bytes = element_bytes
        self.fallback = fallback

    def __call__(self, invocation):
        if invocation.executable_symbol == "__gf_halo_pack":
            return HaloPackExecutable(self.halo, self.element_bytes)
        if invocation.executable_symbol == "__gf_halo_exchange":
            return HaloTransportExecutable(self.halo, self.transport)
        if invocation.executable_symbol == "__gf_halo_unpack":
            return HaloUnpackExecutable(self.halo)
        if self.fallback is None:
            raise KeyError(
                f"no executable for compiler task {invocation.executable_symbol!r}")
        return self.fallback(invocation)

    @staticmethod
    def allocate_communication(plan) -> dict[str, HaloSlot]:
        return {
            requirement.name: HaloSlot()
            for requirement in plan.resources
            if requirement.memory_space == "remote"
        }


class HaloExchangeExecutable:
    """Bundle executable used to bind a typed ``halo_exchange`` task."""

    def __init__(self, halo: HaloMap, transport: NeighborTransport, element_bytes: int):
        self.halo = halo
        self.transport = transport
        self.element_bytes = element_bytes

    def launch(self, *, owned_data) -> HaloBuffer:
        return exchange_halo(
            self.halo, owned_data, element_bytes=self.element_bytes,
            transport=self.transport,
        )


class DistributedCompletion:
    """Completion for communication progressed by the runtime.

    Python futures expose ``done()`` whereas Tiga bundle execution uses
    the provider-neutral ``ready``/``wait`` contract.  This adapter keeps the
    executor below the distributed runtime boundary.
    """

    def __init__(
        self, future: Future[HaloBuffer], timing: dict[str, int] | None = None
    ) -> None:
        self._future = future
        self._timing = {} if timing is None else timing

    @property
    def ready(self) -> bool:
        return self._future.done()

    def wait(self) -> None:
        self._future.result()

    def result(self) -> HaloBuffer:
        return self._future.result()

    @property
    def started_ns(self) -> int | None:
        return self._timing.get("started_ns")

    @property
    def finished_ns(self) -> int | None:
        return self._timing.get("finished_ns")


class DistributedRuntime:
    """Own asynchronous communication progress below graph kernels."""

    def __init__(self, transport: NeighborTransport, *, progress_threads: int = 1):
        if progress_threads <= 0:
            raise ValueError("progress_threads must be positive")
        self.transport = transport
        self._executor = ThreadPoolExecutor(max_workers=progress_threads)
        self._context_token = None
        self._last_execution_trace: dict[str, object] | None = None
        # Public execution has one policy: complete halo exchange, then compute.
        # Explicit False is reserved for historical overlap regression probes;
        # transport preferences must never change the public default.
        self._force_serialized: bool | None = None

    @classmethod
    def from_provider(
        cls, name: str, *, progress_threads: int = 1, **options
    ) -> "DistributedRuntime":
        """Bind a deployment transport plugin below the unchanged graph API."""
        from .registry import create_transport

        return cls(
            create_transport(name, **options), progress_threads=progress_threads)

    def exchange_halo(
        self, halo: HaloMap, owned_data, *, element_bytes: int
    ) -> DistributedCompletion:
        """Start halo progress and return a provider-neutral completion."""
        timing: dict[str, int] = {}

        def launch() -> HaloBuffer:
            timing["started_ns"] = time.perf_counter_ns()
            try:
                return exchange_halo(
                    halo, owned_data, element_bytes=element_bytes,
                    transport=self.transport,
                )
            finally:
                timing["finished_ns"] = time.perf_counter_ns()

        return DistributedCompletion(self._executor.submit(launch), timing)

    @property
    def last_execution_trace(self) -> dict[str, object] | None:
        """Return timings from the latest automatic sharded execution."""
        return (
            None if self._last_execution_trace is None
            else dict(self._last_execution_trace)
        )

    def close(self) -> None:
        self._executor.shutdown(wait=True)

    def __enter__(self) -> "DistributedRuntime":
        if self._context_token is not None:
            raise RuntimeError("distributed runtime context is already active")
        self._context_token = _ACTIVE_RUNTIME.set(self)
        return self

    def __exit__(self, *_exc) -> None:
        try:
            self.close()
        finally:
            assert self._context_token is not None
            _ACTIVE_RUNTIME.reset(self._context_token)
            self._context_token = None


def current_distributed_runtime() -> DistributedRuntime | None:
    return _ACTIVE_RUNTIME.get()


@dataclass(frozen=True)
class _ShardRowPlan:
    """Cached destination split used by the automatic halo executor.

    Interior rows reference owned sources only. Boundary rows may reference
    ghosts and therefore depend on halo completion.  Row and edge index
    tensors let the ordinary MessagePassing frontend consume both partitions;
    no distributed-only kernel is introduced.
    """

    source_ids: tuple[int, ...]
    local_rows: tuple[int, ...]
    local_columns: tuple[int, ...]
    local_edges: int
    full_graph: Any
    interior_graph: Any | None
    interior_rows: Any | None
    interior_edges: Any | None
    interior_inverse: Any | None
    boundary_graph: Any | None
    boundary_rows: Any | None
    boundary_edges: Any | None
    boundary_inverse: Any | None


def _shard_row_plan(graph, halo: HaloMap, local_rows, columns) -> _ShardRowPlan:
    """Build and cache the physical interior/boundary CSR partition."""
    from ..graph import Graph
    from ..tensor import DType, int32, int64, tensor

    cache = getattr(graph, "_distributed_row_plan_cache", None)
    if cache is None:
        cache = {}
        graph._distributed_row_plan_cache = cache
    topology_inputs = (
        getattr(graph, "_row_ptr", None), getattr(graph, "_col_idx", None),
    )
    topology_versions = tuple(
        (
            id(value),
            getattr(value, "version", getattr(value, "_version", None)),
        )
        for value in topology_inputs if value is not None
    )
    cache_key = (
        halo.rank, halo.world_size, halo.owned_begin, halo.owned_end,
        graph.schema.realization, str(graph.device), topology_versions,
    )
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    local_rows = tuple(int(value) for value in local_rows)
    columns = tuple(int(value) for value in columns)
    source_ids = (
        *range(halo.owned_begin, halo.owned_end), *halo.ghost_ids,
    )
    local_id = {entity: index for index, entity in enumerate(source_ids)}
    local_columns = tuple(local_id[entity] for entity in columns)
    interior = tuple(
        row for row in range(halo.owned_entities)
        if all(
            column < halo.owned_entities
            for column in local_columns[local_rows[row]:local_rows[row + 1]]
        )
    )
    interior_set = set(interior)
    boundary = tuple(
        row for row in range(halo.owned_entities) if row not in interior_set
    )
    index_dtype = graph.schema.index_dtype
    if not isinstance(index_dtype, DType):
        index_dtype = int32 if "int32" in str(index_dtype) else int64
    full_graph = Graph.from_csr(
        tensor(local_rows, dtype=index_dtype, device=str(graph.device)),
        tensor(local_columns, dtype=index_dtype, device=str(graph.device)),
        num_src=len(source_ids), validate="basic",
    )

    def make_partition(rows: tuple[int, ...], *, num_src: int):
        if not rows:
            return None, None, None, None
        row_ptr = [0]
        partition_columns: list[int] = []
        edge_ids: list[int] = []
        for row in rows:
            begin, end = local_rows[row], local_rows[row + 1]
            partition_columns.extend(local_columns[begin:end])
            edge_ids.extend(range(begin, end))
            row_ptr.append(len(partition_columns))
        partition_graph = Graph.from_csr(
            tensor(row_ptr, dtype=index_dtype, device=str(graph.device)),
            tensor(
                partition_columns,
                dtype=index_dtype,
                device=str(graph.device),
            ),
            num_src=num_src,
            validate="basic",
        )
        inverse = [-1] * halo.owned_entities
        for source_row, destination_row in enumerate(rows):
            inverse[destination_row] = source_row
        return (
            partition_graph,
            tensor(rows, dtype=int64, device=str(graph.device)),
            tensor(edge_ids, dtype=int64, device=str(graph.device)),
            tensor(inverse, dtype=int64, device=str(graph.device)),
        )

    interior_graph, interior_rows, interior_edges, interior_inverse = make_partition(
        interior, num_src=halo.owned_entities)
    boundary_graph, boundary_rows, boundary_edges, boundary_inverse = make_partition(
        boundary, num_src=len(source_ids))
    result = _ShardRowPlan(
        source_ids=tuple(source_ids),
        local_rows=local_rows,
        local_columns=local_columns,
        local_edges=len(columns),
        full_graph=full_graph,
        interior_graph=interior_graph,
        interior_rows=interior_rows,
        interior_edges=interior_edges,
        interior_inverse=interior_inverse,
        boundary_graph=boundary_graph,
        boundary_rows=boundary_rows,
        boundary_edges=boundary_edges,
        boundary_inverse=boundary_inverse,
    )
    cache[cache_key] = result
    return result


def execute_sharded_message_passing(
    kernel,
    *,
    graph,
    src: Mapping[str, Any],
    dst: Mapping[str, Any],
    edge: Mapping[str, Any],
    params: Mapping[str, Any],
):
    """Execute one destination-sharded static CSR forward program.

    Complete halo exchange before computing all owned rows in one local call.
    Native autograd reverses ghost cotangents. The interior/boundary split is
    retained only for explicitly enabled internal regression probes.
    """
    from ..graph import Graph
    from ..runtime import Buffer, Device, DeviceType, Stream
    from ..tensor import Tensor, int64, tensor
    from ..tensor.core import _Expr

    runtime = current_distributed_runtime()
    if runtime is None:
        raise NotImplementedError(
            "distributed MessagePassing needs an active DistributedRuntime")
    placement = graph.placement
    graph_device = Device.parse(str(graph.device))
    if placement is None or placement.mesh.size != runtime.transport.world_size:
        raise ValueError("graph mesh and active transport topology disagree")
    if graph.schema.lifecycle != "static" or graph.schema.realization not in {
        "materialized_csr", "paged_csr"
    }:
        raise NotImplementedError(
            "automatic sharded MessagePassing currently requires static CSR")
    if graph.schema.num_src != graph.schema.num_dst:
        raise NotImplementedError(
            "automatic sharded MessagePassing currently requires one entity domain")
    fields = {**src, **dst, **edge}
    if any(not isinstance(value, Tensor) for value in fields.values()):
        raise TypeError("sharded MessagePassing fields must be tiga.Tensor values")
    if any(value.device != graph_device or not value.is_contiguous
           for value in fields.values()):
        raise NotImplementedError(
            "sharded fields must be contiguous native storage on the graph device")

    topology_inputs = (
        getattr(graph, "_row_ptr", None), getattr(graph, "_col_idx", None),
    )
    topology_versions = tuple(
        (
            id(value),
            getattr(value, "version", getattr(value, "_version", None)),
        )
        for value in topology_inputs if value is not None
    )
    topology_cache = getattr(graph, "_distributed_topology_cache", None)
    if topology_cache is None:
        topology_cache = {}
        graph._distributed_topology_cache = topology_cache
    topology_key = (
        runtime.transport.rank, runtime.transport.world_size,
        graph.schema.realization, topology_versions,
    )
    topology = topology_cache.get(topology_key)
    if topology is None:
        if graph.schema.realization == "paged_csr":
            halos = collective_paged_halo_maps(
                graph, world_size=runtime.transport.world_size)
            halo = halos[runtime.transport.rank]
            local_rows, columns = graph.paged_rows(
                halo.owned_begin, halo.owned_end)
        else:
            rows_tensor, columns_tensor = graph.resolve_csr()
            rows = tuple(int(value) for value in rows_tensor.tolist())
            global_columns = tuple(
                int(value) for value in columns_tensor.tolist())
            halos = collective_halo_maps(
                rows, global_columns, num_entities=graph.schema.num_dst,
                world_size=runtime.transport.world_size,
            )
            halo = halos[runtime.transport.rank]
            edge_begin, edge_end = rows[halo.owned_begin], rows[halo.owned_end]
            local_rows = tuple(
                rows[row] - edge_begin
                for row in range(halo.owned_begin, halo.owned_end + 1)
            )
            columns = global_columns[edge_begin:edge_end]
        topology = (halo, tuple(local_rows), tuple(columns))
        topology_cache[topology_key] = topology
    halo, local_rows, columns = topology
    row_plan = _shard_row_plan(graph, halo, local_rows, columns)
    source_ids = row_plan.source_ids
    local_rows = row_plan.local_rows
    local_columns = row_plan.local_columns
    local_edges = row_plan.local_edges

    def runtime_buffer(value: Tensor) -> tuple[Buffer, bool]:
        value.realize()
        buffer = value._buffer
        if isinstance(buffer, Buffer):
            return buffer, False
        address = getattr(buffer, "address", None)
        size = getattr(buffer, "nbytes", None)
        device = getattr(buffer, "device", None)
        if not isinstance(address, int) or not isinstance(size, int):
            raise NotImplementedError(
                "sharded fields require addressable provider storage")
        return Buffer.wrap_address(
            address, size, device=device, owner=buffer), True

    def physical_bytes(value: Tensor) -> bytes:
        buffer, temporary = runtime_buffer(value)
        try:
            return buffer.read(
                offset=value.offset * value.dtype.itemsize, bytes=value.nbytes)
        finally:
            if temporary:
                buffer.close()

    device_direct = (
        graph_device.type == DeviceType.CUDA
        and isinstance(runtime.transport, DeviceBufferTransport)
    )
    prefer_overlap = runtime._force_serialized is False
    host_overlap = (
        graph_device.type in {DeviceType.CPU, DeviceType.CUDA}
        and not device_direct
        and prefer_overlap
    )
    device_overlap = device_direct and prefer_overlap
    communication_stream = Stream(graph_device) if device_direct else None

    # Device halo storage must outlive every operation enqueued on the
    # communication stream.  Cleanup is deliberately separated from enqueue:
    # synchronizing inside ``device_direct_halo`` would serialize halo traffic
    # before the compiler can launch the independent interior partition.
    device_completions: list[Any] = []
    device_temporary_buffers: list[Buffer] = []
    device_receive_buffers: list[Buffer] = []
    device_retained_values: list[Tensor] = []

    def release_device_halo_temporaries() -> None:
        for completion in device_completions:
            close = getattr(completion, "close", None)
            if callable(close):
                close()
        device_completions.clear()
        for buffer in device_receive_buffers:
            buffer.close()
        device_receive_buffers.clear()
        for buffer in device_temporary_buffers:
            buffer.close()
        device_temporary_buffers.clear()
        device_retained_values.clear()

    def enqueue_device_direct_halo(value: Tensor) -> Buffer:
        assert communication_stream is not None
        source, source_temporary = runtime_buffer(value)
        row_bytes = value.dtype.itemsize * value.numel // value.shape[0]
        sends: list[tuple[int, DeviceBufferSlice]] = []
        receive_buffers: list[tuple[tuple[int, ...], Buffer]] = []
        if source_temporary:
            device_temporary_buffers.append(source)
        for peer, ids in halo.send_to:
            local_ids = tuple(entity - halo.owned_begin for entity in ids)
            if not local_ids:
                continue
            if local_ids == tuple(
                range(local_ids[0], local_ids[0] + len(local_ids))
            ):
                region = DeviceBufferSlice(
                    source,
                    (value.offset + local_ids[0] * value.strides[0])
                    * value.dtype.itemsize,
                    len(local_ids) * row_bytes,
                )
            else:
                indices = tensor(local_ids, dtype=int64, device=graph_device)
                packed = value.gather(indices)
                packed.realize()
                device_retained_values.append(packed)
                packed_buffer, temporary = runtime_buffer(packed)
                if temporary:
                    device_temporary_buffers.append(packed_buffer)
                region = DeviceBufferSlice(
                    packed_buffer,
                    packed.offset * packed.dtype.itemsize,
                    packed.nbytes,
                )
            sends.append((peer, region))

        receives: list[tuple[int, DeviceBufferSlice]] = []
        for peer, ids in halo.receive_from:
            buffer = Buffer(len(ids) * row_bytes, device=graph_device)
            receive_buffers.append((ids, buffer))
            device_receive_buffers.append(buffer)
            receives.append((peer, DeviceBufferSlice(buffer, 0, buffer.nbytes)))

        device_completions.append(runtime.transport.exchange_device(
            tuple(sends), tuple(receives), stream=communication_stream))
        combined = Buffer(
            value.nbytes + len(halo.ghost_ids) * row_bytes,
            device=graph_device,
        )
        device_completions.append(source.copy_to(
            combined,
            stream=communication_stream,
            source_offset=value.offset * value.dtype.itemsize,
            bytes=value.nbytes,
        ))
        ghost_position = {
            entity: index for index, entity in enumerate(halo.ghost_ids)
        }
        for ids, received in receive_buffers:
            positions = tuple(ghost_position[entity] for entity in ids)
            runs: list[tuple[int, int, int]] = []
            run_source = run_destination = run_count = 0
            for source_row, destination_row in enumerate(positions):
                if run_count and destination_row == run_destination + run_count:
                    run_count += 1
                    continue
                if run_count:
                    runs.append((run_source, run_destination, run_count))
                run_source = source_row
                run_destination = destination_row
                run_count = 1
            if run_count:
                runs.append((run_source, run_destination, run_count))
            for source_row, destination_row, count in runs:
                device_completions.append(received.copy_to(
                    combined,
                    stream=communication_stream,
                    source_offset=source_row * row_bytes,
                    destination_offset=value.nbytes + destination_row * row_bytes,
                    bytes=count * row_bytes,
                ))
        return combined

    for name, value in src.items():
        if value.shape[0] != halo.owned_entities:
            raise ValueError(
                f"src.{name} must contain {halo.owned_entities} owned rows")
    for name, value in dst.items():
        if value.shape[0] != halo.owned_entities:
            raise ValueError(
                f"dst.{name} must contain {halo.owned_entities} owned rows")
    for name, value in edge.items():
        if value.shape[0] != local_edges:
            raise ValueError(
                f"edge.{name} must contain {local_edges} rank-local edges")

    def launch_partition(partition_graph, row_index, edge_index, source_fields):
        if partition_graph is None:
            return None
        return kernel(
            graph=partition_graph,
            src=source_fields,
            dst={name: value.gather(row_index) for name, value in dst.items()},
            edge={name: value.gather(edge_index) for name, value in edge.items()},
            **dict(params),
        )

    # Host transports progress in the runtime worker while the main thread
    # compiles/executes rows whose sources are entirely owned.  Starting every
    # field before the first interior launch preserves overlap even when one
    # kernel consumes multiple source fields.
    pending_halos: dict[str, tuple[Tensor, bytes, DistributedCompletion]] = {}
    device_halos: dict[str, Buffer] = {}
    interior_output = None
    interior_started_ns: int | None = None
    interior_finished_ns: int | None = None
    communication_enqueued_ns: int | None = None
    communication_wait_started_ns: int | None = None
    communication_wait_finished_ns: int | None = None

    def realize_interior_partition() -> None:
        nonlocal interior_output, interior_started_ns, interior_finished_ns
        interior_output = launch_partition(
            row_plan.interior_graph,
            row_plan.interior_rows,
            row_plan.interior_edges,
            src,
        )
        if interior_output is not None:
            # Tensor execution is lazy. Realization is the ordering point that
            # submits/executes independent interior work before the halo wait.
            interior_started_ns = time.perf_counter_ns()
            interior_output.realize()
            interior_finished_ns = time.perf_counter_ns()

    if host_overlap:
        for name in sorted(src):
            value = src[name]
            owned = physical_bytes(value)
            element_bytes = (
                value.dtype.itemsize * value.numel // value.shape[0]
                if value.shape[0] else value.dtype.itemsize
            )
            pending_halos[name] = (
                value,
                owned,
                runtime.exchange_halo(
                    halo, owned, element_bytes=element_bytes),
            )
        try:
            realize_interior_partition()
        except BaseException:
            # Drain peer communication before propagating a compute failure so
            # the opposite rank cannot remain blocked in its exchange.
            for _value, _owned, completion in pending_halos.values():
                try:
                    completion.wait()
                except Exception:
                    pass
            raise

    if device_direct:
        assert communication_stream is not None
        try:
            # Enqueue all fields first so one field cannot serialize another.
            for name in sorted(src):
                device_halos[name] = enqueue_device_direct_halo(src[name])
            communication_enqueued_ns = time.perf_counter_ns()
            if device_overlap:
                # Native CUDA executables own an independent compute stream;
                # NCCL/D2D halo work remains ordered on communication_stream.
                realize_interior_partition()
            communication_wait_started_ns = time.perf_counter_ns()
            communication_stream.synchronize()
            communication_wait_finished_ns = time.perf_counter_ns()
        except BaseException:
            try:
                communication_stream.synchronize()
            except Exception:
                pass
            release_device_halo_temporaries()
            for buffer in device_halos.values():
                buffer.close()
            communication_stream.close()
            raise
        release_device_halo_temporaries()

    local_src = {}
    try:
        for name in sorted(src):
            value = src[name]
            if device_direct:
                combined = device_halos[name]
                transport_kind = "device-direct"
            elif host_overlap:
                _pending_value, owned, completion = pending_halos[name]
                ghosts = completion.result()
                combined = Buffer(
                    len(owned) + len(ghosts.data), device=graph_device)
                combined.write(owned + ghosts.data)
                transport_kind = "host-staged"
            else:
                owned = physical_bytes(value)
                ghosts = exchange_halo(
                    halo, owned,
                    element_bytes=(
                        value.dtype.itemsize * value.numel // value.shape[0]
                    ),
                    transport=runtime.transport,
                )
                combined = Buffer(
                    len(owned) + len(ghosts.data), device=graph_device)
                combined.write(owned + ghosts.data)
                transport_kind = "host-staged"
            local_src[name] = Tensor(
                (len(source_ids), *value.shape[1:]), dtype=value.dtype,
                device=graph_device, buffer=combined,
                requires_grad=value.requires_grad,
                expression=(
                    _Expr(
                        "distributed_halo_snapshot", (value,),
                        (
                            ("halo", halo),
                            ("owned_rows", halo.owned_entities),
                            ("transport", transport_kind),
                        ),
                    )
                    if value.requires_grad else None
                ), version=value.version,
            )
    finally:
        if communication_stream is not None:
            communication_stream.close()

    if host_overlap:
        communication_starts = [
            completion.started_ns
            for _value, _owned, completion in pending_halos.values()
            if completion.started_ns is not None
        ]
        communication_ends = [
            completion.finished_ns
            for _value, _owned, completion in pending_halos.values()
            if completion.finished_ns is not None
        ]
        communication_started_ns = (
            min(communication_starts) if communication_starts else None)
        communication_finished_ns = (
            max(communication_ends) if communication_ends else None)
        overlap_ns = 0
        if (communication_started_ns is not None
                and communication_finished_ns is not None
                and interior_started_ns is not None
                and interior_finished_ns is not None):
            overlap_ns = max(
                0,
                min(communication_finished_ns, interior_finished_ns)
                - max(communication_started_ns, interior_started_ns),
            )
        runtime._last_execution_trace = {
            "schema": "tiga.distributed-execution-trace.v1",
            "schedule": ("interior||host-staged-halo->boundary"
                         if graph_device.type == DeviceType.CUDA
                         else "interior||halo->boundary"),
            "timing_kind": "host-wall-clock",
            "transport_kind": "host-staged",
            "interior_rows": (
                0 if row_plan.interior_rows is None
                else row_plan.interior_rows.shape[0]
            ),
            "boundary_rows": (
                0 if row_plan.boundary_rows is None
                else row_plan.boundary_rows.shape[0]
            ),
            "communication_started_ns": communication_started_ns,
            "communication_finished_ns": communication_finished_ns,
            "interior_started_ns": interior_started_ns,
            "interior_finished_ns": interior_finished_ns,
            "host_overlap_ms": overlap_ns / 1e6,
            # Host progress can overlap a synchronous CUDA realization, but
            # its wall-clock span includes launch/JIT overhead, not just GPU
            # execution. GPU concurrency requires provider events/profiling.
            "measured_overlap_ms": (None if graph_device.type == DeviceType.CUDA
                                    else overlap_ns / 1e6),
        }
    elif device_overlap:
        runtime._last_execution_trace = {
            "schema": "tiga.distributed-execution-trace.v1",
            "schedule": "interior||device-halo->boundary",
            "timing_kind": "host-submit-order",
            "interior_rows": (
                0 if row_plan.interior_rows is None
                else row_plan.interior_rows.shape[0]
            ),
            "boundary_rows": (
                0 if row_plan.boundary_rows is None
                else row_plan.boundary_rows.shape[0]
            ),
            "communication_enqueued_ns": communication_enqueued_ns,
            "communication_wait_started_ns": communication_wait_started_ns,
            "communication_wait_finished_ns": communication_wait_finished_ns,
            "interior_started_ns": interior_started_ns,
            "interior_finished_ns": interior_finished_ns,
            # These host timestamps establish submission/dependency order. A
            # CUDA profiler or a provider event pair is required to measure
            # actual device concurrency, so do not manufacture a duration.
            "measured_overlap_ms": None,
        }
    else:
        runtime._last_execution_trace = {
            "schema": "tiga.distributed-execution-trace.v1",
            "schedule": (
                "device-direct-serialized" if device_direct
                else "host-staged-serialized"
            ),
            "interior_rows": 0,
            "boundary_rows": halo.owned_entities,
            "measured_overlap_ms": 0.0,
        }

    if host_overlap or device_overlap:
        boundary_output = launch_partition(
            row_plan.boundary_graph,
            row_plan.boundary_rows,
            row_plan.boundary_edges,
            local_src,
        )
        if interior_output is None:
            assert boundary_output is not None
            return boundary_output
        if boundary_output is None:
            return interior_output
        # The two row sets are disjoint. The redundant destination/inverse maps
        # let the compiler emit a linear forward placement and a linear gather
        # VJP, avoiding the generic O(num_rows * partition_rows) segment path.
        return (
            interior_output._scatter_rows(
                row_plan.interior_rows,
                row_plan.interior_inverse,
                halo.owned_entities,
            )
            + boundary_output._scatter_rows(
                row_plan.boundary_rows,
                row_plan.boundary_inverse,
                halo.owned_entities,
            )
        )

    return kernel(
        graph=row_plan.full_graph, src=local_src, dst=dict(dst),
        edge=dict(edge), **dict(params),
    )


__all__ = [
    "DeviceBufferSlice", "DeviceBufferTransport",
    "DistributedCompletion", "DistributedRuntime", "DistributedTaskResolver",
    "FixedByteTransport", "HaloBuffer", "HaloExchangeExecutable", "HaloPackExecutable", "HaloSlot",
    "HaloTransportExecutable", "HaloUnpackExecutable", "NeighborTransport",
    "PackedHalo", "PipeTransport", "ReceivedHalo", "ShardedField",
    "collective_halo_maps", "current_distributed_runtime", "exchange_halo",
    "collective_paged_halo_maps",
    "exchange_packed", "execute_sharded_message_passing", "pack_halo",
    "reverse_halo_device", "reverse_halo_values", "unpack_halo",
]
