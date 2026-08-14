"""Torch-free transport and exact owner/ghost halo exchange primitives."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import ContextVar
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

from .model import HaloMap, derive_halo_map


_ACTIVE_RUNTIME: ContextVar["DistributedRuntime | None"] = ContextVar(
    "graphforge_distributed_runtime", default=None)


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
    reduced into owned rows with ordinary GraphForge Tensor primitives, so the
    consumer remains provider-independent and no device value is decoded by
    Python.
    """
    from ..runtime import Buffer, DeviceType, Stream
    from ..tensor import Tensor, int64, tensor

    if not isinstance(cotangent, Tensor) or cotangent.device.type != DeviceType.CUDA:
        raise TypeError("device reverse halo requires a CUDA gf.Tensor")
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

    Python futures expose ``done()`` whereas GraphForge bundle execution uses
    the provider-neutral ``ready``/``wait`` contract.  This adapter keeps the
    executor below the distributed runtime boundary.
    """

    def __init__(self, future: Future[HaloBuffer]) -> None:
        self._future = future

    @property
    def ready(self) -> bool:
        return self._future.done()

    def wait(self) -> None:
        self._future.result()

    def result(self) -> HaloBuffer:
        return self._future.result()


class DistributedRuntime:
    """Own asynchronous communication progress below graph kernels."""

    def __init__(self, transport: NeighborTransport, *, progress_threads: int = 1):
        if progress_threads <= 0:
            raise ValueError("progress_threads must be positive")
        self.transport = transport
        self._executor = ThreadPoolExecutor(max_workers=progress_threads)
        self._context_token = None

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
        return DistributedCompletion(
            self._executor.submit(
                exchange_halo, halo, owned_data, element_bytes=element_bytes,
                transport=self.transport,
            )
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

    This is the first end-to-end conformance path for the public Graph.halo
    semantics.  It intentionally supports native CPU, one-hop forward fields
    and rank-local edge arrays. Unsupported gradients/layouts/providers fail
    before communication.
    """
    from ..graph import Graph
    from ..runtime import Buffer, DeviceType, Stream
    from ..tensor import Tensor, int64, tensor
    from ..tensor.core import _Expr

    runtime = current_distributed_runtime()
    if runtime is None:
        raise NotImplementedError(
            "distributed MessagePassing needs an active DistributedRuntime")
    placement = graph.placement
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
        raise TypeError("sharded MessagePassing fields must be graphforge.Tensor values")
    if any(value.device != graph.device or not value.is_contiguous
           for value in fields.values()):
        raise NotImplementedError(
            "sharded fields must be contiguous native storage on the graph device")

    if graph.schema.realization == "paged_csr":
        halos = collective_paged_halo_maps(
            graph, world_size=runtime.transport.world_size)
        halo = halos[runtime.transport.rank]
        local_rows, columns = graph.paged_rows(
            halo.owned_begin, halo.owned_end)
        local_edges = len(columns)
    else:
        rows_tensor, columns_tensor = graph.resolve_csr()
        rows = tuple(int(value) for value in rows_tensor.tolist())
        global_columns = tuple(int(value) for value in columns_tensor.tolist())
        halos = collective_halo_maps(
            rows, global_columns, num_entities=graph.schema.num_dst,
            world_size=runtime.transport.world_size,
        )
        halo = halos[runtime.transport.rank]
        edge_begin, edge_end = rows[halo.owned_begin], rows[halo.owned_end]
        local_edges = edge_end - edge_begin
        local_rows = tuple(
            rows[row] - edge_begin
            for row in range(halo.owned_begin, halo.owned_end + 1)
        )
        columns = global_columns[edge_begin:edge_end]
    source_ids = (*range(halo.owned_begin, halo.owned_end), *halo.ghost_ids)
    local_id = {entity: index for index, entity in enumerate(source_ids)}
    local_columns = tuple(
        local_id[entity] for entity in columns)

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
        graph.device.type == DeviceType.CUDA
        and isinstance(runtime.transport, DeviceBufferTransport)
    )
    communication_stream = Stream(graph.device) if device_direct else None

    def device_direct_halo(value: Tensor) -> Buffer:
        assert communication_stream is not None
        source, source_temporary = runtime_buffer(value)
        row_bytes = value.dtype.itemsize * value.numel // value.shape[0]
        sends: list[tuple[int, DeviceBufferSlice]] = []
        packed_values: list[Tensor] = []
        packed_temporaries: list[Buffer] = []
        receive_buffers: list[tuple[tuple[int, ...], Buffer]] = []
        completions: list[Any] = []
        try:
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
                    indices = tensor(
                        local_ids, dtype=int64, device=graph.device)
                    packed = value.gather(indices)
                    packed.realize()
                    packed_values.append(packed)
                    packed_buffer, temporary = runtime_buffer(packed)
                    if temporary:
                        packed_temporaries.append(packed_buffer)
                    region = DeviceBufferSlice(
                        packed_buffer,
                        packed.offset * packed.dtype.itemsize,
                        packed.nbytes,
                    )
                sends.append((peer, region))

            receives: list[tuple[int, DeviceBufferSlice]] = []
            for peer, ids in halo.receive_from:
                buffer = Buffer(len(ids) * row_bytes, device=graph.device)
                receive_buffers.append((ids, buffer))
                receives.append(
                    (peer, DeviceBufferSlice(buffer, 0, buffer.nbytes)))

            completions.append(runtime.transport.exchange_device(
                tuple(sends), tuple(receives), stream=communication_stream))
            combined = Buffer(
                value.nbytes + len(halo.ghost_ids) * row_bytes,
                device=graph.device,
            )
            completions.append(source.copy_to(
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
                    completions.append(received.copy_to(
                        combined,
                        stream=communication_stream,
                        source_offset=source_row * row_bytes,
                        destination_offset=(
                            value.nbytes + destination_row * row_bytes
                        ),
                        bytes=count * row_bytes,
                    ))
            communication_stream.synchronize()
            return combined
        finally:
            # Every provider operation and D2D copy is complete after the
            # stream synchronization above. On an exception, wait for any
            # successfully enqueued work before releasing temporary storage.
            if completions:
                try:
                    communication_stream.synchronize()
                except Exception:
                    pass
            for _ids, buffer in receive_buffers:
                buffer.close()
            for buffer in packed_temporaries:
                buffer.close()
            if source_temporary:
                source.close()

    local_src = {}
    try:
        for name in sorted(src):
            value = src[name]
            if value.shape[0] != halo.owned_entities:
                raise ValueError(
                    f"src.{name} must contain {halo.owned_entities} owned rows")
            if device_direct:
                combined = device_direct_halo(value)
                transport_kind = "device-direct"
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
                    len(owned) + len(ghosts.data), device=graph.device)
                combined.write(owned + ghosts.data)
                transport_kind = "host-staged"
            local_src[name] = Tensor(
                (len(source_ids), *value.shape[1:]), dtype=value.dtype,
                device=graph.device, buffer=combined,
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

    local_dst = {}
    for name, value in dst.items():
        if value.shape[0] != halo.owned_entities:
            raise ValueError(
                f"dst.{name} must contain {halo.owned_entities} owned rows")
        local_dst[name] = value
    local_edge = {}
    for name, value in edge.items():
        if value.shape[0] != local_edges:
            raise ValueError(
                f"edge.{name} must contain {local_edges} rank-local edges")
        local_edge[name] = value
    local_graph = Graph.from_csr(
        tensor(local_rows, dtype=int64, device=graph.device),
        tensor(local_columns, dtype=int64, device=graph.device),
        num_src=len(source_ids),
        validate="full",
    )
    return kernel(
        graph=local_graph, src=local_src, dst=local_dst,
        edge=local_edge, **dict(params),
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
