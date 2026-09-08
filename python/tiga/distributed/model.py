from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence


@dataclass(frozen=True)
class DeviceMesh:
    """A declarative device mesh used by Tiga planning.

    This object deliberately does not initialize process groups.  A runtime
    may bind it to a Torch ``DeviceMesh`` or a vendor communicator when an
    executable is launched.
    """

    device_type: str
    shape: tuple[int, ...]
    names: tuple[str, ...] = ()

    def __init__(
        self,
        device_type: str,
        shape: int | tuple[int, ...],
        *,
        names: tuple[str, ...] = (),
    ) -> None:
        normalized = (shape,) if isinstance(shape, int) else tuple(shape)
        if not device_type:
            raise ValueError("device_type must be non-empty")
        if not normalized or any(
            not isinstance(extent, int) or extent <= 0 for extent in normalized
        ):
            raise ValueError("mesh shape must contain positive integers")
        if names and len(names) != len(normalized):
            raise ValueError("mesh names must match the mesh rank")
        if len(set(names)) != len(names):
            raise ValueError("mesh names must be unique")
        object.__setattr__(self, "device_type", device_type)
        object.__setattr__(self, "shape", normalized)
        object.__setattr__(self, "names", tuple(names))

    @property
    def size(self) -> int:
        return math.prod(self.shape)

    def specialization_key(self) -> tuple[object, ...]:
        return self.device_type, self.shape, self.names


@dataclass(frozen=True)
class ByDestination:
    """Assign destination entities and final reducer state to mesh shards."""

    mesh_axis: int | str = 0
    balance: Literal["edges", "entities", "auto"] = "auto"

    def __post_init__(self) -> None:
        if not isinstance(self.mesh_axis, (int, str)):
            raise TypeError("mesh_axis must be an integer or a named mesh axis")
        if self.balance not in {"edges", "entities", "auto"}:
            raise ValueError("balance must be 'edges', 'entities', or 'auto'")

    def specialization_key(self) -> tuple[object, ...]:
        return "by_destination", self.mesh_axis, self.balance


@dataclass(frozen=True)
class GraphPlacement:
    """Logical ownership and ghost requirements attached to a Graph snapshot."""

    mesh: DeviceMesh
    partition: ByDestination
    halo_depth: int | Literal["auto"]

    def __post_init__(self) -> None:
        if self.halo_depth != "auto" and (
            not isinstance(self.halo_depth, int) or self.halo_depth < 0
        ):
            raise ValueError("halo depth must be a non-negative integer or 'auto'")
        axis = self.partition.mesh_axis
        if isinstance(axis, int) and not -len(self.mesh.shape) <= axis < len(self.mesh.shape):
            raise ValueError("partition mesh_axis is outside the device mesh rank")
        if isinstance(axis, str) and axis not in self.mesh.names:
            raise ValueError(f"unknown named mesh axis {axis!r}")

    def specialization_key(self) -> tuple[object, ...]:
        return (
            *self.mesh.specialization_key(),
            *self.partition.specialization_key(),
            self.halo_depth,
        )


@dataclass(frozen=True)
class HaloMap:
    """Concrete local owner/ghost map for one destination-partitioned CSR."""

    rank: int
    world_size: int
    owned_begin: int
    owned_end: int
    ghost_ids: tuple[int, ...]
    receive_from: tuple[tuple[int, tuple[int, ...]], ...]
    send_to: tuple[tuple[int, tuple[int, ...]], ...]

    @property
    def owned_entities(self) -> int:
        return self.owned_end - self.owned_begin

    @property
    def ghost_entities(self) -> int:
        return len(self.ghost_ids)

    def bytes_for(self, itemsize: int, trailing_elements: int = 1) -> int:
        if itemsize <= 0 or trailing_elements <= 0:
            raise ValueError("itemsize and trailing_elements must be positive")
        return self.ghost_entities * itemsize * trailing_elements


def destination_owner(entity: int, entities: int, world_size: int) -> int:
    if not 0 <= entity < entities or world_size <= 0:
        raise ValueError("entity/world_size is outside the partition domain")
    base, remainder = divmod(entities, world_size)
    wide = (base + 1) * remainder
    if entity < wide:
        return entity // (base + 1)
    return remainder + (entity - wide) // base if base else remainder


def owned_range(entities: int, world_size: int, rank: int) -> tuple[int, int]:
    if entities < 0 or world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("invalid destination partition")
    base, remainder = divmod(entities, world_size)
    begin = rank * base + min(rank, remainder)
    return begin, begin + base + (rank < remainder)


def derive_halo_map(
    row_ptr: Sequence[int],
    col_idx: Sequence[int],
    *,
    num_entities: int,
    world_size: int,
    rank: int,
    peer_requests: Sequence[Sequence[int]] | None = None,
) -> HaloMap:
    """Derive exact ghosts from owned CSR rows without a framework dependency.

    ``peer_requests[p]`` is the list of locally owned entity IDs requested by
    peer ``p`` after the compiler-planned counts/IDs exchange. Omitting it is
    useful for local planning and leaves ``send_to`` empty.
    """
    rows = tuple(int(value) for value in row_ptr)
    columns = tuple(int(value) for value in col_idx)
    if len(rows) != num_entities + 1 or not rows or rows[0] != 0 or rows[-1] != len(columns):
        raise ValueError("row_ptr does not describe num_entities CSR rows")
    if any(left > right for left, right in zip(rows, rows[1:])):
        raise ValueError("row_ptr must be monotonic")
    if any(value < 0 or value >= num_entities for value in columns):
        raise ValueError("col_idx contains an out-of-range entity")
    begin, end = owned_range(num_entities, world_size, rank)
    ghosts = sorted({
        source
        for source in columns[rows[begin]:rows[end]]
        if not begin <= source < end
    })
    receives: list[tuple[int, tuple[int, ...]]] = []
    for owner in range(world_size):
        ids = tuple(
            source for source in ghosts
            if destination_owner(source, num_entities, world_size) == owner
        )
        if ids:
            receives.append((owner, ids))
    sends: list[tuple[int, tuple[int, ...]]] = []
    if peer_requests is not None:
        if len(peer_requests) != world_size:
            raise ValueError("peer_requests must contain one entry per rank")
        for peer, requested in enumerate(peer_requests):
            unique = tuple(sorted(set(int(value) for value in requested)))
            if any(not begin <= value < end for value in unique):
                raise ValueError("peer requested an entity not owned by this rank")
            if unique:
                sends.append((peer, unique))
    return HaloMap(
        rank=rank,
        world_size=world_size,
        owned_begin=begin,
        owned_end=end,
        ghost_ids=tuple(ghosts),
        receive_from=tuple(receives),
        send_to=tuple(sends),
    )
