"""Declarative mesh, partition and halo placement models."""

from .model import (
    ByDestination, DeviceMesh, GraphPlacement, HaloMap,
    derive_halo_map, destination_owner, owned_range,
)
from .transport import (
    DeviceBufferSlice, DeviceBufferTransport,
    DistributedCompletion, DistributedRuntime, DistributedTaskResolver,
    FixedByteTransport, HaloBuffer, HaloExchangeExecutable, HaloPackExecutable, HaloSlot,
    HaloTransportExecutable, HaloUnpackExecutable, NeighborTransport,
    PackedHalo, PipeTransport, ReceivedHalo, ShardedField,
    collective_halo_maps, collective_paged_halo_maps, exchange_halo,
    exchange_packed, pack_halo, unpack_halo,
)
from .mpi import MPITransport, MPITransportProvider
from .nccl import NCCLTransport, NCCLTransportProvider, unique_id as nccl_unique_id
from .registry import (
    TransportCapabilities, TransportProvider, create_transport,
    discover_transports, get_transport_provider, register_transport,
    transport_conformance,
)

# Built-in provider remains lazy with respect to mpi4py itself.
register_transport("mpi", MPITransportProvider(), replace=True)
register_transport("nccl", NCCLTransportProvider(), replace=True)

__all__ = [
    "ByDestination", "DeviceMesh", "GraphPlacement", "HaloMap",
    "derive_halo_map", "destination_owner", "owned_range",
    "DeviceBufferSlice", "DeviceBufferTransport",
    "DistributedCompletion", "DistributedRuntime", "DistributedTaskResolver",
    "FixedByteTransport", "HaloBuffer", "HaloExchangeExecutable", "HaloPackExecutable", "HaloSlot",
    "HaloTransportExecutable", "HaloUnpackExecutable", "NeighborTransport",
    "PackedHalo", "PipeTransport", "ReceivedHalo", "ShardedField",
    "collective_halo_maps", "collective_paged_halo_maps", "exchange_halo",
    "exchange_packed", "pack_halo",
    "unpack_halo",
    "MPITransport", "MPITransportProvider", "NCCLTransport",
    "NCCLTransportProvider", "nccl_unique_id", "TransportCapabilities",
    "TransportProvider", "create_transport", "discover_transports",
    "get_transport_provider", "register_transport", "transport_conformance",
]
