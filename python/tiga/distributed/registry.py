"""Entry-point registry for deployment-owned communication transports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import metadata
from typing import Callable, Protocol, runtime_checkable

from .transport import NeighborTransport


@dataclass(frozen=True)
class TransportCapabilities:
    name: str
    backends: tuple[str, ...]
    device_direct: bool
    asynchronous: bool
    object_messages: bool = True
    abi_version: int = 1

    def __post_init__(self) -> None:
        if self.abi_version != 1:
            raise ValueError("unsupported Tiga transport ABI")
        if not self.name or not self.backends:
            raise ValueError("transport capabilities require a name and backend")


@runtime_checkable
class TransportProvider(Protocol):
    capabilities: TransportCapabilities

    def create(self, **options) -> NeighborTransport: ...


_TRANSPORTS: dict[str, TransportProvider | Callable[[], TransportProvider]] = {}
_DISCOVERED = False


def register_transport(name: str, provider, *, replace: bool = False) -> None:
    if not name:
        raise ValueError("transport provider name must not be empty")
    if name in _TRANSPORTS and not replace:
        raise ValueError(f"transport provider {name!r} is already registered")
    _TRANSPORTS[name] = provider


def discover_transports() -> tuple[str, ...]:
    global _DISCOVERED
    if not _DISCOVERED:
        for entry in metadata.entry_points(group="tiga.transport"):
            register_transport(entry.name, entry.load(), replace=True)
        _DISCOVERED = True
    return tuple(sorted(_TRANSPORTS))


def get_transport_provider(name: str) -> TransportProvider:
    discover_transports()
    try:
        candidate = _TRANSPORTS[name]
    except KeyError as error:
        raise LookupError(
            f"Tiga transport provider {name!r} is unavailable; installed: "
            f"{', '.join(sorted(_TRANSPORTS)) or 'none'}"
        ) from error
    provider = candidate() if callable(candidate) and not hasattr(candidate, "create") else candidate
    if not isinstance(provider, TransportProvider):
        raise TypeError(f"transport provider {name!r} does not implement the ABI")
    return provider


def create_transport(name: str, **options) -> NeighborTransport:
    transport = get_transport_provider(name).create(**options)
    if not isinstance(transport, NeighborTransport):
        raise TypeError(f"transport provider {name!r} returned an invalid transport")
    return transport


def transport_conformance(provider: TransportProvider) -> dict[str, object]:
    if not isinstance(provider, TransportProvider):
        raise TypeError("provider does not implement TransportProvider")
    if not isinstance(provider.capabilities, TransportCapabilities):
        raise TypeError("transport capabilities have the wrong type")
    return asdict(provider.capabilities)


__all__ = [
    "TransportCapabilities", "TransportProvider", "create_transport",
    "discover_transports", "get_transport_provider", "register_transport",
    "transport_conformance",
]
