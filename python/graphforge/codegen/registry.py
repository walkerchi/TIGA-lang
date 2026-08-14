"""Provider plugin registry for vendor compiler/runtime implementations."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from typing import Any, Callable, Mapping, Protocol, runtime_checkable


@dataclass(frozen=True)
class ProviderCapabilities:
    name: str
    target_family: str
    devices: tuple[str, ...]
    input_ir: str
    artifacts: tuple[str, ...]
    supports_async: bool
    supports_distributed: bool
    abi_version: int = 1

    def __post_init__(self) -> None:
        if self.abi_version != 1:
            raise ValueError("unsupported GraphForge provider ABI")
        if not self.name or not self.target_family or not self.devices:
            raise ValueError("provider capabilities must name a target/device")
        if self.input_ir not in {"ttir", "llvm", "msl", "vendor-ir"}:
            raise ValueError("provider input_ir is not recognized")


@runtime_checkable
class CodegenProvider(Protocol):
    capabilities: ProviderCapabilities

    def compile(
        self, module: str, *, target: Any | None,
        options: Mapping[str, Any],
    ) -> Any: ...


_PROVIDERS: dict[str, Callable[[], CodegenProvider] | CodegenProvider] = {}
_DISCOVERED = False


def register_provider(
    name: str, factory: Callable[[], CodegenProvider] | CodegenProvider, *,
    replace: bool = False,
) -> None:
    if not name:
        raise ValueError("provider name must not be empty")
    if name in _PROVIDERS and not replace:
        raise ValueError(f"provider {name!r} is already registered")
    _PROVIDERS[name] = factory


def discover_providers() -> tuple[str, ...]:
    global _DISCOVERED
    if not _DISCOVERED:
        for entry in metadata.entry_points(group="graphforge.codegen"):
            register_provider(entry.name, entry.load(), replace=True)
        _DISCOVERED = True
    return tuple(sorted(_PROVIDERS))


def get_provider(name: str) -> CodegenProvider:
    discover_providers()
    try:
        factory = _PROVIDERS[name]
    except KeyError as error:
        raise LookupError(
            f"GraphForge codegen provider {name!r} is unavailable; installed: "
            f"{', '.join(sorted(_PROVIDERS)) or 'none'}"
        ) from error
    provider = factory() if callable(factory) and not hasattr(factory, "compile") else factory
    if not isinstance(provider, CodegenProvider):
        raise TypeError(f"provider {name!r} does not implement CodegenProvider")
    return provider


def provider_conformance(provider: CodegenProvider) -> dict[str, object]:
    """Validate the stable ABI without claiming target hardware correctness."""
    if not isinstance(provider, CodegenProvider):
        raise TypeError("provider does not implement CodegenProvider")
    capabilities = provider.capabilities
    if not isinstance(capabilities, ProviderCapabilities):
        raise TypeError("provider capabilities have the wrong type")
    return {
        "abi_version": capabilities.abi_version,
        "name": capabilities.name,
        "target_family": capabilities.target_family,
        "devices": capabilities.devices,
        "input_ir": capabilities.input_ir,
        "artifacts": capabilities.artifacts,
        "supports_async": capabilities.supports_async,
        "supports_distributed": capabilities.supports_distributed,
    }


__all__ = [
    "CodegenProvider", "ProviderCapabilities", "discover_providers",
    "get_provider", "provider_conformance", "register_provider",
]
