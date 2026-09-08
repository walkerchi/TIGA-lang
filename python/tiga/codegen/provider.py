from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import importlib.metadata
import platform

@dataclass(frozen=True)
class ProviderIdentity:
    """Versioned identity included in compiled-variant and cache metadata."""

    name: str
    target_family: str
    provider_version: str
    torch_version: str
    compiler_revision: str | None = None
    target: str | None = None

    def cache_key(self) -> tuple[str, ...]:
        return (
            self.name,
            self.target_family,
            self.provider_version,
            self.torch_version,
            self.compiler_revision or "unknown",
            self.target or "unknown",
        )

    def display_name(self) -> str:
        return f"{self.name}@{self.provider_version}"


@lru_cache(maxsize=8)
def triton_provider_identity(target=None) -> ProviderIdentity:
    """Describe the active Triton distribution without assuming NVIDIA."""
    try:
        version = importlib.metadata.version("triton")
    except importlib.metadata.PackageNotFoundError:
        version = "unavailable"
    try:
        torch_version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        torch_version = "not-installed"

    runtime = "host"
    vendor = platform.machine()
    target_text = None
    compiler_revision = None
    try:
        from triton.compiler.compiler import make_backend
        if target is None:
            from triton.runtime import driver

            target = driver.active.get_current_target()
        runtime = str(target.backend)
        vendor = {
            "cuda": "nvidia",
            "hip": "amd",
        }.get(runtime, runtime)
        target_text = f"{runtime}:{target.arch}:warp{target.warp_size}"
        backend_hash = make_backend(target).hash()
        compiler_revision = hashlib.sha256(
            backend_hash.encode()).hexdigest()[:16]
    except Exception:
        # Identity inspection must not make importing Tiga require a GPU
        # or an initialized vendor runtime. The unavailable target is still a
        # distinct cache identity and will never be used for a GPU executable.
        # Without an initialized provider runtime we deliberately keep a host
        # identity. Optional framework metadata must not guess a GPU target.
        pass
    return ProviderIdentity(
        name=f"triton-{vendor}",
        target_family=f"{runtime}:{vendor}",
        provider_version=version,
        torch_version=torch_version,
        compiler_revision=compiler_revision,
        target=target_text,
    )
