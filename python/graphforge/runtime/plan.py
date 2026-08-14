"""Typed executable-bundle plans emitted by ``gf-task-to-bundle``.

The compiler owns task topology and static scheduling metadata. Providers own
executable construction. This module is the deliberately small, Torch-free
binding seam between those responsibilities.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from types import MappingProxyType
from typing import Any, Callable, Mapping

from .bundle import (
    AccessMode,
    ArgumentBinding,
    ExecutableBundle,
    KernelInvocation,
    ResourceAccess,
)


_SCHEMA = "graphforge.executable-bundle-plan.v1"


@dataclass(frozen=True)
class PlannedInvocation:
    """One compiler-planned invocation before provider executable binding."""

    name: str
    task_kind: str
    phase: str
    executable_symbol: str
    depends_on: tuple[str, ...]
    arguments: tuple[ArgumentBinding, ...]
    accesses: tuple[ResourceAccess, ...]
    snapshot_version: int
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class ResourceRequirement:
    """Compiler-planned physical resource to allocate or externally bind."""

    name: str
    memory_space: str
    layout: str
    device: str
    capacity_bytes: int
    snapshot_version: int
    external: bool

    def __post_init__(self) -> None:
        if not self.name or not self.memory_space or not self.layout or not self.device:
            raise ValueError("resource requirement names must not be empty")
        if self.capacity_bytes < 0 or self.snapshot_version < 0:
            raise ValueError("resource capacity/version must be non-negative")


class _JoinExecutable:
    @staticmethod
    def launch() -> None:
        return None


class _ReleaseExecutable:
    """Logical lifetime end; provider allocators own actual recycling."""

    @staticmethod
    def launch(instance: Any) -> None:
        close = getattr(instance, "close", None)
        if close is not None:
            close()


class _TransferExecutable:
    """Execute a compiler-planned transfer through the owning hierarchy."""

    @staticmethod
    def launch(source: Any, destination: Any) -> None:
        runtime = getattr(source, "_runtime", None)
        if runtime is None or runtime is not getattr(destination, "_runtime", None):
            raise ValueError("transfer instances must share a hierarchy runtime")
        runtime.transfer(source, destination).wait()


class ExecutableBundlePlan:
    """Validated provider-neutral task plan serialized by the native compiler."""

    def __init__(
        self,
        invocations: tuple[PlannedInvocation, ...],
        *,
        resources: tuple[ResourceRequirement, ...] = (),
        terminals: tuple[str, ...],
    ) -> None:
        if not invocations:
            raise ValueError("bundle plan requires at least one invocation")
        self.invocations = invocations
        self.resources = resources
        self.terminals = terminals
        names = tuple(invocation.name for invocation in invocations)
        if len(names) != len(set(names)):
            raise ValueError("bundle plan invocation names must be unique")
        known = set(names)
        resource_names = tuple(resource.name for resource in resources)
        if len(resource_names) != len(set(resource_names)):
            raise ValueError("bundle plan resource names must be unique")
        if not terminals or not set(terminals) <= known:
            raise ValueError("bundle plan terminals must name known invocations")
        for invocation in invocations:
            if not invocation.executable_symbol:
                raise ValueError(
                    f"invocation {invocation.name!r} has no executable symbol"
                )
            if invocation.phase not in {"materialize", "execute"}:
                raise ValueError(
                    f"invocation {invocation.name!r} has invalid phase"
                )
            unknown = set(invocation.depends_on) - known
            if unknown:
                raise ValueError(
                    f"invocation {invocation.name!r} has unknown dependencies: "
                    f"{sorted(unknown)}"
                )
            if any(
                access.snapshot_version != invocation.snapshot_version
                for access in invocation.accesses
            ):
                raise ValueError(
                    f"invocation {invocation.name!r} mixes snapshot versions"
                )

    @classmethod
    def parse(cls, serialized: str | bytes) -> "ExecutableBundlePlan":
        """Parse the versioned JSON emitted at the native compiler boundary."""
        try:
            document = json.loads(serialized)
        except (TypeError, ValueError) as error:
            raise ValueError("invalid executable-bundle plan JSON") from error
        if not isinstance(document, dict) or document.get("schema") != _SCHEMA:
            raise ValueError(f"unsupported executable-bundle plan schema")
        raw_invocations = document.get("invocations")
        raw_resources = document.get("resources", [])
        raw_terminals = document.get("terminals")
        if not isinstance(raw_resources, list) or not isinstance(raw_invocations, list) or not isinstance(
            raw_terminals, list
        ):
            raise ValueError("bundle plan invocations/terminals must be arrays")

        resources: list[ResourceRequirement] = []
        for raw in raw_resources:
            if not isinstance(raw, dict):
                raise ValueError("bundle plan resource must be an object")
            try:
                resources.append(ResourceRequirement(**raw))
            except (TypeError, ValueError) as error:
                raise ValueError("malformed bundle plan resource") from error

        invocations: list[PlannedInvocation] = []
        fixed_keys = {
            "name",
            "task_kind",
            "phase",
            "executable_symbol",
            "depends_on",
            "arguments",
            "accesses",
            "snapshot_version",
        }
        for raw in raw_invocations:
            if not isinstance(raw, dict):
                raise ValueError("bundle plan invocation must be an object")
            try:
                version = raw["snapshot_version"]
                accesses = tuple(
                    ResourceAccess(
                        binding=entry["binding"],
                        mode=AccessMode(entry["mode"]),
                        snapshot_version=entry["snapshot_version"],
                        partitioning=entry.get("partitioning"),
                        partition=entry.get("partition"),
                    )
                    for entry in raw["accesses"]
                )
                arguments = tuple(
                    ArgumentBinding(
                        parameter=entry["parameter"],
                        resource=entry["resource"],
                    )
                    for entry in raw["arguments"]
                )
                invocation = PlannedInvocation(
                    name=raw["name"],
                    task_kind=raw["task_kind"],
                    phase=raw["phase"],
                    executable_symbol=raw["executable_symbol"],
                    depends_on=tuple(raw["depends_on"]),
                    arguments=arguments,
                    accesses=accesses,
                    snapshot_version=version,
                    metadata=MappingProxyType(
                        {key: value for key, value in raw.items() if key not in fixed_keys}
                    ),
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("malformed bundle plan invocation") from error
            if not isinstance(version, int) or version < 0:
                raise ValueError("invocation snapshot_version must be non-negative")
            invocations.append(invocation)
        if not all(isinstance(name, str) and name for name in raw_terminals):
            raise ValueError("bundle plan terminal names must be non-empty strings")
        return cls(
            tuple(invocations), resources=tuple(resources),
            terminals=tuple(raw_terminals)
        )

    def allocate(
        self,
        allocator: Callable[[ResourceRequirement], Any],
    ) -> dict[str, Any]:
        """Allocate compiler-owned resources; external resources stay unbound."""
        return {
            requirement.name: allocator(requirement)
            for requirement in self.resources
            if not requirement.external
        }

    def allocate_hierarchy(self, runtime) -> dict[str, Any]:
        """Allocate non-external resources in the hierarchy runtime."""
        return self.allocate(runtime.allocate_requirement)

    def bind(
        self,
        resolver: Callable[[PlannedInvocation], Any],
    ) -> ExecutableBundle:
        """Resolve provider executables and construct the verified runtime DAG."""
        return self._bind(resolver, self.invocations, verify_terminals=True)

    def bind_phase(
        self,
        phase: str,
        resolver: Callable[[PlannedInvocation], Any],
    ) -> ExecutableBundle:
        """Bind one compiler phase; cross-phase events are satisfied preconditions."""
        if phase not in {"materialize", "execute"}:
            raise ValueError("phase must be 'materialize' or 'execute'")
        selected = tuple(
            invocation for invocation in self.invocations
            if invocation.phase == phase
        )
        if not selected:
            raise ValueError(f"bundle plan has no {phase!r} invocations")
        return self._bind(resolver, selected, verify_terminals=False)

    def _bind(self, resolver, selected, *, verify_terminals):
        versions = {invocation.snapshot_version for invocation in selected}
        if len(versions) != 1:
            raise ValueError("one executable bundle cannot mix snapshot versions")
        runtime_invocations = []
        required_bindings: list[str] = []
        selected_names = {invocation.name for invocation in selected}
        for invocation in selected:
            for binding in (
                argument.resource for argument in invocation.arguments
            ):
                if binding not in required_bindings:
                    required_bindings.append(binding)
            executable = (
                _JoinExecutable()
                if invocation.executable_symbol == "__gf_join"
                else _ReleaseExecutable()
                if invocation.executable_symbol == "__gf_release"
                else _TransferExecutable()
                if invocation.executable_symbol == "__gf_transfer"
                else resolver(invocation)
            )
            runtime_invocations.append(
                KernelInvocation(
                    name=invocation.name,
                    executable=executable,
                    arguments=invocation.arguments,
                    accesses=invocation.accesses,
                    depends_on=tuple(
                        dependency for dependency in invocation.depends_on
                        if dependency in selected_names
                    ),
                    task_kind=invocation.task_kind,
                )
            )
        bundle = ExecutableBundle(
            runtime_invocations,
            required_bindings=required_bindings,
            snapshot_version=next(iter(versions)),
        )
        derived_terminals = tuple(
            bundle.invocations[index].name for index in bundle._terminals
        )
        if verify_terminals and set(derived_terminals) != set(self.terminals):
            raise ValueError(
                "serialized terminals disagree with the invocation dependency DAG"
            )
        return bundle


__all__ = [
    "ExecutableBundlePlan", "PlannedInvocation", "ResourceRequirement"
]
