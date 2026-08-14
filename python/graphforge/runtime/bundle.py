"""Provider-neutral executable bundles and dependency submission.

This is the runtime counterpart of ``gf.task``.  It deliberately knows
nothing about Torch, Triton, CUDA, MPI, or NVMe: compiler plans bind named
resources to compiled executable objects, while a provider turns dependency
completions into its native stream/event operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


class AccessMode(str, Enum):
    READ = "read"
    WRITE = "write"
    READ_WRITE = "read_write"

    @property
    def writes(self) -> bool:
        return self is not AccessMode.READ


@dataclass(frozen=True)
class ResourceAccess:
    """One versioned logical resource accessed by an invocation."""

    binding: str
    mode: AccessMode | str
    snapshot_version: int = 0
    partitioning: str | None = None
    partition: str | None = None

    def __post_init__(self) -> None:
        if not self.binding:
            raise ValueError("resource binding must not be empty")
        if not isinstance(self.mode, AccessMode):
            object.__setattr__(self, "mode", AccessMode(self.mode))
        if not isinstance(self.snapshot_version, int) or self.snapshot_version < 0:
            raise ValueError("snapshot_version must be a non-negative integer")
        if (self.partitioning is None) != (self.partition is None):
            raise ValueError("partitioning and partition must be specified together")
        if self.partitioning == "" or self.partition == "":
            raise ValueError("partitioning/partition names must not be empty")


@dataclass(frozen=True)
class ArgumentBinding:
    """Bind one executable ABI argument to a bundle resource name."""

    parameter: str
    resource: str

    def __post_init__(self) -> None:
        if not self.parameter or not self.resource:
            raise ValueError("argument parameter/resource names must not be empty")


@dataclass(frozen=True)
class KernelInvocation:
    """A compiled executable launch in a bundle DAG."""

    name: str
    executable: Any
    arguments: tuple[ArgumentBinding, ...]
    accesses: tuple[ResourceAccess, ...]
    depends_on: tuple[str, ...] = ()
    task_kind: str = "local"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("invocation name must not be empty")
        if self.executable is None:
            raise ValueError("invocation executable must not be None")
        parameters = [argument.parameter for argument in self.arguments]
        if len(parameters) != len(set(parameters)):
            raise ValueError(f"invocation {self.name!r} has duplicate ABI parameters")
        if len(self.depends_on) != len(set(self.depends_on)):
            raise ValueError(f"invocation {self.name!r} has duplicate dependencies")


@runtime_checkable
class Completion(Protocol):
    @property
    def ready(self) -> bool: ...

    def wait(self) -> None: ...


@runtime_checkable
class SubmissionProvider(Protocol):
    """Backend hook for one launch and its already-resolved dependencies."""

    def submit(
        self,
        invocation: KernelInvocation,
        arguments: Mapping[str, Any],
        wait_for: Sequence[Completion],
        *,
        will_be_waited: bool = False,
    ) -> Completion: ...


@runtime_checkable
class PreparedInvocation(Protocol):
    def submit(
        self,
        resources: tuple[Any, ...],
        wait_for: Sequence[Completion],
        *,
        will_be_waited: bool = False,
    ) -> Completion: ...


@runtime_checkable
class PreparingProvider(SubmissionProvider, Protocol):
    def prepare(
        self,
        invocation: KernelInvocation,
        argument_slots: tuple[tuple[str, int], ...],
    ) -> PreparedInvocation: ...


class ImmediateCompletion:
    """Completion used by the synchronous CPU/debug provider."""

    __slots__ = ()

    @property
    def ready(self) -> bool:
        return True

    def wait(self) -> None:
        return None


class SynchronousProvider:
    """Reference submission provider; useful for CPU and runtime tests."""

    def submit(
        self,
        invocation: KernelInvocation,
        arguments: Mapping[str, Any],
        wait_for: Sequence[Completion],
        *,
        will_be_waited: bool = False,
    ) -> Completion:
        del will_be_waited
        for completion in wait_for:
            completion.wait()
        executable = invocation.executable
        launch = getattr(executable, "launch", None)
        if launch is None:
            if not callable(executable):
                raise TypeError(
                    f"executable for invocation {invocation.name!r} has no launch()"
                )
            launch = executable
        launch(**arguments)
        return ImmediateCompletion()

    def prepare(self, invocation, argument_slots):
        return _PreparedSynchronousInvocation(
            getattr(invocation.executable, "launch", invocation.executable),
            argument_slots,
        )


class _PreparedSynchronousInvocation:
    def __init__(self, launch, argument_slots):
        self._launch = launch
        self._argument_slots = argument_slots

    def submit(self, resources, wait_for, *, will_be_waited=False):
        del will_be_waited
        for completion in wait_for:
            completion.wait()
        self._launch(**{
            parameter: resources[slot]
            for parameter, slot in self._argument_slots
        })
        return ImmediateCompletion()


@dataclass(frozen=True)
class BundleSubmission:
    """Completion handles returned from one bundle submission."""

    completions: Mapping[str, Completion]
    terminal: tuple[Completion, ...]

    @property
    def ready(self) -> bool:
        return all(completion.ready for completion in self.terminal)

    def wait(self) -> None:
        for completion in self.terminal:
            completion.wait()


class ExecutableBundle:
    """A verified DAG of compiled executable invocations.

    Conflicting accesses must be ordered explicitly.  The runtime never
    invents a dependency because doing so would hide compiler scheduling and
    could invalidate communication/overlap decisions.
    """

    def __init__(
        self,
        invocations: Sequence[KernelInvocation],
        *,
        required_bindings: Sequence[str],
        snapshot_version: int = 0,
    ) -> None:
        if not isinstance(snapshot_version, int) or snapshot_version < 0:
            raise ValueError("bundle snapshot_version must be non-negative")
        self.invocations = tuple(invocations)
        self.required_bindings = tuple(required_bindings)
        self.snapshot_version = snapshot_version
        if len(self.required_bindings) != len(set(self.required_bindings)):
            raise ValueError("bundle required_bindings must be unique")
        self._order, self._terminals = self._verify()
        terminal_set = set(self._terminals)
        self._will_be_waited = tuple(
            index not in terminal_set for index in range(len(self.invocations)))

    def structural_key(self) -> tuple[object, ...]:
        """Return the executable-independent task topology/cache key."""
        return (
            self.snapshot_version,
            self.required_bindings,
            tuple(
                (
                    invocation.name,
                    invocation.task_kind,
                    tuple(
                        (argument.parameter, argument.resource)
                        for argument in invocation.arguments
                    ),
                    tuple(
                        (
                            access.binding,
                            access.mode.value,
                            access.snapshot_version,
                            access.partitioning,
                            access.partition,
                        )
                        for access in invocation.accesses
                    ),
                    invocation.depends_on,
                )
                for invocation in self.invocations
            ),
        )

    @property
    def structural_hash(self) -> str:
        encoded = json.dumps(
            self.structural_key(), separators=(",", ":"), sort_keys=False)
        return hashlib.sha256(encoded.encode()).hexdigest()

    def explain(self) -> str:
        lines = [
            f"executable bundle: invocations={len(self.invocations)} "
            f"snapshot={self.snapshot_version}",
            f"structure hash: {self.structural_hash[:16]}",
            f"bindings: {', '.join(self.required_bindings)}",
        ]
        for index in self._order:
            invocation = self.invocations[index]
            dependencies = ",".join(invocation.depends_on) or "ready"
            accesses = ", ".join(
                f"{access.binding}:{access.mode.value}"
                + (
                    f"[{access.partitioning}/{access.partition}]"
                    if access.partitioning is not None else ""
                )
                for access in invocation.accesses
            )
            lines.append(
                f"  {invocation.name} ({invocation.task_kind}) <- "
                f"{dependencies}; {accesses or 'no resources'}"
            )
        return "\n".join(lines)

    @property
    def execution_order(self) -> tuple[str, ...]:
        return tuple(self.invocations[index].name for index in self._order)

    def _verify(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        names = [invocation.name for invocation in self.invocations]
        if len(names) != len(set(names)):
            raise ValueError("bundle invocation names must be unique")
        by_name = {name: index for index, name in enumerate(names)}
        required = set(self.required_bindings)
        successors: list[list[int]] = [[] for _ in names]
        indegree = [0] * len(names)
        for index, invocation in enumerate(self.invocations):
            used = {binding.resource for binding in invocation.arguments}
            used.update(access.binding for access in invocation.accesses)
            missing = sorted(used - required)
            if missing:
                raise ValueError(
                    f"invocation {invocation.name!r} references undeclared "
                    f"bindings {missing}"
                )
            for access in invocation.accesses:
                if access.snapshot_version != self.snapshot_version:
                    raise ValueError(
                        f"invocation {invocation.name!r} accesses snapshot "
                        f"{access.snapshot_version}, bundle owns "
                        f"{self.snapshot_version}"
                    )
            for dependency in invocation.depends_on:
                if dependency not in by_name:
                    raise ValueError(
                        f"invocation {invocation.name!r} depends on unknown "
                        f"invocation {dependency!r}"
                    )
                source = by_name[dependency]
                if source == index:
                    raise ValueError("an invocation cannot depend on itself")
                successors[source].append(index)
                indegree[index] += 1
        queue = [index for index, degree in enumerate(indegree) if degree == 0]
        order: list[int] = []
        while queue:
            index = queue.pop(0)
            order.append(index)
            for successor in successors[index]:
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    queue.append(successor)
        if len(order) != len(names):
            raise ValueError("bundle dependency graph contains a cycle")

        reachable = [[False] * len(names) for _ in names]
        for source in reversed(order):
            for target in successors[source]:
                reachable[source][target] = True
                for descendant in range(len(names)):
                    reachable[source][descendant] |= reachable[target][descendant]
        accesses = [invocation.accesses for invocation in self.invocations]
        for left in range(len(names)):
            for right in range(left + 1, len(names)):
                conflicts = set()
                for left_access in accesses[left]:
                    for right_access in accesses[right]:
                        if left_access.binding != right_access.binding:
                            continue
                        if not (left_access.mode.writes or right_access.mode.writes):
                            continue
                        proven_disjoint = (
                            left_access.partitioning is not None
                            and left_access.partitioning == right_access.partitioning
                            and left_access.partition != right_access.partition
                        )
                        if not proven_disjoint:
                            conflicts.add(left_access.binding)
                if conflicts and not (
                    reachable[left][right] or reachable[right][left]
                ):
                    raise ValueError(
                        f"unordered resource hazard between {names[left]!r} and "
                        f"{names[right]!r}: {sorted(conflicts)}"
                    )
        terminals = tuple(index for index, edges in enumerate(successors) if not edges)
        return tuple(order), terminals

    def submit(
        self,
        provider: SubmissionProvider,
        bindings: Mapping[str, Any],
    ) -> BundleSubmission:
        missing = sorted(set(self.required_bindings) - bindings.keys())
        if missing:
            raise KeyError(f"bundle submission is missing bindings {missing}")
        # Single-launch bundles are the overwhelmingly common hot path. Their
        # DAG and ABI were verified at construction, so avoid allocating an
        # argument MappingProxy and dependency tuple on every submission.
        if len(self._order) == 1:
            invocation = self.invocations[self._order[0]]
            arguments = {
                argument.parameter: bindings[argument.resource]
                for argument in invocation.arguments
            }
            completion = provider.submit(
                invocation, arguments, (), will_be_waited=False)
            if not isinstance(completion, Completion):
                raise TypeError(
                    f"provider returned no Completion for {invocation.name!r}"
                )
            completions = MappingProxyType({invocation.name: completion})
            return BundleSubmission(completions, (completion,))

        completions: dict[str, Completion] = {}
        for index in self._order:
            invocation = self.invocations[index]
            arguments = {
                argument.parameter: bindings[argument.resource]
                for argument in invocation.arguments
            }
            wait_for = tuple(completions[name] for name in invocation.depends_on)
            completion = provider.submit(
                invocation, arguments, wait_for,
                will_be_waited=self._will_be_waited[index],
            )
            if not isinstance(completion, Completion):
                raise TypeError(
                    f"provider returned no Completion for {invocation.name!r}"
                )
            completions[invocation.name] = completion
        return BundleSubmission(
            completions=MappingProxyType(completions),
            terminal=tuple(completions[names] for names in (
                self.invocations[index].name for index in self._terminals
            )),
        )

    def prepare(self, provider: PreparingProvider) -> "PreparedBundle":
        """Resolve names/DAG once and return a fixed-slot submission plan."""
        slots = {name: index for index, name in enumerate(self.required_bindings)}
        prepared = tuple(
            provider.prepare(
                invocation,
                tuple(
                    (argument.parameter, slots[argument.resource])
                    for argument in invocation.arguments
                ),
            )
            for invocation in self.invocations
        )
        dependency_indices = tuple(
            tuple(
                next(
                    index for index, candidate in enumerate(self.invocations)
                    if candidate.name == dependency
                )
                for dependency in invocation.depends_on
            )
            for invocation in self.invocations
        )
        return PreparedBundle(self, prepared, dependency_indices)


class PreparedBundle:
    """Name-free, fixed-slot executable bundle used by warm dispatch."""

    def __init__(
        self,
        bundle: ExecutableBundle,
        invocations: tuple[PreparedInvocation, ...],
        dependencies: tuple[tuple[int, ...], ...],
    ) -> None:
        self.bundle = bundle
        self.invocations = invocations
        self.dependencies = dependencies

    def submit_resources(self, resources: tuple[Any, ...]) -> BundleSubmission:
        if len(resources) != len(self.bundle.required_bindings):
            raise ValueError(
                f"prepared bundle requires {len(self.bundle.required_bindings)} "
                f"resources, received {len(resources)}"
            )
        completions: list[Completion | None] = [None] * len(self.invocations)
        for index in self.bundle._order:
            wait_for = tuple(
                completions[dependency] for dependency in self.dependencies[index]
            )
            completion = self.invocations[index].submit(
                resources,
                wait_for,  # type: ignore[arg-type]
                will_be_waited=self.bundle._will_be_waited[index],
            )
            if not isinstance(completion, Completion):
                raise TypeError(
                    "prepared provider returned no Completion for "
                    f"{self.bundle.invocations[index].name!r}"
                )
            completions[index] = completion
        mapped = MappingProxyType({
            invocation.name: completions[index]
            for index, invocation in enumerate(self.bundle.invocations)
        })
        return BundleSubmission(
            mapped,  # type: ignore[arg-type]
            tuple(completions[index] for index in self.bundle._terminals),  # type: ignore[arg-type]
        )


__all__ = [
    "AccessMode",
    "ArgumentBinding",
    "BundleSubmission",
    "Completion",
    "ExecutableBundle",
    "ImmediateCompletion",
    "KernelInvocation",
    "PreparedBundle",
    "PreparedInvocation",
    "PreparingProvider",
    "ResourceAccess",
    "SubmissionProvider",
    "SynchronousProvider",
]
