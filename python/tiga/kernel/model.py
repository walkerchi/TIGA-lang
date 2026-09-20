from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AnalysisFinding:
    """One machine-readable compiler/verifier/cost-model diagnostic."""

    stage: str
    disposition: str
    message: str
    source_location: str | None = None
    affected_resource: str | None = None
    target_contract: str | None = None
    estimate: str | None = None
    suggested_action: str | None = None

    def __post_init__(self) -> None:
        if not self.stage or not self.message:
            raise ValueError("analysis findings require a stage and message")
        if self.disposition not in {
            "accepted", "rejected", "warning", "unknown", "remark"
        }:
            raise ValueError("invalid analysis finding disposition")


@dataclass(frozen=True)
class MachineSchedule:
    """Verified provider-neutral machine schedule extracted from ``gf.kernel``."""

    operation: str
    source_location: str
    kind: str
    block_rows: int
    block_neighbors: int
    num_warps: int
    pipeline_stages: int
    target_contract: str
    resources: tuple[str, ...]
    roles: tuple[str, ...]
    handoffs: tuple[str, ...]
    instructions: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.operation.startswith("gf_kernel."):
            raise ValueError("machine schedule operation must belong to gf_kernel")
        if min(
            self.block_rows, self.block_neighbors,
            self.num_warps, self.pipeline_stages,
        ) <= 0:
            raise ValueError("machine schedule dimensions must be positive")
        if not all((self.resources, self.roles, self.handoffs, self.instructions)):
            raise ValueError("machine schedule contract must be complete")

    @property
    def findings(self) -> tuple[AnalysisFinding, ...]:
        admitted = AnalysisFinding(
            stage="machine-schedule",
            disposition="accepted",
            message=f"admitted {self.kind}",
            source_location=self.source_location,
            affected_resource=", ".join(self.resources),
            target_contract=self.target_contract,
            estimate=(
                f"rows={self.block_rows}, neighbors={self.block_neighbors}, "
                f"warps={self.num_warps}, stages={self.pipeline_stages}"
            ),
        )
        if self.pipeline_stages > 1:
            pipeline = AnalysisFinding(
                stage="pipeline",
                disposition="accepted",
                message="compiler-controlled multi-stage pipeline admitted",
                source_location=self.source_location,
                affected_resource=", ".join(self.handoffs),
                target_contract=self.target_contract,
                estimate=f"{self.pipeline_stages} stages",
            )
        else:
            pipeline = AnalysisFinding(
                stage="pipeline",
                disposition="unknown",
                message="no compiler-controlled asynchronous pipeline admitted",
                source_location=self.source_location,
                affected_resource=", ".join(self.handoffs),
                target_contract=self.target_contract,
                estimate="1 stage",
                suggested_action=(
                    "do not claim producer/consumer overlap from provider "
                    "instruction scheduling alone"
                ),
            )
        return admitted, pipeline


@dataclass(frozen=True)
class CompiledVariant:
    key: tuple[object, ...]
    backend: str
    provider: str
    domain_plan: str
    passes: tuple[str, ...]
    lowering: str | None = None
    provider_key: tuple[str, ...] = ()
    artifacts: dict[str, str] = field(default_factory=dict)
    remarks: tuple[str, ...] = ()
    findings: tuple[AnalysisFinding, ...] = ()
    schedules: tuple[MachineSchedule, ...] = ()

    @property
    def diagnostics(self) -> tuple[AnalysisFinding, ...]:
        """Return structured findings, adapting legacy remarks losslessly."""
        schedule_findings = tuple(
            finding for schedule in self.schedules for finding in schedule.findings
        )
        return (*schedule_findings, *self.findings, *(
            AnalysisFinding("planning", "remark", remark)
            for remark in self.remarks
        ))

    def ir(self, stage: str = "domain") -> str:
        if stage in {"domain", "domain-plan"}:
            return self.domain_plan
        stage = {"iteration": "iter", "gf.iter": "iter",
                 "gf.kernel": "kernel",
                 "gf.task": "task",
                 "gf.kernel.ttir": "kernel_ttir"}.get(stage, stage)
        artifact = self.artifacts.get(stage)
        if isinstance(artifact, str):
            return artifact
        available = ", ".join(["domain", *sorted(self.artifacts)])
        raise KeyError(f"IR/artifact stage {stage!r} is unavailable; have: {available}")

    def code(self, kind: str) -> str:
        try:
            return self.artifacts[kind]
        except KeyError as exc:
            raise KeyError(
                f"artifact {kind!r} is unavailable for provider {self.provider!r}"
            ) from exc

    def explain(self) -> str:
        lines = [
            f"backend: {self.backend}",
            f"provider: {self.provider}",
            f"lowering: {self.lowering or '(unspecified)'}",
            f"passes: {', '.join(self.passes) if self.passes else '(none)'}",
        ]
        if self.provider_key:
            lines.append(f"provider cache key: {' | '.join(self.provider_key)}")
        lines.extend(
            f"{finding.disposition}: [{finding.stage}] {finding.message}"
            for finding in self.diagnostics
        )
        return "\n".join(lines)


class Kernel:
    def __init__(self) -> None:
        self._variants: dict[tuple[object, ...], CompiledVariant] = {}
        self._last_variant: CompiledVariant | None = None
        self._cache_hits = 0
        self._cache_misses = 0

    @property
    def variants(self) -> tuple[CompiledVariant, ...]:
        return tuple(self._variants.values())

    @property
    def last_variant(self) -> CompiledVariant:
        if self._last_variant is None:
            raise RuntimeError("kernel has not been called, so no JIT variant exists")
        return self._last_variant

    def ir(self, stage: str = "domain") -> str:
        return self.last_variant.ir(stage)

    def code(self, kind: str = "ptx") -> str:
        return self.last_variant.code(kind)

    def explain(self) -> str:
        return "\n".join((
            self.last_variant.explain(),
            f"variant cache: hits={self._cache_hits}, misses={self._cache_misses}",
        ))

    @property
    def diagnostics(self) -> tuple[AnalysisFinding, ...]:
        return self.last_variant.diagnostics

    @property
    def schedules(self) -> tuple[MachineSchedule, ...]:
        return self.last_variant.schedules

    @property
    def cache_info(self) -> dict[str, int]:
        """Kernel variant lookups, including capture before native execution.

        These counters do not measure native compilation. Deferred Tensor
        results report the actual backend and executable cache lookup in
        ``Tensor.execution`` after realization.
        """
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "variants": len(self._variants),
        }

    def _record_variant(self, variant: CompiledVariant) -> None:
        if not variant.schedules:
            kernel_ir = variant.artifacts.get("kernel")
            if isinstance(kernel_ir, str) and "gf_kernel." in kernel_ir:
                from ..compiler.schedule import schedules_from_mlir

                schedules = schedules_from_mlir(kernel_ir)
                if schedules:
                    from dataclasses import replace

                    variant = replace(variant, schedules=schedules)
        self._variants[variant.key] = variant
        self._last_variant = variant

    def _lookup_variant(self, key: tuple[Any, ...]) -> CompiledVariant | None:
        variant = self._variants.get(key)
        if variant is not None:
            self._last_variant = variant
        return variant
