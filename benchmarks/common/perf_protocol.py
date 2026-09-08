"""Shared performance acceptance helpers.

This module intentionally contains no Torch/Triton imports so the SOTA gate can
be unit tested without accelerator initialization.
"""

from __future__ import annotations

import hashlib
import random
import statistics
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence


BUCKET_FIELDS = (
    "topology",
    "locality",
    "cache",
    "nodes",
    "edges",
    "features",
    "index_dtype",
)


@dataclass(frozen=True)
class GateResult:
    topology: str
    locality: str
    cache: str
    nodes: int
    edges: int
    features: int
    index_dtype: str
    candidate: str
    baseline: str | None
    candidate_ms: float
    baseline_ms: float | None
    speedup_vs_sota: float | None
    speedup_ci_low: float | None
    speedup_ci_high: float | None
    threshold: float
    passed: bool
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _read(item: object, name: str):
    if isinstance(item, Mapping):
        return item[name]
    return getattr(item, name)


def _optional_read(item: object, name: str):
    if isinstance(item, Mapping):
        return item.get(name)
    return getattr(item, name, None)


def _bootstrap_speedup_ci(
    baseline_samples: Sequence[float],
    candidate_samples: Sequence[float],
    *,
    seed_material: object,
    resamples: int,
) -> tuple[float, float]:
    if len(baseline_samples) < 2 or len(candidate_samples) < 2:
        raise ValueError("at least two raw samples per provider are required")
    digest = hashlib.sha256(repr(seed_material).encode()).digest()
    generator = random.Random(int.from_bytes(digest[:8], "little"))
    ratios = []
    for _ in range(resamples):
        baseline = statistics.median(
            generator.choice(baseline_samples)
            for _ in range(len(baseline_samples)))
        candidate = statistics.median(
            generator.choice(candidate_samples)
            for _ in range(len(candidate_samples)))
        ratios.append(baseline / candidate)
    ratios.sort()
    low = ratios[int(0.025 * (resamples - 1))]
    high = ratios[int(0.975 * (resamples - 1))]
    return low, high


def evaluate_sota_gates(
    results: Sequence[object],
    candidates: Iterable[str],
    *,
    baselines: set[str] | None = None,
    ignored_providers: set[str] | None = None,
    threshold: float = 1.0,
    bootstrap_resamples: int = 2000,
) -> list[GateResult]:
    """Compare every candidate in every represented bucket to its fastest peer.

    A baseline set of ``None`` means every measured performance provider except
    candidates and explicitly ignored semantic/reference implementations.
    """

    candidate_set = set(candidates)
    if not candidate_set:
        return []
    if threshold <= 0:
        raise ValueError("SOTA threshold must be positive")
    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive")
    ignored = {"tiga.reference"}
    if ignored_providers is not None:
        ignored.update(ignored_providers)

    grouped: dict[tuple[object, ...], dict[str, object]] = {}
    seen_providers = set()
    for item in results:
        key = tuple(_read(item, name) for name in BUCKET_FIELDS)
        provider = str(_read(item, "provider"))
        seen_providers.add(provider)
        grouped.setdefault(key, {})[provider] = item

    missing = candidate_set - seen_providers
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"gate candidate provider was not measured: {names}")

    gates = []
    for key, providers in sorted(grouped.items(), key=lambda pair: repr(pair[0])):
        for candidate in sorted(candidate_set & providers.keys()):
            candidate_item = providers[candidate]
            candidate_ms = float(_read(candidate_item, "milliseconds"))
            eligible = {
                name: item
                for name, item in providers.items()
                if name not in candidate_set
                and name not in ignored
                and (baselines is None or name in baselines)
            }
            common = dict(zip(BUCKET_FIELDS, key))
            if not eligible:
                gates.append(GateResult(
                    **common,
                    candidate=candidate,
                    baseline=None,
                    candidate_ms=candidate_ms,
                    baseline_ms=None,
                    speedup_vs_sota=None,
                    speedup_ci_low=None,
                    speedup_ci_high=None,
                    threshold=threshold,
                    passed=False,
                    reason="no eligible baseline was measured in this bucket",
                ))
                continue
            baseline, baseline_item = min(
                eligible.items(), key=lambda pair: float(
                    _read(pair[1], "milliseconds")))
            baseline_ms = float(_read(baseline_item, "milliseconds"))
            speedup = baseline_ms / candidate_ms
            baseline_samples = _optional_read(baseline_item, "samples_ms")
            candidate_samples = _optional_read(candidate_item, "samples_ms")
            reason = None
            ci_low = ci_high = None
            if baseline_samples is None or candidate_samples is None:
                passed = False
                reason = "raw samples are required for the confidence gate"
            else:
                try:
                    ci_low, ci_high = _bootstrap_speedup_ci(
                        baseline_samples,
                        candidate_samples,
                        seed_material=(key, candidate, baseline),
                        resamples=bootstrap_resamples,
                    )
                    passed = speedup >= threshold and ci_low >= threshold
                except ValueError as error:
                    passed = False
                    reason = str(error)
            gates.append(GateResult(
                **common,
                candidate=candidate,
                baseline=baseline,
                candidate_ms=candidate_ms,
                baseline_ms=baseline_ms,
                speedup_vs_sota=speedup,
                speedup_ci_low=ci_low,
                speedup_ci_high=ci_high,
                threshold=threshold,
                passed=passed,
                reason=reason,
            ))
    return gates
