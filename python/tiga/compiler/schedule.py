"""Typed access to machine schedules produced and verified in native MLIR.

This module never infers scheduling decisions from pass names or log strings.
The native binding parses verified ``gf_kernel`` IR and returns only attributes
owned by the dialect's schedule ABI.
"""

from __future__ import annotations

from collections.abc import Mapping

from ..kernel.model import MachineSchedule


def schedules_from_mlir(module: str) -> tuple[MachineSchedule, ...]:
    if not isinstance(module, str) or not module.strip():
        raise TypeError("kernel schedule inspection requires non-empty MLIR")
    try:
        from tiga import _graphforge_compiler as native
    except ImportError:
        import _graphforge_compiler as native
    records = native.kernel_schedules(module)
    schedules = []
    for record in records:
        if not isinstance(record, Mapping):
            raise RuntimeError("native schedule inspector returned an invalid record")
        schedules.append(MachineSchedule(
            operation=str(record["operation"]),
            source_location=str(record["source_location"]),
            kind=str(record["kind"]),
            block_rows=int(record["block_rows"]),
            block_neighbors=int(record["block_neighbors"]),
            num_warps=int(record["num_warps"]),
            pipeline_stages=int(record["pipeline_stages"]),
            target_contract=str(record["target_contract"]),
            resources=tuple(map(str, record["resources"])),
            roles=tuple(map(str, record["roles"])),
            handoffs=tuple(map(str, record["handoffs"])),
            instructions=tuple(map(str, record["instructions"])),
        ))
    return tuple(schedules)


__all__ = ["schedules_from_mlir"]
