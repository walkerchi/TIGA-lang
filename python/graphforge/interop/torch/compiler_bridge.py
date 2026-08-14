"""Torch interoperability around the canonical GraphForge MLIR compiler.

Graph programs are captured as typed Python descriptors and constructed by
the native C++ OpBuilder. Target-independent passes run in-process in one
MLIRContext. Text is used as a debug snapshot and at the explicit
``gf-translate`` provider boundary, where it isolates GraphForge's pinned MLIR
from a provider compiler's potentially different MLIR build. ``gf-opt`` stays
available for standalone textual-IR debugging and compatibility tests.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory

from .graph import DenseCellDirectory, Graph
from ...reducer import OnlineSoftmaxReducer


@dataclass(frozen=True)
class MLIRStages:
    domain: str
    iteration: str
    kernel: str
    task: str | None = None
    provider_ttir: str | None = None


@dataclass(frozen=True)
class KernelTTIRPlan:
    module: str
    entry: str
    block_rows: int
    num_warps: int
    abi: tuple[str, ...]

    def grid(self, num_rows: int) -> tuple[int, int, int]:
        return ((num_rows + self.block_rows - 1) // self.block_rows, 1, 1)


class _NativeDomainModule(str):
    """Serialized Domain view carrying in-process pass snapshots."""

    stages: tuple[str, str, str, str]

    def __new__(cls, stages: tuple[str, str, str, str]):
        value = super().__new__(cls, stages[0])
        value.stages = stages
        return value


def message_passing_domain_mlir(
    *,
    kernel,
    graph: Graph,
    src: dict[str, object],
    dst: dict[str, object],
    edge: dict[str, object],
    params: dict[str, object],
    kernel_name: str,
    directory: DenseCellDirectory | None = None,
) -> str:
    """Capture one MessagePassing invocation into verified native Domain IR."""
    from .graph import from_native
    from ...compiler.domain_capture import (
        capture_message_passing,
        capture_structured_message_passing,
    )
    from ...compiler.native import domain_stages

    graph = from_native(graph)
    if isinstance(kernel.reducer, OnlineSoftmaxReducer):
        descriptor = capture_structured_message_passing(
            kernel=kernel,
            graph=graph,
            src=src,
            dst=dst,
            edge=edge,
            params=params,
            kernel_name=kernel_name,
        )
    else:
        descriptor = capture_message_passing(
            kernel=kernel,
            graph=graph,
            src=src,
            dst=dst,
            edge=edge,
            params=params,
            kernel_name=kernel_name,
            directory=directory,
        )
    return _NativeDomainModule(domain_stages(descriptor))


def structured_message_bindings(kernel, params):
    """Return structured message capture and its ABI field/parameter order."""
    from ...compiler.domain_capture import structured_message_bindings as capture

    return capture(kernel, params)


def find_gf_opt(explicit: str | os.PathLike[str] | None = None) -> str | None:
    from ...compiler.toolchain import find_gf_opt as discover

    return discover(explicit)


def find_gf_translate(
    explicit: str | os.PathLike[str] | None = None,
    *,
    next_to: str | os.PathLike[str] | None = None,
) -> str | None:
    from ...compiler.toolchain import find_gf_translate as discover

    return discover(explicit, next_to=next_to)


def lower_kernel_to_ttir(
    kernel_module: str,
    *,
    gf_translate: str | os.PathLike[str] | None = None,
) -> str:
    """Serialize verified Kernel IR across the provider process boundary."""
    tool = find_gf_translate(gf_translate)
    if tool is None:
        raise FileNotFoundError(
            "gf-translate was not found; set GRAPHFORGE_TRANSLATE or pass "
            "gf_translate explicitly"
        )
    with TemporaryDirectory(prefix="graphforge-kernel-") as directory:
        path = Path(directory) / "kernel.mlir"
        path.write_text(kernel_module)
        result = subprocess.run(
            [tool, str(path), "-gf-kernel-to-ttir"],
            check=False,
            text=True,
            capture_output=True,
        )
    if result.returncode != 0:
        raise RuntimeError(f"gf-translate failed:\n{result.stderr}")
    return result.stdout


def lower_kernel_to_ttir_plan(
    kernel_module: str,
    *,
    gf_translate: str | os.PathLike[str] | None = None,
) -> KernelTTIRPlan:
    return parse_kernel_ttir_plan(
        lower_kernel_to_ttir(kernel_module, gf_translate=gf_translate)
    )


def parse_kernel_ttir_plan(module: str) -> KernelTTIRPlan:
    """Read the compact launch manifest prepended to serialized TTIR."""
    match = re.match(
        r"// graphforge\.launch entry=(\S+) block_rows=(\d+) "
        r"num_warps=(\d+) abi=([A-Za-z0-9_,]+)\n",
        module,
    )
    if match is None:
        raise RuntimeError("gf-translate output has no GraphForge launch manifest")
    return KernelTTIRPlan(
        module=module,
        entry=match.group(1),
        block_rows=int(match.group(2)),
        num_warps=int(match.group(3)),
        abi=tuple(match.group(4).split(",")),
    )


def lower_mlir_stages(
    domain_module: str,
    *,
    gf_opt: str | os.PathLike[str] | None = None,
) -> MLIRStages:
    """Verify serialized Domain IR and return each canonical pass stage."""
    if isinstance(domain_module, _NativeDomainModule):
        domain, iteration, kernel, task = domain_module.stages
        return MLIRStages(
            domain=domain,
            iteration=iteration,
            kernel=kernel,
            task=task if '"gf_task.' in task else None,
        )
    tool = find_gf_opt(gf_opt)
    if tool is None:
        raise FileNotFoundError(
            "gf-opt was not found; set GRAPHFORGE_OPT or pass gf_opt explicitly"
        )

    def run(path: Path, passes: tuple[str, ...]) -> str:
        result = subprocess.run(
            [tool, str(path), *passes],
            check=False,
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"gf-opt failed ({' '.join(passes)}):\n{result.stderr}"
            )
        return result.stdout

    with TemporaryDirectory(prefix="graphforge-mlir-") as directory:
        path = Path(directory) / "module.mlir"
        path.write_text(domain_module)
        domain = run(path, ("-gf-verify-domain",))
        iteration = run(
            path, ("-gf-verify-domain", "-gf-lower-domain-to-iter")
        )
        kernel = run(
            path,
            (
                "-gf-verify-domain",
                "-gf-lower-domain-to-iter",
                "-gf-lower-iter-to-kernel",
                "-gf-select-kernel-schedule",
            ),
        )
        task = run(
            path,
            (
                "-gf-verify-domain",
                "-gf-lower-domain-to-iter",
                "-gf-lower-iter-to-kernel",
                "-gf-select-kernel-schedule",
                "-gf-plan-degree-buckets",
                "-gf-plan-split-rows",
                "-gf-decompose-degree-worklists",
                "-gf-plan-distributed-tasks",
            ),
        )
    return MLIRStages(
        domain=domain,
        iteration=iteration,
        kernel=kernel,
        task=task if '"gf_task.' in task else None,
    )


__all__ = [
    "KernelTTIRPlan",
    "MLIRStages",
    "find_gf_opt",
    "find_gf_translate",
    "lower_kernel_to_ttir",
    "lower_kernel_to_ttir_plan",
    "lower_mlir_stages",
    "message_passing_domain_mlir",
    "parse_kernel_ttir_plan",
    "structured_message_bindings",
]
