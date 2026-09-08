"""Native task-IR to provider-neutral executable-bundle translation."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from ..runtime import ExecutableBundlePlan
from .toolchain import find_gf_translate


def translate_task_bundle(
    task_ir: str,
    *,
    gf_translate: str | os.PathLike[str] | None = None,
) -> ExecutableBundlePlan:
    """Translate verified ``gf_task`` IR into a typed runtime plan.

    Text crosses only the intentionally isolated native MLIR/provider boundary;
    scheduling decisions remain encoded and verified in task IR.
    """
    if not isinstance(task_ir, str) or '"gf_task.' not in task_ir:
        raise ValueError("task_ir must contain a Tiga task plan")
    tool = find_gf_translate(gf_translate)
    if tool is None:
        raise FileNotFoundError(
            "gf-translate was not found; install tiga-lang or set "
            "TIGA_TRANSLATE"
        )
    with TemporaryDirectory(prefix="tiga-task-") as directory:
        path = Path(directory) / "task.mlir"
        path.write_text(task_ir)
        result = subprocess.run(
            [tool, str(path), "-gf-task-to-bundle"],
            check=False,
            text=True,
            capture_output=True,
        )
    if result.returncode != 0:
        raise RuntimeError(f"gf-task-to-bundle failed:\n{result.stderr}")
    return ExecutableBundlePlan.parse(result.stdout)


__all__ = ["translate_task_bundle"]
