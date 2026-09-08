"""Shared capability probes for distributed tests.

Module level stays stdlib-only so torch-free CI jobs can collect every test
module importing these helpers; optional dependencies are probed lazily.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys


def mpi_launcher() -> str | None:
    """Return an mpiexec path when mpi4py and a launcher are both usable."""
    try:
        from mpi4py import MPI  # noqa: F401
    except (ImportError, RuntimeError):
        return None
    mpiexec = shutil.which("mpiexec")
    if mpiexec is None:
        # Wheels may install the launcher next to the active interpreter
        # without activating that directory in PATH.
        candidate = Path(os.path.dirname(sys.executable)) / "mpiexec"
        mpiexec = str(candidate) if candidate.exists() else None
    return mpiexec


def cuda_device_available(device: str) -> bool:
    try:
        import tiga as gf

        gf.runtime.cuda_compute_capability(device)
    except (ModuleNotFoundError, RuntimeError):
        return False
    return True


def nccl_skip_reason(device: str = "cuda:0") -> str | None:
    """Return None when CUDA plus NCCL unique-id creation works in-process."""
    try:
        import tiga as gf
        from tiga.distributed import nccl_unique_id

        gf.runtime.cuda_compute_capability(device)
        nccl_unique_id()
    except (ModuleNotFoundError, RuntimeError) as error:
        return f"NCCL/CUDA runtime is unavailable: {error}"
    return None
