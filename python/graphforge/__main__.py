"""Print a compact GraphForge compiler installation report."""

from __future__ import annotations

import importlib.metadata

from ._version import __version__
from .compiler.toolchain import find_gf_opt, find_gf_translate
from .runtime import cuda_compute_capability


def main() -> None:
    print(f"GraphForge compiler {__version__}")
    try:
        torch_version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        print("Torch adapter: not installed")
    else:
        print(f"Torch adapter: {torch_version}")
    try:
        major, minor, warp = cuda_compute_capability()
    except RuntimeError:
        print("CUDA runtime: unavailable")
    else:
        print(f"CUDA runtime: sm_{major}{minor}, warp {warp}")
    print(f"gf-opt: {find_gf_opt() or 'not found'}")
    print(f"gf-translate: {find_gf_translate() or 'not found'}")


if __name__ == "__main__":
    main()
