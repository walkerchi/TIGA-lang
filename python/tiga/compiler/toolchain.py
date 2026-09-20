"""Discovery of wheel-bundled or editable-build Tiga compiler tools."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys


def find_tool(
    name: str,
    *,
    explicit: str | os.PathLike[str] | None = None,
    next_to: str | os.PathLike[str] | None = None,
) -> str | None:
    if explicit is not None:
        path = Path(explicit)
        return str(path) if path.is_file() else None
    environment_name = {
        "gf-opt": "TIGA_OPT",
        "gf-translate": "TIGA_TRANSLATE",
    }.get(name)
    if environment_name:
        configured = os.environ.get(environment_name)
        if configured and Path(configured).is_file():
            return configured
    candidates: list[Path] = []
    if next_to is not None:
        candidates.append(Path(next_to).with_name(name))
    package = Path(__file__).resolve().parents[1]
    repository = Path(__file__).resolve().parents[3]
    candidates.append(package / "bin" / name)
    # Prefer installed companions; never override a wheel with a newer build
    # accidentally found next to site-packages. Editable tools follow the
    # same ABI-compatible extension build, not an independent mtime contest.
    for path in candidates:
        if path.is_file():
            return str(path)
    from .native import _compatible_extension
    loaded = sys.modules.get("tiga._tiga_compiler")
    extension = getattr(loaded, "__file__", None)
    if extension:
        companion = Path(extension).parent.parent / "bin" / name
        if companion.is_file():
            return str(companion)
        return None  # Never mix a loaded editable frontend with another build.
    extensions = [path for path in repository.glob(
        "build/*/python_bindings/_tiga_compiler*.so")
        if _compatible_extension(path)]
    if extensions:
        extension = max(extensions, key=lambda path: path.stat().st_mtime_ns)
        companion = extension.parent.parent / "bin" / name
        if companion.is_file():
            return str(companion)
        return None
    # Standalone compiler builds need not include the Python extension.
    available = list(repository.glob(f"build/*/bin/{name}"))
    if available and not extensions and not extension:
        return str(max(available, key=lambda path: path.stat().st_mtime_ns))
    return shutil.which(name)


def find_gf_opt(explicit: str | os.PathLike[str] | None = None) -> str | None:
    return find_tool("gf-opt", explicit=explicit)


def find_gf_translate(
    explicit: str | os.PathLike[str] | None = None,
    *,
    next_to: str | os.PathLike[str] | None = None,
) -> str | None:
    return find_tool("gf-translate", explicit=explicit, next_to=next_to)


__all__ = ["find_gf_opt", "find_gf_translate", "find_tool"]
