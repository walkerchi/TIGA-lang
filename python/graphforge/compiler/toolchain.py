"""Discovery of wheel-bundled or editable-build GraphForge compiler tools."""

from __future__ import annotations

import os
from pathlib import Path
import shutil


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
        "gf-opt": "GRAPHFORGE_OPT",
        "gf-translate": "GRAPHFORGE_TRANSLATE",
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
    candidates.extend(repository.glob(f"build/*/bin/{name}"))
    available = [path for path in candidates if path.is_file()]
    if available:
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
