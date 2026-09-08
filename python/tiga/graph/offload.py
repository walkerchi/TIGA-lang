"""Budget-driven automatic graph offload.

While ``auto_offload`` is active (or ``TIGA_GRAPH_RAM_BUDGET`` is set),
``Graph.from_csr`` — the funnel every explicit CSR builder goes through —
compares the CSR byte size against the RAM budget. A graph that exceeds the
budget is persisted as a versioned ``.gfg`` and reopened as a ``paged_csr``
graph, so execution streams bounded destination-row pages with prefetch
instead of materializing the topology in RAM.

The offload directory is per-process and removed at exit; set
``TIGA_SPILL_DIR`` to keep offloaded graphs across processes.

v1 boundary: only CPU graphs offload (the paged executor is host-side), and
only the topology is paged — node/edge fields stay in RAM.
"""

from __future__ import annotations

import atexit
import contextvars
import itertools
import os
from pathlib import Path
import shutil
import tempfile

_BUDGET_BYTES: "contextvars.ContextVar[int | None]" = contextvars.ContextVar(
    "graphforge_graph_ram_budget", default=None)
_BUDGET_ENV = "TIGA_GRAPH_RAM_BUDGET"

# Paged execution builds page-local CSR graphs through from_csr; those must
# never offload recursively.
_IN_PAGED: "contextvars.ContextVar[bool]" = contextvars.ContextVar(
    "graphforge_in_paged_execution", default=False)

_offload_dir: Path | None = None
_counter = itertools.count()


def _parse_bytes(value: object, name: str) -> int:
    try:
        result = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer byte count") from None
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


class auto_offload:
    """Context manager: page CSR graphs larger than ``ram`` bytes to disk."""

    def __init__(self, ram: int) -> None:
        self._budget = _parse_bytes(ram, "auto_offload ram budget")
        self._token = None

    def __enter__(self) -> "auto_offload":
        self._token = _BUDGET_BYTES.set(self._budget)
        return self

    def __exit__(self, *_exc) -> None:
        _BUDGET_BYTES.reset(self._token)


def current_budget() -> int | None:
    """The active RAM budget in bytes: context manager, else env, else None."""
    budget = _BUDGET_BYTES.get()
    if budget is not None:
        return budget
    override = os.environ.get(_BUDGET_ENV)
    if override is None:
        return None
    return _parse_bytes(override, _BUDGET_ENV)


def _offload_root() -> Path:
    """Per-process offload directory (or the shared spill store)."""
    global _offload_dir
    override = os.environ.get("TIGA_SPILL_DIR")
    if override is not None:
        path = Path(override) / "graphs"
        path.mkdir(parents=True, exist_ok=True)
        return path
    if _offload_dir is None:
        _offload_dir = Path(
            tempfile.mkdtemp(prefix="tiga-graph-offload-"))
        atexit.register(shutil.rmtree, _offload_dir, ignore_errors=True)
    return _offload_dir


def maybe_offload(graph) -> object:
    """Return a paged replacement for ``graph`` when it exceeds the budget."""
    from ..runtime import DeviceType  # lazy: graph ↔ runtime import cycle

    budget = current_budget()
    if budget is None or _IN_PAGED.get() or \
            graph.device.type is not DeviceType.CPU:
        return graph
    row_ptr, col_idx = graph.resolve_csr()
    if row_ptr.numel * row_ptr.dtype.itemsize + \
            col_idx.numel * col_idx.dtype.itemsize <= budget:
        return graph
    from .storage import PagedCSRStore  # lazy: keep module import leaf-only

    destination = _offload_root() / f"auto-{os.getpid()}-{next(_counter)}.gfg"
    PagedCSRStore.create(graph, destination)
    return graph.open(destination, device=graph.device)


__all__ = ["auto_offload", "current_budget", "maybe_offload"]
