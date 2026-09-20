"""Execution-scoped accounting shared by native buffers and temporary storage.

Limits cover live Tiga-owned allocations made in the scope, not process RSS,
external framework storage, allocator pools created earlier, or compiler memory.
"""

from __future__ import annotations

import contextvars
import gc
from collections import OrderedDict
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
import re
from threading import RLock
import weakref

_CURRENT = contextvars.ContextVar("tiga_execution", default=None)


def parse_bytes(value: int | str) -> int:
    if isinstance(value, bool):
        raise ValueError("memory size must be a byte count or an IEC size")
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, str):
        match = re.fullmatch(r"\s*(\d+)\s*(B|KiB|MiB|GiB|TiB)?\s*", value)
        if match:
            return int(match[1]) * (1024 ** ("B", "KiB", "MiB", "GiB", "TiB").index(match[2] or "B"))
    raise ValueError("memory size must be non-negative bytes or e.g. '8GiB'")


def current_execution():
    return _CURRENT.get()


class Reservation:
    def __init__(self, execution, tier, size):
        self.execution, self.tier, self.size = execution, tier, size
        self.closed = False

    def close(self):
        if not self.closed:
            with self.execution._lock:
                if not self.closed:
                    self.execution._live[self.tier] -= self.size
                    self.closed = True


class execution:
    """Select default device, managed allocation limits and paging controls.

Existing tensors keep their device and ownership. Exiting the context does not
invalidate returned values. Optional LRU spills idle whole Tensor values;
unsupported kernel working sets still fail before allocation.
"""

    def __init__(self, *, device="cpu", memory=None, spill_dir=None,
                 page_rows=100_000, prefetch_depth=2, eviction="error"):
        from . import Device
        from .hierarchy import MemoryTier

        self.device = Device.parse(device)
        if eviction not in {"error", "lru"}:
            raise ValueError("eviction must be 'error' or 'lru'")
        if eviction == "lru" and spill_dir is None:
            raise ValueError("automatic eviction requires an explicit spill_dir")
        self.eviction = eviction
        self.budgets = {}
        for key, value in (memory or {}).items():
            tier = MemoryTier.parse(key).value
            if tier not in {"device", "host-pinned", "ram", "nvme"}:
                raise ValueError(f"{tier} is not a runtime allocation budget")
            if tier in self.budgets:
                raise ValueError(f"duplicate memory tier: {tier}")
            self.budgets[tier] = parse_bytes(value)
        for name, value in (("page_rows", page_rows), ("prefetch_depth", prefetch_depth)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.page_rows, self.prefetch_depth = page_rows, prefetch_depth
        self.spill_dir = None if spill_dir is None else Path(spill_dir)
        self._lock = RLock()
        self._live = {tier: 0 for tier in ("device", "host-pinned", "ram", "nvme")}
        self._peak = dict(self._live)
        self._token = None
        self._resident = OrderedDict()
        self._pins = {}
        self._evicting = False
        self._evictions = 0
        self._restores = 0

    def track(self, value):
        if self.eviction != "lru":
            return
        with self._lock:
            key = id(value)
            owner = weakref.ref(self)
            def forget(reference):
                active = owner()
                if active is not None:
                    with active._lock:
                        if active._resident.get(key) is reference:
                            active._resident.pop(key, None)
            self._resident[key] = weakref.ref(value, forget)
            self._resident.move_to_end(key)

    @contextmanager
    def pin(self, root):
        """Serialize managed operations and pin their materialized frontier."""
        with self._lock:
            values, pending = {}, [root]
            while pending:
                value = pending.pop()
                if id(value) in values:
                    continue
                values[id(value)] = value
                if value._buffer is None and value._spill is None and value._expr is not None:
                    pending.extend(value._expr.operands)
            for key, value in values.items():
                previous = self._pins.get(key)
                self._pins[key] = (value, 1 if previous is None else previous[1] + 1)
                self.track(value)
            try:
                yield tuple(values.values())
            finally:
                for key in values:
                    value, count = self._pins[key]
                    if count == 1:
                        del self._pins[key]
                    else:
                        self._pins[key] = (value, count - 1)

    def _make_room(self, tier, size, limit):
        if self.eviction != "lru" or tier not in {"ram", "device"} or self._evicting:
            return
        # An operation larger than the budget cannot be fixed by evicting data.
        if size > limit:
            return
        self._evicting = True
        try:
            for key, reference in list(self._resident.items()):
                if self._live[tier] + size <= limit:
                    break
                value = reference()
                if value is None:
                    self._resident.pop(key, None)
                    continue
                buffer = value._buffer
                reservation = getattr(buffer, "_reservation", None)
                if (key in self._pins or reservation is None or reservation.closed or
                        reservation.execution is not self or reservation.tier != tier):
                    continue
                if any(pinned._buffer is buffer for pinned, _ in self._pins.values()):
                    continue
                # Do not keep a local strong reference to the allocation while
                # checking whether spill actually released it (views may not).
                del buffer
                value.spill()
                self._evictions += 1
        finally:
            self._evicting = False

    def reserve(self, tier: str, size: int):
        with self._lock:
            live = self._live[tier] + size
            limit = self.budgets.get(tier)
            if limit is not None and live > limit:
                # Python compiler walkers/prepared launchers can form cycles.
                # Reclaim unreachable values before treating them as live
                # working-set pressure or spilling useful resident values.
                gc.collect()
                live = self._live[tier] + size
            if limit is not None and live > limit:
                self._make_room(tier, size, limit)
                live = self._live[tier] + size
            if limit is not None and live > limit:
                raise MemoryError(
                    f"Tiga managed {tier} budget exceeded: {live} > {limit} bytes; "
                    "spill/release resident values or use a supported paged plan. "
                    "This limit is not process RSS.")
            self._live[tier] = live
            self._peak[tier] = max(live, self._peak[tier])
        return Reservation(self, tier, size)

    def memory_report(self):
        with self._lock:
            return {"scope": "allocations-created-in-this-execution",
                    "excludes": ["external buffers", "earlier allocations", "Python/compiler memory", "OS page cache", "I/O staging bytes"],
                    "live_bytes": dict(self._live), "peak_bytes": dict(self._peak),
                    "budget_bytes": dict(self.budgets),
                    "automatic_tensor_eviction": self.eviction == "lru",
                    "eviction_policy": self.eviction,
                    "evictions": self._evictions, "restores": self._restores}

    def __enter__(self):
        if self._token is not None:
            raise RuntimeError("the same execution context cannot be re-entered")
        if current_execution() is not None:
            raise RuntimeError("nested execution budgets are not supported")
        self._token = _CURRENT.set(self)
        return self

    def __exit__(self, *_exc):
        _CURRENT.reset(self._token)
        self._token = None


def reserve(tier, size):
    active = current_execution()
    return None if active is None else active.reserve(tier, size)


def default_device():
    active = current_execution()
    return "cpu" if active is None else active.device


def managed_operation(function):
    """Guard synchronous Tensor entry points only when paging is enabled."""
    @wraps(function)
    def guarded(self, *args, **kwargs):
        active = current_execution()
        if active is None or active.eviction != "lru":
            return function(self, *args, **kwargs)
        with active.pin(self) as frontier:
            # Native capture treats materialized leaves as storage bindings.
            # Restore spilled inputs before capture, not after addresses bind.
            for value in frontier:
                if value is not self and value._spill is not None:
                    value.realize()
            result = function(self, *args, **kwargs)
            if self.ready_event is not None:
                self.ready_event.wait()
            active.track(self)
            return result
    return guarded
