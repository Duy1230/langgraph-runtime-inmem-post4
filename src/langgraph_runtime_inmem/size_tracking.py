"""Lightweight estimated serialized-size accounting for in-memory data."""

from __future__ import annotations

import pickle
import sys
import threading
from collections.abc import Hashable
from typing import Any


class SizeTracker:
    """Track per-item byte estimates without serializing again at deletion."""

    def __init__(self) -> None:
        self._entries: dict[Hashable, int] = {}
        self._total_bytes = 0
        self._lock = threading.RLock()

    @property
    def total_bytes(self) -> int:
        with self._lock:
            return self._total_bytes

    def set_size(
        self,
        entry: Hashable,
        size_bytes: int,
    ) -> None:
        """Insert or replace one tracked entry."""
        size_bytes = max(0, int(size_bytes))
        with self._lock:
            old_size = self._entries.get(entry, 0)
            self._entries[entry] = size_bytes
            delta = size_bytes - old_size
            self._total_bytes += delta

    def remove(self, entry: Hashable) -> int:
        """Remove one entry and return its tracked byte estimate."""
        with self._lock:
            size_bytes = self._entries.pop(entry, 0)
            self._total_bytes -= size_bytes
            return size_bytes

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._total_bytes = 0


def estimate_pickle_size(value: Any) -> int:
    """Return a best-effort pickle size without affecting the write operation."""
    try:
        return len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception:  # noqa: BLE001 - instrumentation must never break a write
        return sys.getsizeof(value)


def format_size(size_bytes: int | None) -> str:
    """Format a byte estimate for sweep logs."""
    if size_bytes is None:
        return "unknown"
    value = float(max(0, size_bytes))
    units = ("B", "KB", "MB", "GB", "TB")
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:,.1f} {unit}" if unit != "B" else f"{int(value):,} B"
        value /= 1024
    return f"{int(size_bytes):,} B"


def size_log_fields(
    deleted_size_bytes: int | None,
    total_before_bytes: int | None,
) -> dict[str, int | float | str | None]:
    """Build consistent human-readable and numeric TTL sweep log fields."""
    if deleted_size_bytes is None or total_before_bytes is None:
        ratio = None
    elif total_before_bytes <= 0:
        ratio = 0.0
    else:
        ratio = round(100 * deleted_size_bytes / total_before_bytes, 1)
    return {
        "deleted_size": format_size(deleted_size_bytes),
        "deleted_size_bytes": deleted_size_bytes,
        "total_before": format_size(total_before_bytes),
        "total_before_bytes": total_before_bytes,
        "deleted_ratio": "unknown" if ratio is None else f"{ratio:.1f}%",
        "deleted_ratio_percent": ratio,
    }
