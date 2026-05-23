"""Process-local in-memory cache for leaderboard rows.

Design (locked by spec):
- Dict keyed by project_id → {mtime: float, row: LeaderboardRow}.
- On each request, stat() each final_report.json.  If mtime matches the
  cached entry, use the cached row.  If mtime is newer or no entry exists,
  re-read + parse and update the cache.
- A threading.Lock() guards the dict so concurrent FastAPI requests are safe.
- No TTL — mtime IS the invalidation signal (D4).
- Stale entries (run directory deleted) are evicted at the top of the
  aggregate call (D6).

This module is intentionally free of any FastAPI / route imports so it can
be unit-tested without spinning up an app.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# _cache: project_id → {"mtime": float, "row": <LeaderboardRow-compatible dict>}
# We store the raw parsed dict (not the Pydantic model) so the cache stays
# serialisation-agnostic.  The route layer re-hydrates to LeaderboardRow after
# the cache hit.
_cache: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()


def _stat_mtime(path: Path) -> float | None:
    """Return the file's mtime as a float, or None if it doesn't exist."""
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _load_json(path: Path) -> dict | None:
    """Read and JSON-parse *path*, returning None on any error."""
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("leaderboard_cache: failed to read %s — %s", path, exc)
        return None


def evict_missing(runs_root: Path) -> None:
    """Remove cache entries whose run directory no longer exists on disk.

    Called once per aggregate request before the per-run loop so we never
    hand back data for a deleted run.
    """
    with _lock:
        to_drop = [
            pid for pid in list(_cache)
            if not (runs_root / pid).is_dir()
        ]
        for pid in to_drop:
            del _cache[pid]


def get_or_load(project_id: str, fr_path: Path) -> dict | None:
    """Return the cached parsed dict for *project_id*, reloading if stale.

    Returns None when the file cannot be read or does not exist.
    The returned value is the raw JSON dict (not yet coerced to LeaderboardRow).
    """
    current_mtime = _stat_mtime(fr_path)
    if current_mtime is None:
        # File gone since the directory scan — drop stale entry if any.
        with _lock:
            _cache.pop(project_id, None)
        return None

    with _lock:
        entry = _cache.get(project_id)
        if entry is not None and entry["mtime"] == current_mtime:
            return entry["data"]

    # Cache miss or stale — parse outside the lock (I/O should not hold it).
    data = _load_json(fr_path)
    if data is None:
        return None

    with _lock:
        # Double-check: another thread may have populated while we were
        # reading.  Only overwrite if our mtime is still the freshest.
        existing = _cache.get(project_id)
        if existing is None or existing["mtime"] <= current_mtime:
            _cache[project_id] = {"mtime": current_mtime, "data": data}

    return data


def clear() -> None:
    """Wipe the entire cache.  Useful in tests to ensure isolation."""
    with _lock:
        _cache.clear()


__all__ = [
    "evict_missing",
    "get_or_load",
    "clear",
]
