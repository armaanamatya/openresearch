"""C2 — mid-run orphan-resource guard (2026-06-16).

When binding's per-primitive daemon-thread timeout abandons a ``run_experiment``
worker thread (``binding.py`` ``FuturesTimeoutError`` path), the experiment's
subprocess **process group keeps running** and holding GPU/VRAM — starving the
retry the timeout just authorised. The exec layer already launches its training
subprocesses with ``start_new_session=True`` (their own process group) and kills
them on *their own* stall/timeout; the gap is when the OUTER binding timeout fires
*first*, so the inner kill never runs.

This module is a tiny thread-safe registry the exec layer populates (the session
leader's pid == its pgid) so the binding timeout handler can SIGKILL the whole
tree on abandonment — reusing the cell runner's process-group-kill.

``REPROLAB_ORPHAN_GUARD`` is **default-OFF**: registration/deregistration is
cheap always-on bookkeeping, but :func:`kill_orphans` is a no-op (returns 0) when
the flag is off, so behavior is byte-for-byte today. Deregistration on normal
completion keeps the set small and bounds PID-recycle risk; every kill is
fail-soft per-pgid (a dead/recycled-away group just no-ops).
"""

from __future__ import annotations

import os
import signal
import threading

_lock = threading.Lock()
# Session-leader pids of live experiment subprocesses (pid == pgid because they
# were spawned with start_new_session=True).
_active_pgids: set[int] = set()


def orphan_guard_enabled() -> bool:
    """True iff ``REPROLAB_ORPHAN_GUARD`` opts the kill ON (default OFF)."""
    return os.environ.get("REPROLAB_ORPHAN_GUARD", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def register(pid: int) -> None:
    """Record a spawned experiment subprocess's process group (best-effort).

    ``pid`` must be a ``start_new_session=True`` session leader (pid == pgid).
    Always runs (cheap bookkeeping) so the registry is populated even with the
    flag off — only the kill is gated.
    """
    try:
        with _lock:
            _active_pgids.add(int(pid))
    except (TypeError, ValueError):
        pass


def deregister(pid: int) -> None:
    """Drop a reaped subprocess's process group from the registry (no-op if absent)."""
    try:
        with _lock:
            _active_pgids.discard(int(pid))
    except (TypeError, ValueError):
        pass


def kill_orphans() -> int:
    """SIGKILL every registered process group; return the count signalled.

    No-op (returns 0) unless ``REPROLAB_ORPHAN_GUARD`` is on — so the default is
    byte-for-byte today. Drains the registry first (under the lock) so a kill and
    a concurrent deregister can't double-act. Each ``killpg`` is fail-soft: a
    process group that already exited just raises ``ProcessLookupError`` and is
    skipped.
    """
    if not orphan_guard_enabled():
        return 0
    with _lock:
        pgids = list(_active_pgids)
        _active_pgids.clear()
    killed = 0
    for pgid in pgids:
        try:
            os.killpg(pgid, signal.SIGKILL)
            killed += 1
        except (ProcessLookupError, PermissionError, OSError):
            pass
    return killed
