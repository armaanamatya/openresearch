"""Best-effort descendant-process cleanup before an unconditional ``os._exit``.

Several call sites hard-exit the process (``os._exit``) to route around a
known claude-agent-sdk / asyncio atexit hang on WSL2 (a bare
``subprocess.wait()`` deadlocking in ``futex_wait_queue`` once interpreter
shutdown starts). ``os._exit`` skips normal interpreter teardown — including
whatever child-process reaping the SDK would otherwise perform — so a still
-running claude-agent-sdk executor subprocess (and its own children) can be
orphaned mid-flight, left burning API tokens with nothing left to observe or
bound it.

``terminate_children_then_exit`` is the shared, flag-gated remedy: a bounded
best-effort SIGTERM -> short grace -> SIGKILL sweep of this process's
descendants, run immediately before the same hard ``os._exit`` the caller
would otherwise have made directly. It never performs an unbounded wait —
that is exactly the hang ``os._exit`` exists to route around — and any error
anywhere in the sweep falls through to ``os._exit(code)`` regardless.

Flag: ``OPENRESEARCH_HARDEXIT_CLEANUP`` (default OFF, checked at each call
site). Call sites keep their bare ``os._exit(code)`` when the flag is off;
this module is only imported/invoked when it's on.
"""

from __future__ import annotations

import logging
import os
import time
from typing import NoReturn

logger = logging.getLogger(__name__)

# Short and bounded on purpose: long enough for a cooperative subprocess to
# exit on SIGTERM, short enough to never resemble the unbounded wait() this
# whole module exists to avoid.
_GRACE_S = 2.0
_POLL_S = 0.05


def terminate_children_then_exit(code: int) -> NoReturn:
    """Best-effort terminate this process's descendants, then ``os._exit(code)``.

    Fail-soft by construction: any exception anywhere in the sweep is
    swallowed and control still falls through to ``os._exit(code)``, so this
    can never turn a hard-exit site into a hang or a crash.
    """
    try:
        _terminate_descendants(_GRACE_S)
    except Exception:  # noqa: BLE001 — cleanup must never block the hard exit
        try:
            logger.debug(
                "terminate_children_then_exit: cleanup raised; exiting anyway",
                exc_info=True,
            )
        except Exception:  # noqa: BLE001 — even logging must not block the exit
            pass
    os._exit(code)


def _terminate_descendants(grace_s: float) -> None:
    """SIGTERM this process's descendants, wait *grace_s* (bounded), SIGKILL stragglers.

    ``psutil`` first (clean recursive child-tree enumeration — same idiom as
    ``local_process.py::_proc_tree_cpu_seconds``); falls back to a stdlib
    ``/proc`` walk (Linux only) when psutil is not importable or raises.
    """
    try:
        import psutil  # convenience: recursive child tree + a bounded wait_procs

        me = psutil.Process(os.getpid())
        children = me.children(recursive=True)
        if not children:
            return
        for child in children:
            try:
                child.terminate()  # SIGTERM
            except Exception:  # noqa: BLE001 — a vanished/unreachable child is fine
                continue
        # Bounded: psutil.wait_procs returns at latest after `timeout` seconds,
        # never blocks indefinitely — the load-bearing property here.
        _gone, alive = psutil.wait_procs(children, timeout=grace_s)
        for child in alive:
            try:
                child.kill()  # SIGKILL
            except Exception:  # noqa: BLE001
                continue
        return
    except Exception:  # noqa: BLE001 — psutil absent/broken -> /proc fallback
        pass
    _terminate_descendants_proc_fallback(grace_s)


def _direct_children(parent_pid: int) -> list[int]:
    """Direct child PIDs of *parent_pid* via a single ``/proc`` scan (Linux only).

    Any read/parse error for an individual entry is skipped, not raised — a
    process can vanish mid-scan. An unreadable/absent ``/proc`` yields ``[]``.
    """
    found: list[int] = []
    try:
        entries = os.listdir("/proc")
    except OSError:
        return found
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat", encoding="utf-8", errors="replace") as fh:
                data = fh.read()
            # comm (2nd field) is parenthesized and may itself contain ')';
            # split on the LAST ')' so the field right after it is reliably ppid.
            ppid = int(data.rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            continue
        if ppid == parent_pid:
            found.append(int(entry))
    return found


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True  # e.g. EPERM: exists but not signalable by us — treat as alive
    return True


def _terminate_descendants_proc_fallback(grace_s: float) -> None:
    """Stdlib-only descendant sweep for environments without a usable psutil."""
    import signal

    pid = os.getpid()
    descendants: set[int] = set()
    frontier = _direct_children(pid)
    while frontier:
        descendants.update(frontier)
        frontier = [
            child
            for parent in frontier
            for child in _direct_children(parent)
            if child not in descendants
        ]
    if not descendants:
        return

    for child_pid in descendants:
        try:
            os.kill(child_pid, signal.SIGTERM)
        except OSError:
            pass

    deadline = time.monotonic() + grace_s
    remaining = set(descendants)
    while remaining and time.monotonic() < deadline:
        remaining = {p for p in remaining if _pid_alive(p)}
        if remaining:
            time.sleep(_POLL_S)

    for child_pid in remaining:
        try:
            os.kill(child_pid, signal.SIGKILL)
        except OSError:
            pass
