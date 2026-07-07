"""Tests for OPENRESEARCH_HARDEXIT_CLEANUP (BUG B): terminate_children_then_exit.

``terminate_children_then_exit`` is ``NoReturn`` (it always ends in
``os._exit``), so it must never be called directly in the pytest process
itself. These tests either:

(a) exercise the real SIGTERM -> grace -> SIGKILL sweep via the internal
    ``_terminate_descendants`` helper against a genuinely spawned child
    process (this is the actual cleanup logic; ``terminate_children_then_exit``
    is a thin, always-exits wrapper around it), or
(b) monkeypatch ``os._exit`` to a stub that raises instead of exiting,
    mirroring the existing pattern for the other hard-exit sites (see
    ``tests/test_cli_default_mode.py`` / ``tests/rdr/test_controller.py``).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

from backend.agents.rlm import process_cleanup


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True  # e.g. EPERM: exists but not signalable — treat as alive
    return True


def _wait_until(predicate, *, timeout_s: float = 5.0, poll_s: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll_s)
    return predicate()


# ---------------------------------------------------------------------------
# Real subprocess kill — the actual cleanup logic
# ---------------------------------------------------------------------------

def test_terminate_descendants_kills_a_real_lingering_child() -> None:
    """The real SIGTERM->grace->SIGKILL sweep stops a spawned child that would
    otherwise linger (a 60s sleep — far longer than this test).

    Liveness is checked via ``child.poll()`` (a non-blocking ``waitpid``),
    NOT ``os.kill(pid, 0)`` — for a DIRECT child of this test process, a
    SIGKILL'd-but-unreaped child is a zombie that still answers signal-probes
    truthfully as "exists" until its parent (this test) reaps it via
    wait()/poll(). psutil's ``wait_procs`` inside ``_terminate_descendants``
    reaps psutil's own handles, not this test's ``Popen`` object, so the
    liveness check here must reap independently.
    """
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        assert _wait_until(lambda: child.poll() is None), "child failed to start"

        process_cleanup._terminate_descendants(grace_s=2.0)

        assert _wait_until(lambda: child.poll() is not None), (
            "child should be gone after the cleanup sweep"
        )
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


def test_terminate_descendants_kills_a_grandchild_too() -> None:
    """``children(recursive=True)`` covers grandchildren, not just direct children."""
    script = (
        "import subprocess, sys, time\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "print(p.pid, flush=True)\n"
        "time.sleep(60)\n"
    )
    parent = subprocess.Popen(
        [sys.executable, "-c", script], stdout=subprocess.PIPE, text=True,
    )
    grandchild_pid: int | None = None
    try:
        line = parent.stdout.readline()
        grandchild_pid = int(line.strip())

        assert _wait_until(lambda: _alive(grandchild_pid)), "grandchild failed to start"

        process_cleanup._terminate_descendants(grace_s=2.0)

        assert _wait_until(lambda: not _alive(grandchild_pid)), (
            "grandchild should be cleaned up too, not just the direct child"
        )
    finally:
        if parent.poll() is None:
            parent.kill()
        parent.wait(timeout=5)
        if grandchild_pid is not None and _alive(grandchild_pid):
            try:
                os.kill(grandchild_pid, 9)
            except OSError:
                pass


def test_terminate_descendants_noop_when_no_children() -> None:
    """A process with no children returns immediately — no error, no hang."""
    process_cleanup._terminate_descendants(grace_s=0.5)  # must not raise


# ---------------------------------------------------------------------------
# terminate_children_then_exit orchestration (mocked — never actually exits)
# ---------------------------------------------------------------------------

def test_terminate_children_then_exit_runs_cleanup_then_os_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple] = []
    monkeypatch.setattr(
        process_cleanup, "_terminate_descendants",
        lambda grace_s: calls.append(("cleanup", grace_s)),
    )

    def _fake_exit(code):
        calls.append(("exit", code))
        raise SystemExit(code)

    monkeypatch.setattr(process_cleanup.os, "_exit", _fake_exit)

    with pytest.raises(SystemExit):
        process_cleanup.terminate_children_then_exit(7)

    assert calls == [("cleanup", process_cleanup._GRACE_S), ("exit", 7)]


def test_terminate_children_then_exit_fails_soft_on_cleanup_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ANY error in cleanup must still fall through to os._exit(code)."""
    def _boom(grace_s):
        raise RuntimeError("cleanup exploded")

    monkeypatch.setattr(process_cleanup, "_terminate_descendants", _boom)

    calls: list[int] = []

    def _fake_exit(code):
        calls.append(code)
        raise SystemExit(code)

    monkeypatch.setattr(process_cleanup.os, "_exit", _fake_exit)

    with pytest.raises(SystemExit):
        process_cleanup.terminate_children_then_exit(3)

    assert calls == [3]
