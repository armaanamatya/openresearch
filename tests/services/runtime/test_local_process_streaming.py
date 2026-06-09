"""Tests for the 2026-06-08 execution-reliability redesign — Pillars 1 & 2 in
``backend.services.runtime.local_process.LocalProcessBackend.exec``.

``exec`` was rewritten to STREAM both pipes to a live log + heartbeat sidecar and to
enforce a **stall** window (no liveness → ``exec_stalled``) independently from the
**hard** budget cap (the ``timeout`` param → ``exec_timeout``). Liveness = ANY of
{new stdout/stderr line, checkpoint mtime bump, GPU-util on pinned physical ids,
process-tree CPU-util}. Every augmentation is fail-soft.

These are integration-y (real subprocesses) but FAST: every stall/timeout window is
1–3 s. They follow the repo pattern of a sync test function wrapping ``asyncio.run`` of
the coroutine (see test_gpu_device_ids.py). The sandbox carries NO gpu_device_ids and
``OPENRESEARCH_EXPERIMENT_GPU_LIVENESS=0`` so nothing queries a real GPU (the host's GPU is
never touched — ``_gpu_busy_blocking(())`` short-circuits on empty device_ids anyway).
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from backend.services.runtime import local_process
from backend.services.runtime.interface import (
    RuntimeCauseKind,
    Sandbox,
    SandboxConfig,
)
from backend.services.runtime.local_process import LocalProcessBackend


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_sandbox(tmp_path: Path) -> Sandbox:
    """A local Sandbox with NO GPU device IDs (mirrors test_gpu_device_ids helper)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg = SandboxConfig(
        project_id="test-proj",
        run_id="test-run",
        project_root=tmp_path,
        gpu_device_ids=(),
    )
    return Sandbox(
        sandbox_id="local-test-proj-test-run",
        name="local-test-proj-test-run",
        image="local-process",
        config=cfg,
        created_at=datetime(2026, 6, 8, 0, 0, 0, tzinfo=timezone.utc),
    )


@pytest.fixture(autouse=True)
def _no_gpu_liveness(monkeypatch: pytest.MonkeyPatch) -> None:
    """No real GPU in CI: disable GPU-util polling by default (test 7 re-enables it)."""
    monkeypatch.setenv("OPENRESEARCH_EXPERIMENT_GPU_LIVENESS", "0")


def _run(sandbox: Sandbox, command: str, timeout: int):
    return asyncio.run(LocalProcessBackend().exec(sandbox, command, timeout=timeout))


# ---------------------------------------------------------------------------
# 1. Normal exit + streaming to live log + heartbeat sidecar
# ---------------------------------------------------------------------------


def test_normal_exit_streams_to_live_log_and_heartbeat(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENRESEARCH_EXPERIMENT_STALL_S", "0")
    sandbox = _make_sandbox(tmp_path)

    r = _run(sandbox, "echo hello && echo done", timeout=30)

    assert r.exit_code == 0
    assert r.succeeded
    assert "hello" in r.stdout
    assert "done" in r.stdout

    live = tmp_path / ".exec_live.log"
    assert live.exists()
    assert "hello" in live.read_text(encoding="utf-8")

    hb = tmp_path / ".exec_heartbeat.json"
    assert hb.exists()
    data = json.loads(hb.read_text(encoding="utf-8"))
    assert "lines" in data


# ---------------------------------------------------------------------------
# 2. exec_stalled — a silent sleep with no liveness trips the (short) stall window
# ---------------------------------------------------------------------------


def test_exec_stalled_on_silent_sleep(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENRESEARCH_EXPERIMENT_STALL_S", "1")
    sandbox = _make_sandbox(tmp_path)

    t0 = time.monotonic()
    r = _run(sandbox, "sleep 30", timeout=120)
    elapsed = time.monotonic() - t0

    assert elapsed < 15, f"stall should fire well before the 30 s sleep; took {elapsed:.1f}s"
    assert r.timed_out is True
    assert r.cause_kind == RuntimeCauseKind.exec_stalled


# ---------------------------------------------------------------------------
# 3. exec_timeout — a CPU-busy loop hits the hard cap (NOT classified as stalled)
# ---------------------------------------------------------------------------


def test_exec_timeout_hard_cap_on_cpu_busy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Disable the stall window so only the hard timeout can fire; the job is CPU-busy
    # (so it would never be "stalled" anyway) and must be cut at the budget cap.
    monkeypatch.setenv("OPENRESEARCH_EXPERIMENT_STALL_S", "0")
    sandbox = _make_sandbox(tmp_path)

    t0 = time.monotonic()
    r = _run(sandbox, "while true; do :; done", timeout=2)
    elapsed = time.monotonic() - t0

    assert r.cause_kind == RuntimeCauseKind.exec_timeout
    assert r.timed_out is True
    assert 1.5 <= elapsed <= 8.0, f"expected ~2 s hard cap, got {elapsed:.1f}s"


# ---------------------------------------------------------------------------
# 4. CPU liveness keeps a quiet-but-computing job alive through the stall window
# ---------------------------------------------------------------------------


def test_cpu_liveness_keeps_busy_job_alive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Stall window (2 s) < job length (5 s), but the job is CPU-busy with NO output,
    # so the process-tree CPU-time delta is the liveness signal that prevents a kill.
    monkeypatch.setenv("OPENRESEARCH_EXPERIMENT_STALL_S", "2")
    sandbox = _make_sandbox(tmp_path)

    t0 = time.monotonic()
    # Pure-bash busy spin: GNU `timeout` does not exist on macOS, where the
    # old `timeout 5 …` exited command-not-found in ~6ms and the busy loop
    # never ran (test only passed on Linux; audit 2026-06-09). $SECONDS ticks
    # on whole-second boundaries, so N ticks elapse in [N-1, N] real seconds —
    # spin 6 ticks to land in [5, 6]s, safely inside the 4.5–12s assertion.
    r = _run(
        sandbox,
        "bash -c 'end=$((SECONDS+6)); while [ $SECONDS -lt $end ]; do :; done'; true",
        timeout=60,
    )
    elapsed = time.monotonic() - t0

    # Must NOT be killed for stalling — it ran to completion on CPU liveness alone.
    assert r.cause_kind != RuntimeCauseKind.exec_stalled
    assert r.cause_kind is None or r.cause_kind == RuntimeCauseKind.command_failed
    assert 4.5 <= elapsed <= 12.0, f"expected ~5 s run, got {elapsed:.1f}s"


# ---------------------------------------------------------------------------
# 5. No process-group orphan — the group kill reaps the child sleep
# ---------------------------------------------------------------------------


def test_no_sleep_orphan_after_stall_kill(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENRESEARCH_EXPERIMENT_STALL_S", "1")
    sandbox = _make_sandbox(tmp_path)

    # A unique per-test token so the count isolates THIS run's process tree from any
    # unrelated `sleep` processes on a shared host (a bare `pgrep -x sleep` count is
    # noisy on a multi-tenant box). The token lives in the script path → `pgrep -f`
    # matches our `sh <token>.sh` (and its child sleep) and nothing else.
    token = f"reprolab_orphan_{uuid.uuid4().hex}"
    script = tmp_path / f"{token}.sh"
    script.write_text("sleep 30\n", encoding="utf-8")

    def _count() -> int | None:
        if shutil.which("pgrep") is None:
            return None
        try:
            out = subprocess.run(
                ["pgrep", "-c", "-f", token],
                capture_output=True,
                text=True,
                timeout=5,
            )
            # pgrep -c prints a count even when 0 (rc=1 when no match); parse stdout.
            return int((out.stdout or "0").strip() or "0")
        except Exception:  # noqa: BLE001
            return None

    before = _count()
    r = _run(sandbox, f"bash {script}", timeout=120)
    assert r.cause_kind == RuntimeCauseKind.exec_stalled
    time.sleep(0.5)  # let the SIGKILL'd group fully reap
    after = _count()

    if before is None or after is None:
        pytest.skip("pgrep unavailable — cannot verify process-group reaping")
    # The group kill must reap our whole tree (the wrapper script + its child sleep),
    # leaving zero of OUR marked processes — and never more than we started with.
    assert after <= before, (
        f"the stalled job's process tree must be reaped by the group kill (not leaked): "
        f"before={before} after={after}"
    )
    assert after == 0, f"orphaned process from the stalled run survived: {after} left"


# ---------------------------------------------------------------------------
# 6. Retained-text cap — ExecResult bounds stdout; live log keeps everything
# ---------------------------------------------------------------------------


def test_retained_text_is_capped_but_live_log_keeps_all(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENRESEARCH_EXPERIMENT_STALL_S", "0")
    sandbox = _make_sandbox(tmp_path)

    line = "A" * 44  # fixed 44-char ASCII line → 45 bytes with newline
    # 8000 * 45 B ≈ 360 KB > the 256 KB head+tail retention cap.
    r = _run(sandbox, f"for i in $(seq 1 8000); do echo {line}; done", timeout=30)

    assert r.exit_code == 0
    assert len(r.stdout) <= 270 * 1024, f"retained stdout not capped: {len(r.stdout)} bytes"
    assert "truncated" in r.stdout

    live = tmp_path / ".exec_live.log"
    assert live.exists()
    # The live log retains the FULL stream (plus a header line), so it is larger than
    # the capped ExecResult text.
    assert live.stat().st_size > len(r.stdout)


# ---------------------------------------------------------------------------
# 7. GPU-liveness fail-soft (poll raises) + CPU probe on a nonexistent pid
# ---------------------------------------------------------------------------


def test_gpu_liveness_poll_exception_is_swallowed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Re-enable GPU liveness (the autouse fixture disabled it) and make the poll raise;
    # the monitor must swallow it and let the happy path complete.
    monkeypatch.setenv("OPENRESEARCH_EXPERIMENT_GPU_LIVENESS", "1")
    monkeypatch.setenv("OPENRESEARCH_EXPERIMENT_STALL_S", "0")

    def _boom(*_a, **_k):
        raise RuntimeError("nvidia-smi exploded")

    monkeypatch.setattr(local_process, "_gpu_busy_blocking", _boom)

    sandbox = _make_sandbox(tmp_path)
    # `echo` then a brief quiet phase: the second monitor poll has no output/CPU/ckpt
    # liveness, so it reaches the (now-raising) GPU poll — proving the exception is
    # swallowed rather than aborting exec.
    r = _run(sandbox, "echo hello; sleep 2", timeout=60)

    assert r.exit_code == 0
    assert r.succeeded
    assert "hello" in r.stdout

    # CPU probe on a nonexistent pid must fail-soft (None or a float, never raise).
    cpu = local_process._proc_tree_cpu_seconds(99999999)
    assert cpu is None or isinstance(cpu, float)
