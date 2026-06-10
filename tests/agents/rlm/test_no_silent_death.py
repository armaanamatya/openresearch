"""Regression tests for the 2026-06-01 "no-silent-death" hardening.

The SDAR run prj_09047604e591d969 hung ~3h inside one synchronous run_experiment
matrix with --max-wall-clock unset (all soft bounds collapsed to None, the watchdog
unarmed), the operator killed it, and NO final_report was ever written. Three fixes
close that gap; this suite pins each contract:

  C-B1  ``_arm_watchdog`` arms an always-on hard-ceiling backstop even when no
        explicit wall-clock is set, so a wedged run still ships a report; disabled
        only by ``OPENRESEARCH_WATCHDOG_HARD_CEILING_S=0``.
  C-B2  ``gpu_cell_runner.run_matrix`` honours an ``overall_timeout_s`` budget — it
        stops launching cells past the deadline (status ``timeout``) and clamps each
        in-flight cell's timeout to the time remaining, so the matrix can't run for
        hours.
  KILL  a SIGTERM finalizer ships a partial report on a graceful kill (the case the
        operator hit), reusing the shared ``_hard_stop_with_report`` path.
"""
from __future__ import annotations

import signal
import threading
import time
from unittest.mock import MagicMock

import pytest

from backend.agents.rlm import run
import backend.agents.rlm.gpu_cell_runner as gcr


# ---------------------------------------------------------------------------
# C-B1 — _watchdog_hard_ceiling_s + _arm_watchdog
# ---------------------------------------------------------------------------

class TestWatchdogHardCeiling:
    def test_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("OPENRESEARCH_WATCHDOG_HARD_CEILING_S", raising=False)
        assert run._watchdog_hard_ceiling_s() == run._WATCHDOG_HARD_CEILING_DEFAULT_S

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("OPENRESEARCH_WATCHDOG_HARD_CEILING_S", "3600")
        assert run._watchdog_hard_ceiling_s() == 3600.0

    def test_zero_disables(self, monkeypatch):
        monkeypatch.setenv("OPENRESEARCH_WATCHDOG_HARD_CEILING_S", "0")
        assert run._watchdog_hard_ceiling_s() == 0.0

    def test_empty_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("OPENRESEARCH_WATCHDOG_HARD_CEILING_S", "")
        assert run._watchdog_hard_ceiling_s() == run._WATCHDOG_HARD_CEILING_DEFAULT_S

    def test_malformed_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("OPENRESEARCH_WATCHDOG_HARD_CEILING_S", "not-a-number")
        assert run._watchdog_hard_ceiling_s() == run._WATCHDOG_HARD_CEILING_DEFAULT_S


class TestArmWatchdog:
    def test_armed_even_without_wall_clock(self, tmp_path, monkeypatch):
        """The bug: a None deadline used to return None (no backstop)."""
        monkeypatch.delenv("OPENRESEARCH_WATCHDOG_HARD_CEILING_S", raising=False)
        t = run._arm_watchdog(
            None, project_dir=tmp_path, emit=MagicMock(), iteration_count=lambda: 0
        )
        try:
            # Sleep-robust watchdog (ported 2026-06-09): a polling handle, not
            # a threading.Timer (Timer waits on a monotonic clock that pauses
            # during macOS sleep). The handle keeps the .interval contract.
            assert t is not None
            assert t.interval == pytest.approx(
                run._WATCHDOG_HARD_CEILING_DEFAULT_S + run._WATCHDOG_GRACE_S
            )
        finally:
            if t is not None:
                t.cancel()

    def test_disabled_when_ceiling_zero(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENRESEARCH_WATCHDOG_HARD_CEILING_S", "0")
        t = run._arm_watchdog(
            None, project_dir=tmp_path, emit=MagicMock(), iteration_count=lambda: 0
        )
        assert t is None  # operator opted fully out

    def test_explicit_deadline_takes_precedence(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENRESEARCH_WATCHDOG_HARD_CEILING_S", "99999")
        t = run._arm_watchdog(
            123.0, project_dir=tmp_path, emit=MagicMock(), iteration_count=lambda: 0
        )
        try:
            assert t.interval == pytest.approx(123.0 + run._WATCHDOG_GRACE_S)
        finally:
            if t is not None:
                t.cancel()


# ---------------------------------------------------------------------------
# KILL — _hard_stop_with_report + SIGTERM finalizer
# ---------------------------------------------------------------------------

class _ExitCalled(Exception):
    def __init__(self, code: int) -> None:
        self.code = code


class TestHardStopWithReport:
    def test_writes_report_and_exits(self, tmp_path, monkeypatch):
        def _raise_exit(code: int) -> None:
            raise _ExitCalled(code)

        monkeypatch.setattr(run.os, "_exit", _raise_exit)
        emit = MagicMock()
        with pytest.raises(_ExitCalled) as ei:
            run._hard_stop_with_report(
                project_dir=tmp_path,
                emit=emit,
                done=3,
                summary="partial summary",
                status_error="terminated",
                exit_code=143,
            )
        assert ei.value.code == 143
        # The whole point: a report is ALWAYS left behind.
        assert (tmp_path / "final_report.json").exists()
        emit.assert_called_once()


class TestSigtermFinalizer:
    def test_installed_on_main_thread_and_restorable(self, tmp_path):
        prev = signal.getsignal(signal.SIGTERM)
        try:
            ret = run._install_sigterm_finalizer(
                project_dir=tmp_path, emit=MagicMock(), iteration_count=lambda: 0
            )
            current = signal.getsignal(signal.SIGTERM)
            assert callable(current)
            assert current is not prev  # our finalizer is now installed
            # the install returns the prior handler so the caller can restore it
            assert ret == prev
        finally:
            signal.signal(signal.SIGTERM, prev)

    def test_noop_off_main_thread(self, tmp_path):
        result: dict[str, object] = {}

        def worker() -> None:
            result["ret"] = run._install_sigterm_finalizer(
                project_dir=tmp_path, emit=MagicMock(), iteration_count=lambda: 0
            )

        t = threading.Thread(target=worker)
        t.start()
        t.join()
        assert result["ret"] is None  # signals can't be set off the main thread


# ---------------------------------------------------------------------------
# C-B2 — run_matrix overall_timeout_s
# ---------------------------------------------------------------------------

class TestRunMatrixOverallTimeout:
    def test_overall_timeout_bounds_the_matrix(self, tmp_path, monkeypatch):
        captured_timeouts: list[float | None] = []

        def fake_subprocess(*, cell, cell_script, gpu_id, output_dir,
                            batch_scale, grad_checkpoint, timeout_s, log_path):
            # Model a cell that always consumes its (clamped) budget then is killed.
            captured_timeouts.append(timeout_s)
            time.sleep(min(timeout_s if timeout_s else 0.4, 0.4))
            return -9, "killed: TIMEOUT after budget"

        monkeypatch.setattr(gcr, "_run_cell_subprocess", fake_subprocess)

        cells = [{"id": f"c{i}"} for i in range(4)]
        t0 = time.monotonic()
        res = gcr.run_matrix(
            cells, "train_cell.py", output_root=str(tmp_path), gpus=["0"],
            per_cell_timeout_s=100.0, overall_timeout_s=0.3,
        )
        wall = time.monotonic() - t0

        # Every cell still has a result entry (never raises, never drops a cell).
        assert set(res.keys()) == {"c0", "c1", "c2", "c3"}
        statuses = [res[f"c{i}"]["status"] for i in range(4)]
        # The matrix did NOT run 4 * 100s; it returned within a small multiple of
        # the overall budget.
        assert wall < 5.0
        # At least one cell was skipped pre-launch (fewer launches than cells) OR
        # killed at the deadline — recorded as "timeout".
        assert len(captured_timeouts) < 4
        assert statuses.count("timeout") >= 1
        # Launched cells had their timeout clamped below the per-cell ceiling.
        assert all(t is None or t <= 100.0 for t in captured_timeouts)
        assert any(t is not None and t < 100.0 for t in captured_timeouts)

    def test_no_overall_timeout_is_unbounded(self, tmp_path, monkeypatch):
        """overall_timeout_s=None must preserve the prior per-cell-only behaviour."""
        seen: list[float | None] = []

        def fake_subprocess(*, timeout_s, **kw):
            seen.append(timeout_s)
            return 0, "ok"

        monkeypatch.setattr(gcr, "_run_cell_subprocess", fake_subprocess)
        monkeypatch.setattr(gcr, "_load_metrics", lambda d: {"ok": True})
        cells = [{"id": "c0"}, {"id": "c1"}]
        res = gcr.run_matrix(
            cells, "train_cell.py", output_root=str(tmp_path), gpus=["0", "1"],
            per_cell_timeout_s=42.0, overall_timeout_s=None,
        )
        assert all(res[c]["status"] == "ok" for c in ("c0", "c1"))
        # With no overall budget, the per-cell timeout passes through unchanged.
        assert seen == [42.0, 42.0]
