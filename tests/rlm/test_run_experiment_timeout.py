"""Pin the run_experiment timeout default + REPROLAB_RUN_EXPERIMENT_TIMEOUT_S
env-var override (2026-05-23 evening fix).

B2 of the paper sweep (prj_77b7294aed1bf872) sat for the full 2-hour
default cap because the model upgraded its baseline to a real VLM training
that was CPU-infeasible. We reduced the default to 1800 s (30 min) and
made it env-var tunable so users with genuinely long experiments can
extend without re-deploying.

This is a smoke-shape test — it patches the dependencies and reads the
computed timeout value passed to `.result(timeout=...)` without actually
launching docker. The full run_experiment is exercised in the stub
primitives tests + the E2E paper-sweep runs.
"""

from __future__ import annotations

import os
from unittest.mock import patch, MagicMock

import pytest


def test_default_aggregate_timeout_is_None_no_cap(monkeypatch, tmp_path):
    """PR-μ Solution B: mode-scaled default cap replaces the old "no cap" policy.
    Without env var or run-budget, resolve_experiment_timeout_s returns the
    mode-specific default (7200 for efficient/unknown, 21600 for max)."""
    # Ensure env var is unset for this test
    monkeypatch.delenv("REPROLAB_RUN_EXPERIMENT_TIMEOUT_S", raising=False)

    # Capture the timeout value passed to .result(...) by patching the
    # ThreadPoolExecutor's submitted future. We don't want to actually run
    # docker — just measure the cap selection logic.
    from backend.agents.rlm import primitives

    captured: dict = {}
    real_executor = primitives.concurrent.futures.ThreadPoolExecutor

    class _CapturingFuture:
        def result(self, timeout=None):
            captured["timeout"] = timeout
            # Return a fail-soft dict so run_experiment returns cleanly
            return {"success": False, "metrics": {}, "error": "test stub"}

    class _CapturingExecutor:
        def __init__(self, *a, **kw): pass
        def submit(self, *a, **kw): return _CapturingFuture()
        def shutdown(self, *a, **kw): pass

    # Minimal RunContext with the fields run_experiment reads
    ctx = MagicMock()
    ctx.project_id = "prj_test_timeout"
    ctx.project_dir = tmp_path
    ctx.runs_root = tmp_path
    ctx.sandbox_mode = "docker"
    ctx.run_budget = None
    # _timeout_for clamps against run_budget.max_wall_clock_seconds; with None
    # budget it returns the cap value as-is.
    ctx.deadline_monotonic = None
    # remaining_s() returns None when there is no run-budget deadline — so
    # _timeout_for returns the cap as-is. (MagicMock would return a MagicMock,
    # breaking min(cap_s, remaining) with a TypeError.)
    ctx.remaining_s = MagicMock(return_value=None)

    # Set up minimal commands.json so run_experiment doesn't early-exit
    code_dir = tmp_path / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    (code_dir / "commands.json").write_text('["echo hi"]')

    # No Dockerfile → skips the rebuild branch
    with patch.object(primitives.concurrent.futures, "ThreadPoolExecutor", _CapturingExecutor):
        primitives.run_experiment(str(code_dir), env_id="stub:latest", ctx=ctx)

    assert "timeout" in captured, "timeout was never read off the future"
    # PR-μ Solution B: default is now 7200 (efficient mode) when neither env var
    # nor execution_mode is set. The old "no cap" (None) was replaced.
    assert captured["timeout"] == 7200, (
        f"expected default mode cap of 7200; got {captured['timeout']!r}"
    )


def test_env_var_override_takes_effect(monkeypatch, tmp_path):
    """REPROLAB_RUN_EXPERIMENT_TIMEOUT_S=600 → cap becomes 600 s."""
    monkeypatch.setenv("REPROLAB_RUN_EXPERIMENT_TIMEOUT_S", "600")

    from backend.agents.rlm import primitives

    captured: dict = {}

    class _CapturingFuture:
        def result(self, timeout=None):
            captured["timeout"] = timeout
            return {"success": False, "metrics": {}, "error": "test stub"}

    class _CapturingExecutor:
        def __init__(self, *a, **kw): pass
        def submit(self, *a, **kw): return _CapturingFuture()
        def shutdown(self, *a, **kw): pass

    ctx = MagicMock()
    ctx.project_id = "prj_test_env_override"
    ctx.project_dir = tmp_path
    ctx.runs_root = tmp_path
    ctx.sandbox_mode = "docker"
    ctx.run_budget = None
    ctx.deadline_monotonic = None
    # remaining_s() returns None when there is no run-budget deadline — so
    # _timeout_for returns the cap as-is. (MagicMock would return a MagicMock,
    # breaking min(cap_s, remaining) with a TypeError.)
    ctx.remaining_s = MagicMock(return_value=None)

    code_dir = tmp_path / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    (code_dir / "commands.json").write_text('["echo hi"]')

    with patch.object(primitives.concurrent.futures, "ThreadPoolExecutor", _CapturingExecutor):
        primitives.run_experiment(str(code_dir), env_id="stub:latest", ctx=ctx)

    assert captured.get("timeout") == 600.0, (
        f"env var REPROLAB_RUN_EXPERIMENT_TIMEOUT_S=600 was ignored; "
        f"got {captured.get('timeout')!r}."
    )


def test_invalid_env_var_falls_back_to_default(monkeypatch, tmp_path):
    """REPROLAB_RUN_EXPERIMENT_TIMEOUT_S=garbage → falls back silently."""
    monkeypatch.setenv("REPROLAB_RUN_EXPERIMENT_TIMEOUT_S", "not-a-number")

    from backend.agents.rlm import primitives

    captured: dict = {}

    class _CapturingFuture:
        def result(self, timeout=None):
            captured["timeout"] = timeout
            return {"success": False, "metrics": {}, "error": "test stub"}

    class _CapturingExecutor:
        def __init__(self, *a, **kw): pass
        def submit(self, *a, **kw): return _CapturingFuture()
        def shutdown(self, *a, **kw): pass

    ctx = MagicMock()
    ctx.project_id = "prj_test_bad_env"
    ctx.project_dir = tmp_path
    ctx.runs_root = tmp_path
    ctx.sandbox_mode = "docker"
    ctx.run_budget = None
    ctx.deadline_monotonic = None
    # remaining_s() returns None when there is no run-budget deadline — so
    # _timeout_for returns the cap as-is. (MagicMock would return a MagicMock,
    # breaking min(cap_s, remaining) with a TypeError.)
    ctx.remaining_s = MagicMock(return_value=None)

    code_dir = tmp_path / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    (code_dir / "commands.json").write_text('["echo hi"]')

    with patch.object(primitives.concurrent.futures, "ThreadPoolExecutor", _CapturingExecutor):
        primitives.run_experiment(str(code_dir), env_id="stub:latest", ctx=ctx)

    # Invalid env var → falls back to mode default (7200 for efficient/unknown,
    # per PR-μ Solution B which replaced the old "no cap" policy).
    assert captured.get("timeout") == 7200
