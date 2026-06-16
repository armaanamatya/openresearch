"""RunContext.execution_mode threading + resolver (C1, 2026-06-16).

Regression for the silently-dropped ``--execution-mode max``: RunContext now
carries ``execution_mode`` (threaded from ExecutionProfile.mode by run.py, or
autoloaded from REPROLAB_EXECUTION_MODE), and resolve_experiment_timeout_s
reads it FIRST. Previously the field did not exist, so the 6h cap applied only
when the env var happened to be exported.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.agents.rlm.context import RunContext
from backend.agents.rlm.primitives import (
    EXPERIMENT_TIMEOUT_BY_MODE,
    _DEFAULT_EXPERIMENT_TIMEOUT_S,
    resolve_experiment_timeout_s,
)


def _ctx(**overrides) -> RunContext:
    defaults = dict(
        project_id="p1",
        project_dir=Path("/tmp"),
        runs_root=Path("/tmp/runs"),
        dashboard=None,
        cost_ledger=None,
        llm_client=None,
        provider="anthropic",
        model="claude-sonnet-4-6",
    )
    defaults.update(overrides)
    return RunContext(**defaults)


@pytest.fixture(autouse=True)
def _clear_exec_env(monkeypatch):
    # Isolate from the ambient shell — these env vars otherwise leak into
    # __post_init__ autoload and the resolver fallback.
    monkeypatch.delenv("REPROLAB_EXECUTION_MODE", raising=False)
    monkeypatch.delenv("REPROLAB_RUN_EXPERIMENT_TIMEOUT_S", raising=False)


class TestExecutionModeContext:
    def test_default_is_none(self):
        assert _ctx().execution_mode is None

    def test_resolver_default_when_unset(self):
        # No ctx.execution_mode, no env → the resolver's default cap.
        assert resolve_experiment_timeout_s(_ctx()) == _DEFAULT_EXPERIMENT_TIMEOUT_S

    def test_threaded_max_applies_6h_cap(self):
        # The bug this fixes: max must resolve to the 6h cap, not the 2h default.
        ctx = _ctx(execution_mode="max")
        assert ctx.execution_mode == "max"
        assert resolve_experiment_timeout_s(ctx) == EXPERIMENT_TIMEOUT_BY_MODE["max"]
        assert EXPERIMENT_TIMEOUT_BY_MODE["max"] != _DEFAULT_EXPERIMENT_TIMEOUT_S

    def test_threaded_efficient(self):
        ctx = _ctx(execution_mode="efficient")
        assert resolve_experiment_timeout_s(ctx) == EXPERIMENT_TIMEOUT_BY_MODE["efficient"]

    def test_post_init_autoloads_from_env(self, monkeypatch):
        # Any construction site that didn't thread it still gets the env value.
        monkeypatch.setenv("REPROLAB_EXECUTION_MODE", "max")
        ctx = _ctx()  # caller did NOT pass execution_mode
        assert ctx.execution_mode == "max"
        assert resolve_experiment_timeout_s(ctx) == EXPERIMENT_TIMEOUT_BY_MODE["max"]

    def test_threaded_value_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("REPROLAB_EXECUTION_MODE", "efficient")
        ctx = _ctx(execution_mode="max")  # threaded value beats env autoload
        assert ctx.execution_mode == "max"
        assert resolve_experiment_timeout_s(ctx) == EXPERIMENT_TIMEOUT_BY_MODE["max"]
