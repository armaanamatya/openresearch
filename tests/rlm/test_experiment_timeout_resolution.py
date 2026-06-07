"""Resolution order: env override > mode default > catch-all default, clamped
by ctx.remaining_s() only when finite."""
import math
import os
import pytest
from unittest.mock import MagicMock
from backend.agents.rlm.primitives import resolve_experiment_timeout_s, EXPERIMENT_TIMEOUT_BY_MODE

@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_RUN_EXPERIMENT_TIMEOUT_S", raising=False)

def _ctx(*, remaining_s=math.inf, execution_mode="efficient"):
    ctx = MagicMock()
    ctx.remaining_s.return_value = remaining_s
    ctx.execution_mode = execution_mode
    return ctx

def test_env_override_wins():
    os.environ["OPENRESEARCH_RUN_EXPERIMENT_TIMEOUT_S"] = "1234"
    try:
        assert resolve_experiment_timeout_s(_ctx(execution_mode="efficient")) == 1234
    finally:
        del os.environ["OPENRESEARCH_RUN_EXPERIMENT_TIMEOUT_S"]

def test_efficient_mode_default():
    assert resolve_experiment_timeout_s(_ctx(execution_mode="efficient")) == 7200

def test_max_mode_default():
    assert resolve_experiment_timeout_s(_ctx(execution_mode="max")) == 21600

def test_unknown_mode_falls_back():
    assert resolve_experiment_timeout_s(_ctx(execution_mode=None)) == 7200

def test_clamps_to_finite_remaining():
    """remaining_s=3600 with max mode (21600 default) should clamp to 3600."""
    result = resolve_experiment_timeout_s(_ctx(remaining_s=3600.0, execution_mode="max"))
    assert result == 3600

def test_does_not_clamp_to_infinite_remaining():
    """Infinite remaining_s means no --max-wall-clock; clamp must be a no-op."""
    assert resolve_experiment_timeout_s(_ctx(remaining_s=math.inf, execution_mode="max")) == 21600

def test_env_override_also_clamped_to_finite_remaining():
    os.environ["OPENRESEARCH_RUN_EXPERIMENT_TIMEOUT_S"] = "100000"
    try:
        result = resolve_experiment_timeout_s(_ctx(remaining_s=3600.0, execution_mode="efficient"))
        assert result == 3600
    finally:
        del os.environ["OPENRESEARCH_RUN_EXPERIMENT_TIMEOUT_S"]
