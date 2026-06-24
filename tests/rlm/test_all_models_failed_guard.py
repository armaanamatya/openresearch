"""Tests for the all-models-failed postflight guard (Workstream C Fix 1).

The monolithic ``run_experiment`` path sets ``success = all(r.succeeded …)`` —
subprocess exit only. An experiment where EVERY model errored at load (e.g.
``per_model.qwen3 = {status:"failed", accuracy:0.0}``) still exits 0 and reports
``success=true`` — a latent fake-green. The existing completeness guard only
fires when per_model entries are EMPTY placeholders; an error-bearing entry with
a ``0.0`` numeric passes ``_per_model_has_measured_value``. The degenerate-training
guard only judges ``_OK_STATUSES`` models, so it skips a ``status:"failed"`` model.

This guard closes the narrow gap: per_model is non-empty but NO entry has an ok
status, yet success=true → repairable failure ``all_models_failed``.

DEFAULT-OFF behind ``OPENRESEARCH_PER_MODEL_STATUS_GATE`` (1/true/yes = ON); unset
is byte-for-byte today.
"""

from backend.agents.rlm.primitives import (
    _OK_STATUSES,
    _all_models_failed_violation,
)

_FLAG = "OPENRESEARCH_PER_MODEL_STATUS_GATE"


def _all_failed_result() -> dict:
    return {
        "metrics": {
            "per_model": {
                "qwen3": {"status": "failed", "error": "ValueError", "accuracy": 0.0},
                "qwen2": {"status": "error", "accuracy": 0.0},
            }
        }
    }


def test_all_failed_with_flag_on_fires(monkeypatch):
    monkeypatch.setenv(_FLAG, "1")
    out = _all_models_failed_violation(_all_failed_result())
    assert out is not None
    cls, msg = out
    assert cls == "all_models_failed"
    # message names the offending models
    assert "qwen3" in msg
    assert "qwen2" in msg


def test_flag_off_unset_returns_none_byte_for_byte(monkeypatch):
    """CRITICAL regression: unset flag → None → all-failed row still success=true today."""
    monkeypatch.delenv(_FLAG, raising=False)
    assert _all_models_failed_violation(_all_failed_result()) is None


def test_flag_explicit_off_returns_none(monkeypatch):
    monkeypatch.setenv(_FLAG, "0")
    assert _all_models_failed_violation(_all_failed_result()) is None


def test_one_ok_status_with_flag_on_returns_none(monkeypatch):
    """≥1 ok status → no false positive."""
    monkeypatch.setenv(_FLAG, "1")
    ok_status = sorted(_OK_STATUSES)[0]
    result = {
        "metrics": {
            "per_model": {
                "qwen3": {"status": "failed", "accuracy": 0.0},
                "qwen2": {"status": ok_status, "accuracy": 0.71},
            }
        }
    }
    assert _all_models_failed_violation(result) is None


def test_empty_per_model_with_flag_on_returns_none(monkeypatch):
    """Empty per_model is the completeness guard's job, not this one."""
    monkeypatch.setenv(_FLAG, "1")
    assert _all_models_failed_violation({"metrics": {"per_model": {}}}) is None


def test_no_per_model_key_with_flag_on_returns_none(monkeypatch):
    monkeypatch.setenv(_FLAG, "1")
    assert _all_models_failed_violation({"metrics": {"status": "done"}}) is None


def test_per_model_not_a_dict_with_flag_on_returns_none(monkeypatch):
    monkeypatch.setenv(_FLAG, "1")
    assert _all_models_failed_violation({"metrics": {"per_model": ["qwen3"]}}) is None


def test_no_metrics_with_flag_on_returns_none(monkeypatch):
    monkeypatch.setenv(_FLAG, "1")
    assert _all_models_failed_violation({}) is None
