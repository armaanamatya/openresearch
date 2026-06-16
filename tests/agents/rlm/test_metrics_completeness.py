"""Metrics-completeness guard (2026-05-30, rubric-scoring-fidelity spec).

A success=True run that wrote a placeholder / unpopulated metrics.json (a
non-terminal status, or per_model entries all empty) is flipped to a repairable
`incomplete_metrics` failure so the rubric never grades a half-finished
experiment ~0 on eval/result/execution. Deterministic; no sandbox/LLM.
"""
from __future__ import annotations

import pytest

from backend.agents.rlm.failure_classifier import FAILURE_CLASSES, classify_failure
from backend.agents.rlm.primitives import (
    _RUN_EXPERIMENT_REPAIRABLE_FAILURES,
    _metrics_completeness_violation,
    _per_model_has_measured_value,
)


def _v(metrics):
    return _metrics_completeness_violation({"success": True, "metrics": metrics})


def test_non_terminal_status_is_incomplete():
    out = _v({"status": "running", "per_model": {"m": {"accuracy": 0.5}}})
    assert out is not None and out[0] == "incomplete_metrics" and "non-terminal" in out[1]


@pytest.mark.parametrize("status", ["running", "in_progress", "in-progress", "pending", "started", "queued"])
def test_all_non_terminal_statuses(status):
    assert _v({"status": status, "per_model": {"m": {}}})[0] == "incomplete_metrics"


def test_empty_per_model_is_incomplete():
    out = _v({"status": "completed", "per_model": {"qwen3_1_7b": {}, "qwen2_5_3b": {}}})
    assert out is not None and out[0] == "incomplete_metrics" and "placeholder" in out[1]


def test_populated_per_model_is_ok():
    assert _v({"status": "completed", "per_model": {"m": {"final_accuracy": 0.42, "final_reward": 0.2}}}) is None


def test_terminal_status_top_level_metric_ok():
    assert _v({"status": "ok", "final_accuracy": 0.5}) is None  # single-model, real number


def test_genuinely_empty_metrics_deferred_elsewhere():
    assert _v({}) is None
    assert _metrics_completeness_violation({"success": True}) is None


def test_partial_one_model_measured_is_ok():
    # if ANY per-model entry carries a measured value, it's not a placeholder run
    assert _v({"status": "done", "per_model": {"good": {"reward": 0.3}, "bad": {}}}) is None


def test_toggle_off(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_METRICS_COMPLETENESS_CHECK", "0")
    assert _v({"status": "running", "per_model": {"m": {}}}) is None


def test_per_model_has_measured_value():
    assert _per_model_has_measured_value({}) is False
    assert _per_model_has_measured_value({"accuracy": 0.5}) is True
    assert _per_model_has_measured_value({"training_curves": {"reward": [0.1, 0.2]}}) is True
    assert _per_model_has_measured_value({"nested": {"deep": {"steps": 50}}}) is True
    assert _per_model_has_measured_value({"flag": True, "name": "x"}) is False  # bool/str aren't numbers


def test_incomplete_metrics_registered_and_repairable():
    assert "incomplete_metrics" in FAILURE_CLASSES
    assert "incomplete_metrics" in _RUN_EXPERIMENT_REPAIRABLE_FAILURES
    cls, suggestion = classify_failure({"failure_class": "incomplete_metrics"})
    assert cls == "incomplete_metrics" and suggestion
