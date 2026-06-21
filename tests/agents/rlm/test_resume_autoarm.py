"""Tests for _maybe_auto_arm_cell_resume (6d-T1 spot-restart cell-resume auto-arm)."""

import os

import pytest

from backend.agents.rlm.run import _maybe_auto_arm_cell_resume


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Ensure OPENRESEARCH_RESUME_CELLS is unset before each test."""
    monkeypatch.delenv("OPENRESEARCH_RESUME_CELLS", raising=False)


def test_fresh_empty_dir_returns_false(tmp_path):
    """Fresh empty project dir — no prior attempt, must not arm."""
    result = _maybe_auto_arm_cell_resume(tmp_path)
    assert result is False
    assert os.environ.get("OPENRESEARCH_RESUME_CELLS") is None


def test_rlm_state_no_final_report_arms(tmp_path):
    """Prior rlm_state/ exists but no final_report.json → incomplete run → arm."""
    (tmp_path / "rlm_state").mkdir()
    result = _maybe_auto_arm_cell_resume(tmp_path)
    assert result is True
    assert os.environ.get("OPENRESEARCH_RESUME_CELLS") == "1"


def test_experiment_runs_jsonl_no_final_report_arms(tmp_path):
    """Prior experiment_runs.jsonl exists but no final_report.json → incomplete run → arm."""
    (tmp_path / "experiment_runs.jsonl").write_text("")
    result = _maybe_auto_arm_cell_resume(tmp_path)
    assert result is True
    assert os.environ.get("OPENRESEARCH_RESUME_CELLS") == "1"


def test_rlm_state_with_final_report_does_not_arm(tmp_path):
    """Run already finished (final_report.json present) — must not arm."""
    (tmp_path / "rlm_state").mkdir()
    (tmp_path / "final_report.json").write_text("{}")
    result = _maybe_auto_arm_cell_resume(tmp_path)
    assert result is False
    assert os.environ.get("OPENRESEARCH_RESUME_CELLS") is None


def test_explicit_zero_wins_over_prior_state(tmp_path, monkeypatch):
    """Explicit OPENRESEARCH_RESUME_CELLS=0 must be respected (operator override)."""
    monkeypatch.setenv("OPENRESEARCH_RESUME_CELLS", "0")
    (tmp_path / "rlm_state").mkdir()
    result = _maybe_auto_arm_cell_resume(tmp_path)
    assert result is False
    assert os.environ.get("OPENRESEARCH_RESUME_CELLS") == "0"


def test_explicit_one_already_set_returns_false(tmp_path, monkeypatch):
    """OPENRESEARCH_RESUME_CELLS=1 already set — returns False (didn't arm this call)."""
    monkeypatch.setenv("OPENRESEARCH_RESUME_CELLS", "1")
    (tmp_path / "rlm_state").mkdir()
    result = _maybe_auto_arm_cell_resume(tmp_path)
    assert result is False
    assert os.environ.get("OPENRESEARCH_RESUME_CELLS") == "1"
