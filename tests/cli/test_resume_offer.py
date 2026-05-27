"""Tests for PR-π Module D — CLI orphan sweep + resume offer.

Tests the _offer_resume helper from backend.cli:
  - Detects an interrupted prior run and prompts on TTY.
  - Returns False when no prior run exists.
  - Returns False when stdin is not a TTY.
"""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from backend.cli import _offer_resume, _count_iterations, _read_last_rubric


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_status(run_dir: Path, payload: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "demo_status.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_iterations(run_dir: Path, count: int) -> None:
    state_dir = run_dir / "rlm_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    lines = "\n".join(json.dumps({"i": i}) for i in range(count)) + "\n"
    (state_dir / "iterations.jsonl").write_text(lines, encoding="utf-8")


# ---------------------------------------------------------------------------
# _offer_resume
# ---------------------------------------------------------------------------

def test_resume_offer_on_interrupted_prior_run(tmp_path: Path) -> None:
    """When status=interrupted, isatty()=True, user answers 'y' → returns True."""
    run_dir = tmp_path / "prj_X"
    _write_status(run_dir, {
        "projectId": "prj_X",
        "status": "interrupted",
        "runMode": "rlm",
    })
    _write_iterations(run_dir, 2)

    with (
        patch("sys.stdin") as mock_stdin,
        patch("builtins.input", return_value="y"),
    ):
        mock_stdin.isatty.return_value = True
        result = _offer_resume(run_dir)

    assert result is True


def test_no_resume_offer_when_no_prior_run(tmp_path: Path) -> None:
    """Empty project_dir (no demo_status.json) → returns False immediately."""
    run_dir = tmp_path / "prj_nonexistent"
    # Do NOT create run_dir at all.

    result = _offer_resume(run_dir)

    assert result is False


def test_no_resume_offer_on_non_tty(tmp_path: Path) -> None:
    """When isatty()=False, no prompt is shown and function returns False."""
    run_dir = tmp_path / "prj_Y"
    _write_status(run_dir, {
        "projectId": "prj_Y",
        "status": "interrupted",
        "runMode": "rlm",
    })

    with (
        patch("sys.stdin") as mock_stdin,
        patch("builtins.input") as mock_input,
    ):
        mock_stdin.isatty.return_value = False
        result = _offer_resume(run_dir)

    assert result is False
    mock_input.assert_not_called()


def test_no_resume_offer_when_status_not_interrupted(tmp_path: Path) -> None:
    """When prior run status is 'failed' (not interrupted) → no offer."""
    run_dir = tmp_path / "prj_Z"
    _write_status(run_dir, {"projectId": "prj_Z", "status": "failed"})

    with patch("builtins.input") as mock_input:
        result = _offer_resume(run_dir)

    assert result is False
    mock_input.assert_not_called()


def test_resume_offer_declined_returns_false(tmp_path: Path) -> None:
    """User answers 'n' → function returns False."""
    run_dir = tmp_path / "prj_decline"
    _write_status(run_dir, {"projectId": "prj_decline", "status": "interrupted"})

    with (
        patch("sys.stdin") as mock_stdin,
        patch("builtins.input", return_value="n"),
    ):
        mock_stdin.isatty.return_value = True
        result = _offer_resume(run_dir)

    assert result is False


# ---------------------------------------------------------------------------
# _count_iterations
# ---------------------------------------------------------------------------

def test_count_iterations_empty(tmp_path: Path) -> None:
    run_dir = tmp_path / "prj_empty"
    run_dir.mkdir()
    assert _count_iterations(run_dir) == 0


def test_count_iterations_with_lines(tmp_path: Path) -> None:
    run_dir = tmp_path / "prj_iters"
    _write_iterations(run_dir, 3)
    assert _count_iterations(run_dir) == 3


# ---------------------------------------------------------------------------
# _read_last_rubric
# ---------------------------------------------------------------------------

def test_read_last_rubric_from_final_report(tmp_path: Path) -> None:
    run_dir = tmp_path / "prj_rubric"
    run_dir.mkdir()
    (run_dir / "final_report.json").write_text(
        json.dumps({"rubric": {"overall_score": 0.45}}), encoding="utf-8"
    )
    assert abs(_read_last_rubric(run_dir) - 0.45) < 1e-6


def test_read_last_rubric_no_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "prj_no_rubric"
    run_dir.mkdir()
    assert _read_last_rubric(run_dir) == 0.0
