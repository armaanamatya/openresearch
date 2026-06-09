"""Tests for PR-π Module D — CLI orphan sweep + interrupted-run detection.

Tests the _detect_interrupted_run helper from backend.cli (which replaced
_offer_resume: the old [Y/n] prompt promised an RLM resume that has no read
path — args.resume is consumed only by the rdr handler — so the CLI now
detects and reports instead of prompting; audit 2026-06-09):
  - Returns (iterations, last_rubric) for an interrupted prior run.
  - Returns None when no prior run exists / status is not interrupted.
  - Never prompts.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from backend.cli import _detect_interrupted_run, _count_iterations, _read_last_rubric


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
# _detect_interrupted_run
# ---------------------------------------------------------------------------

def test_detects_interrupted_prior_run(tmp_path: Path) -> None:
    """status=interrupted → (iteration count, last rubric) is returned."""
    run_dir = tmp_path / "prj_X"
    _write_status(run_dir, {
        "projectId": "prj_X",
        "status": "interrupted",
        "runMode": "rlm",
    })
    _write_iterations(run_dir, 2)

    result = _detect_interrupted_run(run_dir)

    assert result == (2, 0.0)


def test_none_when_no_prior_run(tmp_path: Path) -> None:
    """Empty project_dir (no demo_status.json) → None immediately."""
    run_dir = tmp_path / "prj_nonexistent"
    # Do NOT create run_dir at all.
    assert _detect_interrupted_run(run_dir) is None


def test_none_when_status_not_interrupted(tmp_path: Path) -> None:
    """Prior run status 'failed' (not interrupted) → None."""
    run_dir = tmp_path / "prj_Z"
    _write_status(run_dir, {"projectId": "prj_Z", "status": "failed"})
    assert _detect_interrupted_run(run_dir) is None


def test_none_on_corrupt_status_file(tmp_path: Path) -> None:
    run_dir = tmp_path / "prj_corrupt"
    run_dir.mkdir()
    (run_dir / "demo_status.json").write_text("{not json", encoding="utf-8")
    assert _detect_interrupted_run(run_dir) is None


def test_detection_never_prompts(tmp_path: Path) -> None:
    """Detection must be side-effect free — no input() even on interrupted."""
    run_dir = tmp_path / "prj_no_prompt"
    _write_status(run_dir, {"projectId": "prj_no_prompt", "status": "interrupted"})

    with patch("builtins.input") as mock_input:
        result = _detect_interrupted_run(run_dir)

    assert result == (0, 0.0)
    mock_input.assert_not_called()


def test_reports_last_rubric_from_final_report(tmp_path: Path) -> None:
    run_dir = tmp_path / "prj_with_rubric"
    _write_status(run_dir, {"projectId": "prj_with_rubric", "status": "interrupted"})
    (run_dir / "final_report.json").write_text(
        json.dumps({"rubric": {"overall_score": 0.45}}), encoding="utf-8"
    )

    result = _detect_interrupted_run(run_dir)

    assert result is not None
    assert abs(result[1] - 0.45) < 1e-6


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
