"""Tests for the .preserved GC-protection marker on completed runs."""

from __future__ import annotations

import json
from pathlib import Path

from backend.agents.rlm.report import RLMFinalReport, write_final_report_rlm


def _base_report(verdict: str = "partial", with_rubric: bool = True) -> RLMFinalReport:
    rubric: dict = {}
    if with_rubric:
        rubric = {"overall_score": 0.42, "meets_target": False}
    return RLMFinalReport(
        verdict=verdict,
        rubric=rubric,
        baseline_metrics={"test_err_pct": 1.2},
        reproduction_summary="ok",
        iterations=3,
        paper={"id": "1207.0580", "title": "Dropout"},
    )


def _seed_evidence(project_dir: Path) -> None:
    """Evidence gate (ported 2026-06-09): a partial/reproduced verdict needs a
    real success+metrics row in experiment_runs.jsonl or write_final_report_rlm
    downgrades it to failed (and then writes no .preserved marker)."""
    (project_dir / "experiment_runs.jsonl").write_text(
        json.dumps({"success": True, "metrics": {"test_err_pct": 1.2}}) + "\n",
        encoding="utf-8",
    )


def test_preserved_marker_written_on_partial(tmp_path: Path) -> None:
    _seed_evidence(tmp_path)
    write_final_report_rlm(_base_report(verdict="partial"), tmp_path)
    marker = tmp_path / ".preserved"
    assert marker.exists()
    payload = json.loads(marker.read_text())
    assert payload["verdict"] == "partial"
    assert payload["rubric_overall_score"] == 0.42
    assert payload["paper_id"] == "1207.0580"
    assert payload["paper_title"] == "Dropout"
    assert payload["iterations"] == 3
    assert payload["schema_version"] == 1
    assert "preserved_at_utc" in payload


def test_preserved_marker_written_on_reproduced(tmp_path: Path) -> None:
    _seed_evidence(tmp_path)
    write_final_report_rlm(_base_report(verdict="reproduced"), tmp_path)
    assert (tmp_path / ".preserved").exists()


def test_preserved_marker_not_written_on_failed(tmp_path: Path) -> None:
    write_final_report_rlm(_base_report(verdict="failed", with_rubric=False), tmp_path)
    assert not (tmp_path / ".preserved").exists()
    # The canonical report should still exist — a failed run is on-disk
    # data; only the GC-protection marker is skipped.
    assert (tmp_path / "final_report.json").exists()


def test_preserved_marker_handles_missing_rubric(tmp_path: Path) -> None:
    _seed_evidence(tmp_path)
    write_final_report_rlm(_base_report(verdict="partial", with_rubric=False), tmp_path)
    marker = tmp_path / ".preserved"
    payload = json.loads(marker.read_text())
    assert payload["rubric_overall_score"] is None


def test_preserved_marker_is_atomic(tmp_path: Path) -> None:
    # _atomic_write uses a sibling .tmp file then os.replace. Make sure the
    # marker doesn't leave a half-written .preserved.tmp behind on success.
    _seed_evidence(tmp_path)
    write_final_report_rlm(_base_report(verdict="partial"), tmp_path)
    assert (tmp_path / ".preserved").exists()
    assert not (tmp_path / ".preserved.tmp").exists()
