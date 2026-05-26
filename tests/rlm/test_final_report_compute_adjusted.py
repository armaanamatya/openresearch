"""β3 Task 4: final_report.json carries compute_adjusted_score + compute_scope.

Tests:
- write_final_report_rlm serializes rubric with compute_adjusted_score on max mode
- write_final_report_rlm serializes rubric with compute_adjusted_score on clipped run
- Backward-compat: old rubric dicts without compute_adjusted_score still round-trip
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.agents.rlm.report import RLMFinalReport, write_final_report_rlm
from backend.agents.schemas import ComputeScope


def _make_report(rubric_payload: dict, verdict: str = "completed") -> RLMFinalReport:
    """Construct a minimal RLMFinalReport with the given rubric dict."""
    return RLMFinalReport(
        paper={"id": "test_paper", "title": "Test"},
        verdict=verdict,
        reproduction_summary="test summary",
        baseline_metrics={"acc": 0.88},
        rubric=rubric_payload,
    )


def test_final_report_emits_compute_adjusted_on_max_mode(tmp_path: Path):
    """Max mode: compute_adjusted_score equals overall_score (no clipping)."""
    rubric_payload = {
        "overall_score": 0.7,
        "compute_adjusted_score": 0.7,
        "compute_scope": None,
        "meets_target": False,
        "target_score": 0.6,
        "areas": [{"area": "x", "score": 0.7, "compute_adjusted_score": 0.7, "weight": 1.0}],
    }
    report = _make_report(rubric_payload, verdict="partial")
    write_final_report_rlm(report, tmp_path)

    saved = json.loads((tmp_path / "final_report.json").read_text())
    assert saved["rubric"]["compute_adjusted_score"] == pytest.approx(0.7)
    assert saved["rubric"]["compute_scope"] is None
    assert saved["rubric"]["overall_score"] == pytest.approx(0.7)


def test_final_report_emits_compute_adjusted_on_clipped_run(tmp_path: Path):
    """Clipped run: overall_score (raw) and compute_adjusted_score differ."""
    scope = ComputeScope(
        is_clipped=True, paper_epochs=45, actual_epochs=5,
        rationale="efficient", metric_floors=[]
    )
    rubric_payload = {
        "overall_score": 0.4,
        "compute_adjusted_score": 0.85,
        "compute_scope": scope.model_dump(),
        "meets_target": False,
        "target_score": 0.6,
        "areas": [{"area": "x", "score": 0.4, "compute_adjusted_score": 0.85, "weight": 1.0}],
    }
    report = _make_report(rubric_payload, verdict="partial")
    write_final_report_rlm(report, tmp_path)

    saved = json.loads((tmp_path / "final_report.json").read_text())
    assert saved["rubric"]["overall_score"] == pytest.approx(0.4)
    assert saved["rubric"]["compute_adjusted_score"] == pytest.approx(0.85)
    assert saved["rubric"]["compute_scope"]["is_clipped"] is True


def test_final_report_backward_compat_no_compute_adjusted(tmp_path: Path):
    """Old rubric dict without compute_adjusted_score round-trips fine."""
    rubric_payload = {
        "overall_score": 0.6,
        "meets_target": True,
        "target_score": 0.6,
        "areas": [],
    }
    report = _make_report(rubric_payload, verdict="reproduced")
    write_final_report_rlm(report, tmp_path)

    saved = json.loads((tmp_path / "final_report.json").read_text())
    # The rubric dict may or may not have compute_adjusted_score depending on
    # whether it was populated during the run. The key requirement: the file
    # is valid JSON and overall_score is preserved.
    assert saved["rubric"]["overall_score"] == pytest.approx(0.6)
    # No KeyError on missing field
    _ = saved["rubric"].get("compute_adjusted_score")
