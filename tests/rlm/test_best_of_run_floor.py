"""F-11: build_final_report must apply the best-of-run floor BEFORE reconciling the
verdict, so a late regression / degraded self-report can't cap the verdict below the
score the run actually achieved (the no-amend path that leaf_scorer.amend_final_report
does not cover).
"""
from __future__ import annotations

import json

from backend.agents.rlm.report import (
    _apply_best_of_run_floor,
    _reconcile_verdict_against_evidence,
)


def _write_events(project_dir, scores):
    lines = [json.dumps({"type": "rubric_score", "payload": {"overall_score": s}}) for s in scores]
    (project_dir / "dashboard_events.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_floor_bumps_to_best_recorded(tmp_path):
    _write_events(tmp_path, [0.18, 0.55, 0.40])
    out = _apply_best_of_run_floor({"overall_score": 0.0}, tmp_path)
    assert out["overall_score"] == 0.55
    assert out["best_of_run"] is True


def test_floor_noop_when_current_is_higher(tmp_path):
    _write_events(tmp_path, [0.30])
    out = _apply_best_of_run_floor({"overall_score": 0.62}, tmp_path)
    assert out["overall_score"] == 0.62
    assert "best_of_run" not in out


def test_floor_noop_when_no_events(tmp_path):
    out = _apply_best_of_run_floor({"overall_score": 0.2}, tmp_path)
    assert out == {"overall_score": 0.2}


def test_floor_handles_none_current(tmp_path):
    _write_events(tmp_path, [0.33])
    out = _apply_best_of_run_floor({"overall_score": None}, tmp_path)
    assert out["overall_score"] == 0.33
    assert out["best_of_run"] is True


def test_f11_floored_score_keeps_reproduced_verdict(tmp_path):
    # Self-reported 0.45 (<0.5) WOULD downgrade reproduced->partial; the run recorded
    # a best of 0.55 mid-run. Flooring BEFORE the reconcile keeps the verdict.
    _write_events(tmp_path, [0.45, 0.55])
    floored = _apply_best_of_run_floor({"overall_score": 0.45}, tmp_path)
    assert floored["overall_score"] == 0.55
    verdict, reason = _reconcile_verdict_against_evidence(
        "reproduced",
        baseline_metrics={"accuracy": 0.55},
        rubric=floored,
        primitive_trace={"by_primitive": {"run_experiment": 1}},
    )
    assert verdict == "reproduced"
    assert reason is None


def test_f11_unfloored_low_score_downgrades():
    # Control (pre-F-11 behavior): against the raw self-reported 0.45 the reconcile
    # downgrades to partial — proving flooring BEFORE the reconcile is what matters.
    verdict, reason = _reconcile_verdict_against_evidence(
        "reproduced",
        baseline_metrics={"accuracy": 0.55},
        rubric={"overall_score": 0.45},
        primitive_trace={"by_primitive": {"run_experiment": 1}},
    )
    assert verdict == "partial"
    assert reason is not None
