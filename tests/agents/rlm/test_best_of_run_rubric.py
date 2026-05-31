"""Best-of-run rubric floor (2026-05-30).

build_final_report surfaces the BEST rubric score the run recorded (from
dashboard_events.jsonl) so the RLM loop over-iterating into a degraded final
state can't bury a real result — the SDAR run self-reported 0.0 despite scoring
~0.18 mid-run. Reading from disk makes it salvage-capable. Deterministic.
"""
from __future__ import annotations

import json

from backend.agents.rlm.report import _best_recorded_rubric_score


def _write(tmp_path, lines):
    (tmp_path / "dashboard_events.jsonl").write_text("\n".join(lines), encoding="utf-8")
    return tmp_path


def test_picks_max_not_last(tmp_path):
    _write(tmp_path, [
        json.dumps({"event_type": "rubric_score", "payload": {"overall_score": 0.1737}}),
        json.dumps({"event_type": "primitive_call", "primitive": "run_experiment"}),
        json.dumps({"event_type": "rubric_score", "payload": {"overall_score": 0.1797}}),
        json.dumps({"event_type": "rubric_score", "payload": {"overall_score": 0.05}}),  # degraded last
    ])
    assert _best_recorded_rubric_score(tmp_path) == 0.1797  # the best, not the last


def test_none_when_absent_or_empty(tmp_path):
    assert _best_recorded_rubric_score(tmp_path) is None  # no file
    _write(tmp_path, [])
    assert _best_recorded_rubric_score(tmp_path) is None


def test_ignores_non_rubric_and_malformed_lines(tmp_path):
    _write(tmp_path, [
        json.dumps({"event_type": "primitive_call"}),
        "not json at all",
        json.dumps({"event_type": "rubric_score", "payload": {"overall_score": 0.3}}),
        json.dumps({"event_type": "rubric_score", "payload": {"score": 0.42}}),  # alt key
    ])
    assert _best_recorded_rubric_score(tmp_path) == 0.42


def test_handles_null_and_non_numeric_scores(tmp_path):
    _write(tmp_path, [
        json.dumps({"event_type": "rubric_score", "payload": {"overall_score": None}}),
        json.dumps({"event_type": "rubric_score", "payload": {"overall_score": "n/a"}}),
        json.dumps({"event_type": "rubric_score", "payload": {"overall_score": 0.2}}),
    ])
    assert _best_recorded_rubric_score(tmp_path) == 0.2
