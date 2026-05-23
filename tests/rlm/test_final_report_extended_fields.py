"""Tests for the extended fields on RLMFinalReport (spec §4.5):
models, mode, started_at, completed_at."""

from backend.agents.rlm.report import RLMFinalReport


def test_default_values_are_filled():
    report = RLMFinalReport()
    assert report.mode == "rlm"
    assert report.models == {
        "planner": None, "executor": None, "verifier": None, "grader": None,
    }
    assert report.started_at is None
    assert report.completed_at is None


def test_populated_values_round_trip():
    report = RLMFinalReport(
        mode="rlm",
        models={
            "planner": "gpt-5",
            "executor": "claude-sonnet-4-6",
            "verifier": None,
            "grader": None,
        },
        started_at="2026-05-23T04:10:09+00:00",
        completed_at="2026-05-23T04:12:33+00:00",
    )
    dumped = report.model_dump()
    assert dumped["mode"] == "rlm"
    assert dumped["models"]["planner"] == "gpt-5"
    assert dumped["models"]["executor"] == "claude-sonnet-4-6"
    assert dumped["started_at"] == "2026-05-23T04:10:09+00:00"
    assert dumped["completed_at"] == "2026-05-23T04:12:33+00:00"


def test_legacy_json_back_compat():
    """A pre-extension final_report.json missing the new keys must still parse."""
    legacy = {
        "paper": {},
        "verdict": "failed",
        "reproduction_summary": "x",
        "baseline_metrics": {},
        "paper_claims": {},
        "rubric": {"overall_score": 0.0, "meets_target": False, "areas": []},
        "improvements": [],
        "primitive_trace": {},
        "cost": {"llm_usd": 0.0, "primitives": 0.0},
        "iterations": 0,
    }
    report = RLMFinalReport.model_validate(legacy)
    assert report.mode == "rlm"
    assert report.models["planner"] is None
    assert report.started_at is None
