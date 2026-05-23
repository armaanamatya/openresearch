"""Tests for live_runs RLM bridging (handoff P1-I8 / T15).

finalize_benchmark was previously buried inside the embedded _python_script
string — it existed only as dynamically-evaluated code and could not be unit-
tested.  T15 extracts it as a module-level function and ensures it handles both
the RLM schema (baseline_metrics, rubric.overall_score, cost.llm_usd) and the
legacy SDK schema without breaking either path.
"""

from __future__ import annotations

import json


# ---------------------------------------------------------------------------
# RLM-schema tests (the load-bearing regression guard for P1-I8)
# ---------------------------------------------------------------------------


def test_finalize_benchmark_handles_rlm_schema(tmp_path):
    """Symptom: REST /runs/arxiv RLM runs return empty benchmark.

    finalize_benchmark read SDK-schema keys (metrics, paper_title) that don't
    exist on RLMFinalReport (baseline_metrics, paper.title) — so the bridged
    benchmark was {} for every RLM run (handoff P1-I8 / T15). Verify: an
    RLM-shaped final_report.json produces a populated benchmark.
    """
    from backend.services.events.live_runs import finalize_benchmark

    # An RLM-shaped final_report.json — mirror the actual RLMFinalReport schema.
    report = {
        "paper": {"id": "ftrl", "title": "FTRL"},
        "verdict": "partial",
        "reproduction_summary": "...",
        "baseline_metrics": {"mean_reward": 487.3},
        "rubric": {"overall_score": 0.45, "meets_target": False},
        "cost": {"llm_usd": 0.012, "primitives": 0.0},
    }
    (tmp_path / "final_report.json").write_text(json.dumps(report), encoding="utf-8")

    summary = finalize_benchmark(tmp_path)
    bench = summary["benchmark"]
    assert bench is not None
    assert bench["verdict"] == "partial"
    assert bench["rubric_score"] == 0.45
    assert bench["metrics"] == {"mean_reward": 487.3}
    assert bench["cost_usd"] == 0.012


def test_finalize_benchmark_rlm_baseline_metrics_key_alone(tmp_path):
    """baseline_metrics key alone (without verdict) triggers RLM path."""
    from backend.services.events.live_runs import finalize_benchmark

    report = {
        "baseline_metrics": {"accuracy": 0.87},
        "rubric": {"overall_score": 0.70},
        "cost": {"llm_usd": 0.005},
    }
    (tmp_path / "final_report.json").write_text(json.dumps(report), encoding="utf-8")

    summary = finalize_benchmark(tmp_path)
    bench = summary["benchmark"]
    assert bench is not None
    assert bench["metrics"] == {"accuracy": 0.87}
    assert bench["rubric_score"] == 0.70
    assert bench["cost_usd"] == 0.005
    # verdict absent → empty string
    assert bench["verdict"] == ""


def test_finalize_benchmark_rlm_missing_optional_fields(tmp_path):
    """RLM report with minimal keys must not raise — optional fields default safely."""
    from backend.services.events.live_runs import finalize_benchmark

    report = {"verdict": "failed", "rubric": {}}
    (tmp_path / "final_report.json").write_text(json.dumps(report), encoding="utf-8")

    summary = finalize_benchmark(tmp_path)
    bench = summary["benchmark"]
    assert bench is not None
    assert bench["verdict"] == "failed"
    assert bench["rubric_score"] is None
    assert bench["metrics"] == {}
    assert bench["cost_usd"] is None


# ---------------------------------------------------------------------------
# Fail-soft / degrade tests
# ---------------------------------------------------------------------------


def test_finalize_benchmark_handles_missing_report(tmp_path):
    """finalize_benchmark must degrade fail-soft on a missing/empty report."""
    from backend.services.events.live_runs import finalize_benchmark

    summary = finalize_benchmark(tmp_path)
    assert summary["benchmark"] is None


def test_finalize_benchmark_handles_corrupt_json(tmp_path):
    """A corrupt final_report.json must produce benchmark=None, not raise."""
    from backend.services.events.live_runs import finalize_benchmark

    (tmp_path / "final_report.json").write_text("{not valid json", encoding="utf-8")

    summary = finalize_benchmark(tmp_path)
    assert summary["benchmark"] is None


# ---------------------------------------------------------------------------
# SDK schema — preserved legacy behaviour
# ---------------------------------------------------------------------------


def test_finalize_benchmark_handles_sdk_schema(tmp_path):
    """SDK-shaped reports (rubric_overall_score key) must not be misdetected as RLM."""
    from backend.services.events.live_runs import finalize_benchmark

    report = {
        "rubric_overall_score": 0.82,
        "primary_metric": "mean_reward",
        "paper_primary_target": 475.0,
        "reproduction_primary_value": 492.3,
        "reproduction_delta_vs_paper": 17.3,
        "reproduction_status": "reproduced_with_caveats",
        "rubric_verification": {"overall_score": 0.82, "meets_target": True, "areas": []},
    }
    (tmp_path / "final_report.json").write_text(json.dumps(report), encoding="utf-8")

    summary = finalize_benchmark(tmp_path)
    bench = summary["benchmark"]
    assert bench is not None
    # SDK path: overallScore is rubric_overall_score * 100
    assert bench["overallScore"] == 82.0
    assert bench["verdict"] == "reproduced_with_caveats"
    assert bench["targetMetric"] == "mean_reward"
    assert bench["ourRubricScore"] == 0.82


# ---------------------------------------------------------------------------
# CLI --project-id flag smoke test
# ---------------------------------------------------------------------------


def test_reproduce_argparse_accepts_project_id():
    """--project-id must be accepted by the reproduce subcommand argparse config."""
    import argparse
    import sys

    # We only want to parse; avoid actually importing backend heavy deps by
    # importing the module and calling the parser setup via parse_known_args.
    # Use a minimal argv that won't trigger any I/O.
    from backend import cli  # noqa: F401 — side-effect: registers defaults

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    reproduce = sub.add_parser("reproduce")
    reproduce.add_argument("source")
    reproduce.add_argument("--project-id", default=None)

    args = reproduce.parse_args(["some_paper.pdf", "--project-id", "ui_rlm_test123"])
    assert args.project_id == "ui_rlm_test123"


def test_reproduce_defaults_include_project_id():
    """_REPRODUCE_DEFAULTS must include 'project_id' so Namespace callers don't KeyError."""
    from backend.cli import _REPRODUCE_DEFAULTS

    assert "project_id" in _REPRODUCE_DEFAULTS
    assert _REPRODUCE_DEFAULTS["project_id"] is None
