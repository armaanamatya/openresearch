"""Unit tests for scripts/rlm_root_ab.py — parse_run_metrics + aggregate.

All tests use synthetic run dirs in tmp_path; no credentials or subprocesses
are involved.  The synthetic event shapes match the three wire formats observed
in real dashboard_events.jsonl files:

  Shape 1 (newer harness):
    {"event": "run_warning", "code": "<code>", "message": "...", "timestamp": "..."}

  Shape 2 (older harness):
    {"event": "run_warning", "data": {"code": "<code>", "message": "..."}, "ts": "..."}

  Shape 3 (watchdog/infra):
    {"event": "run_warning", "data": {"reason": "stale_run", ...}, "ts": "..."}

  primitive_call:
    {"event": "primitive_call", "primitive": "<name>", "status": "start"|"ok"|"error",
     "args_summary": {...}, "result_summary": "...", "iteration": null, ...}

  experiment_completed (failure_class surface):
    {"event": "experiment_completed", "data": {"failure_class": "<class>", ...}}

  repl_iteration:
    {"event": "repl_iteration", "iteration": N, "response": "..."}

  run_complete:
    {"event": "run_complete", "status": "completed", "iterations": N,
     "rubric_score": F, "cost_usd": F, "final_report_path": "..."}

  final_report.json:
    {"verdict": "...", "rubric": {"overall_score": F, "meets_target": bool}, ...}
"""
from __future__ import annotations

import json
import pathlib
import sys


# Ensure the repo root is on sys.path so the script module is importable.
_REPO_ROOT = pathlib.Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.rlm_root_ab import aggregate, parse_run_metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_events(run_dir: pathlib.Path, events: list[dict]) -> None:
    events_path = run_dir / "dashboard_events.jsonl"
    with events_path.open("w", encoding="utf-8") as fh:
        for ev in events:
            fh.write(json.dumps(ev) + "\n")


def _write_report(run_dir: pathlib.Path, report: dict) -> None:
    (run_dir / "final_report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )


def _make_run_dir(tmp_path: pathlib.Path, name: str = "run1") -> pathlib.Path:
    d = tmp_path / name
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# Test 1: warning code tallying — all three wire shapes
# ---------------------------------------------------------------------------

def test_run_warning_counts_three_shapes(tmp_path: pathlib.Path) -> None:
    """All three run_warning shapes are counted correctly."""
    d = _make_run_dir(tmp_path)
    _write_events(d, [
        # Shape 1: top-level code
        {"event": "run_warning", "code": "forced_iteration",
         "message": "score below target", "timestamp": "2026-01-01T00:00:00Z"},
        {"event": "run_warning", "code": "forced_iteration",
         "message": "still below", "timestamp": "2026-01-01T00:01:00Z"},
        # Shape 2: nested data.code
        {"event": "run_warning", "data": {"code": "compute_scope_invalid",
         "message": "scope must be dict"}, "ts": "2026-01-01T00:02:00Z"},
        # Shape 3: nested data.reason (watchdog)
        {"event": "run_warning", "data": {"reason": "stale_run",
         "stale_seconds": 610.0}, "ts": "2026-01-01T00:03:00Z"},
        # run_complete
        {"event": "run_complete", "status": "completed", "iterations": 3,
         "rubric_score": 0.5, "cost_usd": 0.0,
         "final_report_path": str(d / "final_report.json")},
    ])
    _write_report(d, {
        "verdict": "partial",
        "rubric": {"overall_score": 0.5, "meets_target": False},
    })

    m = parse_run_metrics(d)

    assert m["run_warning_counts"]["forced_iteration"] == 2
    assert m["run_warning_counts"]["compute_scope_invalid"] == 1
    assert m["run_warning_counts"]["stale_run"] == 1
    assert m["iterations"] == 3


# ---------------------------------------------------------------------------
# Test 2: arg_contract_blocks count via experiment_completed.data.failure_class
# ---------------------------------------------------------------------------

def test_arg_contract_blocks_from_experiment_completed(tmp_path: pathlib.Path) -> None:
    """arg_contract_blocks counts experiment_completed events with failure_class=arg_contract."""
    d = _make_run_dir(tmp_path)
    _write_events(d, [
        # arg_contract block from the guard
        {"event": "experiment_completed", "data": {
            "failure_class": "arg_contract",
            "success": False,
            "error": "arg guard blocked",
        }},
        # non-arg_contract failure (should not count)
        {"event": "experiment_completed", "data": {
            "failure_class": "preflight_blocked",
            "success": False,
        }},
        # another arg_contract
        {"event": "experiment_completed", "data": {
            "failure_class": "arg_contract",
            "success": False,
        }},
        {"event": "run_complete", "status": "failed", "iterations": 2,
         "rubric_score": 0.0, "cost_usd": 0.0,
         "final_report_path": str(d / "final_report.json")},
    ])
    _write_report(d, {"verdict": "failed",
                      "rubric": {"overall_score": 0.0, "meets_target": False}})

    m = parse_run_metrics(d)

    assert m["arg_contract_blocks"] == 2
    assert m["fabrication_suspected"] == 0


# ---------------------------------------------------------------------------
# Test 3: fabrication_suspected count — both sources
# ---------------------------------------------------------------------------

def test_fabrication_suspected_from_warning_and_experiment(tmp_path: pathlib.Path) -> None:
    """fabrication_suspected is incremented from both run_warning code and experiment failure_class."""
    d = _make_run_dir(tmp_path)
    _write_events(d, [
        # run_warning with a fabrication-related code (top-level shape)
        {"event": "run_warning", "code": "fabrication_suspected",
         "message": "metrics look fabricated", "timestamp": "2026-01-01T00:00:00Z"},
        # experiment_completed with a fabrication-related failure_class
        {"event": "experiment_completed", "data": {
            "failure_class": "fabrication_suspected",
            "success": False,
        }},
        {"event": "run_complete", "status": "failed", "iterations": 1,
         "rubric_score": 0.0, "cost_usd": 0.0,
         "final_report_path": str(d / "final_report.json")},
    ])
    _write_report(d, {"verdict": "failed",
                      "rubric": {"overall_score": 0.0, "meets_target": False}})

    m = parse_run_metrics(d)

    assert m["fabrication_suspected"] == 2
    assert m["run_warning_counts"].get("fabrication_suspected", 0) == 1


# ---------------------------------------------------------------------------
# Test 4: first_implement_baseline_iteration detection
# ---------------------------------------------------------------------------

def test_first_implement_baseline_iteration(tmp_path: pathlib.Path) -> None:
    """first_implement_baseline_iteration is the repl_iteration count up to the first
    implement_baseline start event."""
    d = _make_run_dir(tmp_path)
    _write_events(d, [
        {"event": "repl_iteration", "iteration": 1, "response": "reading paper"},
        {"event": "primitive_call", "primitive": "understand_section",
         "status": "start", "args_summary": {"arg0": "str[1000]"},
         "result_summary": None, "iteration": None},
        {"event": "primitive_call", "primitive": "understand_section",
         "status": "ok", "args_summary": {},
         "result_summary": "dict[ambiguities, datasets]", "iteration": None},
        {"event": "repl_iteration", "iteration": 2, "response": "planning"},
        {"event": "repl_iteration", "iteration": 3, "response": "implement"},
        # First implement_baseline — at this point repl_iter_count = 3
        {"event": "primitive_call", "primitive": "implement_baseline",
         "status": "start", "args_summary": {"arg0": "dict[3]"},
         "result_summary": None, "iteration": None},
        {"event": "primitive_call", "primitive": "implement_baseline",
         "status": "ok", "args_summary": {},
         "result_summary": "str[57]", "iteration": None},
        {"event": "repl_iteration", "iteration": 4, "response": "run"},
        # Second implement_baseline — should NOT overwrite first
        {"event": "primitive_call", "primitive": "implement_baseline",
         "status": "start", "args_summary": {"arg0": "dict[4]"},
         "result_summary": None, "iteration": None},
        {"event": "run_complete", "status": "partial", "iterations": 4,
         "rubric_score": 0.3, "cost_usd": 0.0,
         "final_report_path": str(d / "final_report.json")},
    ])
    _write_report(d, {"verdict": "partial",
                      "rubric": {"overall_score": 0.3, "meets_target": False}})

    m = parse_run_metrics(d)

    # repl_iter_count=3 when first implement_baseline start was seen
    assert m["first_implement_baseline_iteration"] == 3
    assert m["iterations"] == 4  # from run_complete


# ---------------------------------------------------------------------------
# Test 5: verdict / overall_score extraction (rubric-nested schema)
# ---------------------------------------------------------------------------

def test_final_report_verdict_and_score(tmp_path: pathlib.Path) -> None:
    """verdict, overall_score, meets_target come from final_report.rubric."""
    d = _make_run_dir(tmp_path)
    # Only final_report.json — no events file at all
    _write_report(d, {
        "verdict": "reproduced",
        "overall_score": None,   # top-level is None (real schema)
        "meets_target": None,    # top-level is None (real schema)
        "iterations": 7,
        "rubric": {
            "overall_score": 0.725,
            "meets_target": True,
            "rubric_source": "auto",
        },
    })

    m = parse_run_metrics(d)

    assert m["verdict"] == "reproduced"
    assert abs(m["overall_score"] - 0.725) < 1e-9
    assert m["meets_target"] is True
    # No events → zero counts
    assert m["run_warning_counts"] == {}
    assert m["arg_contract_blocks"] == 0
    assert m["iterations"] is None  # no run_complete event


# ---------------------------------------------------------------------------
# Test 6: fail-soft on missing files
# ---------------------------------------------------------------------------

def test_fail_soft_missing_files(tmp_path: pathlib.Path) -> None:
    """parse_run_metrics returns zero/None values when both files are absent."""
    d = _make_run_dir(tmp_path)
    # Neither dashboard_events.jsonl nor final_report.json exists.

    m = parse_run_metrics(d)

    assert m["run_warning_counts"] == {}
    assert m["arg_contract_blocks"] == 0
    assert m["fabrication_suspected"] == 0
    assert m["iterations"] is None
    assert m["first_implement_baseline_iteration"] is None
    assert m["verdict"] is None
    assert m["overall_score"] is None
    assert m["meets_target"] is None


# ---------------------------------------------------------------------------
# Test 7: fail-soft on garbled JSON lines
# ---------------------------------------------------------------------------

def test_fail_soft_garbled_lines(tmp_path: pathlib.Path) -> None:
    """Garbled / partial JSON lines are skipped; valid lines are still parsed."""
    d = _make_run_dir(tmp_path)
    events_path = d / "dashboard_events.jsonl"
    events_path.write_text(
        "THIS IS NOT JSON\n"
        + json.dumps({"event": "run_warning", "code": "sdk_aclose_loop",
                      "message": "deadlock", "timestamp": "2026-01-01T00:00:00Z"})
        + "\n"
        + "{broken json here\n"
        + json.dumps({"event": "run_complete", "status": "failed", "iterations": 2,
                      "rubric_score": 0.0, "cost_usd": 0.0,
                      "final_report_path": "x"})
        + "\n",
        encoding="utf-8",
    )
    # Corrupted final_report.json
    (d / "final_report.json").write_text("{bad json", encoding="utf-8")

    m = parse_run_metrics(d)

    assert m["run_warning_counts"]["sdk_aclose_loop"] == 1
    assert m["iterations"] == 2
    assert m["verdict"] is None  # bad json → None


# ---------------------------------------------------------------------------
# Test 8: aggregate — mean/distribution across trials
# ---------------------------------------------------------------------------

def test_aggregate_mean_and_distribution() -> None:
    """aggregate computes per-metric means and verdict distribution correctly."""
    trial_metrics = [
        {
            "run_warning_counts": {"forced_iteration": 2, "compute_scope_invalid": 1},
            "arg_contract_blocks": 1,
            "fabrication_suspected": 0,
            "iterations": 3,
            "first_implement_baseline_iteration": 2,
            "overall_score": 0.4,
            "meets_target": False,
            "verdict": "partial",
        },
        {
            "run_warning_counts": {"forced_iteration": 4},
            "arg_contract_blocks": 0,
            "fabrication_suspected": 1,
            "iterations": 5,
            "first_implement_baseline_iteration": 3,
            "overall_score": 0.6,
            "meets_target": True,
            "verdict": "reproduced",
        },
        {
            "run_warning_counts": {"forced_iteration": 0, "compute_scope_invalid": 1},
            "arg_contract_blocks": 2,
            "fabrication_suspected": 0,
            "iterations": 4,
            "first_implement_baseline_iteration": None,  # no impl_baseline seen
            "overall_score": None,
            "meets_target": None,
            "verdict": "failed",
        },
    ]

    agg = aggregate(trial_metrics)

    assert agg["n"] == 3

    # forced_iteration mean: (2+4+0)/3
    expected_fi_mean = (2 + 4 + 0) / 3
    assert abs(agg["run_warning_counts_mean"]["forced_iteration"] - expected_fi_mean) < 1e-9

    # compute_scope_invalid mean: (1+0+1)/3
    expected_csi_mean = (1 + 0 + 1) / 3
    assert abs(agg["run_warning_counts_mean"]["compute_scope_invalid"] - expected_csi_mean) < 1e-9

    # arg_contract_blocks_mean: (1+0+2)/3
    assert abs(agg["arg_contract_blocks_mean"] - (1 + 0 + 2) / 3) < 1e-9

    # fabrication_suspected_mean: (0+1+0)/3
    assert abs(agg["fabrication_suspected_mean"] - (0 + 1 + 0) / 3) < 1e-9

    # iterations_mean: (3+5+4)/3 = 4.0
    assert abs(agg["iterations_mean"] - 4.0) < 1e-9

    # first_implement_baseline_iteration_mean: None excluded → (2+3)/2 = 2.5
    assert abs(agg["first_implement_baseline_iteration_mean"] - 2.5) < 1e-9

    # overall_score_mean: None excluded → (0.4+0.6)/2 = 0.5
    assert abs(agg["overall_score_mean"] - 0.5) < 1e-9

    # verdict distribution
    assert agg["verdict_distribution"]["partial"] == 1
    assert agg["verdict_distribution"]["reproduced"] == 1
    assert agg["verdict_distribution"]["failed"] == 1

    # meets_target_count
    assert agg["meets_target_count"] == 1
