"""Tests for the terminal-stop wiring (comp 4b, 2026-05-31).

A terminal cell-matrix stop (oom_shrink_exhausted / capacity_exhausted) must:
  1. cause run.py's tool wrapper to call policy.note_terminal_failure (so
     forced_iteration accepts the next FINAL_VAR — stop + report, no re-OOM loop);
  2. stash ctx._terminal_stop_reason so build_final_report surfaces it;
and final_report.json must carry the structured stop_reason.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from backend.agents.rlm.run import _record_last_primitive_result_tools
from backend.agents.rlm.report import RLMFinalReport, _terminal_stop_reason_from_disk


def _policy_spy():
    calls = {"terminal": [], "repair": []}
    policy = SimpleNamespace(
        note_terminal_failure=lambda fc: calls["terminal"].append(fc),
        record_repair_attempt=lambda fc: calls["repair"].append(fc),
    )
    return policy, calls


def _wrap_one(result_dict, ctx):
    policy, calls = _policy_spy()
    tools = {"run_experiment": {"tool": lambda **kw: result_dict}}
    wrapped = _record_last_primitive_result_tools(tools, ctx, [policy])
    wrapped["run_experiment"]["tool"](code_path="x", env_id="y")
    return calls


def test_terminal_oom_notes_failure_and_stashes_stop_reason():
    ctx = SimpleNamespace()
    calls = _wrap_one({
        "outcome": "partial_evidence", "success": False,
        "failure_class": "oom_shrink_exhausted",
        "stop_reason": {"kind": "oom_shrink_exhausted", "detail": "all cells OOM"},
        "metrics": {"per_model": {}},
    }, ctx)
    assert calls["terminal"] == ["oom_shrink_exhausted"]
    assert calls["repair"] == []  # partial_evidence is NOT repairable
    assert ctx._terminal_stop_reason["kind"] == "oom_shrink_exhausted"


def test_capacity_exhausted_via_failure_class_only():
    ctx = SimpleNamespace()
    calls = _wrap_one({
        "outcome": "partial_evidence", "success": False,
        "failure_class": "capacity_exhausted", "metrics": {"scope": {"gaps": []}},
    }, ctx)
    assert calls["terminal"] == ["capacity_exhausted"]
    assert ctx._terminal_stop_reason["kind"] == "capacity_exhausted"


def test_non_terminal_repairable_does_not_note_terminal():
    ctx = SimpleNamespace()
    calls = _wrap_one({
        "outcome": "repairable", "success": False,
        "failure_class": "silent_oom",  # repairable, NOT terminal
        "metrics": {},
    }, ctx)
    assert calls["terminal"] == []
    assert calls["repair"] == ["silent_oom"]
    assert not hasattr(ctx, "_terminal_stop_reason")


def test_final_report_schema_carries_stop_reason():
    r = RLMFinalReport(verdict="failed", stop_reason={"kind": "oom_shrink_exhausted", "detail": "x"})
    assert r.stop_reason["kind"] == "oom_shrink_exhausted"
    assert r.model_dump()["stop_reason"]["kind"] == "oom_shrink_exhausted"
    # default is None on a normal run
    assert RLMFinalReport(verdict="reproduced").stop_reason is None


def test_terminal_stop_reason_from_disk_recovers_last(tmp_path):
    lines = [
        json.dumps({"result": {"success": True, "metrics": {}}}),
        json.dumps({"result": {"stop_reason": {"kind": "oom_shrink_exhausted", "detail": "a"}}}),
        json.dumps({"result": {"stop_reason": {"kind": "capacity_exhausted", "detail": "b"}}}),
    ]
    (tmp_path / "experiment_runs.jsonl").write_text("\n".join(lines), encoding="utf-8")
    sr = _terminal_stop_reason_from_disk(tmp_path)
    assert sr["kind"] == "capacity_exhausted"  # last one wins


def test_terminal_stop_reason_from_disk_none_when_absent(tmp_path):
    assert _terminal_stop_reason_from_disk(tmp_path) is None
    (tmp_path / "experiment_runs.jsonl").write_text(
        json.dumps({"result": {"success": True}}), encoding="utf-8")
    assert _terminal_stop_reason_from_disk(tmp_path) is None
