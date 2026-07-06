"""T8: _synth_result_from_summary projects an honest report from evidence,
even when rubric_score is None, instead of failing a scoreless-but-evidenced
run. Only a genuinely-empty run (no score, no evidence) ships 'failed'.

See tests/rlm/test_run_lifecycle_primary.py::TestSynthResultFromSummary for
the pre-existing scored-path coverage this must not regress.
"""
import json

from backend.agents.rlm.run import _synth_result_from_summary


class _Ctx:
    project_id = "p1"
    cost_ledger = []
    def remaining_s(self): return 999


def test_projects_report_from_scored_summary():
    summary = {"rubric_score": 0.46,
               "verify_result": {"overall_score": 0.46, "target_score": 0.456,
                                 "meets_target": True},
               "driven": ["implement_baseline", "run_experiment", "verify_against_rubric"]}
    res = _synth_result_from_summary(summary, _Ctx())
    payload = json.loads(res.response)
    assert payload["verdict"] in ("reproduced", "partial")
    assert payload.get("rubric") or payload.get("overall_score") is not None


def test_completed_but_unscored_with_evidence_is_not_failed():
    summary = {"rubric_score": None,
               "verify_result": {"overall_score": None},
               "driven": ["implement_baseline", "run_experiment"],
               "has_evidence": True}
    res = _synth_result_from_summary(summary, _Ctx())
    payload = json.loads(res.response)
    assert payload["verdict"] != "failed"  # honest partial, not a scoreless failure


def test_genuinely_empty_run_fails_honestly():
    summary = {"rubric_score": None, "verify_result": None, "driven": [], "has_evidence": False}
    res = _synth_result_from_summary(summary, _Ctx())
    payload = json.loads(res.response)
    assert payload["verdict"] == "failed"
