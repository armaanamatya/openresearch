"""
Unit tests for Phase 1e Unit C — EvidenceVector held-out non-regression
admission gate (backend/agents/rlm/held_out_gate.py).
Pure unit tests — no network, no filesystem, no subprocess.
"""

from backend.agents.rlm.held_out_gate import CandidateLesson, ReplayCase, admit, evidence_vector  # noqa: F401
from backend.agents.rlm.failure_attribution import attribute_failure


def _cand():
    return CandidateLesson(attribution=attribute_failure({"failure_class": "missing_module"}), patch={"hint": "add flash_attn"})


def _case(preds):  # expected (baseline) predicates
    return ReplayCase(id="c1", expected_predicates=preds)


def test_promotes_on_improvement_no_regression():
    baseline = {"backed_by_ledger": True, "provenance_present": False, "metrics_non_degenerate": True,
                "metric_keys_real": True, "run_level_clean": True}
    improved = dict(baseline, provenance_present=True)   # one predicate improved, veto True, none regressed
    out = admit(_cand(), [_case(baseline)], apply_fn=lambda c, case: improved)
    assert out.admission_state == "active"


def test_rejects_on_any_regression():
    baseline = {"backed_by_ledger": True, "provenance_present": True, "metrics_non_degenerate": True,
                "metric_keys_real": True, "run_level_clean": True}
    regressed = dict(baseline, metric_keys_real=False)   # a held-out predicate regressed
    out = admit(_cand(), [_case(baseline)], apply_fn=lambda c, case: regressed)
    assert out.admission_state == "rejected"


def test_rejects_on_veto_false_even_if_others_improve():
    baseline = {"backed_by_ledger": False, "provenance_present": False, "metrics_non_degenerate": True,
                "metric_keys_real": True, "run_level_clean": True}
    fabricated = dict(baseline, backed_by_ledger=True, provenance_present=True, run_level_clean=False)  # veto fails
    out = admit(_cand(), [_case(baseline)], apply_fn=lambda c, case: fabricated)
    assert out.admission_state == "rejected"   # validator veto is absolute — never a scalar override


def test_no_improvement_stays_rejected():
    baseline = {"backed_by_ledger": True, "provenance_present": True, "metrics_non_degenerate": True,
                "metric_keys_real": True, "run_level_clean": True}
    out = admit(_cand(), [_case(baseline)], apply_fn=lambda c, case: dict(baseline))  # identical, nothing improved
    assert out.admission_state == "rejected"
