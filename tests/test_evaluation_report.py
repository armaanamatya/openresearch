"""Track E Task 5: typed EvaluationReport adapter + ScorecardRow model.

``EvaluationReport`` COMPOSES the run report (spec §6.1/§6.3) — it copies
``final_report.json``'s ``verdict`` read-only and never recomputes it, and its
``gate_caps()`` helper is a downward-only PRE-``decide()`` input (like
``claim_gate_cap``), never a post-decide writer. This suite locks: the
read-only verdict copy, the ``ScorecardRow`` shape, gate-cap downward-only
behavior (display/pass rows never gate), and the composite staying an
optional, non-authoritative display float.
"""

import json

from backend.evals.evaluation_report import EvaluationReport, ScorecardRow


def _run(tmp_path, verdict="partial"):
    (tmp_path / "final_report.json").write_text(json.dumps({"verdict": verdict, "baseline_metrics": {}}))
    return tmp_path


def test_evaluation_report_copies_verdict_read_only(tmp_path):
    er = EvaluationReport.from_run(_run(tmp_path, verdict="inconclusive"))
    assert er.verdict == "inconclusive"  # copied, not recomputed


def test_scorecard_row_shape():
    r = ScorecardRow(dimension="numerical_reproduction", status="pass",
                     provenance="agent_measured", gates=True, evidence_refs=["metrics.json#accuracy"], detail="")
    assert r.gates is True and r.status == "pass"


def test_gate_cap_is_downward_only_and_never_a_verdict_write(tmp_path):
    er = EvaluationReport.from_run(_run(tmp_path, verdict="reproduced"))
    er.scorecard = [ScorecardRow(dimension="execution_completeness", status="unmeasured",
                                 provenance="evaluator_computed", gates=True, evidence_refs=[], detail="")]
    cap = er.gate_caps()
    assert cap in ("partial", "inconclusive")  # a downward cap, computed BEFORE decide(), never written back
    # the report object's verdict field is untouched by gate_caps()
    assert er.verdict == "reproduced"


def test_display_rows_never_gate(tmp_path):
    er = EvaluationReport.from_run(_run(tmp_path))
    er.scorecard = [ScorecardRow(dimension="paper_understanding", status="display",
                                 provenance="evaluator_computed", gates=False, evidence_refs=[], detail="x")]
    assert er.gate_caps() is None  # display rows contribute no cap


def test_composite_is_display_only_optional(tmp_path):
    er = EvaluationReport.from_run(_run(tmp_path))
    assert er.composite is None or isinstance(er.composite, float)  # never a verdict driver
