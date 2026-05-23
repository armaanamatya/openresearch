"""Unit tests for backend.agents.rlm.report.

Coverage:
- JSON response → valid RLMFinalReport
- Python-repr dict string (FINAL_VAR str()-ification) → recovered via ast.literal_eval
- Garbage string → `failed` verdict, no crash
- Cost reconciliation: sums usage_summary + cost_ledger
- Empty baseline_metrics tolerated
- write_final_report_rlm writes both files and they are valid

Spec: §11 (2026-05-21-rlm-phase3-orchestrator-design.md).
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

import pytest

from rlm.core.types import RLMChatCompletion, UsageSummary, ModelUsageSummary

from backend.agents.rlm.report import RLMFinalReport, build_final_report, write_final_report_rlm
from backend.agents.resilience.cost import CostLedgerEntry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_usage(total_cost: float = 0.0) -> UsageSummary:
    return UsageSummary(
        model_usage_summaries={
            "test-model": ModelUsageSummary(
                total_calls=1,
                total_input_tokens=100,
                total_output_tokens=50,
                total_cost=total_cost,
            )
        }
    )


def _make_result(response: str, total_cost: float = 0.0) -> RLMChatCompletion:
    return RLMChatCompletion(
        root_model="test-model",
        prompt={"paper_text": "some text"},
        response=response,
        usage_summary=_make_usage(total_cost),
        execution_time=1.0,
        metadata=None,
    )


def _record_run_experiment(ctx):
    """Append a run_experiment ledger entry so the honesty guard treats the
    run's baseline_metrics as backed by a real execution. Returns ``ctx``."""
    ctx.cost_ledger.append(CostLedgerEntry(
        timestamp=datetime.now(timezone.utc),
        agent_id="run_experiment",
        attempt_index=0,
        provider="anthropic",
        model="test-model",
    ))
    return ctx


_BASE_REPORT_DICT = {
    "paper": {"id": "2512.24601", "title": "Test Paper"},
    "verdict": "reproduced",
    "reproduction_summary": "Baseline reproduced within 2% of claimed accuracy.",
    "baseline_metrics": {"accuracy": 0.92},
    "paper_claims": {"accuracy": 0.94},
    "rubric": {"overall_score": 0.88, "meets_target": True, "areas": []},
    "improvements": [{"tag": "lr-tuning", "outcome": "pending"}],
    "primitive_trace": {"understand_section": 2},
    "cost": {"llm_usd": 0.05, "root": 0.04, "sub": 0.01, "primitives": 0.0},
    "iterations": 5,
}


# ---------------------------------------------------------------------------
# Tests: parsing
# ---------------------------------------------------------------------------


class TestBuildFinalReportParsing:
    """Tests for response-parsing logic in build_final_report."""

    def test_json_response(self, make_context, tmp_path):
        """A valid JSON response is parsed into a correct RLMFinalReport."""
        ctx = _record_run_experiment(make_context(tmp_path))
        result = _make_result(json.dumps(_BASE_REPORT_DICT))
        report = build_final_report(result, ctx=ctx)

        assert isinstance(report, RLMFinalReport)
        assert report.verdict == "reproduced"
        assert report.paper == {"id": "2512.24601", "title": "Test Paper"}
        assert report.iterations == 5
        assert report.baseline_metrics == {"accuracy": 0.92}

    def test_python_repr_fallback(self, make_context, tmp_path):
        """A Python-repr-stringified dict (from FINAL_VAR str()) is recovered via ast.literal_eval."""
        ctx = _record_run_experiment(make_context(tmp_path))
        # Simulate what LocalREPL._final_var does: str() on the dict
        raw = str(_BASE_REPORT_DICT)
        result = _make_result(raw)
        report = build_final_report(result, ctx=ctx)

        assert isinstance(report, RLMFinalReport)
        assert report.verdict == "reproduced"
        assert report.reproduction_summary == _BASE_REPORT_DICT["reproduction_summary"]

    def test_garbage_response_gives_failed_verdict(self, make_context, tmp_path):
        """An un-parseable response produces a `failed` verdict without crashing."""
        ctx = make_context(tmp_path)
        result = _make_result("this is not json {{{{{")
        report = build_final_report(result, ctx=ctx)

        assert isinstance(report, RLMFinalReport)
        assert report.verdict == "failed"
        # The raw text is surfaced in reproduction_summary
        assert "this is not json" in report.reproduction_summary

    def test_garbage_response_no_exception(self, make_context, tmp_path):
        """build_final_report never raises, even on completely invalid input."""
        ctx = make_context(tmp_path)
        for bad_input in ["", "null", "[]", "42", "\x00\x01\x02"]:
            result = _make_result(bad_input)
            # Should not raise
            report = build_final_report(result, ctx=ctx)
            assert isinstance(report, RLMFinalReport)

    def test_missing_verdict_field_downgrades_to_partial(self, make_context, tmp_path):
        """A parsed dict missing `verdict` gets a `partial` verdict (honest default)."""
        ctx = make_context(tmp_path)
        partial_dict = {k: v for k, v in _BASE_REPORT_DICT.items() if k != "verdict"}
        result = _make_result(json.dumps(partial_dict))
        report = build_final_report(result, ctx=ctx)

        assert report.verdict == "partial"

    def test_unknown_verdict_downgrades_to_partial(self, make_context, tmp_path):
        """A parsed dict with an unknown verdict string is down-graded to `partial`."""
        ctx = make_context(tmp_path)
        d = dict(_BASE_REPORT_DICT, verdict="unknown_value")
        result = _make_result(json.dumps(d))
        report = build_final_report(result, ctx=ctx)

        assert report.verdict == "partial"

    def test_empty_baseline_metrics_tolerated(self, make_context, tmp_path):
        """Empty baseline_metrics in the response does not raise."""
        ctx = make_context(tmp_path)
        d = dict(_BASE_REPORT_DICT, baseline_metrics={})
        result = _make_result(json.dumps(d))
        report = build_final_report(result, ctx=ctx)

        assert report.baseline_metrics == {}
        assert isinstance(report, RLMFinalReport)

    def test_absent_baseline_metrics_defaults_to_empty_dict(self, make_context, tmp_path):
        """A response without baseline_metrics gets an empty dict default."""
        ctx = make_context(tmp_path)
        d = {k: v for k, v in _BASE_REPORT_DICT.items() if k != "baseline_metrics"}
        result = _make_result(json.dumps(d))
        report = build_final_report(result, ctx=ctx)

        assert report.baseline_metrics == {}

    def test_root_model_none_accepted(self, make_context, tmp_path):
        """root_model=None (default) is accepted without error."""
        ctx = make_context(tmp_path)
        result = _make_result(json.dumps(_BASE_REPORT_DICT))
        report = build_final_report(result, ctx=ctx, root_model=None)

        assert isinstance(report, RLMFinalReport)


# ---------------------------------------------------------------------------
# Tests: honesty guard — a result section must be backed by its primitive
# ---------------------------------------------------------------------------


class TestHonestyGuard:
    """build_final_report must not present results the root never measured."""

    def test_unbacked_baseline_metrics_dropped(self, make_context, tmp_path):
        """When run_experiment never ran, root-supplied baseline_metrics are
        dropped and a 'reproduced' verdict is downgraded to 'partial'."""
        ctx = make_context(tmp_path)  # empty ledger — run_experiment never ran
        result = _make_result(json.dumps(_BASE_REPORT_DICT))
        report = build_final_report(result, ctx=ctx)

        assert report.baseline_metrics == {}
        assert report.verdict == "partial"
        assert "honesty guard" in report.reproduction_summary.lower()

    def test_backed_baseline_metrics_kept(self, make_context, tmp_path):
        """baseline_metrics survive when run_experiment is in the ledger."""
        ctx = _record_run_experiment(make_context(tmp_path))
        result = _make_result(json.dumps(_BASE_REPORT_DICT))
        report = build_final_report(result, ctx=ctx)

        assert report.baseline_metrics == {"accuracy": 0.92}
        assert report.verdict == "reproduced"

    def test_primitive_trace_comes_from_the_ledger_not_the_root(
        self, make_context, tmp_path
    ):
        """primitive_trace reflects the authoritative cost ledger — not the
        root model's self-reported (and observed-unreliable) trace."""
        ctx = make_context(tmp_path)
        ctx.cost_ledger.append(CostLedgerEntry(
            timestamp=datetime.now(timezone.utc),
            agent_id="understand_section",
            attempt_index=0,
            provider="anthropic",
            model="test-model",
        ))
        # _BASE_REPORT_DICT self-reports primitive_trace={"understand_section": 2}
        result = _make_result(json.dumps(_BASE_REPORT_DICT))
        report = build_final_report(result, ctx=ctx)

        assert report.primitive_trace["by_primitive"] == {"understand_section": 1}
        assert report.primitive_trace["calls"] == 1


# ---------------------------------------------------------------------------
# Tests: cost reconciliation
# ---------------------------------------------------------------------------


class TestCostReconciliation:
    """Tests for cost summing from usage_summary + cost_ledger."""

    def test_usage_summary_cost_included(self, make_context, tmp_path):
        """RLM usage_summary total_cost contributes to report.cost['llm_usd']."""
        ctx = make_context(tmp_path)
        result = _make_result(json.dumps(_BASE_REPORT_DICT), total_cost=0.123)
        report = build_final_report(result, ctx=ctx)

        # T7/M-BUDGET: the false root/sub split is dropped — llm_usd is the honest total.
        assert report.cost["llm_usd"] == pytest.approx(0.123, abs=1e-7)
        assert "root" not in report.cost and "sub" not in report.cost

    def test_cost_ledger_entries_summed(self, make_context, tmp_path):
        """Entries in ctx.cost_ledger contribute to cost['primitives'] and 'llm_usd'."""
        ctx = make_context(tmp_path)
        # Append a real cost ledger entry
        entry = CostLedgerEntry(
            timestamp=datetime.now(timezone.utc),
            agent_id="test-agent",
            attempt_index=0,
            provider="anthropic",
            model="claude-3-haiku-20240307",
            input_tokens=100,
            output_tokens=50,
            estimated_usd=0.05,
        )
        ctx.cost_ledger.append(entry)

        result = _make_result(json.dumps(_BASE_REPORT_DICT), total_cost=0.10)
        report = build_final_report(result, ctx=ctx)

        assert report.cost["primitives"] == pytest.approx(0.05, abs=1e-7)
        assert report.cost["llm_usd"] == pytest.approx(0.15, abs=1e-7)

    def test_zero_cost_ledger_tolerated(self, make_context, tmp_path):
        """An empty cost ledger produces 0.0 primitives cost — no crash."""
        ctx = make_context(tmp_path)
        result = _make_result(json.dumps(_BASE_REPORT_DICT), total_cost=0.0)
        report = build_final_report(result, ctx=ctx)

        assert report.cost["llm_usd"] == pytest.approx(0.0, abs=1e-7)
        assert report.cost["primitives"] == pytest.approx(0.0, abs=1e-7)


# ---------------------------------------------------------------------------
# Tests: write_final_report_rlm
# ---------------------------------------------------------------------------


class TestWriteFinalReport:
    """Tests for the atomic file writer."""

    def _build_report(self) -> RLMFinalReport:
        return RLMFinalReport(**_BASE_REPORT_DICT)

    def test_writes_json_and_md(self, tmp_path):
        """write_final_report_rlm creates both final_report.json and final_report.md."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        report = self._build_report()

        json_path, md_path = write_final_report_rlm(report, project_dir)

        assert json_path == project_dir / "final_report.json"
        assert md_path == project_dir / "final_report.md"
        assert json_path.exists()
        assert md_path.exists()

    def test_json_is_valid_and_round_trips(self, tmp_path):
        """The written JSON parses back to an equivalent RLMFinalReport."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        report = self._build_report()
        json_path, _ = write_final_report_rlm(report, project_dir)

        raw = json_path.read_text(encoding="utf-8")
        parsed = json.loads(raw)
        restored = RLMFinalReport.model_validate(parsed)

        assert restored.verdict == report.verdict
        assert restored.iterations == report.iterations
        assert restored.paper == report.paper

    def test_md_contains_verdict_and_rubric_score(self, tmp_path):
        """The Markdown file contains the verdict and rubric score."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        report = self._build_report()
        _, md_path = write_final_report_rlm(report, project_dir)

        md = md_path.read_text(encoding="utf-8")
        assert "REPRODUCED" in md
        assert "0.880" in md  # rubric overall_score = 0.88

    def test_md_contains_paper_title(self, tmp_path):
        """The Markdown file includes the paper title."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        report = self._build_report()
        _, md_path = write_final_report_rlm(report, project_dir)

        md = md_path.read_text(encoding="utf-8")
        assert "Test Paper" in md

    def test_write_creates_project_dir(self, tmp_path):
        """write_final_report_rlm creates project_dir if it does not exist."""
        project_dir = tmp_path / "new_project_dir"
        assert not project_dir.exists()
        report = self._build_report()
        write_final_report_rlm(report, project_dir)
        assert project_dir.exists()
        assert (project_dir / "final_report.json").exists()

    def test_no_tmp_file_left_behind(self, tmp_path):
        """No .tmp file is left after write_final_report_rlm."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        report = self._build_report()
        write_final_report_rlm(report, project_dir)

        tmp_files = list(project_dir.glob("*.tmp"))
        assert tmp_files == [], f"Leftover tmp files: {tmp_files}"

    def test_failed_verdict_report_written(self, tmp_path):
        """A `failed` verdict report is written without error."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        report = RLMFinalReport(verdict="failed", reproduction_summary="Run failed.")

        json_path, md_path = write_final_report_rlm(report, project_dir)

        assert json_path.exists()
        md = md_path.read_text(encoding="utf-8")
        assert "REPRODUCTION FAILED" in md


# ---------------------------------------------------------------------------
# Tests: evidence-based verdict reconciliation (T6 / P0-I9)
# ---------------------------------------------------------------------------


class TestEvidenceBasedVerdictReconciliation:
    """Symptom: ftrl run scored 0.0 yet self-reported verdict='reproduced'.

    The pre-T6 honesty guard only dropped fabricated metrics; it did not
    downgrade an over-claimed verdict when (a) run_experiment never ran,
    (b) baseline_metrics is empty, or (c) rubric.overall_score < 0.5
    (handoff P0-I9 / plan T6).
    """

    def test_verdict_downgraded_when_evidence_contradicts(self, make_context, tmp_path):
        """Verify: a 'reproduced' claim with rubric score 0.0 (but run_experiment
        DID run, with measured metrics) still downgrades to 'partial' due to the
        score threshold.
        """
        ctx = _record_run_experiment(make_context(tmp_path))
        raw = json.dumps({
            "verdict": "reproduced",
            "baseline_metrics": {"mean_reward": 487.3},  # has metrics
            "rubric": {"overall_score": 0.0, "meets_target": False},  # but score == 0
            "paper": {"id": "ftrl"},
        })
        result = _make_result(raw)
        report = build_final_report(result, ctx=ctx)

        assert report.verdict == "partial"  # downgraded by evidence guard
        assert "0.000 < 0.5" in report.reproduction_summary  # reason surfaced

    def test_verdict_not_downgraded_when_score_sufficient(self, make_context, tmp_path):
        """A 'reproduced' claim with rubric score >= 0.5, non-empty metrics, and
        run_experiment in the ledger is NOT downgraded."""
        ctx = _record_run_experiment(make_context(tmp_path))
        raw = json.dumps({
            "verdict": "reproduced",
            "baseline_metrics": {"accuracy": 0.92},
            "rubric": {"overall_score": 0.75, "meets_target": True},
            "paper": {"id": "test"},
        })
        result = _make_result(raw)
        report = build_final_report(result, ctx=ctx)

        assert report.verdict == "reproduced"

    def test_partial_verdict_not_downgraded_by_evidence_guard(self, make_context, tmp_path):
        """'partial' verdict is passed through unchanged regardless of evidence."""
        ctx = make_context(tmp_path)  # run_experiment never ran
        raw = json.dumps({
            "verdict": "partial",
            "baseline_metrics": {},
            "rubric": {"overall_score": 0.0, "meets_target": False},
            "paper": {"id": "test"},
        })
        result = _make_result(raw)
        report = build_final_report(result, ctx=ctx)

        assert report.verdict == "partial"

    def test_failed_verdict_not_touched_by_evidence_guard(self, make_context, tmp_path):
        """'failed' verdict is passed through unchanged."""
        ctx = make_context(tmp_path)
        raw = json.dumps({
            "verdict": "failed",
            "baseline_metrics": {},
            "rubric": {"overall_score": 0.0, "meets_target": False},
            "paper": {"id": "test"},
        })
        result = _make_result(raw)
        report = build_final_report(result, ctx=ctx)

        assert report.verdict == "failed"

    def test_no_run_experiment_downgrades_reproduced(self, make_context, tmp_path):
        """'reproduced' with no run_experiment entry is downgraded (via the
        metric-fabrication guard which fires first when metrics are present,
        then evidence guard for the no-metrics case)."""
        ctx = make_context(tmp_path)  # empty ledger
        raw = json.dumps({
            "verdict": "reproduced",
            "baseline_metrics": {},  # no metrics, no run_experiment
            "rubric": {"overall_score": 0.0, "meets_target": False},
            "paper": {"id": "test"},
        })
        result = _make_result(raw)
        report = build_final_report(result, ctx=ctx)

        assert report.verdict == "partial"

    def test_no_run_experiment_in_isolation_downgrades_reproduced(self, make_context, tmp_path):
        """Symptom: a 'reproduced' verdict with a high score but where run_experiment
        never ran could slip past the evidence guard if only the score and metrics
        branches were tested.

        Pin the never-ran branch: empty baseline_metrics (so the prior
        metric-fabrication guard does NOT fire), high rubric score, and
        no run_experiment entry on the ledger. Both "never ran" and
        "no measured baseline metrics" fire — the score branch does not.
        """
        # Empty baseline_metrics in payload — metric-fabrication guard is a no-op.
        # High overall_score — score branch is a no-op.
        # No _record_run_experiment(ctx) — never-ran + empty-metrics branches fire.
        raw = json.dumps({
            "verdict": "reproduced",
            "baseline_metrics": {},
            "rubric": {"overall_score": 0.8, "meets_target": True},
            "paper": {"id": "p"},
        })
        ctx = make_context(tmp_path)  # no _record_run_experiment(ctx) call

        result = _make_result(raw)
        report = build_final_report(result, ctx=ctx)
        assert report.verdict == "partial"
        assert "run_experiment never ran" in report.reproduction_summary
        # Score branch must NOT fire — score is 0.8, well above 0.5:
        assert "< 0.5" not in report.reproduction_summary

    def test_empty_baseline_metrics_in_isolation_downgrades_reproduced(self, make_context, tmp_path):
        """Symptom: a 'reproduced' verdict with a high score and a run_experiment
        call on the ledger but baseline_metrics={} in the parsed payload could
        slip past the evidence guard if only the score branch were tested.

        Pin the empty-metrics branch in isolation: run_experiment DID run
        (so the never-ran branch is a no-op), score is high (so the score
        branch is a no-op), but the parsed payload claims reproduction with
        no measured metrics.
        """
        raw = json.dumps({
            "verdict": "reproduced",
            "baseline_metrics": {},
            "rubric": {"overall_score": 0.75, "meets_target": True},
            "paper": {"id": "p"},
        })
        ctx = make_context(tmp_path)
        _record_run_experiment(ctx)  # never-ran branch is a no-op

        result = _make_result(raw)
        report = build_final_report(result, ctx=ctx)
        assert report.verdict == "partial"
        assert "no measured baseline metrics" in report.reproduction_summary
        # Score branch must NOT fire — score is 0.75, well above 0.5:
        assert "< 0.5" not in report.reproduction_summary
        # Never-ran branch must NOT fire — run_experiment is on the ledger:
        assert "run_experiment never ran" not in report.reproduction_summary


def test_render_markdown_graded_defaults_to_zero_not_leaf_count():
    """Symptom: a rubric dict with no `graded` key falsely claims full N/N coverage.

    _render_markdown used `rubric.get('graded', leaf_count)` as a default — a
    missing `graded` field rendered "N/N rubric leaves graded" (review M1 / T27).
    Verify: default is 0, not leaf_count.
    """
    from backend.agents.rlm.report import RLMFinalReport, _render_markdown
    report = RLMFinalReport(rubric={"leaf_count": 5, "overall_score": 0.5})
    md = _render_markdown(report)
    assert "0/5 rubric leaves graded" in md, (
        "missing `graded` field should default to 0, not the leaf_count"
    )
