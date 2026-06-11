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

from backend.agents.rlm.report import (
    RLMFinalReport,
    _latest_successful_experiment_record,
    _metric_provenance_enabled,
    build_final_report,
    write_final_report_rlm,
)
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

    def test_calibration_recompute_is_opt_in(self, tmp_path, monkeypatch):
        """Writing a non-failed report must not rewrite global calibration by default."""
        import backend.services.pricing.calibration as calibration

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        report = self._build_report()
        monkeypatch.delenv("REPROLAB_UPDATE_CALIBRATION", raising=False)

        def fail_recompute(*args, **kwargs):
            raise AssertionError("calibration recompute should be opt-in")

        monkeypatch.setattr(calibration, "recompute_calibration", fail_recompute)

        write_final_report_rlm(report, project_dir)

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


# ---------------------------------------------------------------------------
# C2c second pass — unscored default rubric must read as null, not fabricated
#
# Symptom uncovered 2026-05-22: two RunPod reproductions died at the first
# Anthropic call (no API credit). Their final_report.json carried
# rubric.meets_target=false despite never being scored — a fabricated boolean
# the audit's original C2c finding had not traced. The C2c amend-write fix
# only touched the post-scoring path; the pre-scoring default (in
# RLMFinalReport.rubric default_factory + _HONEST_DEFAULTS["rubric"]) still
# hardcoded {"overall_score": 0.0, "meets_target": False}, which persisted
# verbatim to disk when the run failed before reaching the scorer.
#
# These tests pin the unscored honesty contract: a brand-new RLMFinalReport
# rubric reads as null on every field that requires scoring, and the markdown
# renderer surfaces "not scored" instead of "below target" / 0.000.
# ---------------------------------------------------------------------------


def test_rlm_final_report_default_rubric_is_unscored_null():
    """A fresh RLMFinalReport carries null score / target / degraded / meets_target.

    Persisted via write_final_report_rlm on a no-score path (run died before
    amend_final_report), this is what readers actually see on disk. None is
    honest; 0.0 + False is a fabricated claim of "scored zero, below target."
    """
    report = RLMFinalReport()
    assert report.rubric["overall_score"] is None
    assert report.rubric["meets_target"] is None
    assert report.rubric["target_score"] is None
    assert report.rubric["degraded"] is None
    assert report.rubric["areas"] == []


def test_render_markdown_unscored_run_says_not_scored(tmp_path):
    """An unscored run renders "not scored", never "0.000 (✘ below target)".

    The 2026-05-22 RunPod failures wrote a rubric block of
    {"overall_score": 0.0, "meets_target": False, ...} to disk. The markdown
    renderer turned that into "**Overall score:** 0.000  (✘ below target)" —
    a precise-looking number for a score that does not exist. With the C2c
    second-pass fix, an unscored rubric (None on overall_score) must instead
    render "not scored".
    """
    from backend.agents.rlm.report import RLMFinalReport, _render_markdown
    report = RLMFinalReport(verdict="failed", iterations=0)
    md = _render_markdown(report)

    assert "not scored" in md, (
        "unscored run must render 'not scored', not a fabricated overall_score "
        "+ below-target verdict"
    )
    # Isolate the rubric section ("## Rubric Score" → next "## "); the assertion
    # must not be confused by the cost table's $0.000000 entries.
    rubric_start = md.index("## Rubric Score")
    rubric_end_search = md.find("\n## ", rubric_start + 1)
    rubric_section = md[rubric_start:rubric_end_search] if rubric_end_search != -1 else md[rubric_start:]
    assert "0.000" not in rubric_section, (
        "unscored run must not render a 0.000 overall_score — that's a "
        f"claim of 'scored zero' for a run that was never graded.\n"
        f"Rubric section was:\n{rubric_section}"
    )
    assert "below target" not in rubric_section, (
        "unscored run must not claim 'below target' — the target was never "
        "compared against, because there was no score"
    )


def test_render_markdown_scored_run_with_no_target_says_no_target_set():
    """A genuinely-scored run whose rubric has no target_score must render
    "no target set", not "✘ below target".

    This was the original audit C2c — amend_final_report wrote
    meets_target=False when target was missing, flipping a legitimate high
    score to "below target". The fix wrote None instead; the renderer must
    render that None honestly.
    """
    from backend.agents.rlm.report import RLMFinalReport, _render_markdown
    report = RLMFinalReport(
        rubric={
            "overall_score": 0.80,
            "meets_target": None,  # target_score was None → meets_target None
            "target_score": None,
            "leaf_count": 12,
            "graded": 12,
        },
    )
    md = _render_markdown(report)

    assert "0.800" in md
    assert "no target set" in md, (
        "scored run with no target should render 'no target set', not "
        "'below target' which would be a fabricated comparison"
    )
    assert "below target" not in md
    assert "meets target" not in md  # the ✔ variant


def test_amend_final_report_overwrites_unscored_defaults_with_real_scores(tmp_path):
    """The honest-null defaults must not block legitimate scoring.

    Round-trip: write_final_report_rlm(report-with-null-rubric) → on-disk file
    with null fields → amend_final_report(score-dict-with-real-values) →
    on-disk file now has real numeric overall_score / meets_target / etc.
    Without this guarantee, the C2c second-pass fix would have created a
    new bug where successful scoring couldn't overwrite the defaults.
    """
    from backend.agents.rlm.report import RLMFinalReport, write_final_report_rlm
    from backend.evals.paperbench.leaf_scorer import amend_final_report

    # An unscored report — what the failing-early path writes.
    report = RLMFinalReport(
        paper={"id": "test", "title": "Test"},
        verdict="failed",
        iterations=0,
    )
    write_final_report_rlm(report, tmp_path)

    pre = json.loads((tmp_path / "final_report.json").read_text(encoding="utf-8"))
    assert pre["rubric"]["overall_score"] is None
    assert pre["rubric"]["meets_target"] is None

    # The scorer ran later (in some other invocation) and now amends the report.
    amend_final_report(tmp_path, {
        "overall_score": 0.72,
        "rubric_source": "paperbench_bundle",
        "leaf_count": 10,
        "graded": 10,
        "target_score": 0.60,
        "degraded": False,
    })

    post = json.loads((tmp_path / "final_report.json").read_text(encoding="utf-8"))
    assert post["rubric"]["overall_score"] == 0.72
    assert post["rubric"]["meets_target"] is True  # 0.72 >= 0.60
    assert post["rubric"]["target_score"] == 0.60
    assert post["rubric"]["degraded"] is False


class TestMetricProvenance:
    """P3 §5b — baseline_metrics is PROJECTED from the canonical experiment
    artifact (mirrors RDR), not the root's self-attested numbers. The existing
    honesty guard is preserved as the fallback when no successful record exists."""

    def test_projected_from_canonical_record(self, make_context, tmp_path):
        ctx = _record_run_experiment(make_context(tmp_path))
        # Canonical record measured 0.81; the root self-attests 0.92.
        (ctx.project_dir / "experiment_runs.jsonl").write_text(
            json.dumps({
                "success": True,
                "experiment_run_id": "prj-r1",
                "metrics_sha256": "abc123",
                "metrics": {"accuracy": 0.81},
            }) + "\n",
            encoding="utf-8",
        )
        root_dict = {**_BASE_REPORT_DICT, "baseline_metrics": {"accuracy": 0.92}}
        report = build_final_report(_make_result(json.dumps(root_dict)), ctx=ctx)
        assert report.baseline_metrics == {"accuracy": 0.81}   # projected from the artifact
        assert report.reported_metrics == {"accuracy": 0.92}   # root claim preserved, non-authoritative
        assert report.experiment_run_id == "prj-r1"
        assert report.metrics_sha256 == "abc123"

    def test_hatch_off_keeps_root_metrics(self, make_context, tmp_path, monkeypatch):
        monkeypatch.setenv("REPROLAB_METRIC_PROVENANCE", "false")
        ctx = _record_run_experiment(make_context(tmp_path))
        (ctx.project_dir / "experiment_runs.jsonl").write_text(
            json.dumps({"success": True, "experiment_run_id": "r1", "metrics": {"accuracy": 0.81}}) + "\n",
            encoding="utf-8",
        )
        report = build_final_report(_make_result(json.dumps(_BASE_REPORT_DICT)), ctx=ctx)
        # Hatch off → fallback; run_experiment backed via the cost ledger → root's 0.92 kept.
        assert report.baseline_metrics == {"accuracy": 0.92}

    def test_no_record_preserves_honesty_guard(self, make_context, tmp_path):
        """No successful experiment record → the existing honesty guard still
        drops root metrics that run_experiment never backed (no regression)."""
        ctx = make_context(tmp_path)  # NOT _record_run_experiment → run_experiment unbacked
        report = build_final_report(_make_result(json.dumps(_BASE_REPORT_DICT)), ctx=ctx)
        assert report.baseline_metrics == {}

    def test_latest_successful_record_selection(self, tmp_path):
        (tmp_path / "experiment_runs.jsonl").write_text(
            "\n".join([
                json.dumps({"success": True, "experiment_run_id": "r1", "metrics": {"a": 1}}),
                json.dumps({"success": False, "experiment_run_id": "r2", "metrics": {"a": 2}}),
                json.dumps({"success": True, "experiment_run_id": "r3", "metrics": {"a": 3}}),
            ]) + "\n",
            encoding="utf-8",
        )
        rec = _latest_successful_experiment_record(tmp_path)
        assert rec is not None and rec["experiment_run_id"] == "r3" and rec["metrics"] == {"a": 3}

    def test_latest_successful_record_none_when_absent(self, tmp_path):
        assert _latest_successful_experiment_record(tmp_path) is None

    def test_metric_provenance_hatch_parsing(self, monkeypatch):
        monkeypatch.delenv("REPROLAB_METRIC_PROVENANCE", raising=False)
        assert _metric_provenance_enabled() is True
        for off in ("false", "0", "no", "off"):
            monkeypatch.setenv("REPROLAB_METRIC_PROVENANCE", off)
            assert _metric_provenance_enabled() is False, off


class TestExperimentArmStamp:
    """A/B observability (2026-06-11): write_final_report_rlm labels every
    report with its with/without-BES arm so paired runs are explicit."""

    def test_report_carries_control_stamp_by_default(self, tmp_path, monkeypatch):
        from types import SimpleNamespace
        monkeypatch.setattr(
            "backend.config.get_settings",
            lambda: SimpleNamespace(
                bes_enabled=False, bes_candidates_per_cluster=1,
                bes_select_metric="cluster_score",
            ),
        )
        monkeypatch.delenv("REPROLAB_AB_ARM", raising=False)
        monkeypatch.delenv("REPROLAB_AB_PAIR_ID", raising=False)
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        report = RLMFinalReport(**_BASE_REPORT_DICT)

        json_path, _ = write_final_report_rlm(report, project_dir)
        data = json.loads(json_path.read_text())

        assert data["experiment_arm"]["arm"] == "control"
        assert data["experiment_arm"]["bes"]["enabled"] is False

    def test_report_carries_bes_stamp_and_pool(self, tmp_path, monkeypatch):
        from types import SimpleNamespace
        monkeypatch.setattr(
            "backend.config.get_settings",
            lambda: SimpleNamespace(
                bes_enabled=True, bes_candidates_per_cluster=2,
                bes_select_metric="cluster_score",
            ),
        )
        monkeypatch.setenv("REPROLAB_AB_PAIR_ID", "allcnn-ab-1")
        project_dir = tmp_path / "project"
        (project_dir / "rlm_state").mkdir(parents=True)
        (project_dir / "rlm_state" / "bes_candidates.json").write_text(json.dumps({
            "winner": "rlm_impl#1",
            "candidates": [
                {"candidate_id": "rlm_impl#0", "ok": True, "score": 0.3},
                {"candidate_id": "rlm_impl#1", "ok": True, "score": 0.7},
            ],
        }))
        report = RLMFinalReport(**_BASE_REPORT_DICT)

        json_path, _ = write_final_report_rlm(report, project_dir)
        data = json.loads(json_path.read_text())

        stamp = data["experiment_arm"]
        assert stamp["arm"] == "bes"
        assert stamp["ab_pair_id"] == "allcnn-ab-1"
        assert stamp["bes"]["winner"] == "rlm_impl#1"
        assert len(stamp["bes"]["pool"]) == 2


# ---------------------------------------------------------------------------
# Authoritative rubric override + verdict floor (2026-06-11 OmniZip stale-REPL
# report-assembly bug: meets_target=False shipped beside overall 0.656 ≥ 0.6)
# ---------------------------------------------------------------------------

def _stale_partial_report(**overrides) -> RLMFinalReport:
    base = dict(_BASE_REPORT_DICT)
    base["verdict"] = "partial"
    base["rubric"] = {
        "overall_score": 0.656,
        "target_score": 0.6,
        "meets_target": False,  # stale root-supplied value
        "areas": [],
    }
    base.update(overrides)
    return RLMFinalReport(**base)


def _write_eval(project_dir: Path, **extra) -> None:
    payload = {
        "overall_score": 0.65632,
        "target_score": 0.6,
        "meets_target": True,
        "leaf_count": 24,
        "graded": 22,
    }
    payload.update(extra)
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "rubric_evaluation.json").write_text(json.dumps(payload))


class TestAuthoritativeRubricOverride:
    def test_eval_overrides_stale_scalars_and_floors_verdict(self, tmp_path):
        project_dir = tmp_path / "proj"
        _write_eval(project_dir)
        report = _stale_partial_report()

        json_path, _ = write_final_report_rlm(report, project_dir)
        data = json.loads(json_path.read_text())

        assert data["rubric"]["overall_score"] == pytest.approx(0.65632)
        assert data["rubric"]["meets_target"] is True
        assert data["verdict"] == "reproduced"

    def test_hard_stop_keeps_partial_cap(self, tmp_path):
        project_dir = tmp_path / "proj"
        _write_eval(project_dir)
        report = _stale_partial_report(
            stop_reason={"kind": "wall_clock_watchdog", "detail": "hard stop"},
        )

        json_path, _ = write_final_report_rlm(report, project_dir)
        data = json.loads(json_path.read_text())

        # Scalars still corrected from the authoritative eval…
        assert data["rubric"]["meets_target"] is True
        # …but a hard-stopped run can never be floored up to "reproduced".
        assert data["verdict"] == "partial"

    def test_degraded_run_keeps_partial_cap(self, tmp_path):
        project_dir = tmp_path / "proj"
        _write_eval(project_dir)
        report = _stale_partial_report(degraded=True)

        json_path, _ = write_final_report_rlm(report, project_dir)
        data = json.loads(json_path.read_text())

        assert data["verdict"] == "partial"

    def test_no_eval_file_leaves_report_untouched(self, tmp_path):
        project_dir = tmp_path / "proj"
        report = _stale_partial_report()

        json_path, _ = write_final_report_rlm(report, project_dir)
        data = json.loads(json_path.read_text())

        assert data["rubric"]["meets_target"] is False
        assert data["verdict"] == "partial"

    def test_failed_verdict_is_never_floored(self, tmp_path):
        project_dir = tmp_path / "proj"
        _write_eval(project_dir)
        report = _stale_partial_report()
        report.verdict = "failed"

        json_path, _ = write_final_report_rlm(report, project_dir)
        data = json.loads(json_path.read_text())

        assert data["verdict"] == "failed"

    def test_none_defaults_still_filled_regression(self, tmp_path):
        # The original 2026-06-09 behavior: a hard-stopped run shipping the
        # unscored None defaults gets them filled from the eval file.
        project_dir = tmp_path / "proj"
        _write_eval(project_dir)
        base = dict(_BASE_REPORT_DICT)
        base["verdict"] = "partial"
        base["rubric"] = {
            "overall_score": None,
            "meets_target": None,
            "target_score": None,
            "degraded": None,
            "areas": [],
        }
        report = RLMFinalReport(**base)

        json_path, _ = write_final_report_rlm(report, project_dir)
        data = json.loads(json_path.read_text())

        assert data["rubric"]["overall_score"] == pytest.approx(0.65632)
        assert data["rubric"]["target_score"] == 0.6
        assert data["rubric"]["meets_target"] is True
