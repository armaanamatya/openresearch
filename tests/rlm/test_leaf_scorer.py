"""Tests for backend.evals.paperbench.leaf_scorer."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from backend.evals.paperbench.leaf_scorer import (
    DEGRADED_LEAF_CEILING,
    _gather_evidence,
    amend_final_report,
    flatten_leaves,
    roll_up,
    score_reproduction,
)

# ---------------------------------------------------------------------------
# Synthetic rubric fixture
# ---------------------------------------------------------------------------

TINY_TREE = {
    "id": "root",
    "requirements": "root",
    "weight": 1,
    "sub_tasks": [
        {
            "id": "branch-a",
            "requirements": "branch a",
            "weight": 3,
            "sub_tasks": [
                {
                    "id": "leaf-a1",
                    "requirements": "leaf a1",
                    "weight": 1,
                    "sub_tasks": [],
                },
                {
                    "id": "leaf-a2",
                    "requirements": "leaf a2",
                    "weight": 1,
                    "sub_tasks": [],
                },
            ],
        },
        {
            "id": "leaf-b",
            "requirements": "leaf b",
            "weight": 1,
            "sub_tasks": [],
        },
    ],
}


# ---------------------------------------------------------------------------
# Test 1: flatten_leaves count
# ---------------------------------------------------------------------------


def test_flatten_leaves_count():
    leaves = flatten_leaves(TINY_TREE)
    ids = {leaf["id"] for leaf in leaves}
    assert len(leaves) == 3
    assert ids == {"leaf-a1", "leaf-a2", "leaf-b"}


# ---------------------------------------------------------------------------
# Test 2: roll_up weighted math
# ---------------------------------------------------------------------------


def test_roll_up_weighted_math():
    # branch-a has weight=3, leaf-b has weight=1 at root level.
    # branch-a score = (score(a1)*1 + score(a2)*1) / 2
    # root score = (branch_a_score * 3 + leaf_b_score * 1) / 4
    #
    # With: leaf-a1=1.0, leaf-a2=0.0, leaf-b=0.5
    # branch-a = (1.0 + 0.0) / 2 = 0.5
    # root = (0.5*3 + 0.5*1) / 4 = (1.5 + 0.5) / 4 = 2.0/4 = 0.5
    leaf_scores = {"leaf-a1": 1.0, "leaf-a2": 0.0, "leaf-b": 0.5}
    result = roll_up(TINY_TREE, leaf_scores)
    assert abs(result - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# Test 3: score_reproduction with mock client
# ---------------------------------------------------------------------------


class MockLlmClient:
    """Returns canned scores for the synthetic tree leaves."""

    def complete(self, *, system: str, user: str) -> str:
        return json.dumps([
            {"leaf_id": "leaf-a1", "score": 1.0, "justification": "fully satisfied"},
            {"leaf_id": "leaf-a2", "score": 0.0, "justification": "not found"},
            {"leaf_id": "leaf-b", "score": 0.5, "justification": "partial"},
        ])


def test_score_reproduction_overall():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        # Provide a minimal final_report.json so evidence gathering has something.
        # baseline_metrics is populated (an "honest" run) so the C2b degraded cap
        # does not fire — this test exercises the uncapped weighted-roll-up path.
        (run_dir / "final_report.json").write_text(
            json.dumps({
                "reproduction_summary": "test run",
                "baseline_metrics": {"accuracy": 0.5},
            }),
            encoding="utf-8",
        )

        result = score_reproduction(TINY_TREE, run_dir, MockLlmClient(), batch_size=15)

    assert abs(result["overall_score"] - 0.5) < 1e-9
    assert result["leaf_count"] == 3
    assert result["graded"] == 3
    assert result["rubric_source"] == "paperbench_bundle"
    assert len(result["leaf_scores"]) == 3
    assert result["degraded"] is False  # C2b: honest run, cap did not fire


# ---------------------------------------------------------------------------
# Test 4: rubric_source="generated" is propagated to the result dict
# ---------------------------------------------------------------------------


def test_score_reproduction_generated_rubric_source():
    """Passing rubric_source='generated' puts 'generated' in the result dict.

    Locks in: the new rubric_source keyword param of score_reproduction is
    forwarded verbatim — the arXiv self-generated-rubric path sets it so the
    report is honest about its rubric origin.
    """
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        (run_dir / "final_report.json").write_text(
            json.dumps({
                "reproduction_summary": "arxiv run",
                "baseline_metrics": {"some_metric": 1.0},
            }),
            encoding="utf-8",
        )

        result = score_reproduction(
            TINY_TREE, run_dir, MockLlmClient(), rubric_source="generated"
        )

    assert result["rubric_source"] == "generated"


# ---------------------------------------------------------------------------
# Test 5: amend_final_report keeps final_report.md consistent with the JSON
# ---------------------------------------------------------------------------


def _rlm_report_dict() -> dict:
    """A dict carrying every RLMFinalReport field (RLM-mode report shape)."""
    return {
        "paper": {"id": "2510.25013", "title": "Test Paper"},
        "verdict": "partial",
        "reproduction_summary": "ran the baseline",
        "baseline_metrics": {},
        "paper_claims": {},
        "rubric": {"overall_score": 0.999, "meets_target": True, "areas": []},
        "improvements": [],
        "primitive_trace": {"calls": 0, "by_primitive": {}},
        "cost": {"llm_usd": 0.0, "primitives": 0.0},
        "iterations": 3,
    }


def test_amend_final_report_rerenders_markdown():
    """amend_final_report rewrites final_report.md with the leaf score.

    Regression: score_run.py updated only final_report.json, so the markdown
    GET /runs/{id}/final-report serves kept showing the stale in-loop
    verify_against_rubric score. The markdown must track the JSON.
    """
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        (run_dir / "final_report.json").write_text(
            json.dumps(_rlm_report_dict()), encoding="utf-8"
        )
        (run_dir / "final_report.md").write_text(
            "# PARTIAL REPRODUCTION\n\n## Rubric Score\n\n"
            "**Overall score:** 0.999  (meets target)\n\n## Reproduction Summary\n\nx\n",
            encoding="utf-8",
        )
        amend_final_report(run_dir, {
            "overall_score": 0.42, "rubric_source": "generated",
            "leaf_count": 10, "graded": 10,
        })
        md = (run_dir / "final_report.md").read_text(encoding="utf-8")
        report = json.loads((run_dir / "final_report.json").read_text(encoding="utf-8"))

    assert report["rubric"]["overall_score"] == 0.42
    assert "0.420" in md
    assert "self-generated rubric" in md
    assert "10/10 rubric leaves graded" in md
    assert "0.999" not in md  # the stale score is gone


# ---------------------------------------------------------------------------
# P0 honesty guards (C2a / C2b / C2c) from the 2026-05-22 ship-readiness audit.
#
# C2a: _gather_evidence reads "metrics"/"paper_title" — keys that do not exist
#      in RLMFinalReport. Every RLM run was graded against evidence with no
#      metrics and no paper identity. The real keys are "baseline_metrics"/"paper".
#
# C2b: score_reproduction has no degraded-run cap. A metric-less run (the
#      RLMFinalReport.baseline_metrics={} case that survives a failed
#      run_experiment) can still be stamped with an uncapped overall_score —
#      the honest 0.35 ceiling that verify_against_rubric used to enforce was
#      not relocated here.
#
# C2c: amend_final_report hardcodes meets_target=False. A legitimate high score
#      still renders "below target"; the score block is fabricated, not computed.
# ---------------------------------------------------------------------------


def test_gather_evidence_reads_rlmfinalreport_keys(tmp_path):
    """C2a guard: _gather_evidence reads baseline_metrics/paper (the real RLM keys).

    Symptom: _gather_evidence hardcoded the snippet keys to ("reproduction_summary",
    "metrics", "verdict", "paper_title") — none of "metrics"/"paper_title" exist
    in RLMFinalReport, which writes "baseline_metrics" and "paper" (a dict).
    Every RLM run was graded against evidence with no metrics and no paper id.
    """
    # Use the existing _rlm_report_dict() helper but populate baseline_metrics so
    # we can assert their values surface in the evidence text.
    report = _rlm_report_dict()
    report["baseline_metrics"] = {"accuracy": 0.789, "f1": 0.812}
    (tmp_path / "final_report.json").write_text(json.dumps(report), encoding="utf-8")

    evidence = _gather_evidence(tmp_path)

    # baseline_metrics values must appear in the evidence the LLM grader sees
    assert "0.789" in evidence, (
        "baseline_metrics value missing from evidence — _gather_evidence is "
        "probably still reading 'metrics' instead of 'baseline_metrics'"
    )
    assert "accuracy" in evidence
    # paper identity must appear too — RLMFinalReport.paper is a dict (id/title)
    assert "2510.25013" in evidence, (
        "paper id missing from evidence — _gather_evidence is probably still "
        "reading 'paper_title' instead of 'paper'"
    )
    assert "Test Paper" in evidence


class _HighScoreLlmClient:
    """LLM stub that grades every leaf in the TINY_TREE at 0.9.

    Used to prove that even with a lenient grader, a metric-less run is capped
    at the degraded-run ceiling — the honest backstop verify_against_rubric
    used to enforce before consolidating onto score_reproduction.
    """

    def complete(self, *, system: str, user: str) -> str:
        return json.dumps([
            {"leaf_id": "leaf-a1", "score": 0.9, "justification": "looks good"},
            {"leaf_id": "leaf-a2", "score": 0.9, "justification": "looks good"},
            {"leaf_id": "leaf-b", "score": 0.9, "justification": "looks good"},
        ])


class _FailingLlmClient:
    def complete(self, *, system: str, user: str) -> str:
        raise AssertionError("degraded runs should not call the LLM grader")


def test_score_reproduction_caps_degraded_run_at_0_35(tmp_path):
    """C2b guard: a metric-less RLM run is capped at the 0.35 degraded ceiling.

    Symptom: verify_against_rubric used to enforce `min(score, 0.35)` when the
    run produced no measured metrics. That backstop was deleted in 2e1ce37 when
    verify_against_rubric was refactored to delegate to score_reproduction, but
    score_reproduction itself has no equivalent. A run that measured nothing
    can now be stamped with an uncapped LLM-graded score.

    Setup: an RLMFinalReport with baseline_metrics={} (the "run_experiment
    failed / never ran" case), grader returns 0.9 for every leaf. Without the
    cap, the rolled-up score would be 0.9. With the cap, the leaves should be
    clamped to <=0.35 and the overall score with them.
    """
    report = _rlm_report_dict()
    report["baseline_metrics"] = {}  # the metric-less degraded case
    (tmp_path / "final_report.json").write_text(json.dumps(report), encoding="utf-8")

    result = score_reproduction(TINY_TREE, tmp_path, _HighScoreLlmClient())

    # The honest degraded result is bounded by the 0.35 ceiling. The
    # implementation is allowed to short-circuit to zero without spending an LLM
    # grader call because a metric-less run has no measured reproduction signal.
    assert result["overall_score"] <= 0.35 + 1e-9, (
        f"degraded run not capped — overall_score={result['overall_score']}; "
        "the in-loop honesty backstop is missing from score_reproduction"
    )
    # Each individual leaf record must also reflect the cap so the UI's
    # "weak leaves" surface and the rubric block do not show inflated leaf
    # values that contradict the overall_score.
    for rec in result["leaf_scores"]:
        assert rec["score"] <= 0.35 + 1e-9, (
            f"leaf {rec['id']} score {rec['score']} exceeded degraded cap"
        )
    # The result must mark itself degraded so amend_final_report and downstream
    # consumers can surface why the score is capped.
    assert result.get("degraded") is True, (
        "score_reproduction should mark a metric-less run as degraded=True"
    )


def test_score_reproduction_short_circuits_metricless_degraded_runs(tmp_path):
    """Metric-less runs are honest failures; do not spend grader calls on them."""
    report = _rlm_report_dict()
    report["baseline_metrics"] = {}
    (tmp_path / "final_report.json").write_text(json.dumps(report), encoding="utf-8")

    result = score_reproduction(TINY_TREE, tmp_path, _FailingLlmClient())

    assert result["degraded"] is True
    assert result["overall_score"] == 0.0
    assert result["graded"] == 0
    assert {rec["justification"] for rec in result["leaf_scores"]} == {
        "degraded_no_metrics"
    }


def test_score_reproduction_does_not_cap_honest_run(tmp_path):
    """A run with real baseline_metrics is NOT capped — only degraded runs are.

    Paired-test invariant: the C2b cap must not fire on honest runs, otherwise
    every reproduction would be artificially capped at 0.35 and the loop signal
    would be just as broken as before, in the opposite direction.
    """
    report = _rlm_report_dict()
    report["baseline_metrics"] = {"accuracy": 0.91, "loss": 0.07}
    (tmp_path / "final_report.json").write_text(json.dumps(report), encoding="utf-8")

    result = score_reproduction(TINY_TREE, tmp_path, _HighScoreLlmClient())

    # Honest run: LLM said 0.9, no cap applies — TINY_TREE rolls 0.9 up to 0.9.
    assert result["overall_score"] == pytest.approx(0.9), (
        f"honest run was capped — overall_score={result['overall_score']}"
    )
    assert result.get("degraded") is False, (
        "score_reproduction should mark a run with real metrics as degraded=False"
    )


def test_amend_final_report_computes_meets_target_honestly(tmp_path):
    """C2c guard: meets_target reflects the real overall_score vs target_score.

    Symptom: amend_final_report hardcoded "meets_target": False, so a legitimate
    high score still rendered "✘ below target". The block was fabricated, not
    computed. The reverse failure (a low score stamped meets_target=True) was
    impossible only by coincidence of the hardcode direction.

    Setup: a score dict carrying both overall_score (0.80) and target_score (0.60)
    — what score_reproduction now returns. amend_final_report must compute
    meets_target = (0.80 >= 0.60) = True.
    """
    report = _rlm_report_dict()
    (tmp_path / "final_report.json").write_text(json.dumps(report), encoding="utf-8")

    amend_final_report(tmp_path, {
        "overall_score": 0.80,
        "rubric_source": "paperbench_bundle",
        "leaf_count": 12,
        "graded": 12,
        "target_score": 0.60,
    })
    after = json.loads((tmp_path / "final_report.json").read_text(encoding="utf-8"))

    assert after["rubric"]["overall_score"] == 0.80
    assert after["rubric"]["meets_target"] is True, (
        f"meets_target hardcoded — got {after['rubric']['meets_target']} "
        "for overall_score=0.80, target=0.60"
    )
    # The target itself must be persisted so downstream consumers can recompute.
    assert after["rubric"]["target_score"] == 0.60


def test_amend_final_report_meets_target_below(tmp_path):
    """Paired-test: a below-target score writes meets_target=False (not True)."""
    report = _rlm_report_dict()
    (tmp_path / "final_report.json").write_text(json.dumps(report), encoding="utf-8")

    amend_final_report(tmp_path, {
        "overall_score": 0.20,
        "rubric_source": "paperbench_bundle",
        "leaf_count": 12,
        "graded": 12,
        "target_score": 0.60,
    })
    after = json.loads((tmp_path / "final_report.json").read_text(encoding="utf-8"))

    assert after["rubric"]["meets_target"] is False
    assert after["rubric"]["target_score"] == 0.60


def test_amend_final_report_unknown_target_writes_none(tmp_path):
    """No target_score in the score dict → meets_target is None, never a fabricated bool.

    Symptom that motivated the fix: hardcoding False was as wrong as hardcoding
    True would be. When we genuinely do not know the target (e.g. a rubric tree
    without target_score), the honest representation is null.
    """
    report = _rlm_report_dict()
    (tmp_path / "final_report.json").write_text(json.dumps(report), encoding="utf-8")

    amend_final_report(tmp_path, {
        "overall_score": 0.42,
        "rubric_source": "generated",
        "leaf_count": 7,
        "graded": 7,
        "target_score": None,
    })
    after = json.loads((tmp_path / "final_report.json").read_text(encoding="utf-8"))

    assert after["rubric"]["meets_target"] is None
    assert after["rubric"]["target_score"] is None


def test_round_trip_real_rlmfinalreport_through_scorer(tmp_path):
    """The integration test the audit named: RLMFinalReport → score → amend.

    The C2 honesty defects survived for so long precisely because
    test_leaf_scorer.py never round-tripped a real RLMFinalReport-shaped
    final_report.json through the scorer. This test fixes that gap.

    Scenario: a degraded RLM run (run_experiment never wrote metrics) writes
    an RLMFinalReport with baseline_metrics={}. The post-run leaf scorer reads
    the report, the C2a key fix means it sees the (empty) metrics dict and
    paper identity, the C2b degraded cap clamps the lenient LLM grade, and
    the C2c meets_target computation reflects the capped score vs. the target.
    All three defects would re-emerge if this test were ever skipped.
    """
    # Build a real RLMFinalReport in the (degraded) shape and write it via the
    # actual production writer so the on-disk schema is exactly what RLM runs
    # produce — not a hand-rolled approximation.
    from backend.agents.rlm.report import RLMFinalReport, write_final_report_rlm

    report = RLMFinalReport(
        paper={"id": "2510.25013", "title": "Test Paper"},
        verdict="reproduced",  # over-claimed; reconcile_verdict_with_score will downgrade
        reproduction_summary="ran the baseline (no metrics measured)",
        baseline_metrics={},  # the degraded case
        paper_claims={},
        rubric={"overall_score": 0.0, "meets_target": False, "areas": []},
        improvements=[],
        primitive_trace={"calls": 1, "by_primitive": {"understand_section": 1}},
        cost={"llm_usd": 0.01, "primitives": 0.0},
        iterations=3,
    )
    write_final_report_rlm(report, tmp_path)

    # Rubric tree carries target_score=0.60 (PaperBench bundles do).
    rubric_with_target = {**TINY_TREE, "target_score": 0.60}

    # Score with a lenient LLM (would have stamped 0.9 without the C2b cap).
    scored = score_reproduction(rubric_with_target, tmp_path, _HighScoreLlmClient())

    # C2b — degraded run capped at the ceiling.
    assert scored["degraded"] is True
    assert scored["overall_score"] <= DEGRADED_LEAF_CEILING + 1e-9
    # C2c — target_score threaded through.
    assert scored["target_score"] == 0.60

    # Amend the report. This exercises the markdown re-render, the verdict
    # reconciliation (verdict="reproduced" at score<=0.35 → "failed"), and the
    # C2c meets_target computation.
    amend_final_report(tmp_path, scored)

    after = json.loads((tmp_path / "final_report.json").read_text(encoding="utf-8"))
    md = (tmp_path / "final_report.md").read_text(encoding="utf-8")

    # C2b/c: rubric block reflects the capped overall_score and computed meets_target.
    assert after["rubric"]["overall_score"] <= DEGRADED_LEAF_CEILING + 1e-9
    assert after["rubric"]["target_score"] == 0.60
    assert after["rubric"]["meets_target"] is False  # 0.35 < 0.60
    assert after["rubric"]["degraded"] is True

    # Pre-existing reconcile_verdict_with_score honesty work (2e1ce37): a score
    # A metric-less run is now short-circuited to zero, so the over-claimed
    # "reproduced" verdict is downgraded all the way to "failed".
    assert after["verdict"] == "failed"

    # Markdown's "Overall score" line reflects the deterministic degraded score.
    assert "0.000" in md, (
        "expected deterministic degraded overall_score (0.000) in markdown"
    )
    # And the markdown banner reflects the reconciled verdict.
    assert "REPRODUCTION FAILED" in md


def test_amend_final_report_leaves_non_rlm_markdown_untouched():
    """A non-RLM final_report.json must not have its markdown clobbered — the
    RLM markdown renderer only applies to RLM-shaped reports."""
    stale_md = "# Some SDK report\n\nOverall: 0.7\n"
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        (run_dir / "final_report.json").write_text(
            json.dumps({"sdk_field": 1, "another": 2}), encoding="utf-8"
        )
        (run_dir / "final_report.md").write_text(stale_md, encoding="utf-8")
        amend_final_report(run_dir, {
            "overall_score": 0.3, "rubric_source": "paperbench_bundle",
            "leaf_count": 5, "graded": 5,
        })
        md = (run_dir / "final_report.md").read_text(encoding="utf-8")

    assert md == stale_md  # untouched — the RLM-shape guard prevented a re-render


def test_amend_final_report_preserves_in_loop_areas(tmp_path):
    """Symptom: the markdown areas table renders empty after leaf-score amendment.

    amend_final_report used to overwrite report["rubric"] wholesale, dropping the
    in-loop tree-rubric `areas` list (review M2 / T5).  Verify: a seeded areas
    list survives the amendment intact.
    """
    import json
    from backend.evals.paperbench.leaf_scorer import amend_final_report

    report = {
        "verdict": "reproduced",
        "baseline_metrics": {"x": 1.0},
        "rubric": {
            "areas": [{"name": "code", "score": 0.85, "notes": "good"}],
        },
    }
    (tmp_path / "final_report.json").write_text(json.dumps(report), encoding="utf-8")

    amend_final_report(
        tmp_path,
        {
            "overall_score": 0.9,
            "leaf_count": 1,
            "graded": 1,
            "rubric_source": "paperbench_bundle",
            "leaf_scores": [{"id": "L1", "score": 0.9, "justification": "ok"}],
            "degraded": False,
            "target_score": 0.7,
        },
    )

    out = json.loads((tmp_path / "final_report.json").read_text(encoding="utf-8"))
    assert out["rubric"]["meets_target"] is True
    assert out["rubric"]["areas"] == [
        {"name": "code", "score": 0.85, "notes": "good"}
    ]


def test_amend_final_report_rerenders_when_report_lacks_optional_schema_fields(tmp_path):
    """Symptom: a schema addition to RLMFinalReport silently broke markdown re-render.

    The prior guard used `RLMFinalReport.model_fields.issubset(report.keys())`
    — when the schema grew (T21 added primitive_provider + degraded), every
    existing on-disk report fell out of the subset and the markdown was never
    re-rendered (regression discovered by test_amend_final_report_rerenders_markdown).
    Verify: a report dict missing the new optional fields still gets re-rendered.
    """
    import json
    from backend.evals.paperbench.leaf_scorer import amend_final_report

    # A minimal RLM-mode report missing the T21 fields (primitive_provider, degraded).
    minimal = {
        "verdict": "reproduced",
        "baseline_metrics": {},
        "paper": {"id": "x", "title": "Test"},
        "paper_claims": {},
        "rubric": {"overall_score": 0.0, "meets_target": False, "areas": []},
        "improvements": [],
        "primitive_trace": {"calls": 0, "by_primitive": {}},
        "cost": {"llm_usd": 0.0, "primitives": 0.0},
        "iterations": 0,
        "reproduction_summary": "",
    }
    (tmp_path / "final_report.json").write_text(json.dumps(minimal), encoding="utf-8")
    (tmp_path / "final_report.md").write_text("# REPRODUCED\nold content\n", encoding="utf-8")

    amend_final_report(tmp_path, {
        "overall_score": 0.0, "leaf_count": 10, "graded": 10,
        "rubric_source": "paperbench_bundle", "leaf_scores": [],
        "degraded": False, "target_score": 0.0,
    })

    md = (tmp_path / "final_report.md").read_text(encoding="utf-8")
    # The re-render must have run — the original "old content" line is gone.
    assert "old content" not in md, (
        "markdown re-render did not run on a report missing optional schema fields"
    )


def test_is_degraded_run_caps_failed_verdict_even_with_metrics(tmp_path):
    """Symptom: a verdict='failed' run with metrics escapes the 0.35 honesty cap.

    _is_degraded_run only checked baseline_metrics, so a run that measured
    metrics but errored downstream (verdict='failed') was not detected as
    degraded (T4 plan, review C2b).  Verify: such a run IS classified degraded.
    """
    import json
    from backend.evals.paperbench.leaf_scorer import _is_degraded_run

    report = {
        "verdict": "failed",
        "baseline_metrics": {"mean_reward": 1.0},  # non-empty metrics but verdict=failed
    }
    (tmp_path / "final_report.json").write_text(json.dumps(report), encoding="utf-8")

    assert _is_degraded_run(tmp_path) is True


def test_leaf_scorer_parses_response_with_prose_braces():
    """Symptom: prose braces in the LLM response burn a retry.

    First-`{`-to-last-`}` slicing over-grabs on prose braces (review M3 / T26).
    Verify: a response like 'Here {x} is the array: [{...}]' parses on first try.
    """
    from backend.evals.paperbench.leaf_scorer import _parse_batch_response

    raw = 'Note: the {answer} is below.\n[{"leaf_id":"L1","score":0.8,"justification":"x"}]'
    batch = [{"id": "L1"}]
    out = _parse_batch_response(raw, batch)
    assert out[0]["score"] == 0.8
    assert out[0]["justification"] == "x"
