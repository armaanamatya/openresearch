"""TDD tests for three scoring-fidelity correctness fixes (actor-critic harness audit).

Task 1 — Verdict/score consistency at the write_final_report_rlm chokepoint.
Task 2 — meets_target population in the written final_report.json.
Task 3 — OPENRESEARCH_EVIDENCE_GATE split-default collision: per-leaf gate gets
          its own var (OPENRESEARCH_LEAF_EVIDENCE_GATE, default OFF), leaving
          the verdict gate on OPENRESEARCH_EVIDENCE_GATE (default ON, unchanged).
"""

from __future__ import annotations

import json
from pathlib import Path


from backend.agents.rlm.report import RLMFinalReport, write_final_report_rlm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_success_experiment(run_dir: Path, metrics: dict | None = None) -> None:
    """Write an experiment_runs.jsonl with a cleanly-successful row so the
    evidence gate in write_final_report_rlm does NOT fire."""
    if metrics is None:
        metrics = {"acc": 0.9}
    row = json.dumps(
        {
            "success": True,
            "metrics": metrics,
            "experiment_run_id": "test-run-001",
        }
    )
    (run_dir / "experiment_runs.jsonl").write_text(row + "\n", encoding="utf-8")


def _write_and_read(
    report: RLMFinalReport,
    run_dir: Path,
    *,
    run_experiment_calls: int | None = 1,
    run_experiment_ok_calls: int | None = 1,
    run_experiment_partial_timeout_calls: int | None = 0,
) -> dict:
    """Call write_final_report_rlm, then return the written JSON as a dict."""
    write_final_report_rlm(
        report,
        run_dir,
        run_experiment_calls=run_experiment_calls,
        run_experiment_ok_calls=run_experiment_ok_calls,
        run_experiment_partial_timeout_calls=run_experiment_partial_timeout_calls,
    )
    return json.loads((run_dir / "final_report.json").read_text(encoding="utf-8"))


# ===========================================================================
# Task 1 — Verdict/score consistency at the write chokepoint
# ===========================================================================


class TestVerdictScoreConsistencyAtWriteChokepoint:
    """write_final_report_rlm must cap a 'reproduced' verdict to 'partial'
    (or 'failed') when the authoritative overall_score is below the ceiling
    for 'reproduced' (0.60).

    Symptom: pb_ftrl_1779413937 shipped verdict=reproduced at score 0.0.
    """

    def test_reproduced_at_zero_score_downgraded_to_failed(self, tmp_path):
        """verdict=reproduced + score=0.0 → final written verdict is 'failed'
        (score < partial floor 0.15)."""
        _write_success_experiment(tmp_path)
        report = RLMFinalReport(
            verdict="reproduced",
            rubric={
                "overall_score": 0.0,
                "target_score": 0.6,
                "meets_target": None,
                "areas": [],
            },
            baseline_metrics={"acc": 0.9},
        )
        d = _write_and_read(report, tmp_path)
        assert d["verdict"] == "failed", (
            f"Expected 'failed' (score 0.0 is below partial floor), got {d['verdict']!r}. "
            "Symptom: pb_ftrl_1779413937 shipped verdict=reproduced at score 0.0."
        )

    def test_reproduced_at_partial_score_downgraded_to_partial(self, tmp_path):
        """verdict=reproduced + score=0.30 → written verdict is 'partial'
        (score is above partial floor 0.15 but below reproduced threshold 0.60)."""
        _write_success_experiment(tmp_path)
        report = RLMFinalReport(
            verdict="reproduced",
            rubric={
                "overall_score": 0.30,
                "target_score": 0.6,
                "meets_target": None,
                "areas": [],
            },
            baseline_metrics={"acc": 0.9},
        )
        d = _write_and_read(report, tmp_path)
        assert d["verdict"] == "partial", (
            f"Expected 'partial' (score 0.30 is below reproduced threshold), got {d['verdict']!r}."
        )

    def test_reproduced_at_sufficient_score_unchanged(self, tmp_path):
        """verdict=reproduced + score=0.70 → written verdict remains 'reproduced'
        (score is at or above the 0.60 reproduced threshold)."""
        _write_success_experiment(tmp_path)
        report = RLMFinalReport(
            verdict="reproduced",
            rubric={
                "overall_score": 0.70,
                "target_score": 0.6,
                "meets_target": True,
                "areas": [],
            },
            baseline_metrics={"acc": 0.9},
        )
        d = _write_and_read(report, tmp_path)
        assert d["verdict"] == "reproduced", (
            f"Expected 'reproduced' (score 0.70 ≥ 0.60), got {d['verdict']!r}."
        )

    def test_partial_at_below_floor_downgraded_to_failed(self, tmp_path):
        """verdict=partial + score=0.05 → written verdict is 'failed'
        (below partial floor 0.15)."""
        _write_success_experiment(tmp_path)
        report = RLMFinalReport(
            verdict="partial",
            rubric={
                "overall_score": 0.05,
                "target_score": 0.6,
                "meets_target": False,
                "areas": [],
            },
            baseline_metrics={"acc": 0.9},
        )
        d = _write_and_read(report, tmp_path)
        assert d["verdict"] == "failed", (
            f"Expected 'failed' (score 0.05 < partial floor 0.15), got {d['verdict']!r}."
        )

    def test_failed_at_high_score_not_upgraded(self, tmp_path):
        """reconcile_verdict_with_score only downgrades, never upgrades.
        A 'failed' verdict must remain 'failed' even at a high score."""
        _write_success_experiment(tmp_path)
        report = RLMFinalReport(
            verdict="failed",
            rubric={
                "overall_score": 0.95,
                "target_score": 0.6,
                "meets_target": True,
                "areas": [],
            },
            baseline_metrics={"acc": 0.9},
        )
        d = _write_and_read(report, tmp_path)
        assert d["verdict"] == "failed", (
            f"Expected 'failed' (never upgrade), got {d['verdict']!r}."
        )

    def test_none_score_does_not_crash(self, tmp_path):
        """A rubric with overall_score=None (unscored run) must not crash the
        write path — fail-soft, verdict unchanged."""
        _write_success_experiment(tmp_path)
        report = RLMFinalReport(
            verdict="partial",
            rubric={
                "overall_score": None,
                "target_score": None,
                "meets_target": None,
                "areas": [],
            },
            baseline_metrics={"acc": 0.9},
        )
        # Must not raise
        d = _write_and_read(report, tmp_path)
        # With no score we cannot meaningfully enforce a ceiling — the key point
        # is that the write succeeded (no exception) and verdict is still a string.
        assert isinstance(d.get("verdict"), str)


# ===========================================================================
# Task 2 — meets_target population in the written JSON
# ===========================================================================


class TestMeetsTargetPopulation:
    """The FINAL written final_report.json must have meets_target computed from
    the authoritative score vs target — a bool, never None when both are present.

    Symptom: 100% of an old report corpus had meets_target=None.
    """

    def test_meets_target_false_when_score_below_target(self, tmp_path):
        """score 0.40 < target 0.60 → meets_target is False (a bool)."""
        _write_success_experiment(tmp_path)
        report = RLMFinalReport(
            verdict="partial",
            rubric={
                "overall_score": 0.40,
                "target_score": 0.60,
                "meets_target": None,   # starts as None — must be computed
                "areas": [],
            },
            baseline_metrics={"acc": 0.9},
        )
        d = _write_and_read(report, tmp_path)
        rubric = d.get("rubric", {})
        mt = rubric.get("meets_target")
        assert mt is False, (
            f"Expected meets_target=False (0.40 < 0.60), got {mt!r}. "
            "Old corpus symptom: meets_target=None even with real scores."
        )

    def test_meets_target_true_when_score_meets_target(self, tmp_path):
        """score 0.70 >= target 0.60 → meets_target is True (a bool)."""
        _write_success_experiment(tmp_path)
        report = RLMFinalReport(
            verdict="reproduced",
            rubric={
                "overall_score": 0.70,
                "target_score": 0.60,
                "meets_target": None,
                "areas": [],
            },
            baseline_metrics={"acc": 0.9},
        )
        d = _write_and_read(report, tmp_path)
        rubric = d.get("rubric", {})
        mt = rubric.get("meets_target")
        assert mt is True, (
            f"Expected meets_target=True (0.70 >= 0.60), got {mt!r}."
        )

    def test_meets_target_none_when_no_target_score(self, tmp_path):
        """No target_score → meets_target must remain None (honest 'no target set').
        Never fabricate False."""
        _write_success_experiment(tmp_path)
        report = RLMFinalReport(
            verdict="partial",
            rubric={
                "overall_score": 0.40,
                "target_score": None,   # no target defined
                "meets_target": None,
                "areas": [],
            },
            baseline_metrics={"acc": 0.9},
        )
        d = _write_and_read(report, tmp_path)
        rubric = d.get("rubric", {})
        mt = rubric.get("meets_target")
        assert mt is None, (
            f"Expected meets_target=None (no target score defined), got {mt!r}."
        )

    def test_meets_target_none_when_no_overall_score(self, tmp_path):
        """No overall_score → meets_target must remain None (honest 'not scored').
        Never fabricate False."""
        report = RLMFinalReport(
            verdict="failed",
            rubric={
                "overall_score": None,
                "target_score": 0.60,
                "meets_target": None,
                "areas": [],
            },
        )
        d = _write_and_read(report, tmp_path, run_experiment_calls=0, run_experiment_ok_calls=0)
        rubric = d.get("rubric", {})
        mt = rubric.get("meets_target")
        assert mt is None, (
            f"Expected meets_target=None (no overall_score), got {mt!r}."
        )

    def test_meets_target_is_a_bool_type(self, tmp_path):
        """meets_target must be a Python bool (True/False), not a truthy/falsy non-bool."""
        _write_success_experiment(tmp_path)
        report = RLMFinalReport(
            verdict="partial",
            rubric={
                "overall_score": 0.5,
                "target_score": 0.6,
                "meets_target": None,
                "areas": [],
            },
            baseline_metrics={"acc": 0.9},
        )
        d = _write_and_read(report, tmp_path)
        rubric = d.get("rubric", {})
        mt = rubric.get("meets_target")
        # JSON deserializes True/False as Python bool; None stays None
        if mt is not None:
            assert isinstance(mt, bool), f"meets_target must be bool, got {type(mt).__name__}"


# ===========================================================================
# Task 3 — OPENRESEARCH_EVIDENCE_GATE split-default collision
# ===========================================================================


class TestEvidenceGateSplitDefault:
    """The verdict gate (report.py::_apply_evidence_gate) must remain on
    OPENRESEARCH_EVIDENCE_GATE (default ON).  The per-leaf veto gate
    (evidence_gate.py/leaf_scorer.py) must move to OPENRESEARCH_LEAF_EVIDENCE_GATE
    (default OFF) so the two behaviors can be controlled independently.

    Confirmed collision: report.py read default='1' (verdict gate ON), while
    evidence_gate.py and leaf_scorer.py read default='' (leaf veto OFF) —
    same env var, opposite defaults.
    """

    def test_verdict_gate_default_on_without_env_var(self, tmp_path, monkeypatch):
        """The verdict gate must be ON by default — a success-ish verdict with
        no experiment evidence is downgraded without any env var set."""
        monkeypatch.delenv("OPENRESEARCH_EVIDENCE_GATE", raising=False)
        monkeypatch.delenv("OPENRESEARCH_LEAF_EVIDENCE_GATE", raising=False)
        report = RLMFinalReport(
            verdict="partial",
            rubric={"overall_score": 0.5, "target_score": 0.6, "meets_target": False, "areas": []},
            baseline_metrics={"acc": 0.9},
        )
        # No experiment evidence on disk → evidence gate fires by default
        d = _write_and_read(
            report, tmp_path,
            run_experiment_calls=0, run_experiment_ok_calls=0,
        )
        assert d["verdict"] == "failed", (
            f"Verdict gate must be ON by default. Got {d['verdict']!r}. "
            "OPENRESEARCH_EVIDENCE_GATE=<unset> must default to gate-ON."
        )

    def test_verdict_gate_disabled_by_explicit_zero(self, tmp_path, monkeypatch):
        """Setting OPENRESEARCH_EVIDENCE_GATE=0 disables the verdict gate."""
        monkeypatch.setenv("OPENRESEARCH_EVIDENCE_GATE", "0")
        report = RLMFinalReport(
            verdict="partial",
            rubric={"overall_score": 0.5, "target_score": 0.6, "meets_target": False, "areas": []},
            baseline_metrics={"acc": 0.9},
        )
        # No experiment evidence; but the gate is off → verdict not downgraded
        d = _write_and_read(
            report, tmp_path,
            run_experiment_calls=0, run_experiment_ok_calls=0,
        )
        # With gate off, the verdict stays as partial (not downgraded to failed)
        assert d["verdict"] == "partial", (
            f"Verdict gate must be OFF when OPENRESEARCH_EVIDENCE_GATE=0. "
            f"Got {d['verdict']!r}."
        )

    def test_leaf_gate_default_off(self, monkeypatch):
        """The per-leaf evidence gate must be OFF by default —
        evidence_gate_enabled() returns False when the new var is unset."""
        monkeypatch.delenv("OPENRESEARCH_LEAF_EVIDENCE_GATE", raising=False)
        monkeypatch.delenv("OPENRESEARCH_EVIDENCE_GATE", raising=False)
        from backend.agents.rlm.evidence_gate import evidence_gate_enabled
        assert evidence_gate_enabled() is False, (
            "evidence_gate_enabled() must return False by default "
            "(OPENRESEARCH_LEAF_EVIDENCE_GATE unset → gate OFF). "
            "The split-default collision had OPENRESEARCH_EVIDENCE_GATE default=OFF "
            "for the leaf gate but default=ON for the verdict gate — same var."
        )

    def test_leaf_gate_enabled_by_new_var(self, monkeypatch):
        """Setting OPENRESEARCH_LEAF_EVIDENCE_GATE=1 turns on the per-leaf veto."""
        monkeypatch.setenv("OPENRESEARCH_LEAF_EVIDENCE_GATE", "1")
        monkeypatch.delenv("OPENRESEARCH_EVIDENCE_GATE", raising=False)
        from backend.agents.rlm.evidence_gate import evidence_gate_enabled
        assert evidence_gate_enabled() is True, (
            "evidence_gate_enabled() must return True when "
            "OPENRESEARCH_LEAF_EVIDENCE_GATE=1."
        )

    def test_leaf_gate_not_activated_by_verdict_gate_var(self, monkeypatch):
        """Setting OPENRESEARCH_EVIDENCE_GATE=1 must NOT activate the per-leaf gate —
        the two are now independent after the collision fix."""
        monkeypatch.setenv("OPENRESEARCH_EVIDENCE_GATE", "1")
        monkeypatch.delenv("OPENRESEARCH_LEAF_EVIDENCE_GATE", raising=False)
        from backend.agents.rlm.evidence_gate import evidence_gate_enabled
        assert evidence_gate_enabled() is False, (
            "evidence_gate_enabled() must NOT be activated by OPENRESEARCH_EVIDENCE_GATE=1 "
            "after the split-default fix. Use OPENRESEARCH_LEAF_EVIDENCE_GATE=1 for the leaf gate."
        )

    def test_leaf_scorer_gate_default_off(self, monkeypatch):
        """leaf_scorer._evidence_gate_enabled() must also be OFF by default."""
        monkeypatch.delenv("OPENRESEARCH_LEAF_EVIDENCE_GATE", raising=False)
        monkeypatch.delenv("OPENRESEARCH_EVIDENCE_GATE", raising=False)
        from backend.evals.paperbench.leaf_scorer import _evidence_gate_enabled
        assert _evidence_gate_enabled() is False, (
            "leaf_scorer._evidence_gate_enabled() must return False by default "
            "(OPENRESEARCH_LEAF_EVIDENCE_GATE unset → gate OFF)."
        )

    def test_leaf_scorer_gate_enabled_by_new_var(self, monkeypatch):
        """leaf_scorer._evidence_gate_enabled() turns ON with OPENRESEARCH_LEAF_EVIDENCE_GATE=1."""
        monkeypatch.setenv("OPENRESEARCH_LEAF_EVIDENCE_GATE", "1")
        monkeypatch.delenv("OPENRESEARCH_EVIDENCE_GATE", raising=False)
        from backend.evals.paperbench.leaf_scorer import _evidence_gate_enabled
        assert _evidence_gate_enabled() is True, (
            "leaf_scorer._evidence_gate_enabled() must return True when "
            "OPENRESEARCH_LEAF_EVIDENCE_GATE=1."
        )

    def test_both_gates_independent(self, monkeypatch):
        """Verdict gate and leaf gate must be independently controllable.
        Verdict ON + leaf OFF is the intended production default."""
        monkeypatch.delenv("OPENRESEARCH_EVIDENCE_GATE", raising=False)
        monkeypatch.delenv("OPENRESEARCH_LEAF_EVIDENCE_GATE", raising=False)
        from backend.agents.rlm.report import _apply_evidence_gate  # noqa: F401
        from backend.agents.rlm.evidence_gate import evidence_gate_enabled
        from backend.evals.paperbench.leaf_scorer import _evidence_gate_enabled

        # Verdict gate: default ON (no env var set)
        # Leaf gate: default OFF (no env var set)
        assert evidence_gate_enabled() is False
        assert _evidence_gate_enabled() is False
        # Activating the leaf gate must not affect the verdict gate behavior
        monkeypatch.setenv("OPENRESEARCH_LEAF_EVIDENCE_GATE", "1")
        assert evidence_gate_enabled() is True
        assert _evidence_gate_enabled() is True
