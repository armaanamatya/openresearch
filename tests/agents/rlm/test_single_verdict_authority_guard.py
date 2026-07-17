"""Guard tests for the VerdictAuthority sever (Track A §4.3, Task 6 — "the sever").

Covers the brief's three required guard shapes plus an off-state
byte-identical pass for every severed writer:

  (1) **Runtime guard** — a synthetic post-authority mutation of the verdict
      surface must be caught (``verdict_authority.assert_verdict_surface_unchanged``,
      wired as the literal last check in ``write_final_report_rlm`` before the
      atomic write).
  (2) **Static guard** — no historical grade-derived verdict writer
      (``two_axis_report``, ``finalize_regrade``, ``leaf_scorer.amend_final_report``,
      ``run.py``'s hard-stop salvage) mints ``report["verdict"]``/``report.verdict``
      once VerdictAuthority is active. Verified BEHAVIOURALLY — call the real
      function and assert the verdict surface is untouched — rather than by
      grepping source text, which is fragile against refactors and cannot
      prove the code actually behaves as claimed. ``rdr/controller.py``'s
      severed line sits inside a large async pipeline with no isolated call
      surface for a behavioural test; that one site is verified by source
      inspection instead (the closest literal reading of "static guard" for
      that file — the underlying mechanism, ``verdict_authority.is_enabled()``,
      is already proven correct by every other test here).
  (3) **Grade-severance** — on fixed deterministic result_fidelity/evidence
      artifacts, perturbing the LLM grade (``rubric.overall_score``) or
      simulating a grader outage (no rubric at all) never changes
      ``report["verdict"]``.

Every severed writer additionally gets an OFF-state test proving its
pre-Track-A legacy behaviour survives byte-for-byte when either flag is off.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from backend.agents.rlm.report import RLMFinalReport, write_final_report_rlm
from backend.agents.rlm.verdict_authority import (
    PostAuthorityVerdictMutation,
    VERDICT_SURFACE_KEYS,
    assert_verdict_surface_unchanged,
    is_enabled,
)

BOTH_FLAGS = ("OPENRESEARCH_TWO_AXIS_VERDICT", "OPENRESEARCH_VERDICT_AUTHORITY")


def _enable_authority(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENRESEARCH_TWO_AXIS_VERDICT", "1")
    monkeypatch.setenv("OPENRESEARCH_VERDICT_AUTHORITY", "1")


def _disable_authority(monkeypatch: pytest.MonkeyPatch) -> None:
    for flag in BOTH_FLAGS:
        monkeypatch.delenv(flag, raising=False)


def _write_claim_fixture(project_dir: Path, *, accuracy: float = 0.991) -> None:
    """A single genuinely-primary numeric claim (claimed 0.99 +/- 0.01) plus a
    real success+metrics experiment_runs.jsonl row (the evidence-gate signal).
    ``accuracy`` controls whether the claim PASSES (~0.991, default) or FAILS
    (e.g. 0.10) against the deterministic result_fidelity checker.
    """
    (project_dir / "code").mkdir(parents=True, exist_ok=True)
    (project_dir / "rlm_state").mkdir(parents=True, exist_ok=True)
    (project_dir / "code" / "metrics.json").write_text(
        json.dumps({"accuracy": accuracy}), encoding="utf-8"
    )
    (project_dir / "rlm_state" / "repro_spec.json").write_text(
        json.dumps(
            {
                "claims": [
                    {
                        "claim_id": "primary_0",
                        "is_primary": True,
                        "kind": "numeric",
                        "metric_name": "accuracy",
                        "claimed_effect": 0.99,
                        "equivalence_margin": 0.01,
                        "direction": "higher_is_better",
                        "scope": {},
                        "ambiguous": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (project_dir / "experiment_runs.jsonl").write_text(
        json.dumps({"success": True, "metrics": {"accuracy": accuracy}}) + "\n",
        encoding="utf-8",
    )


def _default_rubric(overall_score: float | None, *, meets_target: bool | None = None) -> dict:
    return {
        "overall_score": overall_score,
        "target_score": 0.6,
        "meets_target": meets_target,
        "degraded": None,
        "areas": [],
    }


# --------------------------------------------------------------------------- #
# is_enabled(): the canonical gate requires BOTH flags
# --------------------------------------------------------------------------- #


def test_is_enabled_requires_both_flags(monkeypatch):
    _disable_authority(monkeypatch)
    assert is_enabled() is False

    monkeypatch.setenv("OPENRESEARCH_TWO_AXIS_VERDICT", "1")
    assert is_enabled() is False  # authority sub-flag still off

    monkeypatch.setenv("OPENRESEARCH_VERDICT_AUTHORITY", "1")
    assert is_enabled() is True

    monkeypatch.setenv("OPENRESEARCH_TWO_AXIS_VERDICT", "0")
    assert is_enabled() is False  # master back off, sub-flag alone is not enough


# --------------------------------------------------------------------------- #
# (1) Runtime guard: a synthetic post-authority mutation must be caught
# --------------------------------------------------------------------------- #


def test_synthetic_post_authority_mutation_is_caught():
    stamped = {
        "verdict": "reproduced",
        "implementation_verdict": "faithful",
        "replication_verdict": "replicated",
    }
    mutated = dict(stamped)
    mutated["verdict"] = "partial"  # a hypothetical later writer flipping it
    with pytest.raises(PostAuthorityVerdictMutation):
        assert_verdict_surface_unchanged(stamped, mutated, context="test")


def test_synthetic_mutation_of_a_diagnostic_mirror_is_also_caught():
    stamped = {"verdict": "reproduced", "implementation_verdict": "faithful"}
    mutated = dict(stamped)
    mutated["implementation_verdict"] = "broken"
    with pytest.raises(PostAuthorityVerdictMutation):
        assert_verdict_surface_unchanged(stamped, mutated)


def test_unmutated_surface_passes():
    stamped = {"verdict": "reproduced"}
    assert assert_verdict_surface_unchanged(stamped, dict(stamped)) is None


def test_key_absent_from_stamped_snapshot_is_never_checked():
    # implementation_verdict was never stamped by decide() in this snapshot
    # (it's a two-axis diagnostic, set independently) -- its value in
    # `current` must never raise regardless of what it is.
    assert (
        assert_verdict_surface_unchanged(
            {"verdict": "reproduced"},
            {"verdict": "reproduced", "implementation_verdict": "anything"},
        )
        is None
    )


def test_verdict_surface_keys_is_the_documented_triple():
    assert VERDICT_SURFACE_KEYS == (
        "verdict",
        "implementation_verdict",
        "replication_verdict",
    )


# --------------------------------------------------------------------------- #
# (3) Grade-severance: perturbing the LLM grade must not change the verdict
# --------------------------------------------------------------------------- #


def _shipped(project_dir: Path) -> dict:
    return json.loads((project_dir / "final_report.json").read_text(encoding="utf-8"))


def test_grade_severance_high_score_vs_zero_score_same_verdict(tmp_path, monkeypatch):
    _enable_authority(monkeypatch)

    high_dir = tmp_path / "high"
    _write_claim_fixture(high_dir)
    write_final_report_rlm(
        RLMFinalReport(verdict="partial", rubric=_default_rubric(0.95)),
        high_dir,
        run_experiment_ok_calls=1,
        run_experiment_calls=1,
    )

    low_dir = tmp_path / "low"
    _write_claim_fixture(low_dir)
    write_final_report_rlm(
        RLMFinalReport(verdict="partial", rubric=_default_rubric(0.0)),
        low_dir,
        run_experiment_ok_calls=1,
        run_experiment_calls=1,
    )

    assert _shipped(high_dir)["verdict"] == _shipped(low_dir)["verdict"] == "reproduced"


def test_grade_severance_grader_outage_same_verdict(tmp_path, monkeypatch):
    """No rubric at all (the honest all-None default) -- simulating a grader
    outage / a run that never reached verify_against_rubric -- must yield the
    SAME verdict as a well-graded run, given the same deterministic evidence.
    """
    _enable_authority(monkeypatch)
    project_dir = tmp_path / "no_rubric"
    _write_claim_fixture(project_dir)
    write_final_report_rlm(
        RLMFinalReport(verdict="partial", reproduction_summary="baseline"),
        project_dir,
        run_experiment_ok_calls=1,
        run_experiment_calls=1,
    )
    assert _shipped(project_dir)["verdict"] == "reproduced"


# --------------------------------------------------------------------------- #
# End-to-end write_final_report_rlm: verdict matches an independent decide() call
# --------------------------------------------------------------------------- #


def test_write_final_report_rlm_verdict_matches_independent_decide_call(tmp_path, monkeypatch):
    _enable_authority(monkeypatch)
    project_dir = tmp_path / "run"
    _write_claim_fixture(project_dir)
    write_final_report_rlm(
        RLMFinalReport(verdict="failed", rubric=_default_rubric(0.1)),
        project_dir,
        run_experiment_ok_calls=1,
        run_experiment_calls=1,
    )
    shipped = _shipped(project_dir)

    from backend.agents.rlm import result_fidelity, verdict_authority

    repro_spec = json.loads((project_dir / "rlm_state" / "repro_spec.json").read_text())
    rf = result_fidelity.evaluate(repro_spec, project_dir)
    independent = verdict_authority.decide(result_fidelity=rf, evidence_gate=True, fidelity_certificate=None)

    assert shipped["verdict"] == independent["verdict"] == "reproduced"
    assert shipped["verdict_authority"]["reason"] == independent["reason"]


def test_demo_status_verdict_is_mirrored_when_active(tmp_path, monkeypatch):
    _enable_authority(monkeypatch)
    project_dir = tmp_path / "run"
    _write_claim_fixture(project_dir)
    write_final_report_rlm(
        RLMFinalReport(verdict="partial"),
        project_dir,
        run_experiment_ok_calls=1,
        run_experiment_calls=1,
    )
    shipped = _shipped(project_dir)
    demo_status = json.loads((project_dir / "demo_status.json").read_text(encoding="utf-8"))
    assert demo_status["verdict"] == shipped["verdict"] == "reproduced"


# --------------------------------------------------------------------------- #
# F1: the live TERMINAL _write_demo_status path reads the authority verdict
#     back off final_report.json (fixed centrally) instead of stomping it to
#     the auto-derived "unknown".
# --------------------------------------------------------------------------- #


def test_terminal_write_demo_status_reads_authority_verdict_when_active(tmp_path, monkeypatch):
    _enable_authority(monkeypatch)
    from backend.agents.rlm.run import _write_demo_status

    project_dir = tmp_path / "run"
    project_dir.mkdir()
    (project_dir / "final_report.json").write_text(
        json.dumps({"verdict": "reproduced"}), encoding="utf-8"
    )
    # Terminal write: status "completed", NO explicit verdict= (the exact live
    # run_pipeline_rlm call shape). Legacy auto-derive would stamp "unknown".
    _write_demo_status(project_dir, "completed")
    demo_status = json.loads((project_dir / "demo_status.json").read_text(encoding="utf-8"))
    assert demo_status["verdict"] == "reproduced"  # read back off disk, not auto-derived


def test_terminal_write_demo_status_auto_derives_when_authority_off(tmp_path, monkeypatch):
    _disable_authority(monkeypatch)
    from backend.agents.rlm.run import _write_demo_status

    project_dir = tmp_path / "run"
    project_dir.mkdir()
    # A final_report.json with a verdict exists, but the authority is OFF, so
    # the read-back branch must be entirely skipped => byte-identical legacy
    # auto-derive ("completed" + no explicit verdict => "unknown").
    (project_dir / "final_report.json").write_text(
        json.dumps({"verdict": "reproduced"}), encoding="utf-8"
    )
    _write_demo_status(project_dir, "completed")
    demo_status = json.loads((project_dir / "demo_status.json").read_text(encoding="utf-8"))
    assert demo_status["verdict"] == "unknown"


def test_terminal_write_demo_status_falls_back_when_no_report(tmp_path, monkeypatch):
    """Authority ON but final_report.json absent (an early-failure path that
    never reached finalize) => legacy auto-derive, never a crash."""
    _enable_authority(monkeypatch)
    from backend.agents.rlm.run import _write_demo_status

    project_dir = tmp_path / "run"
    project_dir.mkdir()
    _write_demo_status(project_dir, "failed")
    demo_status = json.loads((project_dir / "demo_status.json").read_text(encoding="utf-8"))
    assert demo_status["verdict"] == "failed"  # status=="failed" auto-derive


def test_terminal_write_demo_status_explicit_verdict_still_wins(tmp_path, monkeypatch):
    """An explicit verdict= arg must always win — the read-back only fills the
    verdict when the caller passed None."""
    _enable_authority(monkeypatch)
    from backend.agents.rlm.run import _write_demo_status

    project_dir = tmp_path / "run"
    project_dir.mkdir()
    (project_dir / "final_report.json").write_text(
        json.dumps({"verdict": "reproduced"}), encoding="utf-8"
    )
    _write_demo_status(project_dir, "completed", verdict="partial")
    demo_status = json.loads((project_dir / "demo_status.json").read_text(encoding="utf-8"))
    assert demo_status["verdict"] == "partial"


def test_live_sequence_finalize_then_terminal_agree_on_verdict(tmp_path, monkeypatch):
    """The full live shape: write_final_report_rlm (stamps the authority
    verdict into final_report.json + its own demo_status mirror), THEN the
    terminal _write_demo_status(project_dir, "completed") that runs after
    finalize. Before F1 the terminal call stomped the mirror back to
    "unknown"; now it reads final_report.json and the two agree."""
    _enable_authority(monkeypatch)
    from backend.agents.rlm.run import _write_demo_status

    project_dir = tmp_path / "run"
    _write_claim_fixture(project_dir)  # passing primary + real evidence -> "reproduced"
    write_final_report_rlm(
        RLMFinalReport(verdict="partial"),
        project_dir,
        run_experiment_ok_calls=1,
        run_experiment_calls=1,
    )
    _write_demo_status(project_dir, "completed")  # the terminal live write

    shipped = _shipped(project_dir)
    demo_status = json.loads((project_dir / "demo_status.json").read_text(encoding="utf-8"))
    assert shipped["verdict"] == "reproduced"
    assert demo_status["verdict"] == shipped["verdict"] == "reproduced"


# --------------------------------------------------------------------------- #
# F2: the authority block fails CLOSED — an error in the authority path ships
#     "inconclusive", NEVER the pre-authority (grade-derived) verdict.
# --------------------------------------------------------------------------- #


def test_authority_error_fails_closed_to_inconclusive_not_grade(tmp_path, monkeypatch):
    _enable_authority(monkeypatch)
    project_dir = tmp_path / "run"
    _write_claim_fixture(project_dir)  # a PASSING primary: absent the raise this is "reproduced"

    # Force decide() to raise. The pre-authority verdict in json_content is the
    # grade-derived "reproduced" (high rubric, real evidence, no gate downgrade)
    # — shipping THAT on error would defeat the sever. Fail-closed must ship
    # "inconclusive" instead.
    from backend.agents.rlm import verdict_authority

    def _boom(**kwargs):
        raise RuntimeError("simulated authority failure")

    monkeypatch.setattr(verdict_authority, "decide", _boom)

    write_final_report_rlm(
        RLMFinalReport(verdict="reproduced", rubric=_default_rubric(0.95, meets_target=True)),
        project_dir,
        run_experiment_ok_calls=1,
        run_experiment_calls=1,
    )
    shipped = _shipped(project_dir)
    assert shipped["verdict"] == "inconclusive"  # NOT "reproduced"
    assert shipped["verdict_authority"]["reason"] == "authority_error"


def test_result_fidelity_error_also_fails_closed(tmp_path, monkeypatch):
    """The raise can originate anywhere in the authority block (here,
    result_fidelity.evaluate before decide() is even reached) — fail-closed
    still ships inconclusive, never the grade."""
    _enable_authority(monkeypatch)
    project_dir = tmp_path / "run"
    _write_claim_fixture(project_dir)

    from backend.agents.rlm import result_fidelity

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated result_fidelity failure")

    monkeypatch.setattr(result_fidelity, "evaluate", _boom)

    write_final_report_rlm(
        RLMFinalReport(verdict="reproduced", rubric=_default_rubric(0.95, meets_target=True)),
        project_dir,
        run_experiment_ok_calls=1,
        run_experiment_calls=1,
    )
    assert _shipped(project_dir)["verdict"] == "inconclusive"


def test_authority_error_fail_closed_mirrors_inconclusive_to_demo_status(tmp_path, monkeypatch):
    """The fail-closed inconclusive must also be what the terminal demo_status
    read-back sees — end-to-end, an authority error can never surface the grade
    anywhere."""
    _enable_authority(monkeypatch)
    from backend.agents.rlm import verdict_authority
    from backend.agents.rlm.run import _write_demo_status

    project_dir = tmp_path / "run"
    _write_claim_fixture(project_dir)
    monkeypatch.setattr(
        verdict_authority, "decide", lambda **k: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    write_final_report_rlm(
        RLMFinalReport(verdict="reproduced", rubric=_default_rubric(0.95, meets_target=True)),
        project_dir,
        run_experiment_ok_calls=1,
        run_experiment_calls=1,
    )
    _write_demo_status(project_dir, "completed")
    demo_status = json.loads((project_dir / "demo_status.json").read_text(encoding="utf-8"))
    assert demo_status["verdict"] == "inconclusive"


def test_contradicted_claim_yields_contradicted_verdict_despite_high_grade(tmp_path, monkeypatch):
    _enable_authority(monkeypatch)
    project_dir = tmp_path / "run"
    _write_claim_fixture(project_dir, accuracy=0.10)  # measured 0.10 vs claimed 0.99+/-0.01 -> fail
    write_final_report_rlm(
        RLMFinalReport(verdict="reproduced", rubric=_default_rubric(0.95, meets_target=True)),
        project_dir,
        run_experiment_ok_calls=1,
        run_experiment_calls=1,
    )
    assert _shipped(project_dir)["verdict"] == "contradicted"


def test_no_repro_spec_yields_inconclusive_not_pass_through(tmp_path, monkeypatch):
    """The RDR/legacy shape: no rlm_state/repro_spec.json at all. decide()
    must yield "inconclusive", never pass through whatever verdict the report
    started with -- this is the design decision behind the RDR controller's
    severed line (step 3 of the brief), proven here without needing RDR's
    full async pipeline."""
    _enable_authority(monkeypatch)
    project_dir = tmp_path / "no_spec"
    (project_dir / "code").mkdir(parents=True)
    (project_dir / "code" / "metrics.json").write_text(json.dumps({"accuracy": 0.99}))
    write_final_report_rlm(
        RLMFinalReport(verdict="reproduced", rubric=_default_rubric(0.95, meets_target=True)),
        project_dir,
        run_experiment_ok_calls=1,
        run_experiment_calls=1,
    )
    assert _shipped(project_dir)["verdict"] == "inconclusive"


def test_claim_gate_cap_reaches_the_authority(tmp_path, monkeypatch):
    _enable_authority(monkeypatch)
    monkeypatch.setenv("OPENRESEARCH_REPORT_CLAIM_GATE", "1")
    project_dir = tmp_path / "run"
    _write_claim_fixture(project_dir)  # accuracy=0.991 -> the repro_spec primary claim PASSES
    write_final_report_rlm(
        RLMFinalReport(
            verdict="reproduced",
            # An ungrounded RESULT claim in the report's own narrative: 0.50 is
            # nowhere near the on-disk 0.991 (well outside the 5% tolerance).
            reproduction_summary="Achieved accuracy of 0.50 on the held-out set.",
            rubric=_default_rubric(0.95, meets_target=True),
        ),
        project_dir,
        run_experiment_ok_calls=1,
        run_experiment_calls=1,
    )
    shipped = _shipped(project_dir)
    # Without the cap this would be "reproduced" (all-pass primary + satisfied
    # evidence gate, per test_write_final_report_rlm_verdict_matches_independent_decide_call)
    # -- the claim-gate cap clamps it downward to "partial".
    assert shipped["verdict"] == "partial"
    assert shipped["verdict_authority"]["claim_gate_cap"] == "partial"
    assert shipped["claim_grounding"]["ungrounded"] >= 1
    # The legacy per-hoc mutation path must NOT have run (no cap-applied note
    # spliced into reproduction_summary the way apply_report_claim_gate would).
    assert "[harness] report claimed" not in shipped["reproduction_summary"]


# --------------------------------------------------------------------------- #
# Off-state: either flag off => byte-identical legacy behaviour
# --------------------------------------------------------------------------- #


def test_off_state_both_flags_off_no_new_keys_legacy_verdict_stands(tmp_path, monkeypatch):
    _disable_authority(monkeypatch)
    project_dir = tmp_path / "run"
    _write_claim_fixture(project_dir)
    write_final_report_rlm(
        RLMFinalReport(verdict="reproduced", rubric=_default_rubric(0.95, meets_target=True)),
        project_dir,
        run_experiment_ok_calls=1,
        run_experiment_calls=1,
    )
    shipped = _shipped(project_dir)
    assert "verdict_authority" not in shipped
    assert shipped["verdict"] == "reproduced"
    assert not (project_dir / "demo_status.json").exists()


def test_off_state_two_axis_only_preserves_legacy_headline_projection(tmp_path, monkeypatch):
    """OPENRESEARCH_TWO_AXIS_VERDICT=1 alone (authority sub-flag OFF): the
    legacy fidelity-projected verdict still reaches the headline -- proving
    the sever is additive (gated on BOTH flags), not a silent behaviour
    change for existing two-axis-only users."""
    monkeypatch.setenv("OPENRESEARCH_TWO_AXIS_VERDICT", "1")
    monkeypatch.delenv("OPENRESEARCH_VERDICT_AUTHORITY", raising=False)
    project_dir = tmp_path / "run"
    _write_claim_fixture(project_dir)
    write_final_report_rlm(
        RLMFinalReport(verdict="reproduced"),
        project_dir,
        run_experiment_ok_calls=1,
        run_experiment_calls=1,
    )
    shipped = _shipped(project_dir)
    # No fidelity_certificate.json on disk -> certificate not green -> impl
    # caps at "partial" -> legacy_verdict projects "partial" onto the headline
    # (the pre-Track-A two_axis_report behaviour, unmodified).
    assert shipped["verdict"] == "partial"
    assert "verdict_authority" not in shipped
    assert not (project_dir / "demo_status.json").exists()


# --------------------------------------------------------------------------- #
# (2) Static/behavioural guard: each severed writer, in isolation
# --------------------------------------------------------------------------- #


def test_two_axis_report_stops_writing_headline_when_authority_active(tmp_path, monkeypatch):
    _enable_authority(monkeypatch)
    project_dir = tmp_path / "run"
    (project_dir / "code").mkdir(parents=True)
    (project_dir / "code" / "metrics.json").write_text("{}")
    report = {"verdict": "reproduced", "rubric": {"areas": []}}

    from backend.agents.rlm.two_axis_report import compute_and_attach

    assert compute_and_attach(report, project_dir) is True
    assert report["verdict"] == "reproduced"  # untouched
    assert report["implementation_verdict"] == "partial"  # diagnostic still computed


def test_two_axis_report_legacy_projection_when_authority_off(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_TWO_AXIS_VERDICT", "1")
    monkeypatch.delenv("OPENRESEARCH_VERDICT_AUTHORITY", raising=False)
    project_dir = tmp_path / "run"
    (project_dir / "code").mkdir(parents=True)
    (project_dir / "code" / "metrics.json").write_text("{}")
    report = {"verdict": "reproduced", "rubric": {"areas": []}}

    from backend.agents.rlm.two_axis_report import compute_and_attach

    assert compute_and_attach(report, project_dir) is True
    assert report["verdict"] == "partial"  # legacy projection still fires


def test_finalize_regrade_does_not_mint_verdict_when_authority_active(tmp_path, monkeypatch):
    _enable_authority(monkeypatch)
    from backend.agents.rlm import finalize_regrade as fr

    (tmp_path / "code").mkdir(parents=True)
    (tmp_path / "code" / "metrics.json").write_text("{}")
    monkeypatch.setattr(fr, "maybe_regrade", lambda ctx, report: {"overall_score": 0.9})

    class _Ctx:
        project_dir = tmp_path

    report = RLMFinalReport(verdict="failed", rubric=_default_rubric(None))
    fresh = fr.regrade_and_emit(_Ctx(), report, emit=lambda *a, **k: None)

    assert fresh == {"overall_score": 0.9}
    assert report.verdict == "failed"  # not reconciled/bumped -- decide() owns it now


def test_finalize_regrade_legacy_bump_when_authority_off(tmp_path, monkeypatch):
    _disable_authority(monkeypatch)
    from backend.agents.rlm import finalize_regrade as fr

    (tmp_path / "code").mkdir(parents=True)
    (tmp_path / "code" / "metrics.json").write_text("{}")
    monkeypatch.setattr(fr, "maybe_regrade", lambda ctx, report: {"overall_score": 0.9})

    class _Ctx:
        project_dir = tmp_path

    report = RLMFinalReport(verdict="failed", rubric=_default_rubric(None))
    fr.regrade_and_emit(_Ctx(), report, emit=lambda *a, **k: None)

    assert report.verdict == "reproduced"  # legacy reconcile_verdict_with_score("reproduced", 0.9)


def test_leaf_scorer_amend_does_not_mint_verdict_when_authority_active(tmp_path, monkeypatch):
    _enable_authority(monkeypatch)
    from backend.evals.paperbench.leaf_scorer import amend_final_report

    run_dir = tmp_path
    (run_dir / "final_report.json").write_text(
        json.dumps({"verdict": "reproduced", "rubric": {}, "overall_score": None})
    )
    amend_final_report(
        run_dir,
        {"overall_score": 0.0, "leaf_count": 1, "graded": 1, "rubric_source": "generated"},
    )
    shipped = _shipped(run_dir)
    assert shipped["verdict"] == "reproduced"  # untouched despite overall_score=0.0


def test_leaf_scorer_amend_legacy_reconcile_when_authority_off(tmp_path, monkeypatch):
    _disable_authority(monkeypatch)
    from backend.evals.paperbench.leaf_scorer import amend_final_report

    run_dir = tmp_path
    (run_dir / "final_report.json").write_text(
        json.dumps({"verdict": "reproduced", "rubric": {}, "overall_score": None})
    )
    amend_final_report(
        run_dir,
        {"overall_score": 0.0, "leaf_count": 1, "graded": 1, "rubric_source": "generated"},
    )
    shipped = _shipped(run_dir)
    assert shipped["verdict"] == "failed"  # legacy reconcile_verdict_with_score("reproduced", 0.0)


def test_salvage_partial_report_no_mint_when_authority_active(tmp_path, monkeypatch):
    _enable_authority(monkeypatch)
    from backend.agents.rlm.run import _salvage_partial_report

    report = RLMFinalReport(verdict="failed", rubric=_default_rubric(0.8))
    score = _salvage_partial_report(report, tmp_path, stop_kind="hard_stop", stop_detail="x")

    assert score == 0.8
    assert report.verdict == "failed"  # untouched -- write_final_report_rlm's decide()
    # call (moments after this in production) owns the verdict now.


def test_salvage_partial_report_legacy_mint_when_authority_off(tmp_path, monkeypatch):
    _disable_authority(monkeypatch)
    from backend.agents.rlm.run import _salvage_partial_report

    report = RLMFinalReport(verdict="failed", rubric=_default_rubric(0.8))
    score = _salvage_partial_report(report, tmp_path, stop_kind="hard_stop", stop_detail="x")

    assert score == 0.8
    assert report.verdict == "partial"  # legacy reconcile_verdict_with_score("partial", 0.8)


def test_rdr_controller_severed_site_gates_on_authority():
    """rdr/controller.py's verdict-mint line (~1417) sits inside a large async
    pipeline (WorkCluster/Artifacts/ctx) with no isolated call surface for a
    behavioural test -- verified by source inspection instead. The mechanism
    itself (verdict_authority.is_enabled()) is already proven correct by every
    other test in this file; this only proves the RDR call site actually
    consults it before falling back to the legacy grade-derived mint.
    """
    from backend.agents.rdr import controller

    src = inspect.getsource(controller)
    needle = 'verdict = reconcile_verdict_with_score("partial", overall_score)'
    idx = src.index(needle)
    window = src[max(0, idx - 600) : idx]
    assert "verdict_authority" in window
    assert "is_enabled()" in window
