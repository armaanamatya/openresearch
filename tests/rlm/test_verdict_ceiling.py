"""Tests for the tier-aware terminal-verdict ceiling.

The invariant under test: a screening-tier run can NEVER mint a `reproduced`
verdict — on ANY invocation path. A false `reproduced` is the catastrophic error
for patent-viability triage.

These tests deliberately pin the three failure modes that made the original
scope-rung-based design unsound (see verdict_ceiling.py's module docstring), so
that a regression toward "the ladder will protect us" is caught here.
"""

from __future__ import annotations

import json

import pytest

from backend.agents.rlm.verdict_ceiling import apply_verdict_ceiling, configured_ceiling


# --- configured_ceiling ----------------------------------------------------


def test_unset_flag_means_no_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENRESEARCH_MAX_TERMINAL_VERDICT", raising=False)
    assert configured_ceiling() is None


def test_empty_flag_means_no_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENRESEARCH_MAX_TERMINAL_VERDICT", "   ")
    assert configured_ceiling() is None


@pytest.mark.parametrize("value", ["failed", "partial", "reproduced"])
def test_recognized_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("OPENRESEARCH_MAX_TERMINAL_VERDICT", value.upper())
    assert configured_ceiling() == value


def test_typod_ceiling_fails_closed_not_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """A typo must NOT silently disable a trust gate.

    This is the whole 'shipped dark, protects nothing' failure class: a config
    that looks like it protects and doesn't. An unrecognized non-empty value
    collapses to a conservative cap, never to 'no cap'.
    """
    monkeypatch.setenv("OPENRESEARCH_MAX_TERMINAL_VERDICT", "screeened")  # typo
    assert configured_ceiling() == "partial"


# --- byte-identical when off ----------------------------------------------


def test_off_is_byte_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENRESEARCH_MAX_TERMINAL_VERDICT", raising=False)
    payload = {"verdict": "reproduced", "overall_score": 0.83}
    before = json.dumps(payload, sort_keys=True)

    out, warning = apply_verdict_ceiling(payload)

    assert warning is None
    assert json.dumps(out, sort_keys=True) == before
    assert "verdict_ceiling" not in out


# --- the clamp -------------------------------------------------------------


def test_screening_tier_cannot_mint_reproduced(monkeypatch: pytest.MonkeyPatch) -> None:
    """THE core invariant."""
    monkeypatch.setenv("OPENRESEARCH_MAX_TERMINAL_VERDICT", "partial")
    payload = {"verdict": "reproduced", "overall_score": 0.78}

    out, warning = apply_verdict_ceiling(payload)

    assert out["verdict"] == "partial"
    assert warning is not None
    assert out["verdict_ceiling"]["applied"] is True
    assert out["verdict_ceiling"]["uncapped_verdict"] == "reproduced"


def test_score_is_preserved_as_the_ranking_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tier 1 exists to RANK. Capping the verdict must not destroy the score."""
    monkeypatch.setenv("OPENRESEARCH_MAX_TERMINAL_VERDICT", "partial")
    payload = {
        "verdict": "reproduced",
        "overall_score": 0.78,
        "compute_adjusted_score": 0.71,
    }

    out, _ = apply_verdict_ceiling(payload)

    assert out["overall_score"] == 0.78
    assert out["compute_adjusted_score"] == 0.71
    # ...and the report stays honest about what it WOULD have said.
    assert out["verdict_ceiling"]["uncapped_verdict"] == "reproduced"


def test_ceiling_never_raises_a_verdict(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ceiling only ever lowers. A 'failed' run does not get promoted to the
    ceiling just because a ceiling exists — otherwise the gate would be a
    laundering mechanism rather than a cap."""
    monkeypatch.setenv("OPENRESEARCH_MAX_TERMINAL_VERDICT", "reproduced")
    payload = {"verdict": "failed", "overall_score": 0.0}

    out, warning = apply_verdict_ceiling(payload)

    assert out["verdict"] == "failed"
    assert warning is None
    assert out["verdict_ceiling"]["applied"] is False


def test_verdict_already_below_ceiling_is_untouched_but_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Tier-1 'partial' must be distinguishable from a Tier-2 'partial'."""
    monkeypatch.setenv("OPENRESEARCH_MAX_TERMINAL_VERDICT", "partial")
    payload = {"verdict": "partial", "overall_score": 0.4}

    out, warning = apply_verdict_ceiling(payload)

    assert out["verdict"] == "partial"
    assert warning is None
    assert out["verdict_ceiling"]["applied"] is False
    assert out["verdict_ceiling"]["ceiling"] == "partial"


def test_unrecognized_verdict_forced_to_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unknown verdict string cannot be PROVEN to sit below the ceiling, so it
    does not get the benefit of the doubt."""
    monkeypatch.setenv("OPENRESEARCH_MAX_TERMINAL_VERDICT", "partial")
    payload = {"verdict": "totally_reproduced_trust_me"}

    out, warning = apply_verdict_ceiling(payload)

    assert out["verdict"] == "partial"
    assert warning is not None


def test_missing_verdict_key_forced_to_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENRESEARCH_MAX_TERMINAL_VERDICT", "failed")
    out, warning = apply_verdict_ceiling({"overall_score": 0.9})

    assert out["verdict"] == "failed"
    assert warning is not None


def test_failed_ceiling_caps_everything(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENRESEARCH_MAX_TERMINAL_VERDICT", "failed")
    out, _ = apply_verdict_ceiling({"verdict": "reproduced", "overall_score": 1.0})

    assert out["verdict"] == "failed"
    assert out["overall_score"] == 1.0  # score still preserved


# --- integration at the write chokepoint -----------------------------------


def _write_exp_runs(project_dir, *, metrics: dict | None = None) -> None:
    """Minimal experiment_runs.jsonl so the (default-ON) evidence gate passes.

    Mirrors tests/rlm/test_evidence_gate_forge.py::_write_exp_runs.
    """
    row = {
        "success": True,
        "experiment_run_id": "run1",
        "metrics": metrics or {"accuracy": 0.9},
    }
    (project_dir / "experiment_runs.jsonl").write_text(
        json.dumps(row) + "\n", encoding="utf-8"
    )


def _reproduced_report():
    from backend.agents.rlm.report import RLMFinalReport

    return RLMFinalReport(
        verdict="reproduced",
        reproduction_summary="Baseline reproduced within tolerance.",
        baseline_metrics={"accuracy": 0.9},
        rubric={"overall_score": 0.78, "meets_target": True, "areas": []},
    )


def test_ceiling_applies_on_the_plain_reproduce_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """THE regression test for the design flaw this module fixes.

    campaign_policy's scope-rung rule gates REPRODUCED — but campaign_policy is
    never imported on the plain `reproduce` path, which is exactly what a cheap
    Tier-1 screen is. The ceiling must therefore be enforced at the report write
    chokepoint, which BOTH paths route through.

    If this test ever fails, a screening run can certify a paper as reproduced.
    """
    from backend.agents.rlm.report import write_final_report_rlm

    monkeypatch.setenv("OPENRESEARCH_MAX_TERMINAL_VERDICT", "partial")
    _write_exp_runs(tmp_path)  # real evidence, so the evidence gate lets it through

    json_path, _ = write_final_report_rlm(
        _reproduced_report(), tmp_path, run_experiment_ok_calls=1
    )

    written = json.loads(json_path.read_text())
    assert written["verdict"] == "partial", (
        "a screening-tier run minted a 'reproduced' verdict on the reproduce path"
    )
    assert written["verdict_ceiling"]["applied"] is True
    assert written["verdict_ceiling"]["uncapped_verdict"] == "reproduced"
    # The ranking signal survives the cap.
    assert written["rubric"]["overall_score"] == 0.78


def test_chokepoint_without_ceiling_is_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from backend.agents.rlm.report import write_final_report_rlm

    monkeypatch.delenv("OPENRESEARCH_MAX_TERMINAL_VERDICT", raising=False)
    _write_exp_runs(tmp_path)

    json_path, _ = write_final_report_rlm(
        _reproduced_report(), tmp_path, run_experiment_ok_calls=1
    )

    written = json.loads(json_path.read_text())
    assert written["verdict"] == "reproduced"
    assert "verdict_ceiling" not in written


def test_ceiling_composes_with_the_evidence_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Defense in depth: with NO evidence, the evidence gate downgrades the
    verdict before the ceiling is even consulted. Both gates only ever
    downgrade, so they commute — and a no-evidence screening run lands at
    'failed', not merely at the ceiling."""
    from backend.agents.rlm.report import write_final_report_rlm

    monkeypatch.setenv("OPENRESEARCH_MAX_TERMINAL_VERDICT", "partial")
    # deliberately NO experiment_runs.jsonl

    json_path, _ = write_final_report_rlm(
        _reproduced_report(), tmp_path, run_experiment_ok_calls=0
    )

    written = json.loads(json_path.read_text())
    assert written["verdict"] == "failed"
    # The ceiling did not need to fire, and must not have LIFTED anything.
    assert written["verdict_ceiling"]["applied"] is False
