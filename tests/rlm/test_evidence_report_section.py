"""Tests for the flag-gated 'Provenance & Evidence' Markdown section.

Release-1 workstream ④ — PURE DISCLOSURE of data the harness already computes
(``evidence_bundle``, ``validation``): zero new trust surface, zero new
computation. Gate: ``OPENRESEARCH_EVIDENCE_REPORT_SECTION`` (default OFF).

Coverage:
- OFF (unset / falsy): ``_render_markdown`` output does not depend AT ALL on
  whether ``evidence_bundle``/``validation`` are populated on the report —
  the whole point of "byte-identical when off".
- ON: the section renders a resolved evidence bundle, an unverified bundle,
  a clean validation panel (with the mandated "no suspicion raised" / never
  "verified correct" copy), a vetoed panel (veto_set + predicate table), and
  the honest "not available" fallbacks when fields are missing/None/malformed.
- Claim-grounding is deliberately never rendered (no such field exists on the
  RLMFinalReport object `_render_markdown` receives — see report_claim_gate.py).
- The section can never raise, even given malformed field shapes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.agents.rlm.report import RLMFinalReport, _render_markdown


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_report(**extra) -> RLMFinalReport:
    """Minimal RLMFinalReport with safe defaults, same shape as the sibling
    telemetry-rendering test (test_render_markdown_telemetry.py)."""
    return RLMFinalReport(
        paper={"id": "test-paper", "title": "Test Paper"},
        verdict="partial",
        reproduction_summary="A test reproduction.",
        baseline_metrics={"accuracy": 0.9},
        paper_claims={"accuracy": 0.91},
        rubric={"overall_score": 0.5, "meets_target": False, "areas": []},
        improvements=[],
        primitive_trace={},
        cost={"primitives": 0.0, "llm_usd": 0.0},
        iterations=1,
        primitive_provider="real",
        degraded=False,
        **extra,
    )


_RESOLVED_BUNDLE = {
    "attempt_id": "run_abc123",
    "ledger_sequence": 3,
    "metrics_sha256": "deadbeef" * 8,
    "code_tree_digest": "cafef00d" * 8,
    "artifact_dir": "/runs/prj_x/outputs/run_abc123",
    "coordinates": {"model_id": "qwen3-1.7b", "env_id": "alfworld"},
}

_UNVERIFIED_BUNDLE = {"status": "bundle_unverified"}

_CLEAN_VALIDATION = {
    "status": "clean",
    "veto_set": [],
    "separation": "independent",
    "panel_models": ["validator:azure:gpt-4o"],
    "evidence_fingerprint": "fp-clean-1",
    "predicates": [
        {"predicate": "not_all_constant", "metric_ref": "mean_reward", "violated": False},
    ],
}

_VETOED_VALIDATION = {
    "status": "vetoed",
    "veto_set": ["mean_reward"],
    "separation": "independent",
    "panel_models": ["validator:azure:gpt-4o"],
    "evidence_fingerprint": "fp-vetoed-1",
    "predicates": [
        {
            "predicate": "not_all_constant",
            "metric_ref": "mean_reward",
            "violated": True,
            "detail": "all 5 recorded values equal 0.0",
        },
        {"predicate": "provenance_present", "metric_ref": "accuracy", "violated": False},
    ],
}


# ---------------------------------------------------------------------------
# OFF state — byte-identical regardless of what evidence data exists
# ---------------------------------------------------------------------------


def test_off_by_default_no_section(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENRESEARCH_EVIDENCE_REPORT_SECTION", raising=False)
    report = _make_report(evidence_bundle=_RESOLVED_BUNDLE, validation=_CLEAN_VALIDATION)
    md = _render_markdown(report, project_dir=tmp_path)
    assert "## Provenance & Evidence" not in md
    assert "no suspicion raised" not in md


@pytest.mark.parametrize("falsy", ["", "0", "false", "no", "off", "garbage"])
def test_off_for_every_falsy_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, falsy: str
) -> None:
    monkeypatch.setenv("OPENRESEARCH_EVIDENCE_REPORT_SECTION", falsy)
    report = _make_report(evidence_bundle=_RESOLVED_BUNDLE, validation=_VETOED_VALIDATION)
    md = _render_markdown(report, project_dir=tmp_path)
    assert "## Provenance & Evidence" not in md


def test_off_state_byte_identical_regardless_of_evidence_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The load-bearing byte-identical-when-off guarantee: with the flag
    unset, a report that HAS a resolved evidence_bundle + a clean validation
    verdict renders EXACTLY the same Markdown as a report where both fields
    are at their untouched defaults (evidence_bundle=None, validation={}).
    """
    monkeypatch.delenv("OPENRESEARCH_EVIDENCE_REPORT_SECTION", raising=False)

    bare = _make_report()
    rich = _make_report(evidence_bundle=_RESOLVED_BUNDLE, validation=_CLEAN_VALIDATION)

    assert bare.evidence_bundle is None
    assert rich.evidence_bundle == _RESOLVED_BUNDLE  # sanity: fixture really differs

    md_bare = _render_markdown(bare, project_dir=tmp_path)
    md_rich = _render_markdown(rich, project_dir=tmp_path)

    assert md_bare == md_rich
    assert md_rich.endswith("_Generated by ReproLab RLM orchestrator (Issue #60)._\n")


def test_off_state_matches_no_project_dir_call_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """Legacy no-project_dir call path is unaffected by the new flag either way."""
    monkeypatch.delenv("OPENRESEARCH_EVIDENCE_REPORT_SECTION", raising=False)
    report = _make_report(evidence_bundle=_RESOLVED_BUNDLE, validation=_VETOED_VALIDATION)
    md = _render_markdown(report)  # no project_dir at all
    assert "## Provenance & Evidence" not in md
    assert "## Rubric Score" in md


# ---------------------------------------------------------------------------
# ON state — the section renders the already-computed fields
# ---------------------------------------------------------------------------


def test_on_renders_resolved_evidence_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENRESEARCH_EVIDENCE_REPORT_SECTION", "1")
    report = _make_report(evidence_bundle=_RESOLVED_BUNDLE, validation={})
    md = _render_markdown(report, project_dir=tmp_path)

    assert "## Provenance & Evidence" in md
    assert "**Evidence bundle:**" in md
    assert "run_abc123" in md
    assert "deadbeef" * 8 in md
    assert "cafef00d" * 8 in md
    assert "Ledger sequence: 3" in md
    assert "model_id=qwen3-1.7b" in md
    assert "env_id=alfworld" in md


def test_on_renders_unverified_bundle_explicitly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENRESEARCH_EVIDENCE_REPORT_SECTION", "true")
    report = _make_report(evidence_bundle=_UNVERIFIED_BUNDLE, validation={})
    md = _render_markdown(report, project_dir=tmp_path)

    assert "Unverified" in md
    assert "run_abc123" not in md


def test_on_renders_not_available_when_bundle_is_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENRESEARCH_EVIDENCE_REPORT_SECTION", "yes")
    report = _make_report(evidence_bundle=None, validation={})
    md = _render_markdown(report, project_dir=tmp_path)

    assert "**Evidence bundle:**" in md
    assert "Not available" in md


def test_on_clean_panel_copy_discipline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A clean panel must read 'no suspicion raised' and must NEVER claim the
    evidence was 'verified correct' — a clean min-aggregation is an absence of
    a caught problem, not a certification."""
    monkeypatch.setenv("OPENRESEARCH_EVIDENCE_REPORT_SECTION", "1")
    report = _make_report(evidence_bundle=None, validation=_CLEAN_VALIDATION)
    md = _render_markdown(report, project_dir=tmp_path)

    assert "no suspicion raised" in md
    assert "verified correct" not in md.lower()
    assert "validator:azure:gpt-4o" in md
    assert "Separation: independent" in md


def test_on_vetoed_panel_lists_veto_set_and_predicates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENRESEARCH_EVIDENCE_REPORT_SECTION", "1")
    report = _make_report(evidence_bundle=None, validation=_VETOED_VALIDATION)
    md = _render_markdown(report, project_dir=tmp_path)

    assert "**Vetoed**" in md
    assert "mean_reward" in md
    assert "not_all_constant" in md
    assert "all 5 recorded values equal 0.0" in md
    # The non-violated predicate is still surfaced as part of the audit trail.
    assert "provenance_present" in md


def test_on_missing_validator_status_renders_missing_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENRESEARCH_EVIDENCE_REPORT_SECTION", "1")
    report = _make_report(
        evidence_bundle=None,
        validation={"status": "missing", "reason": "no fresh verdict", "evidence_fingerprint": "x"},
    )
    md = _render_markdown(report, project_dir=tmp_path)
    assert "no fresh verdict" in md


def test_on_no_validator_configured_renders_not_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENRESEARCH_EVIDENCE_REPORT_SECTION", "1")
    report = _make_report(evidence_bundle=None, validation={})
    md = _render_markdown(report, project_dir=tmp_path)
    assert "**Validation panel:**" in md
    assert "Not available" in md


def test_on_never_renders_claim_grounding_section(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No claim_grounding field exists on RLMFinalReport (report_claim_gate.py
    stamps it onto the serialized JSON dict only) — the renderer must not
    invent a subsection for it."""
    monkeypatch.setenv("OPENRESEARCH_EVIDENCE_REPORT_SECTION", "1")
    assert not hasattr(RLMFinalReport, "model_fields") or "claim_grounding" not in (
        RLMFinalReport.model_fields
    )
    report = _make_report(evidence_bundle=_RESOLVED_BUNDLE, validation=_CLEAN_VALIDATION)
    assert getattr(report, "claim_grounding", None) is None
    md = _render_markdown(report, project_dir=tmp_path)
    assert "Claim grounding" not in md
    assert "claim_grounding" not in md


# ---------------------------------------------------------------------------
# Fail-soft — malformed shapes must never raise
# ---------------------------------------------------------------------------


def test_on_malformed_validation_type_fails_soft(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Direct attribute assignment (as report.py itself does at the write
    chokepoint) can bypass pydantic validation; the renderer must tolerate a
    non-dict value instead of raising."""
    monkeypatch.setenv("OPENRESEARCH_EVIDENCE_REPORT_SECTION", "1")
    report = _make_report(evidence_bundle=None, validation={})
    report.validation = "not-a-dict"  # type: ignore[assignment]

    md = _render_markdown(report, project_dir=tmp_path)  # must not raise
    assert "## Provenance & Evidence" in md
    assert "Not available" in md


def test_on_malformed_bundle_type_fails_soft(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENRESEARCH_EVIDENCE_REPORT_SECTION", "1")
    report = _make_report(validation={})
    report.evidence_bundle = ["not", "a", "dict"]  # type: ignore[assignment]

    md = _render_markdown(report, project_dir=tmp_path)  # must not raise
    assert "## Provenance & Evidence" in md
    assert "Not available" in md


def test_on_predicates_with_missing_optional_keys_fails_soft(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A predicate entry missing 'detail' (the shape write_final_report_rlm
    actually stamps today) must still render cleanly."""
    monkeypatch.setenv("OPENRESEARCH_EVIDENCE_REPORT_SECTION", "1")
    validation = {
        "status": "vetoed",
        "veto_set": ["accuracy"],
        "separation": "degraded",
        "panel_models": ["validator:same-model"],
        "evidence_fingerprint": "fp",
        "predicates": [{"predicate": "provenance_present", "metric_ref": "accuracy", "violated": True}],
    }
    report = _make_report(evidence_bundle=None, validation=validation)
    md = _render_markdown(report, project_dir=tmp_path)
    assert "provenance_present" in md
    assert "accuracy" in md


def test_render_evidence_section_helper_never_raises_on_garbage_predicates() -> None:
    """Direct unit test of the private helper with a deliberately-malformed
    predicates list (non-dict entries mixed in)."""
    from backend.agents.rlm.report import _render_evidence_section  # noqa: PLC0415

    report = _make_report(
        evidence_bundle=None,
        validation={"status": "vetoed", "veto_set": ["x"], "predicates": ["garbage", 42, None]},
    )
    text = _render_evidence_section(report)  # must not raise
    assert "## Provenance & Evidence" in text
