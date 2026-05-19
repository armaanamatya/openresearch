"""Contract test for the partial-final-report path used at Gate 2 fail.

When the orchestrator halts at Gate 2 because the rubric verifier missed
its target, ``_finalize_partial`` writes ``final_report.{json,md}`` via the
existing ``generate_final_report`` + ``write_final_report`` helpers.

We test the *report contract* at the helper-function level: given a
RubricVerification with ``meets_target=False`` and otherwise empty pipeline
state, the writer must land both files on disk, the JSON must parse, the
rubric overall score must round-trip, and ``reproduction_status`` must be a
canonical schema value. This pins the contract without instantiating the
heavy orchestrator class, so the test stays fast and stable across
orchestrator refactors.

See ``docs/design/gate2-finalize-options.md`` (Option A) for the design.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.agents.report_generator import (
    generate_final_report,
    write_final_report,
)
from backend.agents.schemas import RubricAreaScore, RubricVerification


def _partial_verification() -> RubricVerification:
    """Build a RubricVerification with overall well below the 0.70 target."""
    return RubricVerification.from_areas(
        areas=[
            RubricAreaScore(area="method_fidelity", score=0.30, weight=0.4),
            RubricAreaScore(area="experiments", score=0.25, weight=0.3),
            RubricAreaScore(area="evaluation", score=0.20, weight=0.3),
        ],
        rubric_source="generated",
        target_score=0.70,
    )


def test_partial_final_report_writes_files_when_rubric_missed_target(
    tmp_path: Path,
) -> None:
    project_id = "prj_test_partial"
    project_dir = tmp_path / project_id
    project_dir.mkdir(parents=True)

    verification = _partial_verification()
    # Sanity: this fixture must represent the Gate-2-fail case the helper handles.
    assert verification.meets_target is False
    assert verification.overall_score < verification.target_score

    final_report = generate_final_report(
        project_id,
        paper_claim_map=None,
        experiment_artifacts=None,
        improvement_hypotheses=[],
        path_results=[],
        research_map=None,
        project_dir=project_dir,
        baseline_verification=verification,
    )

    write_final_report(final_report, project_dir)

    json_path = project_dir / "final_report.json"
    md_path = project_dir / "final_report.md"
    assert json_path.exists(), "Option A requires final_report.json on disk"
    assert md_path.exists(), "Option A requires final_report.md on disk"

    payload = json.loads(json_path.read_text(encoding="utf-8"))

    # The verifier-supplied baseline score must round-trip into the report's
    # `baseline_rubric_verification` field — that's the path
    # finalize_benchmark() reads from in live_runs.py. (The top-level
    # `rubric_overall_score` is a separate, artifact-derived rubric and is
    # legitimately 0.0 when paper_claim_map and experiment_artifacts are
    # absent — not under test here.)
    baseline_rubric = payload.get("baseline_rubric_verification") or {}
    assert baseline_rubric.get("overall_score") == pytest.approx(
        verification.overall_score, abs=1e-3
    ), "Baseline RubricVerification.overall_score must appear in baseline_rubric_verification"
    assert baseline_rubric.get("meets_target") is False, (
        "Baseline RubricVerification.meets_target must be preserved"
    )

    # reproduction_status must be a canonical schema value (verified | partial | failed).
    # With no paper_metrics and a low rubric, we expect non-verified — but the
    # exact split between partial vs failed depends on _compute_reproduction_score's
    # vacuous-input behavior, which is not under test here.
    assert payload["reproduction_status"] in {"verified", "partial", "failed"}, (
        f"reproduction_status must be a canonical schema value, "
        f"got {payload['reproduction_status']!r}"
    )


def test_partial_final_report_files_are_utf8_safe(tmp_path: Path) -> None:
    """Regression: the report writers must use utf-8 explicitly so non-Latin
    rubric content (arrows, em-dashes) doesn't crash on Windows cp1252.
    Tier 1's encoding cleanup forced this, and the partial path goes through
    the same writers — pin the behavior with a content sample."""
    project_id = "prj_test_partial_utf8"
    project_dir = tmp_path / project_id
    project_dir.mkdir(parents=True)

    verification = RubricVerification.from_areas(
        areas=[
            RubricAreaScore(
                area="method_fidelity",
                score=0.3,
                weight=1.0,
                weak_points=[
                    "Loss curve diverged → NaN at step 12k",
                    "α coefficient mismatched paper Table 1",
                ],
            ),
        ],
        rubric_source="generated",
        target_score=0.70,
    )

    final_report = generate_final_report(
        project_id,
        paper_claim_map=None,
        experiment_artifacts=None,
        improvement_hypotheses=[],
        path_results=[],
        research_map=None,
        project_dir=project_dir,
        baseline_verification=verification,
    )
    write_final_report(final_report, project_dir)

    json_path = project_dir / "final_report.json"
    md_path = project_dir / "final_report.md"

    # Round-trip through utf-8 explicitly — would crash on a cp1252-encoded
    # file the way the original hermes_audit/storage.py bug did.
    raw_json = json_path.read_text(encoding="utf-8")
    json.loads(raw_json)
    md_path.read_text(encoding="utf-8")  # decodable, that's all we need.
