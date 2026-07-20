"""Tests for backend.agents.rlm.attempt_assessment (campaign Unit 4).

Fixtures are synthetic run dirs built under tmp_path -- never under runs/.
Where a fixture's expected value depends on the REAL composed module's own
hashing/attribution logic (evidence_fingerprint, rubric_sha256, attribute_failure),
the test computes the expected value by calling that real module directly rather
than hardcoding a derived hash, so the test stays correct if that module's
internal algorithm ever changes shape (only the CONTRACT is asserted).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.agents.rlm import attempt_assessment
from backend.agents.rlm import evidence_audit, external_validator, failure_attribution
from backend.agents.rlm.attempt_assessment import (
    AttemptAssessment,
    CampaignSpend,
    ReportDigest,
    ValidatorStatus,
)
from backend.agents.rlm.rubric_gen import CANARY_LEAF_ID


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(r) + "\n" for r in rows)
    path.write_text(text)


def _minimal_final_report(run_dir: Path, **overrides: object) -> dict:
    data = {
        "verdict": "partial",
        "scope": {"requested": "", "ran": [], "gaps": []},
        "rubric": {"overall_score": 0.3, "target_score": 0.6, "meets_target": False},
    }
    data.update(overrides)
    _write_json(run_dir / "final_report.json", data)
    return data


def _bare_assessment(*, hard_quarantined: bool, soft_quarantined: bool) -> AttemptAssessment:
    return AttemptAssessment(
        attempt_n=1,
        driver="live_cli",
        project_id="p1",
        directives_sha256="deadbeef",
        final_report=None,
        evidence_predicates={},
        guard_flags={},
        validator=ValidatorStatus(status="missing", fingerprint=None, fresh=False),
        leaf_pass_count=None,
        leaf_vector_ref=None,
        failure_class=None,
        failure_signature=None,
        failure_scope=None,
        cost=CampaignSpend(),
        rubric_sha256_ok=None,
        hard_quarantined=hard_quarantined,
        soft_quarantined=soft_quarantined,
        quarantine_reasons=(),
    )


# ---------------------------------------------------------------------------
# Full clean-run fixture
# ---------------------------------------------------------------------------


def test_clean_run_assesses_unquarantined(tmp_path: Path) -> None:
    run_dir = tmp_path / "attempt_1"
    run_dir.mkdir()

    metrics = {"accuracy": 0.83, "success_rate": 0.9}
    _write_json(run_dir / "code" / "metrics.json", metrics)
    _write_json(run_dir / "code" / "provenance.json", {"ok": True})

    _minimal_final_report(
        run_dir,
        verdict="reproduced",
        scope={"requested": "full paper", "ran": ["qwen3-1.7b/alfworld"], "gaps": []},
        rubric={"overall_score": 0.82, "target_score": 0.6, "meets_target": True},
        implementation_verdict="faithful",
        replication_verdict="reproduced",
        started_at="2026-07-01T08:00:00+00:00",
        completed_at="2026-07-01T09:00:00+00:00",
    )

    exp_row = {
        "timestamp": "2026-07-01T08:30:00+00:00",
        "success": True,
        "metrics": metrics,
        "wall_time_s": 1800.0,
        "model_id": "qwen3-1.7b",
        "eval_env": "alfworld",
    }
    _write_jsonl(run_dir / "experiment_runs.jsonl", [exp_row])

    _write_json(
        run_dir / "rubric_evaluation.json",
        {
            "overall_score": 0.82,
            "leaf_scores": [
                {"id": "l1", "score": 0.9},
                {"id": "l2", "score": 0.4},
                {"id": "l3", "score": 0.5},
            ],
        },
    )

    rubric_tree = {"areas": [{"name": "training", "weight": 1.0}]}
    _write_json(run_dir / "generated_rubric.json", rubric_tree)
    pinned = attempt_assessment.rubric_sha256(rubric_tree)

    _write_jsonl(
        run_dir / "cost_ledger.jsonl",
        [{"cost_usd": 0.5, "primitive": "implement_baseline"},
         {"cost_usd": 1.25, "primitive": "verify_against_rubric"}],
    )

    _write_json(run_dir / "rlm_state" / "gpu_plan.json",
                {"gpu_count": 2, "sku_usd_per_hr": 1.2, "total_usd_per_hr": 2.4})
    _write_json(run_dir / "timing.json", {"gpu_hours": 0.5})
    _write_json(run_dir / "demo_status.json",
                {"startedAt": "2026-07-01T10:00:00+00:00", "completedAt": "2026-07-01T12:00:00+00:00"})

    fp = external_validator.evidence_fingerprint(metrics)
    external_validator.persist_verdict(
        run_dir,
        external_validator.ValidatorVerdict(
            status="clean", veto_set=[], predicates=[],
            panel_models=["azure-foundry:grok-4.3"], separation="independent",
            evidence_fingerprint=fp,
        ),
    )

    expected_fa = failure_attribution.attribute_failure(exp_row, arxiv_id="2605.15155")

    assessment = attempt_assessment.assess_attempt(
        run_dir, attempt_n=1, driver="live_cli", project_id="proj-1",
        directives_sha256="abc123", pinned_rubric_sha256=pinned, arxiv_id="2605.15155",
    )

    assert assessment.attempt_n == 1
    assert assessment.driver == "live_cli"
    assert assessment.project_id == "proj-1"
    assert assessment.directives_sha256 == "abc123"

    assert assessment.final_report == ReportDigest(
        score=0.82, target=0.6, meets_target=True,
        implementation_verdict="faithful", replication_verdict="reproduced",
        verdict="reproduced", stop_reason=None, exclusions=(),
        path=str(run_dir / "final_report.json"),
    )

    assert assessment.evidence_predicates == {
        "backed_by_ledger": True,
        "provenance_present": True,
        "metrics_non_degenerate": True,
        "metric_keys_real": True,
        "rerun_agrees": None,
        "run_level_clean": True,
    }
    assert assessment.guard_flags == {
        "fabrication": False, "all_models_failed": False,
        "env_unavailable": False, "no_learning_signal": False,
        "canary_tripped": False,
    }
    assert assessment.validator == ValidatorStatus(status="clean", fingerprint=fp, fresh=True)
    assert assessment.leaf_pass_count == 2
    assert assessment.leaf_vector_ref == str(run_dir / "rubric_evaluation.json")

    assert assessment.failure_class == expected_fa.root_cause
    assert assessment.failure_signature == expected_fa.signature
    assert assessment.failure_scope == expected_fa.scope

    assert assessment.cost.llm_usd == pytest.approx(1.75)
    assert assessment.cost.gpu_usd == pytest.approx(1.2)
    assert assessment.cost.gpu_hours == pytest.approx(0.5)
    assert assessment.cost.wall_s == pytest.approx(7200.0)

    assert assessment.rubric_sha256_ok is True
    assert assessment.hard_quarantined is False
    assert assessment.soft_quarantined is False
    assert assessment.quarantine_reasons == ()
    assert assessment.grade_usable_for_terminal is True
    assert assessment.usable_for_seeding is True


# ---------------------------------------------------------------------------
# ReportDigest / report_missing
# ---------------------------------------------------------------------------


def test_missing_report_is_report_missing_infra(tmp_path: Path) -> None:
    run_dir = tmp_path / "attempt_2"
    run_dir.mkdir()

    assessment = attempt_assessment.assess_attempt(
        run_dir, attempt_n=2, driver="live_cli", project_id="proj-1",
        directives_sha256="abc123", pinned_rubric_sha256=None,
    )

    assert assessment.final_report is None
    assert assessment.failure_class == "report_missing"
    assert assessment.failure_signature == "report_missing"
    assert assessment.failure_scope == "infra"
    assert assessment.leaf_pass_count is None
    assert assessment.leaf_vector_ref is None
    assert assessment.cost.llm_usd == pytest.approx(0.0)
    assert assessment.cost.gpu_usd == pytest.approx(0.0)
    assert assessment.cost.gpu_hours == pytest.approx(0.0)
    assert assessment.cost.wall_s == pytest.approx(0.0)
    # No experiment rows / dashboard events at all -> guard flags stay False,
    # but the validator is independently absent -> soft-quarantined.
    assert assessment.hard_quarantined is False
    assert assessment.soft_quarantined is True
    assert "validator:missing" in assessment.quarantine_reasons


def test_unparseable_report_is_report_missing_infra(tmp_path: Path) -> None:
    run_dir = tmp_path / "attempt_2b"
    run_dir.mkdir()
    (run_dir / "final_report.json").write_text("not json{{{")

    assessment = attempt_assessment.assess_attempt(
        run_dir, attempt_n=1, driver="live_cli", project_id="proj-1",
        directives_sha256="abc123", pinned_rubric_sha256=None,
    )

    assert assessment.final_report is None
    assert assessment.failure_class == "report_missing"


# ---------------------------------------------------------------------------
# Fabrication guard + sub-source disambiguation
# ---------------------------------------------------------------------------

_FABRICATION_SUBSOURCE_CASES = [
    pytest.param(
        "fabrication_suspected: run_experiment reported success but the result-claiming "
        "metrics are all-zero (keys: accuracy). Re-implement training.",
        "zero_metrics", id="zero_metrics_all_zero",
    ),
    pytest.param(
        "fabrication_suspected: metrics are constant (0.5) across all cells (keys: accuracy).",
        "zero_metrics", id="zero_metrics_constant",
    ),
    pytest.param(
        "fabrication_suspected: metrics carry only placeholder keys (total_length, chunk_count) "
        "and no real paper metric is present.",
        "stub_metrics", id="stub_metrics",
    ),
    pytest.param(
        "fabrication_suspected: peak VRAM 0.512 GiB is below the fabrication floor for claimed "
        "gpu training.",
        "vram", id="vram",
    ),
    pytest.param(
        "fabrication_suspected: metric-semantics violation — accuracy=83.0 outside [0,1].",
        "metric_semantics", id="metric_semantics",
    ),
    pytest.param(
        "fabrication_suspected: eval-metric provenance check failed — sidecar missing.",
        "eval_provenance", id="eval_provenance",
    ),
]


@pytest.mark.parametrize("error_text,expected_subsource", _FABRICATION_SUBSOURCE_CASES)
def test_latest_row_fabrication_hard_quarantines_with_subsource(
    tmp_path: Path, error_text: str, expected_subsource: str
) -> None:
    run_dir = tmp_path / "attempt_3"
    run_dir.mkdir()
    _minimal_final_report(run_dir)
    _write_jsonl(
        run_dir / "experiment_runs.jsonl",
        [{"success": False, "failure_class": "fabrication_suspected", "error": error_text}],
    )

    assessment = attempt_assessment.assess_attempt(
        run_dir, attempt_n=1, driver="live_cli", project_id="proj-1",
        directives_sha256="abc123", pinned_rubric_sha256=None,
    )

    assert assessment.guard_flags["fabrication"] is True
    assert assessment.hard_quarantined is True
    assert f"guard:fabrication:{expected_subsource}" in assessment.quarantine_reasons


def test_latest_row_fabrication_unrecognized_error_falls_back_to_fabrication(tmp_path: Path) -> None:
    run_dir = tmp_path / "attempt_3b"
    run_dir.mkdir()
    _minimal_final_report(run_dir)
    _write_jsonl(
        run_dir / "experiment_runs.jsonl",
        [{"success": False, "failure_class": "fabrication_suspected",
          "error": "fabrication_suspected: unspecified anomaly in the reported metrics."}],
    )

    assessment = attempt_assessment.assess_attempt(
        run_dir, attempt_n=1, driver="live_cli", project_id="proj-1",
        directives_sha256="abc123", pinned_rubric_sha256=None,
    )

    assert "guard:fabrication:fabrication" in assessment.quarantine_reasons


def test_superseded_fabrication_row_does_not_trip(tmp_path: Path) -> None:
    run_dir = tmp_path / "attempt_4"
    run_dir.mkdir()
    _minimal_final_report(run_dir)
    _write_jsonl(
        run_dir / "experiment_runs.jsonl",
        [
            {"success": False, "failure_class": "fabrication_suspected",
             "error": "fabrication_suspected: metrics are all-zero (keys: accuracy)."},
            {"success": True, "metrics": {"accuracy": 0.9}, "wall_time_s": 100.0},
        ],
    )

    assessment = attempt_assessment.assess_attempt(
        run_dir, attempt_n=1, driver="live_cli", project_id="proj-1",
        directives_sha256="abc123", pinned_rubric_sha256=None,
    )

    assert assessment.guard_flags["fabrication"] is False
    assert assessment.hard_quarantined is False
    assert not any(r.startswith("guard:fabrication:") for r in assessment.quarantine_reasons)


def test_all_models_failed_latest_row_trips(tmp_path: Path) -> None:
    run_dir = tmp_path / "attempt_5"
    run_dir.mkdir()
    _minimal_final_report(run_dir)
    _write_jsonl(
        run_dir / "experiment_runs.jsonl",
        [{"success": False, "failure_class": "all_models_failed",
          "error": "all_models_failed: per_model has 2 model(s) (a, b) but NONE ok"}],
    )

    assessment = attempt_assessment.assess_attempt(
        run_dir, attempt_n=1, driver="live_cli", project_id="proj-1",
        directives_sha256="abc123", pinned_rubric_sha256=None,
    )

    assert assessment.guard_flags["all_models_failed"] is True
    assert assessment.hard_quarantined is True
    assert "guard:all_models_failed" in assessment.quarantine_reasons


# ---------------------------------------------------------------------------
# run_warning codes, both dashboard_events.jsonl shapes
# ---------------------------------------------------------------------------


def test_run_warning_codes_read_in_both_shapes(tmp_path: Path) -> None:
    run_dir = tmp_path / "attempt_6"
    run_dir.mkdir()
    _minimal_final_report(run_dir)
    _write_jsonl(
        run_dir / "dashboard_events.jsonl",
        [
            {"event": "run_warning", "timestamp": "2026-07-01T00:00:00+00:00",
             "level": "warn", "code": "no_learning_signal", "message": "flat shape"},
            {"ts": "2026-07-01T00:01:00+00:00", "event": "run_warning",
             "data": {"code": "env_unavailable", "message": "wrapped shape"}},
        ],
    )

    assessment = attempt_assessment.assess_attempt(
        run_dir, attempt_n=1, driver="live_cli", project_id="proj-1",
        directives_sha256="abc123", pinned_rubric_sha256=None,
    )

    assert assessment.guard_flags["no_learning_signal"] is True
    assert assessment.guard_flags["env_unavailable"] is True
    assert assessment.guard_flags["fabrication"] is False
    assert assessment.guard_flags["all_models_failed"] is False


# ---------------------------------------------------------------------------
# Validator (F4)
# ---------------------------------------------------------------------------


def test_validator_missing_is_soft_quarantine(tmp_path: Path) -> None:
    run_dir = tmp_path / "attempt_7"
    run_dir.mkdir()
    _minimal_final_report(run_dir)

    assessment = attempt_assessment.assess_attempt(
        run_dir, attempt_n=1, driver="live_cli", project_id="proj-1",
        directives_sha256="abc123", pinned_rubric_sha256=None,
    )

    assert assessment.validator == ValidatorStatus(status="missing", fingerprint=None, fresh=False)
    assert assessment.soft_quarantined is True
    assert "validator:missing" in assessment.quarantine_reasons
    assert assessment.grade_usable_for_terminal is False


def test_validator_stale_fingerprint_is_soft_quarantine(tmp_path: Path) -> None:
    run_dir = tmp_path / "attempt_8"
    run_dir.mkdir()
    _minimal_final_report(run_dir)
    metrics = {"accuracy": 0.5}
    _write_json(run_dir / "code" / "metrics.json", metrics)
    real_fp = external_validator.evidence_fingerprint(metrics)
    stale_fp = "f" * 64
    assert stale_fp != real_fp
    external_validator.persist_verdict(
        run_dir,
        external_validator.ValidatorVerdict(
            status="clean", veto_set=[], predicates=[], panel_models=["p"],
            separation="independent", evidence_fingerprint=stale_fp,
        ),
    )

    assessment = attempt_assessment.assess_attempt(
        run_dir, attempt_n=1, driver="live_cli", project_id="proj-1",
        directives_sha256="abc123", pinned_rubric_sha256=None,
    )

    assert assessment.validator.status == "clean"
    assert assessment.validator.fresh is False
    assert assessment.soft_quarantined is True
    assert assessment.quarantine_reasons == ("validator:stale",)


def test_validator_vetoed_is_soft_quarantine_with_reason(tmp_path: Path) -> None:
    run_dir = tmp_path / "attempt_9"
    run_dir.mkdir()
    _minimal_final_report(run_dir)
    metrics = {"accuracy": 0.5}
    _write_json(run_dir / "code" / "metrics.json", metrics)
    fp = external_validator.evidence_fingerprint(metrics)
    external_validator.persist_verdict(
        run_dir,
        external_validator.ValidatorVerdict(
            status="vetoed", veto_set=["accuracy"],
            predicates=[external_validator.PredicateVerdict(
                predicate="not_all_constant", metric_ref="accuracy",
                violated=True, detail="constant",
            )],
            panel_models=["p"], separation="independent", evidence_fingerprint=fp,
        ),
    )

    assessment = attempt_assessment.assess_attempt(
        run_dir, attempt_n=1, driver="live_cli", project_id="proj-1",
        directives_sha256="abc123", pinned_rubric_sha256=None,
    )

    assert assessment.validator.status == "vetoed"
    assert assessment.validator.fresh is True
    assert assessment.soft_quarantined is True
    assert assessment.quarantine_reasons == ("validator:vetoed",)


def test_validator_clean_fresh_is_usable(tmp_path: Path) -> None:
    run_dir = tmp_path / "attempt_10"
    run_dir.mkdir()
    _minimal_final_report(run_dir)
    metrics = {"accuracy": 0.5}
    _write_json(run_dir / "code" / "metrics.json", metrics)
    fp = external_validator.evidence_fingerprint(metrics)
    external_validator.persist_verdict(
        run_dir,
        external_validator.ValidatorVerdict(
            status="clean", veto_set=[], predicates=[], panel_models=["p"],
            separation="independent", evidence_fingerprint=fp,
        ),
    )

    assessment = attempt_assessment.assess_attempt(
        run_dir, attempt_n=1, driver="live_cli", project_id="proj-1",
        directives_sha256="abc123", pinned_rubric_sha256=None,
    )

    assert assessment.validator == ValidatorStatus(status="clean", fingerprint=fp, fresh=True)
    assert assessment.soft_quarantined is False
    assert not any(r.startswith("validator:") for r in assessment.quarantine_reasons)


# ---------------------------------------------------------------------------
# Rubric integrity
# ---------------------------------------------------------------------------


def test_rubric_hash_mismatch_hard_quarantines(tmp_path: Path) -> None:
    run_dir = tmp_path / "attempt_11"
    run_dir.mkdir()
    _minimal_final_report(run_dir)
    tree = {"areas": [{"name": "training"}]}
    _write_json(run_dir / "generated_rubric.json", tree)
    wrong_pin = attempt_assessment.rubric_sha256({"areas": [{"name": "something-else"}]})

    assessment = attempt_assessment.assess_attempt(
        run_dir, attempt_n=1, driver="live_cli", project_id="proj-1",
        directives_sha256="abc123", pinned_rubric_sha256=wrong_pin,
    )

    assert assessment.rubric_sha256_ok is False
    assert assessment.hard_quarantined is True
    assert "rubric_integrity_mismatch" in assessment.quarantine_reasons

    # Sub-case: the pin is set but generated_rubric.json is entirely absent.
    run_dir2 = tmp_path / "attempt_11b"
    run_dir2.mkdir()
    _minimal_final_report(run_dir2)
    assessment2 = attempt_assessment.assess_attempt(
        run_dir2, attempt_n=1, driver="live_cli", project_id="proj-1",
        directives_sha256="abc123", pinned_rubric_sha256=wrong_pin,
    )
    assert assessment2.rubric_sha256_ok is False
    assert assessment2.hard_quarantined is True


def test_no_pin_is_none(tmp_path: Path) -> None:
    run_dir = tmp_path / "attempt_12"
    run_dir.mkdir()
    _minimal_final_report(run_dir)
    _write_json(run_dir / "generated_rubric.json", {"areas": []})

    assessment = attempt_assessment.assess_attempt(
        run_dir, attempt_n=1, driver="live_cli", project_id="proj-1",
        directives_sha256="abc123", pinned_rubric_sha256=None,
    )

    assert assessment.rubric_sha256_ok is None
    assert "rubric_integrity_mismatch" not in assessment.quarantine_reasons
    assert assessment.hard_quarantined is False


# ---------------------------------------------------------------------------
# Leaf vector
# ---------------------------------------------------------------------------


def test_leaf_pass_count_threshold(tmp_path: Path) -> None:
    run_dir = tmp_path / "attempt_13"
    run_dir.mkdir()
    _minimal_final_report(run_dir)
    _write_json(
        run_dir / "rubric_evaluation.json",
        {
            "leaf_scores": [
                {"id": "a", "score": 0.5},          # pass, boundary-inclusive
                {"id": "b", "score": 0.499},         # fail
                {"id": "c", "score": 1.0},           # pass
                {"id": "d", "score": 0.0},           # fail
                {"id": "e", "score": "not-a-number"},  # skipped, never raises
                {"id": "f"},                          # skipped (no score key)
            ],
        },
    )

    assessment = attempt_assessment.assess_attempt(
        run_dir, attempt_n=1, driver="live_cli", project_id="proj-1",
        directives_sha256="abc123", pinned_rubric_sha256=None,
    )

    assert assessment.leaf_pass_count == 2
    assert assessment.leaf_vector_ref == str(run_dir / "rubric_evaluation.json")

    run_dir2 = tmp_path / "attempt_13b"
    run_dir2.mkdir()
    _minimal_final_report(run_dir2)
    assessment2 = attempt_assessment.assess_attempt(
        run_dir2, attempt_n=1, driver="live_cli", project_id="proj-1",
        directives_sha256="abc123", pinned_rubric_sha256=None,
    )
    assert assessment2.leaf_pass_count is None
    assert assessment2.leaf_vector_ref is None


# ---------------------------------------------------------------------------
# Cost reconstruction
# ---------------------------------------------------------------------------


def test_cost_reconstruction_without_gpu_plan_is_zero_gpu_usd(tmp_path: Path) -> None:
    run_dir = tmp_path / "attempt_14"
    run_dir.mkdir()
    _minimal_final_report(run_dir)
    _write_jsonl(
        run_dir / "experiment_runs.jsonl",
        [{"success": True, "wall_time_s": 3600.0}, {"success": True, "wall_time_s": 1800.0}],
    )
    _write_jsonl(run_dir / "cost_ledger.jsonl", [{"cost_usd": 0.1}, {"cost_usd": 0.2}])

    assessment = attempt_assessment.assess_attempt(
        run_dir, attempt_n=1, driver="live_cli", project_id="proj-1",
        directives_sha256="abc123", pinned_rubric_sha256=None,
    )

    # gpu_count defaults to 1 with no gpu_plan.json -> gpu_hours is still derived
    # from wall_time_s, but the $/hr rate is unknown -> gpu_usd is forced to 0.
    assert assessment.cost.gpu_hours == pytest.approx(5400.0 / 3600.0)
    assert assessment.cost.gpu_usd == pytest.approx(0.0)
    assert assessment.cost.llm_usd == pytest.approx(0.3)


def test_cost_uses_timing_gpu_hours_when_present_else_rows(tmp_path: Path) -> None:
    run_dir_a = tmp_path / "attempt_15a"
    run_dir_a.mkdir()
    _minimal_final_report(run_dir_a)
    _write_json(run_dir_a / "rlm_state" / "gpu_plan.json", {"gpu_count": 2, "total_usd_per_hr": 3.0})
    _write_json(run_dir_a / "timing.json", {"gpu_hours": 4.0})
    # A wall_time_s that would give a very different fallback (100*2/3600 != 4.0),
    # proving timing.json truly takes priority over the row-sum fallback.
    _write_jsonl(run_dir_a / "experiment_runs.jsonl", [{"success": True, "wall_time_s": 100.0}])

    assessment_a = attempt_assessment.assess_attempt(
        run_dir_a, attempt_n=1, driver="live_cli", project_id="proj-1",
        directives_sha256="abc123", pinned_rubric_sha256=None,
    )
    assert assessment_a.cost.gpu_hours == pytest.approx(4.0)
    assert assessment_a.cost.gpu_usd == pytest.approx(12.0)

    run_dir_b = tmp_path / "attempt_15b"
    run_dir_b.mkdir()
    _minimal_final_report(run_dir_b)
    _write_json(run_dir_b / "rlm_state" / "gpu_plan.json", {"gpu_count": 2, "total_usd_per_hr": 3.0})
    _write_jsonl(
        run_dir_b / "experiment_runs.jsonl",
        [{"success": True, "wall_time_s": 1800.0}, {"success": True, "wall_time_s": 1800.0}],
    )

    assessment_b = attempt_assessment.assess_attempt(
        run_dir_b, attempt_n=1, driver="live_cli", project_id="proj-1",
        directives_sha256="abc123", pinned_rubric_sha256=None,
    )
    assert assessment_b.cost.gpu_hours == pytest.approx(2.0)
    assert assessment_b.cost.gpu_usd == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------


def test_to_dict_from_dict_roundtrip() -> None:
    original = AttemptAssessment(
        attempt_n=3,
        driver="live_cli",
        project_id="proj-9",
        directives_sha256="feedface",
        final_report=ReportDigest(
            score=0.71, target=0.6, meets_target=True,
            implementation_verdict="faithful", replication_verdict="reproduced",
            verdict="reproduced", stop_reason="wall_clock_watchdog",
            exclusions=("imagenet: model unavailable", "coco: dataset unobtainable"),
            path="/runs/proj-9/final_report.json",
        ),
        evidence_predicates={
            "backed_by_ledger": True, "provenance_present": True,
            "metrics_non_degenerate": True, "metric_keys_real": True,
            "rerun_agrees": None, "run_level_clean": True,
        },
        guard_flags={"fabrication": False, "all_models_failed": False,
                     "env_unavailable": True, "no_learning_signal": False},
        validator=ValidatorStatus(status="vetoed", fingerprint="abc123fingerprint", fresh=True),
        leaf_pass_count=7,
        leaf_vector_ref="/runs/proj-9/rubric_evaluation.json",
        failure_class="ok",
        failure_signature="98a323faf531c7a1",
        failure_scope="method",
        cost=CampaignSpend(llm_usd=2.5, gpu_usd=6.25, gpu_hours=2.5, wall_s=9000.0),
        rubric_sha256_ok=True,
        hard_quarantined=False,
        soft_quarantined=True,
        quarantine_reasons=("validator:vetoed",),
    )

    payload = original.to_dict()
    # to_dict() must be genuinely JSON-safe (tuples etc. survive a real json pass).
    round_tripped = json.loads(json.dumps(payload))
    restored = AttemptAssessment.from_dict(round_tripped)

    assert restored == original
    assert isinstance(restored.final_report.exclusions, tuple)
    assert isinstance(restored.quarantine_reasons, tuple)
    assert isinstance(restored.cost, CampaignSpend)
    assert isinstance(restored.validator, ValidatorStatus)


def test_to_dict_from_dict_roundtrip_with_none_report() -> None:
    original = _bare_assessment(hard_quarantined=True, soft_quarantined=False)
    payload = json.loads(json.dumps(original.to_dict()))
    assert "branch_type" not in payload
    assert "is_safety_bracket" not in payload
    restored = AttemptAssessment.from_dict(payload)
    assert restored == original
    assert restored.final_report is None


def test_scheduler_assessment_metadata_is_strict_and_legacy_absence_defaults() -> None:
    original = _bare_assessment(hard_quarantined=False, soft_quarantined=False)
    typed = AttemptAssessment(
        **{**original.__dict__, "branch_type": "ambiguity"}
    )
    assert AttemptAssessment.from_dict(typed.to_dict()) == typed
    assert typed.to_dict()["branch_type"] == "ambiguity"

    safety = AttemptAssessment(
        **{**original.__dict__, "is_safety_bracket": True}
    )
    assert safety.to_dict()["is_safety_bracket"] is True
    assert AttemptAssessment.from_dict(safety.to_dict()) == safety

    legacy = original.to_dict()
    assert AttemptAssessment.from_dict(legacy).branch_type == "faithful"
    assert AttemptAssessment.from_dict(legacy).is_safety_bracket is False

    malformed_branch = {**legacy, "branch_type": "grade-derived"}
    malformed_safety = {**legacy, "is_safety_bracket": "true"}
    with pytest.raises(ValueError):
        AttemptAssessment.from_dict(malformed_branch)
    with pytest.raises(ValueError):
        AttemptAssessment.from_dict(malformed_safety)
    with pytest.raises(ValueError):
        AttemptAssessment(
            **{**original.__dict__, "branch_type": "discovery", "is_safety_bracket": True}
        )


# ---------------------------------------------------------------------------
# Derived usability properties
# ---------------------------------------------------------------------------


def test_grade_usable_for_terminal_and_seeding_derivations() -> None:
    clean = _bare_assessment(hard_quarantined=False, soft_quarantined=False)
    assert clean.grade_usable_for_terminal is True
    assert clean.usable_for_seeding is True

    hard_only = _bare_assessment(hard_quarantined=True, soft_quarantined=False)
    assert hard_only.grade_usable_for_terminal is False
    assert hard_only.usable_for_seeding is False

    soft_only = _bare_assessment(hard_quarantined=False, soft_quarantined=True)
    assert soft_only.grade_usable_for_terminal is False
    assert soft_only.usable_for_seeding is True

    both = _bare_assessment(hard_quarantined=True, soft_quarantined=True)
    assert both.grade_usable_for_terminal is False
    assert both.usable_for_seeding is False


# ---------------------------------------------------------------------------
# Fail-closed evidence-audit exception
# ---------------------------------------------------------------------------


def test_evidence_audit_exception_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "attempt_16"
    run_dir.mkdir()
    _minimal_final_report(run_dir)

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(evidence_audit, "audit_evidence_from_dir", _raise)

    assessment = attempt_assessment.assess_attempt(
        run_dir, attempt_n=1, driver="live_cli", project_id="proj-1",
        directives_sha256="abc123", pinned_rubric_sha256=None,
    )

    assert assessment.evidence_predicates == {
        "backed_by_ledger": False,
        "provenance_present": False,
        "metrics_non_degenerate": False,
        "metric_keys_real": False,
        "rerun_agrees": False,
        "run_level_clean": False,
    }
    assert "evidence_audit_error:RuntimeError" in assessment.quarantine_reasons


# ---------------------------------------------------------------------------
# per_claim + exclusions_detail (spec §12 locked decision 9)
# ---------------------------------------------------------------------------


def test_per_claim_projected_from_reproducibility_block(tmp_path: Path) -> None:
    run_dir = tmp_path / "attempt_17"
    run_dir.mkdir()
    _minimal_final_report(
        run_dir,
        reproducibility={
            "schema_version": 2,
            "per_claim": [
                {
                    "claim_id": "table1_row2_alfworld_acc",
                    "status": "reproduced",
                    "credit": 1.0,
                    "reason": "measured mean within CI of the paper's claim",
                    "eligible": True,
                    "measured_mean": 0.71,
                    "ci_low": 0.65,
                    "ci_high": 0.77,
                },
                {
                    "claim_id": "table1_row3_webshop_acc",
                    "status": "contradicted",
                    "credit": 0.0,
                    "reason": "measured mean far below the paper's claim",
                    "eligible": True,
                    "measured_mean": 0.12,
                    "ci_low": None,
                    "ci_high": None,
                },
            ],
        },
    )

    assessment = attempt_assessment.assess_attempt(
        run_dir, attempt_n=1, driver="live_cli", project_id="proj-1",
        directives_sha256="abc123", pinned_rubric_sha256=None,
    )

    assert assessment.final_report is not None
    assert assessment.final_report.per_claim == (
        {
            "claim_id": "table1_row2_alfworld_acc", "status": "reproduced", "credit": 1.0,
            "eligible": True, "measured_mean": 0.71, "ci_low": 0.65, "ci_high": 0.77,
        },
        {
            "claim_id": "table1_row3_webshop_acc", "status": "contradicted", "credit": 0.0,
            "eligible": True, "measured_mean": 0.12, "ci_low": None, "ci_high": None,
        },
    )
    # `reason` is LLM-authored rationale prose -- it must never survive into
    # the digest (this module's "no prose" contract).
    assert all("reason" not in c for c in assessment.final_report.per_claim)


def test_per_claim_absent_reproducibility_block_is_empty(tmp_path: Path) -> None:
    run_dir = tmp_path / "attempt_17b"
    run_dir.mkdir()
    _minimal_final_report(run_dir)  # no "reproducibility" key at all

    assessment = attempt_assessment.assess_attempt(
        run_dir, attempt_n=1, driver="live_cli", project_id="proj-1",
        directives_sha256="abc123", pinned_rubric_sha256=None,
    )

    assert assessment.final_report is not None
    assert assessment.final_report.per_claim == ()


def test_exclusions_detail_projected_from_metrics_scope(tmp_path: Path) -> None:
    run_dir = tmp_path / "attempt_18"
    run_dir.mkdir()
    _minimal_final_report(run_dir)
    _write_json(
        run_dir / "code" / "metrics.json",
        {
            "accuracy": 0.5,
            "scope": {
                "exclusions": [
                    {
                        "item": "ImageNet", "axis": "dataset", "kind": "capacity_vram",
                        "reason": "24GB budget exceeded", "verified": True,
                        "evidence": "est_vram_gb=40 > budget=24",
                    },
                    {
                        "item": "COCO", "axis": "dataset", "kind": "dataset_dead",
                        "reason": "agent-declared, uncorroborated", "verified": False,
                        "evidence": "",
                    },
                ],
            },
        },
    )

    assessment = attempt_assessment.assess_attempt(
        run_dir, attempt_n=1, driver="live_cli", project_id="proj-1",
        directives_sha256="abc123", pinned_rubric_sha256=None,
    )

    assert assessment.final_report is not None
    assert assessment.final_report.exclusions_detail == (
        {"axis": "dataset", "item": "ImageNet", "kind": "capacity_vram",
         "verified": True, "reason": "24GB budget exceeded"},
        {"axis": "dataset", "item": "COCO", "kind": "dataset_dead",
         "verified": False, "reason": "agent-declared, uncorroborated"},
    )
    # `evidence` (raw arithmetic / OOM signature) must never survive into the
    # digest.
    assert all("evidence" not in e for e in assessment.final_report.exclusions_detail)


def test_exclusions_detail_absent_scope_is_empty(tmp_path: Path) -> None:
    run_dir = tmp_path / "attempt_18b"
    run_dir.mkdir()
    _minimal_final_report(run_dir)
    _write_json(run_dir / "code" / "metrics.json", {"accuracy": 0.5})  # no "scope" key

    assessment = attempt_assessment.assess_attempt(
        run_dir, attempt_n=1, driver="live_cli", project_id="proj-1",
        directives_sha256="abc123", pinned_rubric_sha256=None,
    )

    assert assessment.final_report is not None
    assert assessment.final_report.exclusions_detail == ()


def test_report_digest_per_claim_and_exclusions_detail_roundtrip() -> None:
    original = AttemptAssessment(
        attempt_n=4,
        driver="live_cli",
        project_id="proj-10",
        directives_sha256="cafef00d",
        final_report=ReportDigest(
            score=0.71, target=0.6, meets_target=True,
            implementation_verdict="faithful", replication_verdict="reproduced",
            verdict="reproduced", stop_reason=None,
            exclusions=("legacy string exclusion",),
            path="/runs/proj-10/final_report.json",
            per_claim=(
                {"claim_id": "c1", "status": "reproduced", "credit": 1.0,
                 "eligible": True, "measured_mean": 0.71, "ci_low": 0.65, "ci_high": 0.77},
            ),
            exclusions_detail=(
                {"axis": "dataset", "item": "ImageNet", "kind": "capacity_vram",
                 "verified": True, "reason": "24GB budget exceeded"},
            ),
        ),
        evidence_predicates={},
        guard_flags={},
        validator=ValidatorStatus(status="missing", fingerprint=None, fresh=False),
        leaf_pass_count=None,
        leaf_vector_ref=None,
        failure_class=None,
        failure_signature=None,
        failure_scope=None,
        cost=CampaignSpend(),
        rubric_sha256_ok=None,
        hard_quarantined=False,
        soft_quarantined=False,
        quarantine_reasons=(),
    )

    payload = original.to_dict()
    round_tripped = json.loads(json.dumps(payload))
    restored = AttemptAssessment.from_dict(round_tripped)

    assert restored == original
    assert isinstance(restored.final_report.per_claim, tuple)
    assert isinstance(restored.final_report.per_claim[0], dict)
    assert isinstance(restored.final_report.exclusions_detail, tuple)
    assert isinstance(restored.final_report.exclusions_detail[0], dict)


# ---------------------------------------------------------------------------
# Duration clamp (money-adjacent: feeds cost.wall_s)
# ---------------------------------------------------------------------------


def test_wall_seconds_clamps_negative_duration_to_zero(tmp_path: Path) -> None:
    """U4 review minor: a completedAt before startedAt (clock skew / a bad
    write) must never yield a negative wall-clock spend -- mirrors
    backend/services/pricing/timing.py's max(0.0, ...) clamp."""
    run_dir = tmp_path / "attempt_19"
    run_dir.mkdir()
    _minimal_final_report(run_dir)
    _write_json(
        run_dir / "demo_status.json",
        {"startedAt": "2026-07-01T12:00:00+00:00", "completedAt": "2026-07-01T10:00:00+00:00"},
    )

    assessment = attempt_assessment.assess_attempt(
        run_dir, attempt_n=1, driver="live_cli", project_id="proj-1",
        directives_sha256="abc123", pinned_rubric_sha256=None,
    )

    assert assessment.cost.wall_s == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Canary→ASSESS wiring (spec §10.4, CapCode-style tamper tripwire)
# ---------------------------------------------------------------------------


def test_canary_leaf_credited_hard_quarantines(tmp_path: Path) -> None:
    run_dir = tmp_path / "attempt_20"
    run_dir.mkdir()
    _minimal_final_report(run_dir)
    _write_json(
        run_dir / "rubric_evaluation.json",
        {
            "leaf_scores": [
                {"id": "l1", "score": 0.7},
                {"id": CANARY_LEAF_ID, "score": 0.9},
            ],
        },
    )

    assessment = attempt_assessment.assess_attempt(
        run_dir, attempt_n=1, driver="live_cli", project_id="proj-1",
        directives_sha256="abc123", pinned_rubric_sha256=None,
    )

    assert assessment.guard_flags["canary_tripped"] is True
    assert assessment.hard_quarantined is True
    assert "guard:canary_tripped" in assessment.quarantine_reasons


def test_canary_leaf_zero_score_not_tripped(tmp_path: Path) -> None:
    run_dir = tmp_path / "attempt_21"
    run_dir.mkdir()
    _minimal_final_report(run_dir)
    _write_json(
        run_dir / "rubric_evaluation.json",
        {
            "leaf_scores": [
                {"id": "l1", "score": 0.7},
                {"id": CANARY_LEAF_ID, "score": 0.0},
            ],
        },
    )

    assessment = attempt_assessment.assess_attempt(
        run_dir, attempt_n=1, driver="live_cli", project_id="proj-1",
        directives_sha256="abc123", pinned_rubric_sha256=None,
    )

    assert assessment.guard_flags["canary_tripped"] is False
    assert assessment.hard_quarantined is False
    assert "guard:canary_tripped" not in assessment.quarantine_reasons


def test_canary_absent_leaf_or_no_eval_file_is_false(tmp_path: Path) -> None:
    # A rubric_evaluation.json with no canary leaf at all.
    run_dir = tmp_path / "attempt_22"
    run_dir.mkdir()
    _minimal_final_report(run_dir)
    _write_json(
        run_dir / "rubric_evaluation.json",
        {"leaf_scores": [{"id": "l1", "score": 0.7}]},
    )
    assessment = attempt_assessment.assess_attempt(
        run_dir, attempt_n=1, driver="live_cli", project_id="proj-1",
        directives_sha256="abc123", pinned_rubric_sha256=None,
    )
    assert assessment.guard_flags["canary_tripped"] is False
    assert "guard:canary_tripped" not in assessment.quarantine_reasons

    # No rubric_evaluation.json on disk at all.
    run_dir2 = tmp_path / "attempt_22b"
    run_dir2.mkdir()
    _minimal_final_report(run_dir2)
    assessment2 = attempt_assessment.assess_attempt(
        run_dir2, attempt_n=1, driver="live_cli", project_id="proj-1",
        directives_sha256="abc123", pinned_rubric_sha256=None,
    )
    assert assessment2.guard_flags["canary_tripped"] is False
    assert "guard:canary_tripped" not in assessment2.quarantine_reasons

    # An unparseable rubric_evaluation.json.
    run_dir3 = tmp_path / "attempt_22c"
    run_dir3.mkdir()
    _minimal_final_report(run_dir3)
    (run_dir3 / "rubric_evaluation.json").write_text("not json{{{")
    assessment3 = attempt_assessment.assess_attempt(
        run_dir3, attempt_n=1, driver="live_cli", project_id="proj-1",
        directives_sha256="abc123", pinned_rubric_sha256=None,
    )
    assert assessment3.guard_flags["canary_tripped"] is False
    assert "guard:canary_tripped" not in assessment3.quarantine_reasons
