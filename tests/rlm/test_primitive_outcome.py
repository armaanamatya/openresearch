import json

import pytest

import backend.agents.rlm.primitives as primitives
from backend.agents.rlm.primitives import (
    PrimitiveOutcome,
    build_environment,
    record_candidate_outcome,
    run_experiment,
    understand_section,
)


def test_primitive_outcome_enum_has_expected_values():
    assert {item.value for item in PrimitiveOutcome} == {
        "ok",
        "partial_evidence",
        "repairable",
        "retryable",
        "fatal",
    }


@pytest.mark.parametrize(
    ("sandbox_result", "expected"),
    [
        ({"success": True, "metrics": {"accuracy": 0.91}, "logs": ""}, PrimitiveOutcome.ok),
        (
            {"success": False, "metrics": {"accuracy": 0.42}, "logs": ""},
            PrimitiveOutcome.partial_evidence,
        ),
        (
            {"success": False, "metrics": {}, "logs": "", "failure_class": "code_bug"},
            PrimitiveOutcome.repairable,
        ),
        (
            {"success": False, "metrics": {}, "logs": "", "failure_class": "transient"},
            PrimitiveOutcome.retryable,
        ),
        (
            {"success": False, "metrics": {}, "logs": "", "failure_class": "balance_too_low"},
            PrimitiveOutcome.fatal,
        ),
        # Phase 0C: an all-cells-errored code bug carries a POPULATED metrics dict
        # (aggregate_cell_metrics always returns non-empty) yet must be repairable —
        # it engages the repair floor instead of being mis-typed partial_evidence.
        (
            {"success": False,
             "metrics": {"status": "failed", "per_model": {"qwen3-1.7b": {"status": "failed"}}},
             "logs": "", "failure_class": "cell_execution_error"},
            PrimitiveOutcome.repairable,
        ),
    ],
)
def test_run_experiment_maps_outcomes(make_context, tmp_path, monkeypatch, sandbox_result, expected):
    ctx = make_context(tmp_path)
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "commands.json").write_text(json.dumps(["python train.py"]), encoding="utf-8")

    async def fake_exec(*args, **kwargs):
        return dict(sandbox_result)

    monkeypatch.setattr(primitives, "_execute_in_sandbox", fake_exec)

    result = run_experiment(str(code_dir), "image:tag", ctx=ctx)

    assert result["outcome"] == expected.value


@pytest.mark.parametrize(
    ("retryable", "expected"),
    [(False, PrimitiveOutcome.fatal), (True, PrimitiveOutcome.retryable)],
)
def test_build_environment_runtime_exception_is_caught_and_mapped(
    make_context,
    tmp_path,
    monkeypatch,
    retryable,
    expected,
):
    from backend.services.runtime.interface import RuntimeCauseKind, SandboxRuntimeError

    ctx = make_context(tmp_path)

    async def raises_runtime_error(*args, **kwargs):
        raise SandboxRuntimeError(
            RuntimeCauseKind.backend_unavailable,
            "backend unavailable",
            retryable=retryable,
        )

    monkeypatch.setattr(primitives, "_build_image", raises_runtime_error)

    result = build_environment({"dockerfile": "FROM python:3.11-slim\n"}, ctx=ctx)

    assert result["ok"] is False
    assert result["outcome"] == expected.value
    assert "backend unavailable" in result["error"]


def test_other_primitive_success_defaults_to_ok(make_context, tmp_path):
    ctx = make_context(tmp_path)

    result = understand_section("We train on MNIST with Adam.", ctx=ctx)

    assert result["outcome"] == PrimitiveOutcome.ok.value


def test_record_candidate_outcome_missing_candidate_id_is_repairable(make_context, tmp_path):
    ctx = make_context(tmp_path)

    result = record_candidate_outcome(candidate_id=None, outcome="declined", ctx=ctx)

    assert result["success"] is False
    assert result["outcome"] == PrimitiveOutcome.repairable.value
    assert result["error"] == "candidate_id missing — pass the most recent proposed candidate"


def test_cell_execution_error_with_metrics_is_repairable_not_partial():
    """Phase 0C: an all-cells-errored code-bug result carries a populated metrics
    dict (aggregate_cell_metrics always returns non-empty), but must classify as
    ``repairable`` so it engages the repair-iteration floor — NOT partial_evidence
    (the metrics-first short-circuit would otherwise skip the floor)."""
    result = {
        "success": False,
        "metrics": {"status": "failed", "per_model": {"qwen3-1.7b": {"status": "failed"}}},
        "failure_class": "cell_execution_error",
        "error": "3 cells failed with non-OOM errors",
    }
    assert (
        primitives._classify_run_experiment_outcome(result)
        is PrimitiveOutcome.repairable
    )


def test_genuine_partial_with_metrics_stays_partial_evidence():
    """Phase 0C regression: the fix must stay NARROW — a partial result whose
    failure_class is NOT a metrics-bearing-repairable class (or has none) keeps its
    partial_evidence typing so some-ok/some-bug runs are not over-repaired."""
    # No failure_class, has metrics → partial_evidence (today's behaviour).
    r1 = {"success": False, "metrics": {"accuracy": 0.42}, "logs": ""}
    assert primitives._classify_run_experiment_outcome(r1) is PrimitiveOutcome.partial_evidence
    # A non-cell_execution_error soft class with metrics → still partial_evidence.
    r2 = {"success": False, "metrics": {"accuracy": 0.4}, "failure_class": "insufficient_train_steps"}
    assert primitives._classify_run_experiment_outcome(r2) is PrimitiveOutcome.partial_evidence


def test_cell_execution_error_in_repairable_and_classifier_sets():
    """Phase 0C: the class is registered everywhere the repair machinery looks."""
    from backend.agents.rlm.failure_classifier import FAILURE_CLASSES, classify_failure
    assert "cell_execution_error" in primitives._RUN_EXPERIMENT_REPAIRABLE_FAILURES
    assert "cell_execution_error" in primitives._METRICS_BEARING_REPAIRABLE_FAILURES
    assert "cell_execution_error" in FAILURE_CLASSES
    # The canonical suggested_fix is wired (classify_failure honours a preset class).
    klass, fix = classify_failure({"success": False, "failure_class": "cell_execution_error"})
    assert klass == "cell_execution_error"
    assert fix and "non-OOM" in fix
