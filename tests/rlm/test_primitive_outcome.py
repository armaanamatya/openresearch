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
