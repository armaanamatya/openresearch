import json
from unittest.mock import MagicMock

import pytest
from rlm.core.types import CodeBlock, REPLResult, RLMIteration
from rlm.utils.parsing import format_iteration

from backend.agents.rlm.run import (
    _FatalBackendGateLogger,
    _FatalPrimitiveAbort,
    _finalize_fatal_primitive_abort,
)


def _fatal_result() -> dict:
    return {
        "success": False,
        "metrics": {},
        "error": "RUNPOD_BALANCE_TOO_LOW: add funds",
        "failure_class": "balance_too_low",
        "outcome": "fatal",
    }


def _iteration_with_fatal_result(result: dict) -> RLMIteration:
    return RLMIteration(
        prompt=[],
        response="I will run the experiment.",
        code_blocks=[
            CodeBlock(
                code="result = run_experiment(code_path, env_id)",
                result=REPLResult(stdout="", stderr="", locals={"result": result}),
            )
        ],
    )


def test_fatal_outcome_terminates_before_history_append(make_context, tmp_path):
    ctx = make_context(tmp_path)
    ctx._last_primitive_name = "run_experiment"
    ctx._last_primitive_result = _fatal_result()
    emitted: list[dict] = []
    checkpointer = MagicMock()
    logger = _FatalBackendGateLogger(
        emit=emitted.append,
        checkpointer=checkpointer,
        ctx=ctx,
    )
    iteration = _iteration_with_fatal_result(ctx._last_primitive_result)
    message_history: list[dict] = []

    with pytest.raises(_FatalPrimitiveAbort):
        logger.log(iteration)
        message_history.extend(format_iteration(iteration))

    assert message_history == []
    assert logger.iteration_count == 1
    checkpointer.record.assert_called_once()


def test_fatal_finalize_writes_failed_status_and_partial_report(make_context, tmp_path):
    ctx = make_context(tmp_path)
    result = _fatal_result()
    abort = _FatalPrimitiveAbort(primitive_name="run_experiment", result=result)
    (ctx.project_dir / "experiment_runs.jsonl").write_text(
        json.dumps({
            "timestamp": "2026-05-26T00:00:00+00:00",
            "success": False,
            "metrics": {"accuracy": 0.42},
            "failure_class": "balance_too_low",
        })
        + "\n",
        encoding="utf-8",
    )
    emitted: list[dict] = []

    run_result = _finalize_fatal_primitive_abort(
        abort=abort,
        ctx=ctx,
        iterations=1,
        project_dir=ctx.project_dir,
        emit=emitted.append,
        tools_label="real",
    )

    status = json.loads((ctx.project_dir / "demo_status.json").read_text(encoding="utf-8"))
    report = json.loads((ctx.project_dir / "final_report.json").read_text(encoding="utf-8"))

    assert run_result.status == "failed"
    assert status["status"] == "failed"
    assert status["error"]["outcome"] == "fatal"
    assert status["error"]["primitive"] == "run_experiment"
    assert report["verdict"] == "partial"
    assert report["baseline_metrics"] == {"accuracy": 0.42}
    assert any(event.get("event") == "run_fatal" for event in emitted)
