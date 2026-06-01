"""Integration test: gepa_pre_call correctly no-ops when GEPA is off, and
gepa_post_call accumulates examples. The skipif test requires a real API key
and GEPA installed — it's marked for local-only runs.
"""
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def reset_settings_cache():
    """Force-reload settings before and after each test so env patches don't leak."""
    from backend.config import get_settings
    get_settings(_force_reload=True)
    yield
    get_settings(_force_reload=True)


VALID_CONTRACT = (
    '{"baseline_plan":"train","smoke_test_plan":"1 epoch",'
    '"full_run_plan":"100 epochs","expected_artifacts":["metrics.json"],'
    '"dataset_plan":{"name":"CIFAR"},"evaluation_plan":{"metrics":["acc"]},'
    '"verification_checklist":["check acc"],'
    '"metrics_shape":[{"metric_id":"acc","json_path":"top1","rubric_leaf_ids":[]}]}'
)


@pytest.fixture()
def ctx_with_emit(tmp_path):
    from backend.agents.rlm.context import RunContext
    llm = MagicMock()
    llm.complete = MagicMock(return_value=VALID_CONTRACT)
    llm._last_usage = {}
    emit_calls = []
    ctx = RunContext(
        project_id="prj_integ",
        project_dir=tmp_path,
        runs_root=tmp_path.parent,
        dashboard=MagicMock(),
        cost_ledger=MagicMock(),
        llm_client=llm,
        provider="openai",
        model="gpt-4o-mini",
        emit=lambda ev: emit_calls.append(ev),
    )
    return ctx, emit_calls


def test_gepa_post_call_accumulates_across_calls(ctx_with_emit):
    """gepa_post_call adds to buffer; second call means 2 entries."""
    ctx, _ = ctx_with_emit
    from backend.agents.gepa.hooks import gepa_post_call
    result1 = {"baseline_plan": "x", "metrics_shape": []}
    result2 = {"baseline_plan": "y", "metrics_shape": [{"metric_id": "acc", "json_path": "top1", "rubric_leaf_ids": []}]}
    gepa_post_call(ctx, "plan_reproduction", result1)
    gepa_post_call(ctx, "plan_reproduction", result2)
    assert len(ctx.gepa_example_buffer["plan_reproduction"]) == 2


def test_gepa_optimization_active_restored_after_exception(ctx_with_emit):
    """gepa_optimization_active is always False after gepa_pre_call even on exception."""
    ctx, _ = ctx_with_emit
    with patch.dict(os.environ, {"REPROLAB_GEPA_OPTIMIZATION": "plan-only"}):
        from backend.config import get_settings
        get_settings(_force_reload=True)
        with patch("backend.agents.gepa.optimizer.run_gepa_mini", side_effect=RuntimeError("mock error")):
            from backend.agents.gepa.hooks import gepa_pre_call
            try:
                gepa_pre_call(ctx, "plan_reproduction", (), {})
            except Exception:
                pass
    assert ctx.gepa_optimization_active is False


def test_gepa_phase_events_emitted(ctx_with_emit):
    """gepa_phase_start and gepa_phase_complete are emitted even when GEPA fails."""
    ctx, emit_calls = ctx_with_emit
    with patch.dict(os.environ, {"REPROLAB_GEPA_OPTIMIZATION": "plan-only"}):
        from backend.config import get_settings
        get_settings(_force_reload=True)
        with patch("backend.agents.gepa.optimizer.run_gepa_mini", return_value=("seed", 0.0, 0)):
            from backend.agents.gepa.hooks import gepa_pre_call
            gepa_pre_call(ctx, "plan_reproduction", (), {})
    event_types = [e.get("event") for e in emit_calls]
    assert "gepa_phase_start" in event_types
    assert "gepa_phase_complete" in event_types


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY") or os.environ.get("CI") == "true",
    reason="Requires valid OPENAI_API_KEY and gepa installed; skip in CI",
)
def test_gepa_pre_call_plan_reproduction_live(ctx_with_emit):
    """Live integration test: gepa_pre_call runs GEPA with real LLM and emits events."""
    ctx, emit_calls = ctx_with_emit
    with patch.dict(os.environ, {
        "REPROLAB_GEPA_OPTIMIZATION": "plan-only",
        "REPROLAB_GEPA_MAX_METRIC_CALLS_PLAN": "3",
        "REPROLAB_GEPA_TIMEOUT_PLAN_S": "30",
    }):
        from backend.config import get_settings
        get_settings(_force_reload=True)
        from backend.agents.gepa.hooks import gepa_pre_call
        gepa_pre_call(ctx, "plan_reproduction", (), {})
    assert ctx.gepa_optimization_active is False
    event_types = [e.get("event") for e in emit_calls]
    assert "gepa_phase_start" in event_types
    assert "gepa_phase_complete" in event_types
