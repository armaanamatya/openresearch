"""Tests for gepa_pre_call / gepa_post_call hooks."""
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.agents.rlm.context import RunContext


@pytest.fixture(autouse=True)
def reset_settings_cache():
    """Force-reload settings before and after each test so env patches don't leak."""
    from backend.config import get_settings
    get_settings(_force_reload=True)
    yield
    get_settings(_force_reload=True)


@pytest.fixture()
def ctx():
    llm = MagicMock()
    llm.complete = MagicMock(return_value=(
        '{"baseline_plan":"x","smoke_test_plan":"y",'
        '"full_run_plan":"z","expected_artifacts":["metrics.json"],'
        '"dataset_plan":{"name":"d"},"evaluation_plan":{"metrics":["m"]},'
        '"verification_checklist":["v"],'
        '"metrics_shape":[{"metric_id":"m","json_path":"m","rubric_leaf_ids":[]}]}'
    ))
    llm._last_usage = {}
    dashboard = MagicMock()
    dashboard.primitive_call = MagicMock()
    return RunContext(
        project_id="prj_t",
        project_dir=Path("/tmp/prj_t"),
        runs_root=Path("/tmp/runs"),
        dashboard=dashboard,
        cost_ledger=MagicMock(),
        llm_client=llm,
        provider="openai",
        model="gpt-4o-mini",
        emit=MagicMock(),
    )


def test_gepa_pre_call_skips_unknown_primitive(ctx):
    with patch.dict(os.environ, {"REPROLAB_GEPA_OPTIMIZATION": "on"}):
        from backend.config import get_settings
        get_settings(_force_reload=True)
        from backend.agents.gepa.hooks import gepa_pre_call
        gepa_pre_call(ctx, "run_experiment", (), {})
        assert "run_experiment" not in ctx.gepa_prompt_overrides


def test_gepa_post_call_appends_to_buffer(ctx):
    from backend.agents.gepa.hooks import gepa_post_call
    result = {"baseline_plan": "test", "metrics_shape": []}
    gepa_post_call(ctx, "plan_reproduction", result)
    assert "plan_reproduction" in ctx.gepa_example_buffer
    assert len(ctx.gepa_example_buffer["plan_reproduction"]) == 1


def test_gepa_post_call_unknown_primitive_ignored(ctx):
    from backend.agents.gepa.hooks import gepa_post_call
    gepa_post_call(ctx, "run_experiment", {"ok": True})
    assert "run_experiment" not in ctx.gepa_example_buffer


def test_reentrancy_guard(ctx):
    """gepa_pre_call is no-op when gepa_optimization_active is True."""
    ctx.gepa_optimization_active = True
    with patch.dict(os.environ, {"REPROLAB_GEPA_OPTIMIZATION": "on"}):
        from backend.config import get_settings
        get_settings(_force_reload=True)
        from backend.agents.gepa.hooks import gepa_pre_call
        gepa_pre_call(ctx, "plan_reproduction", (), {})
        assert "plan_reproduction" not in ctx.gepa_prompt_overrides
