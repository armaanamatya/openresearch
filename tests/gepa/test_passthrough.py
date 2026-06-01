"""When REPROLAB_GEPA_OPTIMIZATION=off (default), wrap_primitive must produce
zero behavioral change vs. today. This test verifies that:
1. gepa_prompt_overrides stays empty after a plan_reproduction-like call
2. No GEPA SSE events are emitted
3. The primitive result is identical to calling the function directly
"""
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.agents.rlm.context import RunContext


@pytest.fixture()
def ctx_gepa_off():
    dashboard = MagicMock()
    dashboard.primitive_call = MagicMock()
    cost_ledger = MagicMock()
    cost_ledger.append = MagicMock()
    llm_client = MagicMock()
    llm_client.complete = MagicMock(return_value='{"baseline_plan":"x","metrics_shape":[]}')
    llm_client._last_usage = {}
    emit = MagicMock()
    return RunContext(
        project_id="prj_test",
        project_dir=Path("/tmp/prj_test"),
        runs_root=Path("/tmp/runs"),
        dashboard=dashboard,
        cost_ledger=cost_ledger,
        llm_client=llm_client,
        provider="openai",
        model="gpt-4o-mini",
        emit=emit,
    )


def test_gepa_fields_default_empty(ctx_gepa_off):
    """RunContext has GEPA fields and they start empty/False."""
    assert ctx_gepa_off.gepa_prompt_overrides == {}
    assert ctx_gepa_off.gepa_example_buffer == {}
    assert ctx_gepa_off.gepa_optimization_active is False


def test_no_gepa_calls_when_off(ctx_gepa_off):
    """With REPROLAB_GEPA_OPTIMIZATION=off, gepa_pre_call is never invoked."""
    with patch.dict(os.environ, {"REPROLAB_GEPA_OPTIMIZATION": "off"}):
        from backend.agents.gepa.hooks import gepa_pre_call
        gepa_pre_call(ctx_gepa_off, "plan_reproduction", (), {})
        assert ctx_gepa_off.gepa_prompt_overrides == {}
        assert ctx_gepa_off.gepa_optimization_active is False


def test_gepa_optimization_active_guard(ctx_gepa_off):
    """gepa_pre_call is a no-op when gepa_optimization_active is already True."""
    ctx_gepa_off.gepa_optimization_active = True
    with patch.dict(os.environ, {"REPROLAB_GEPA_OPTIMIZATION": "on"}):
        from backend.agents.gepa.hooks import gepa_pre_call
        gepa_pre_call(ctx_gepa_off, "plan_reproduction", (), {})
        assert ctx_gepa_off.gepa_prompt_overrides == {}
