from __future__ import annotations

from pathlib import Path

import pytest

from backend.agents.dashboard_emitter import DashboardEmitter
from backend.agents.resilience.cost import RunCostLedger
from backend.agents.rlm.binding import build_custom_tools
from backend.agents.rlm.context import RunContext
from backend.agents.rlm.sse_bridge import make_emit
from backend.agents.rlm import context_map as cm


def make_context(tmp_path: Path, project_id: str = "test_proj") -> RunContext:
    """Local RunContext factory (mirrors tests/rlm/conftest.py — different tree)."""
    project_dir = tmp_path / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    dashboard = DashboardEmitter(project_id, tmp_path)
    return RunContext(
        project_id=project_id,
        project_dir=project_dir,
        runs_root=tmp_path,
        dashboard=dashboard,
        emit=make_emit(dashboard),
        cost_ledger=RunCostLedger.load_jsonl(
            project_dir / "cost_ledger.jsonl",
            project_id=project_id,
            attach_path=True,
        ),
        llm_client=None,
        provider="anthropic",
        model="test-model",
    )


def _tool(ctx, name, fn):
    tools = build_custom_tools(ctx, registry={name: fn}, descriptions={name: name})
    return tools[name]["tool"]


def test_orientation_success_writes_map(tmp_path, monkeypatch):
    monkeypatch.setenv("REPROLAB_CONTEXT_MAP", "1")
    ctx = make_context(tmp_path)
    tool = _tool(ctx, "understand_section", lambda s, *, ctx: {"datasets": [{"name": "ALFWorld"}]})
    tool("a paper slice")
    entries = cm.read(ctx.project_dir)["entries"]
    assert any(e["key"] == "understand_section:datasets" for e in entries)


def test_non_orientation_success_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("REPROLAB_CONTEXT_MAP", "1")
    ctx = make_context(tmp_path)
    tool = _tool(ctx, "run_experiment", lambda *, ctx: {"success": True, "metrics": {}})
    tool()
    assert not (ctx.project_dir / "rlm_state" / "context_map.json").exists()


def test_failed_orientation_result_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("REPROLAB_CONTEXT_MAP", "1")
    ctx = make_context(tmp_path)
    # failure-shaped dict → wrap_primitive takes the `failed` branch → hook not reached
    tool = _tool(ctx, "understand_section",
                 lambda s, *, ctx: {"error": "boom", "datasets": [{"name": "X"}]})
    tool("slice")
    assert not (ctx.project_dir / "rlm_state" / "context_map.json").exists()


def test_flag_off_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv("REPROLAB_CONTEXT_MAP", raising=False)
    ctx = make_context(tmp_path)
    tool = _tool(ctx, "understand_section", lambda s, *, ctx: {"datasets": [{"name": "ALFWorld"}]})
    tool("slice")
    assert not (ctx.project_dir / "rlm_state" / "context_map.json").exists()


def test_hook_exception_does_not_break_primitive(tmp_path, monkeypatch):
    monkeypatch.setenv("REPROLAB_CONTEXT_MAP", "1")
    monkeypatch.setattr(cm, "record", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    ctx = make_context(tmp_path)
    tool = _tool(ctx, "understand_section", lambda s, *, ctx: {"datasets": [{"name": "ALFWorld"}]})
    result = tool("slice")  # must still return the primitive's result
    assert result["datasets"][0]["name"] == "ALFWorld"
