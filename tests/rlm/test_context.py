from pathlib import Path

from backend.agents.rlm.context import RunContext


def test_run_context_holds_run_scoped_dependencies(tmp_path: Path):
    from backend.agents.dashboard_emitter import DashboardEmitter
    from backend.agents.resilience.cost import RunCostLedger

    project_dir = tmp_path / "prj"
    project_dir.mkdir()
    ctx = RunContext(
        project_id="prj",
        project_dir=project_dir,
        runs_root=tmp_path,
        dashboard=DashboardEmitter("prj", tmp_path),
        cost_ledger=RunCostLedger.load_jsonl(
            project_dir / "cost_ledger.jsonl", project_id="prj", attach_path=True
        ),
        llm_client=object(),
        provider="anthropic",
        model="test-model",
    )
    assert ctx.project_id == "prj"
    assert ctx.runtime is None
    assert ctx.workspace_service is None


def test_run_context_has_tree_event_counters():
    from backend.agents.rlm.context import RunContext
    ctx = RunContext.__dataclass_fields__
    assert "current_iteration" in ctx and "propose_round" in ctx
