"""Tests for GPU escalation persistence across run_experiment calls (Fix A2)
and experiment_completed dashboard event emission (Fix A1).

Fix A1: _persist_experiment_result must emit an experiment_completed dashboard
event so that Lane γ's foldExperimentCompleted UI handler fires on real runs.

Fix A2: gpu_escalations counter must be persisted in
rlm_state/gpu_escalation_state.json so the per-run cap is honoured even when
run_experiment is called multiple times (e.g. in the RLM repair loop).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from backend.agents.rlm import primitives


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_ctx(tmp_path: Path):
    """Minimal RunContext-like namespace sufficient for these tests."""
    from backend.agents.schemas import GpuPlan, GpuRequirements

    project_dir = tmp_path / "proj1"
    code_dir = project_dir / "code"
    rlm_state = project_dir / "rlm_state"
    code_dir.mkdir(parents=True)
    rlm_state.mkdir(parents=True)
    (code_dir / "commands.json").write_text(json.dumps(["python train.py"]))

    plan = GpuPlan(
        runpod_id="NVIDIA A100 40GB PCIe",
        short_name="a100_40",
        vram_gb=40,
        gpu_count=1,
        cloud_type="COMMUNITY",
        sku_usd_per_hr=1.19,
        total_usd_per_hr=1.19,
        container_disk_gb=50,
        volume_gb=20,
        source="paper",
        requirements=GpuRequirements(
            estimated_vram_gb=32,
            paper_gpu_string="A100",
            paper_gpu_count=1,
            reasoning="",
            confidence=0.9,
        ),
        ladder_remaining=("a100_80", "h100_80"),
        resolved_at="2026-05-23T00:00:00+00:00",
    )
    (rlm_state / "gpu_plan.json").write_text(json.dumps(plan.model_dump(mode="json")))

    return SimpleNamespace(
        project_id="proj1",
        project_dir=project_dir,
        runs_root=tmp_path,
        run_budget=None,
        sandbox_mode="runpod",
        scope_spec=None,
        remaining_s=lambda: None,
    )


# ---------------------------------------------------------------------------
# Fix A2 — escalation count persistence
# ---------------------------------------------------------------------------


def test_persisted_escalation_count_loads_on_run_experiment(tmp_path, monkeypatch):
    """A pre-existing gpu_escalation_state.json with escalations_used=2 must
    prevent further ladder advances when max_escalations=2.

    Scenario: previous run_experiment call already advanced twice; state file
    records escalations_used=2. On the next call the cap is already met and
    the experiment should not escalate again (loop exits after the first OOM).
    """
    monkeypatch.setenv("OPENRESEARCH_DYNAMIC_GPU_MAX_ESCALATIONS", "2")
    import backend.config as _config
    _config._settings_cache = None

    ctx = _make_ctx(tmp_path)
    # Simulate persisted state from a previous run_experiment call.
    state_file = ctx.project_dir / "rlm_state" / "gpu_escalation_state.json"
    state_file.write_text(json.dumps({"escalations_used": 2}), encoding="utf-8")

    call_count = {"n": 0}

    async def always_oom(*args, **kwargs):
        call_count["n"] += 1
        return {
            "success": False,
            "metrics": {},
            "logs": "CUDA out of memory: tried to alloc 80 GB",
        }

    with patch.object(primitives, "_execute_in_sandbox", side_effect=always_oom):
        result = primitives.run_experiment(
            str(ctx.project_dir / "code"), "env-id", ctx=ctx
        )

    _config._settings_cache = None

    # Cap already met from persisted state — only 1 attempt, no escalation.
    assert call_count["n"] == 1, (
        f"Expected exactly 1 attempt (cap already reached), got {call_count['n']}"
    )
    assert result["success"] is False


def test_escalation_count_persists_after_advance(tmp_path, monkeypatch):
    """After a successful ladder advance, gpu_escalation_state.json must be
    written with the updated escalations_used count.
    """
    monkeypatch.setenv("OPENRESEARCH_DYNAMIC_GPU_MAX_ESCALATIONS", "3")
    import backend.config as _config
    _config._settings_cache = None

    ctx = _make_ctx(tmp_path)
    call_count = {"n": 0}

    async def fake_execute(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # First call: OOM to trigger escalation.
            return {
                "success": False,
                "metrics": {},
                "logs": "RuntimeError: CUDA out of memory: tried to alloc 80GB",
            }
        # Second call: success on bigger SKU.
        return {"success": True, "metrics": {"acc": 0.9}, "logs": "done"}

    with patch.object(primitives, "_execute_in_sandbox", side_effect=fake_execute):
        result = primitives.run_experiment(
            str(ctx.project_dir / "code"), "env-id", ctx=ctx
        )

    _config._settings_cache = None

    assert result["success"] is True

    state_file = ctx.project_dir / "rlm_state" / "gpu_escalation_state.json"
    assert state_file.exists(), "gpu_escalation_state.json must be written after ladder advance"
    state = json.loads(state_file.read_text(encoding="utf-8"))
    assert state["escalations_used"] == 1, (
        f"Expected escalations_used=1 after one advance, got {state}"
    )


def test_escalation_count_accumulates_across_calls(tmp_path, monkeypatch):
    """Two sequential run_experiment calls each advancing once accumulate to
    escalations_used=2 in the persisted state file.
    """
    monkeypatch.setenv("OPENRESEARCH_DYNAMIC_GPU_MAX_ESCALATIONS", "5")
    import backend.config as _config
    _config._settings_cache = None

    ctx = _make_ctx(tmp_path)

    async def oom_then_success(*args, **kwargs):
        return {
            "success": False,
            "metrics": {},
            "logs": "CUDA out of memory: tried to alloc 80GB",
        }

    # First call: OOM -> escalates once -> still OOMs, ladder_remaining now "h100_80".
    # We want both calls to escalate once and fail so we can check accumulation.
    call_seq = {"n": 0}

    async def one_oom_then_success(*args, **kwargs):
        call_seq["n"] += 1
        if call_seq["n"] % 2 == 1:
            # Odd calls OOM.
            return {"success": False, "metrics": {}, "logs": "CUDA out of memory"}
        return {"success": True, "metrics": {}, "logs": "ok"}

    with patch.object(primitives, "_execute_in_sandbox", side_effect=one_oom_then_success):
        primitives.run_experiment(str(ctx.project_dir / "code"), "env-id", ctx=ctx)

    state_after_first = json.loads(
        (ctx.project_dir / "rlm_state" / "gpu_escalation_state.json").read_text()
    )
    assert state_after_first["escalations_used"] == 1

    # Reload the gpu_plan (would normally be updated on disk) so second call still has a ladder.
    from backend.agents.schemas import GpuPlan, GpuRequirements
    plan2 = GpuPlan(
        runpod_id="NVIDIA A100 80GB PCIe",
        short_name="a100_80",
        vram_gb=80,
        gpu_count=1,
        cloud_type="COMMUNITY",
        sku_usd_per_hr=2.0,
        total_usd_per_hr=2.0,
        container_disk_gb=80,
        volume_gb=20,
        source="paper",
        requirements=GpuRequirements(
            estimated_vram_gb=70, paper_gpu_string="A100 80GB", paper_gpu_count=1,
            reasoning="", confidence=0.9,
        ),
        ladder_remaining=("h100_80",),
        resolved_at="2026-05-23T00:00:00+00:00",
    )
    (ctx.project_dir / "rlm_state" / "gpu_plan.json").write_text(
        json.dumps(plan2.model_dump(mode="json"))
    )

    call_seq2 = {"n": 0}

    async def oom_then_ok(*args, **kwargs):
        call_seq2["n"] += 1
        if call_seq2["n"] == 1:
            return {"success": False, "metrics": {}, "logs": "CUDA out of memory"}
        return {"success": True, "metrics": {}, "logs": "ok"}

    with patch.object(primitives, "_execute_in_sandbox", side_effect=oom_then_ok):
        primitives.run_experiment(str(ctx.project_dir / "code"), "env-id", ctx=ctx)

    _config._settings_cache = None

    state_after_second = json.loads(
        (ctx.project_dir / "rlm_state" / "gpu_escalation_state.json").read_text()
    )
    assert state_after_second["escalations_used"] == 2, (
        f"Expected cumulative escalations_used=2, got {state_after_second}"
    )


# ---------------------------------------------------------------------------
# Fix A1 — experiment_completed dashboard event
# ---------------------------------------------------------------------------


def test_experiment_completed_event_emitted(tmp_path):
    """_persist_experiment_result must emit an experiment_completed dashboard
    event carrying success, metrics, and per_model fields.
    """
    from types import SimpleNamespace

    project_dir = tmp_path / "proj1"
    project_dir.mkdir(parents=True)

    ctx = SimpleNamespace(
        project_id="proj1",
        project_dir=project_dir,
    )

    result = {
        "success": True,
        "metrics": {"acc": 0.95, "per_model": {"qwen-1.7b": {"acc": 0.95}}},
        "logs": "done",
    }

    emitted: list[dict] = []

    def fake_emit(ctx_arg, *, event_type: str, payload: dict):
        emitted.append({"event_type": event_type, "payload": payload})

    with patch.object(primitives, "_emit_dashboard_event", side_effect=fake_emit):
        primitives._persist_experiment_result(ctx, result)

    experiment_events = [e for e in emitted if e["event_type"] == "experiment_completed"]
    assert len(experiment_events) == 1, (
        f"Expected exactly 1 experiment_completed event, got {len(experiment_events)}"
    )
    payload = experiment_events[0]["payload"]
    assert payload["success"] is True
    assert payload["metrics"] == result["metrics"]
    assert payload["per_model"] == {"qwen-1.7b": {"acc": 0.95}}


def test_experiment_completed_event_emitted_on_failure(tmp_path):
    """experiment_completed is also emitted for failed experiments (success=False)."""
    from types import SimpleNamespace

    project_dir = tmp_path / "proj1"
    project_dir.mkdir(parents=True)

    ctx = SimpleNamespace(project_id="proj1", project_dir=project_dir)

    result = {"success": False, "metrics": {}, "error": "CUDA OOM", "logs": "..."}

    emitted: list[dict] = []

    def fake_emit(ctx_arg, *, event_type: str, payload: dict):
        emitted.append({"event_type": event_type, "payload": payload})

    with patch.object(primitives, "_emit_dashboard_event", side_effect=fake_emit):
        primitives._persist_experiment_result(ctx, result)

    experiment_events = [e for e in emitted if e["event_type"] == "experiment_completed"]
    assert len(experiment_events) == 1
    assert experiment_events[0]["payload"]["success"] is False
    assert experiment_events[0]["payload"]["per_model"] is None


def test_experiment_completed_per_model_is_none_when_absent(tmp_path):
    """per_model field is None when metrics has no per_model key."""
    from types import SimpleNamespace

    project_dir = tmp_path / "proj1"
    project_dir.mkdir(parents=True)

    ctx = SimpleNamespace(project_id="proj1", project_dir=project_dir)

    result = {
        "success": True,
        "metrics": {"acc": 0.8},
        "logs": "",
    }

    emitted: list[dict] = []

    def fake_emit(ctx_arg, *, event_type: str, payload: dict):
        emitted.append({"event_type": event_type, "payload": payload})

    with patch.object(primitives, "_emit_dashboard_event", side_effect=fake_emit):
        primitives._persist_experiment_result(ctx, result)

    experiment_events = [e for e in emitted if e["event_type"] == "experiment_completed"]
    assert len(experiment_events) == 1
    assert experiment_events[0]["payload"]["per_model"] is None
