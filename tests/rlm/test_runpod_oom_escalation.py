from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from backend.agents.schemas import GpuPlan, GpuRequirements


@pytest.fixture
def ctx(tmp_path: Path):
    project_dir = tmp_path / "proj1"
    code_dir = project_dir / "code"
    rlm_state = project_dir / "rlm_state"
    code_dir.mkdir(parents=True)
    rlm_state.mkdir(parents=True)
    (code_dir / "commands.json").write_text(json.dumps(["python train.py"]))
    plan = GpuPlan(
        runpod_id="NVIDIA A100 40GB PCIe", short_name="a100_40", vram_gb=40, gpu_count=1,
        cloud_type="COMMUNITY", sku_usd_per_hr=1.19, total_usd_per_hr=1.19,
        container_disk_gb=50, volume_gb=20, source="paper",
        requirements=GpuRequirements(
            estimated_vram_gb=32, paper_gpu_string="A100",
            paper_gpu_count=1, reasoning="", confidence=0.9,
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
        remaining_s=lambda: None,
    )


def test_oom_escalates_to_next_ladder_rung(ctx):
    """First call OOMs, second call (with escalated plan) succeeds."""
    from backend.agents.rlm import primitives

    call_count = {"n": 0}

    async def fake_execute(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # First call: OOM
            return {"success": False, "metrics": {}, "logs": "RuntimeError: CUDA out of memory: tried to alloc 80GB"}
        # Second call: success on bigger SKU
        return {"success": True, "metrics": {"acc": 0.9}, "logs": "done"}

    with patch.object(primitives, "_execute_in_sandbox", side_effect=fake_execute):
        result = primitives.run_experiment(str(ctx.project_dir / "code"), "env-id", ctx=ctx)

    assert result["success"] is True
    assert call_count["n"] == 2, "experiment must be re-run after escalation"
    # Plan on disk now points to a100_80
    plan_after = json.loads((ctx.project_dir / "rlm_state" / "gpu_plan.json").read_text())
    assert plan_after["short_name"] == "a100_80"


def test_oom_with_empty_ladder_fails_with_cost_summary(ctx):
    """When ladder_remaining is empty AND oom, raise structured failure."""
    from backend.agents.rlm import primitives

    plan_path = ctx.project_dir / "rlm_state" / "gpu_plan.json"
    plan = json.loads(plan_path.read_text())
    plan["ladder_remaining"] = []
    plan_path.write_text(json.dumps(plan))

    async def always_oom(*args, **kwargs):
        return {"success": False, "metrics": {}, "logs": "torch.cuda.OutOfMemoryError"}

    with patch.object(primitives, "_execute_in_sandbox", side_effect=always_oom):
        result = primitives.run_experiment(str(ctx.project_dir / "code"), "env-id", ctx=ctx)

    assert result["success"] is False
    assert "oom" in str(result.get("error", "")).lower() or "escalation" in str(result.get("error", "")).lower()


def test_max_escalations_cap_honored(ctx, monkeypatch):
    """If max_escalations=1, second OOM must not trigger a third attempt."""
    monkeypatch.setenv("REPROLAB_DYNAMIC_GPU_MAX_ESCALATIONS", "1")
    # Bust the settings cache so the env var is picked up.
    import backend.config as _config
    _config._settings_cache = None
    from backend.agents.rlm import primitives

    call_count = {"n": 0}

    async def always_oom(*args, **kwargs):
        call_count["n"] += 1
        return {"success": False, "metrics": {}, "logs": "CUDA out of memory"}

    with patch.object(primitives, "_execute_in_sandbox", side_effect=always_oom):
        result = primitives.run_experiment(str(ctx.project_dir / "code"), "env-id", ctx=ctx)

    # Reset cache so we don't pollute other tests
    _config._settings_cache = None

    # 1 initial + 1 escalation = 2 calls
    assert call_count["n"] == 2
    assert result["success"] is False


def test_non_oom_failure_does_not_escalate(ctx):
    """Generic experiment failure (e.g., ImportError) does NOT trigger escalation."""
    from backend.agents.rlm import primitives

    call_count = {"n": 0}

    async def import_error(*args, **kwargs):
        call_count["n"] += 1
        return {"success": False, "metrics": {}, "logs": "ImportError: No module named 'transformers'"}

    with patch.object(primitives, "_execute_in_sandbox", side_effect=import_error):
        result = primitives.run_experiment(str(ctx.project_dir / "code"), "env-id", ctx=ctx)

    assert call_count["n"] == 1, "non-OOM failure must not escalate"
    assert result["success"] is False
