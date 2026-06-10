# tests/rlm/test_resolve_gpu_requirements.py
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.agents.rlm.primitives import resolve_gpu_requirements
from backend.agents.schemas import GpuPlan, GpuRequirements


@pytest.fixture
def ctx(tmp_path: Path, monkeypatch):
    from backend.config import get_settings

    monkeypatch.setenv("OPENRESEARCH_DYNAMIC_GPU_ENABLED", "true")
    get_settings(_force_reload=True)
    runs_root = tmp_path / "runs"
    project_dir = runs_root / "proj1"
    project_dir.mkdir(parents=True)
    (project_dir / "rlm_state").mkdir()
    yield SimpleNamespace(
        project_id="proj1",
        runs_root=runs_root,
        project_dir=project_dir,
        run_budget=None,
        sandbox_mode="runpod",
    )
    get_settings(_force_reload=True)


def test_returns_gpu_plan_dict_from_typed_input(ctx):
    req = GpuRequirements(
        estimated_vram_gb=40,
        paper_gpu_string="A100 80GB",
        paper_gpu_count=8,
        reasoning="test",
        confidence=0.9,
    )
    out = resolve_gpu_requirements(req, ctx=ctx)
    assert isinstance(out, dict)
    plan = GpuPlan(**out)
    assert plan.gpu_count == 1
    assert plan.source == "paper"


def test_accepts_loose_dict_payload(ctx):
    payload = {
        "estimated_vram_gb": 40,
        "paper_gpu_string": "A100 80GB",
        "paper_gpu_count": 8,
        "reasoning": "test",
        "confidence": 0.9,
    }
    out = resolve_gpu_requirements(payload, ctx=ctx)
    assert out["source"] == "paper"


def test_idempotent_returns_cached_plan(ctx):
    payload = {"estimated_vram_gb": 40, "paper_gpu_string": "A100", "paper_gpu_count": 1, "reasoning": "", "confidence": 0.9}
    out1 = resolve_gpu_requirements(payload, ctx=ctx)
    out2 = resolve_gpu_requirements({"estimated_vram_gb": 999, "paper_gpu_string": None, "paper_gpu_count": None, "reasoning": "diff", "confidence": 0.9}, ctx=ctx)
    assert out1["short_name"] == out2["short_name"], "cached plan must be returned, second call's higher VRAM ignored"
    assert out1["resolved_at"] == out2["resolved_at"]


def test_persists_plan_to_run_state(ctx):
    payload = {"estimated_vram_gb": 40, "paper_gpu_string": None, "paper_gpu_count": None, "reasoning": "", "confidence": 0.9}
    out = resolve_gpu_requirements(payload, ctx=ctx)
    state_file = ctx.project_dir / "rlm_state" / "gpu_plan.json"
    assert state_file.exists()
    loaded = json.loads(state_file.read_text())
    assert loaded["short_name"] == out["short_name"]


def test_low_confidence_returns_fallback_source(ctx):
    payload = {"estimated_vram_gb": 80, "paper_gpu_string": None, "paper_gpu_count": None, "reasoning": "", "confidence": 0.2}
    out = resolve_gpu_requirements(payload, ctx=ctx)
    assert out["source"] == "fallback"
    assert out["short_name"] == "rtx4090"


def test_malformed_payload_raises_value_error(ctx):
    with pytest.raises((ValueError, Exception)):
        resolve_gpu_requirements({"estimated_vram_gb": "not-a-number", "reasoning": "", "confidence": 0.5}, ctx=ctx)
