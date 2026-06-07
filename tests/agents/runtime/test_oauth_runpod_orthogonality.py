"""OAuth orthogonality guard tests for the dynamic GPU selection feature.

These tests assert that the RunPod execution surface is completely decoupled
from Anthropic credentials — the pod runs ML code, not LLM calls.

Spec reference: docs/superpowers/specs/2026-05-23-dynamic-gpu-selection-design.md
Invariant I4: ANTHROPIC_API_KEY must never enter the pod env.
"""
from __future__ import annotations

import inspect


def test_runpod_backend_init_does_not_read_anthropic_api_key(monkeypatch):
    """Guard: RunpodBackend must not read ANTHROPIC_API_KEY when constructed
    under an OAuth-mode run. The pod runs ML code, not LLM calls."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENRESEARCH_RUNPOD_API_KEY", "test-key")
    from backend.services.runtime.runpod_backend import RunpodBackend
    backend = RunpodBackend(api_key="test-key")  # no Anthropic env var present
    # No exception, no read attempt.
    assert backend.api_key == "test-key"


def test_resolve_gpu_requirements_no_anthropic_env_read(monkeypatch, tmp_path):
    """The primitive is pure-Python over Pydantic + catalog; no Anthropic creds needed."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from types import SimpleNamespace
    from backend.agents.rlm.primitives import resolve_gpu_requirements
    proj = tmp_path / "p1"
    (proj / "rlm_state").mkdir(parents=True)
    ctx = SimpleNamespace(
        project_id="p1", project_dir=proj, runs_root=tmp_path,
        run_budget=None, sandbox_mode="runpod",
        vram_override=None,
    )
    out = resolve_gpu_requirements(
        {"estimated_vram_gb": 24, "paper_gpu_string": None,
         "paper_gpu_count": None, "reasoning": "", "confidence": 0.9},
        ctx=ctx,
    )
    # Key contract: returns a plan dict without needing ANTHROPIC_API_KEY at all.
    assert "short_name" in out
    assert out["source"] in ("paper", "fallback", "informational")
    # No Anthropic key needed for the resolver path.


def test_pod_env_injection_excludes_anthropic_credentials():
    """When the pod is created, ANTHROPIC_API_KEY must NOT be injected by default
    into the container env. (Paper code that needs an LLM key must request it
    explicitly via a separate mechanism.)"""
    from backend.agents.rlm import primitives
    src = inspect.getsource(primitives._execute_in_sandbox)
    assert "ANTHROPIC_API_KEY" not in src, \
        "_execute_in_sandbox must not silently leak ANTHROPIC_API_KEY into the pod env"
