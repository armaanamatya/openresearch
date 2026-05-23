"""Pin the compute-constraint guidance in implement_baseline — and verify the
prompt is identical under both API-key and OAuth auth surfaces (parity).

The 2026-05-23 mandate: 'support API only mode as well as OAuth, elegantly
dynamically support both modes'. The compute-constraint prompt addition is a
pure text change at the prompt-build layer — provider-independent by
construction.

ComputeConstraintGuidance: guidance fires based on BOTH sandbox_mode AND
gpu_mode — not a static sandbox-name heuristic — so docker+gpu_mode=max
correctly skips the constraint, and runpod always skips it.

AuthSurfaceParity: the prompt text is identical for `claude` (API) and
`claude-oauth` (subscription) — proving zero auth-surface fork.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.agents.baseline_implementation import run_with_sdk
from backend.agents.schemas import (
    EnvironmentSpec,
    PaperClaimMap,
    ReproductionContract,
)


def _capture_prompt(monkeypatch):
    """Patch collect_agent_text to capture the prompt argument; return the
    captured-list so the test can inspect what would have been sent."""
    captured: list[dict] = []

    async def _fake_collect(agent_name, prompt, **kwargs):
        captured.append({
            "agent": agent_name, "prompt": prompt,
            "model": kwargs.get("model"), "provider": kwargs.get("provider"),
        })
        return ""

    # collect_agent_text is lazy-imported inside run_with_sdk, so patch at its
    # source module (backend.agents.runtime.invoke), not at the consumer.
    monkeypatch.setattr(
        "backend.agents.runtime.invoke.collect_agent_text",
        _fake_collect,
    )
    return captured


def _minimal_inputs(tmp_path: Path):
    """Build the minimum object set run_with_sdk needs."""
    runs_root = tmp_path / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    (runs_root / "prj_test").mkdir(parents=True, exist_ok=True)
    (runs_root / "prj_test" / "code").mkdir(parents=True, exist_ok=True)
    pcm = PaperClaimMap(core_contribution="test paper")
    env = EnvironmentSpec(dockerfile="FROM python:3.11", framework="pytorch")
    contract = None
    return runs_root, pcm, env, contract


class TestDynamicComputeGuidance:
    """_compute_constraint_guidance uses BOTH sandbox_mode AND gpu_mode.

    Decision table pinned here:
    - docker + gpu_mode=None   → constraint (CPU-likely)
    - docker + gpu_mode='off'  → constraint
    - docker + gpu_mode='auto' → constraint (most dev hosts lack GPU)
    - docker + gpu_mode='prefer' → constraint
    - docker + gpu_mode='max'  → NO constraint (user demands GPU)
    - local + gpu_mode=None    → constraint
    - runpod + gpu_mode=None   → NO constraint (runpod has GPU)
    - runpod + gpu_mode='auto' → NO constraint
    - sandbox=None, gpu_mode=None → NO constraint (conservative)
    - sandbox=unknown, gpu_mode=None → NO constraint (conservative)
    """

    @pytest.mark.parametrize("sandbox,gpu_mode,expect_guidance", [
        ("docker",        None,      True),
        ("docker",        "off",     True),
        ("docker",        "auto",    True),
        ("docker",        "prefer",  True),
        ("docker",        "max",     False),
        ("local",         None,      True),
        ("runpod",        None,      False),
        ("runpod",        "auto",    False),
        (None,            None,      False),
        ("unknown_value", None,      False),
    ])
    def test_compute_guidance_parametrized(
        self, sandbox, gpu_mode, expect_guidance, tmp_path, monkeypatch
    ):
        captured = _capture_prompt(monkeypatch)
        runs_root, pcm, env, contract = _minimal_inputs(tmp_path)
        asyncio.run(run_with_sdk(
            "prj_test", runs_root, pcm, env, contract,
            sandbox_mode=sandbox,
            gpu_mode=gpu_mode,
        ))
        assert len(captured) == 1
        prompt = captured[0]["prompt"]
        if expect_guidance:
            assert "COMPUTE CONSTRAINT" in prompt
            assert "--smoke-test" in prompt
            assert "5 min" in prompt
        else:
            assert "COMPUTE CONSTRAINT" not in prompt
            assert "--smoke-test" not in prompt


class TestAuthSurfaceParity:
    """Verify the sandbox-aware prompt is IDENTICAL under both API-key and
    OAuth auth surfaces — proves zero auth fork in the new code path."""

    def _render_prompt_with_provider(self, tmp_path, monkeypatch, provider: str | None, model: str | None):
        captured = _capture_prompt(monkeypatch)
        runs_root, pcm, env, contract = _minimal_inputs(tmp_path)
        asyncio.run(run_with_sdk(
            "prj_test", runs_root, pcm, env, contract,
            sandbox_mode="docker",  # use the CPU path so guidance fires
            provider=provider,
            model=model,
        ))
        return captured[0]["prompt"]

    def test_api_and_oauth_produce_identical_prompts(self, tmp_path, monkeypatch):
        # API-key path: provider="anthropic", model="claude-sonnet-4-6"
        api_prompt = self._render_prompt_with_provider(
            tmp_path, monkeypatch, provider="anthropic", model="claude-sonnet-4-6"
        )
        # OAuth path: provider="anthropic", model="claude-oauth" (the OAuth alias)
        oauth_prompt = self._render_prompt_with_provider(
            tmp_path, monkeypatch, provider="anthropic", model="claude-oauth"
        )
        # Identical text — the prompt is provider-agnostic. The only auth-
        # specific code is BELOW this layer (in collect_agent_text → SDK).
        assert api_prompt == oauth_prompt, (
            "implement_baseline prompt diverged between API and OAuth modes — "
            "the prompt-build layer must be auth-agnostic."
        )

    def test_openai_provider_also_identical_prompt(self, tmp_path, monkeypatch):
        # If a future deployment uses OpenAI for the baseline agent, the
        # prompt MUST still carry the sandbox guidance.
        prompt = self._render_prompt_with_provider(
            tmp_path, monkeypatch, provider="openai", model="gpt-5"
        )
        assert "COMPUTE CONSTRAINT" in prompt
        assert "--smoke-test" in prompt


class TestGpuModePlumbedThroughRunContext:
    """Pin the 2026-05-23 evening fix: RunContext.gpu_mode is threaded from
    ExecutionProfile so _compute_constraint_guidance gets the right signal.
    Without this, ctx.gpu_mode is always None and the dynamic decision
    collapses to "sandbox_mode alone" — same bug as before the helper."""

    def test_runcontext_has_gpu_mode_field(self):
        """RunContext dataclass MUST expose a gpu_mode attribute (default None
        for back-compat). Removing this field reverts dynamic detection to
        "sandbox name alone" — same bug as before."""
        from backend.agents.rlm.context import RunContext
        import dataclasses
        names = {f.name for f in dataclasses.fields(RunContext)}
        assert "gpu_mode" in names, (
            "RunContext.gpu_mode field is required for sandbox-aware baseline "
            "guidance. Without it, ctx.gpu_mode is always None and runpod runs "
            "incorrectly trigger CPU smoke-test guidance."
        )

    def test_gpu_mode_default_is_None_for_backward_compat(self):
        """Existing call sites that don't pass gpu_mode must still work."""
        from backend.agents.rlm.context import RunContext
        from pathlib import Path
        ctx = RunContext(
            project_id="prj_test",
            project_dir=Path("/tmp/test"),
            runs_root=Path("/tmp"),
            dashboard=None,
            cost_ledger=None,
            llm_client=None,
            provider="anthropic",
            model="claude-sonnet-4-6",
        )
        assert ctx.gpu_mode is None, (
            "gpu_mode must default to None for back-compat with call sites "
            "that don't pass it (e.g. existing tests + scripts)."
        )
