"""Pin the sandbox-aware implement_baseline prompt — and verify the prompt is
identical under both API-key and OAuth auth surfaces (parity).

The 2026-05-23 mandate: 'support API only mode as well as OAuth, elegantly
dynamically support both modes'. The sandbox-aware prompt addition is a pure
text change at the prompt-build layer — provider-independent by construction.
These tests pin (a) the CPU guidance fires when sandbox is CPU-only, and
(b) the prompt text is identical for `claude` (API) and `claude-oauth`
(subscription) — proving zero auth-surface fork in the new code path.
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


class TestCpuSandboxGuidance:
    """sandbox_mode='docker' or 'local' → CPU guidance fires."""

    @pytest.mark.parametrize("sandbox", ["docker", "local", "Docker", "SandboxMode.docker"])
    def test_cpu_sandbox_modes_inject_smoke_test_guidance(self, sandbox, tmp_path, monkeypatch):
        captured = _capture_prompt(monkeypatch)
        runs_root, pcm, env, contract = _minimal_inputs(tmp_path)
        asyncio.run(run_with_sdk(
            "prj_test", runs_root, pcm, env, contract,
            sandbox_mode=sandbox,
        ))
        assert len(captured) == 1
        prompt = captured[0]["prompt"]
        assert "SANDBOX CONSTRAINT" in prompt
        assert "CPU-ONLY" in prompt
        assert "--smoke-test" in prompt
        assert "5 minutes" in prompt

    def test_runpod_sandbox_does_NOT_inject_cpu_guidance(self, tmp_path, monkeypatch):
        """sandbox_mode='runpod' has GPU available — no CPU guidance."""
        captured = _capture_prompt(monkeypatch)
        runs_root, pcm, env, contract = _minimal_inputs(tmp_path)
        asyncio.run(run_with_sdk(
            "prj_test", runs_root, pcm, env, contract,
            sandbox_mode="runpod",
        ))
        prompt = captured[0]["prompt"]
        assert "SANDBOX CONSTRAINT" not in prompt
        assert "--smoke-test" not in prompt

    def test_no_sandbox_specified_no_guidance(self, tmp_path, monkeypatch):
        """sandbox_mode=None → no guidance (conservative; only fire when we KNOW it's CPU)."""
        captured = _capture_prompt(monkeypatch)
        runs_root, pcm, env, contract = _minimal_inputs(tmp_path)
        asyncio.run(run_with_sdk(
            "prj_test", runs_root, pcm, env, contract,
            sandbox_mode=None,
        ))
        prompt = captured[0]["prompt"]
        assert "SANDBOX CONSTRAINT" not in prompt


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
        assert "SANDBOX CONSTRAINT" in prompt
        assert "--smoke-test" in prompt
