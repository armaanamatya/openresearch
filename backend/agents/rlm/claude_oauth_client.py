"""rlm ``BaseLM`` subclass that drives the RLM root via Claude Agent SDK + OAuth.

The standard ``rlm.clients.anthropic.AnthropicClient`` requires a literal
``api_key`` constructor argument because it calls ``anthropic.Anthropic(api_key=...)``
directly — incompatible with Claude Code OAuth (no extractable bearer token).

``ClaudeOauthClient`` replaces that path: each ``completion()`` delegates to
``ClaudeLlmClient`` (rlm_query.py), which calls ``claude_agent_sdk.query()`` with
``ClaudeAgentOptions(tools=[], ...)`` — no tool use possible, so the model must
emit text on turn 1. This is correct for the RLM root, which is a pure
text-generation task. ``ClaudeLlmClient`` instances are cached per model to avoid
per-call ThreadPoolExecutor overhead.

Usage tracking is best-effort — the SDK does not return token counts, so per-call
input/output tokens are recorded as 0. The dominant cost signal comes from
``rlm``'s own ``usage_summary`` if a cost-tracking backend is used elsewhere;
for OAuth runs there is no per-token cost to track. See
``ROOT_MODELS["claude-oauth"]`` in ``backend/agents/rlm/models.py``.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from rlm.clients.base_lm import BaseLM
from rlm.core.types import ModelUsageSummary, UsageSummary

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_S = 1800.0  # 30 minutes per completion — bounded


class ClaudeOauthClient(BaseLM):
    """rlm ``BaseLM`` that routes completions through Claude Agent SDK + OAuth.

    Unlike ``AnthropicClient``, this client requires NO ``api_key`` — auth is
    resolved by ``claude-agent-sdk`` from either ``ANTHROPIC_API_KEY`` or the
    Claude Code subscription's OAuth login. Each completion is thread-isolated
    inside ``ClaudeLlmClient.complete()`` (rlm_query.py) via a dedicated
    ThreadPoolExecutor, keeping the SDK's ``aclose()`` race in the worker
    thread. Sub-agent calls via ``collect_agent_text`` additionally route
    through ``backend.agents.runtime.sdk_isolation.run_isolated`` (PR-μ
    Solution A).

    Constructor args:
        model_name: Claude model to use (default ``claude-sonnet-4-6``).
        max_tokens: Honored only for ``BaseLM`` ABI compat — ignored by SDK.
        timeout_s: Per-completion wall-clock cap; default 1800s.
    """

    def __init__(
        self,
        model_name: str | None = "claude-sonnet-4-6",
        max_tokens: int = 32768,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        **kwargs: Any,
    ):
        # NOTE: do NOT call super().__init__(**kwargs) with api_key — BaseLM doesn't take it,
        # and the wrapper provides no api_key at all.
        super().__init__(model_name=model_name or "claude-sonnet-4-6", **kwargs)
        self.model_name = model_name or "claude-sonnet-4-6"
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        # Per-model usage tracking — best-effort (SDK doesn't return token counts).
        self.model_call_counts: dict[str, int] = defaultdict(int)
        self.model_input_tokens: dict[str, int] = defaultdict(int)
        self.model_output_tokens: dict[str, int] = defaultdict(int)
        self.model_total_tokens: dict[str, int] = defaultdict(int)
        # For get_last_usage compatibility
        self._last_model: str = self.model_name
        # Cached per-model ClaudeLlmClient instances — avoids per-call
        # ThreadPoolExecutor overhead and reuses the same SDK session per model.
        self._claude_clients: dict[str, Any] = {}

    def completion(
        self, prompt: str | list[dict[str, Any]], model: str | None = None
    ) -> str:
        """Sync completion. The rlm root loop calls this synchronously.

        Delegates to ``ClaudeLlmClient`` (rlm_query.py) which calls
        ``claude_agent_sdk.query()`` with ``ClaudeAgentOptions(tools=[], ...)`` —
        no tool use possible, so the model must emit text on turn 1. This is
        correct for the RLM root, which is a pure text-generation task (the
        rdr-specific ``_run_sdk_in_thread`` path is for the tool-using
        ``baseline-implementation`` agent and is not appropriate here).

        ``ClaudeLlmClient`` is OAuth-capable (uses Claude Agent SDK, which
        resolves auth from API key OR the ``claude`` CLI's OAuth login) AND
        loop-safe (commit ``0c5fe4d`` added a running-loop guard).
        """
        text_prompt, system = self._render_prompt(prompt)
        resolved_model = model or self.model_name or "claude-sonnet-4-6"

        # Lazily build a per-model ClaudeLlmClient and cache it. The same
        # client instance is reused across completion() calls for the same
        # model — avoids per-call ThreadPoolExecutor overhead.
        client = self._claude_clients.get(resolved_model)
        if client is None:
            from backend.services.context.workspace.tools.rlm_query import (
                ClaudeLlmClient,
            )
            # max_turns=1 is correct here because tools=[] is enforced inside
            # ClaudeLlmClient — the model must emit text on turn 1.
            client = ClaudeLlmClient(model=resolved_model, max_turns=1)
            self._claude_clients[resolved_model] = client

        try:
            text = client.complete(system=system or "", user=text_prompt)
        except Exception as exc:
            logger.warning(
                "ClaudeOauthClient.completion: ClaudeLlmClient.complete failed — %s",
                exc,
            )
            raise

        # Update usage tracking (best-effort — no token counts from SDK).
        self.model_call_counts[resolved_model] += 1
        self._last_model = resolved_model
        return text

    async def acompletion(
        self, prompt: str | list[dict[str, Any]], model: str | None = None
    ) -> str:
        """Async path — rlm calls this for parallel sub-completions if any."""
        import asyncio
        return await asyncio.to_thread(self.completion, prompt, model)

    def get_usage_summary(self) -> UsageSummary:
        """Return best-effort usage summary (no token counts from SDK)."""
        model_summaries: dict[str, ModelUsageSummary] = {}
        for m, calls in self.model_call_counts.items():
            model_summaries[m] = ModelUsageSummary(
                total_calls=calls,
                total_input_tokens=self.model_input_tokens[m],
                total_output_tokens=self.model_output_tokens[m],
            )
        return UsageSummary(model_usage_summaries=model_summaries)

    def get_last_usage(self) -> ModelUsageSummary:
        """Return last call usage (no token counts from SDK)."""
        return ModelUsageSummary(
            total_calls=1,
            total_input_tokens=0,
            total_output_tokens=0,
        )

    @staticmethod
    def _render_prompt(
        prompt: str | list[dict[str, Any]],
    ) -> tuple[str, str | None]:
        """Reduce list-of-messages prompts to (text, system) for the SDK."""
        if isinstance(prompt, str):
            return prompt, None
        if isinstance(prompt, list) and all(isinstance(m, dict) for m in prompt):
            system_parts: list[str] = []
            user_parts: list[str] = []
            for msg in prompt:
                role = msg.get("role")
                content = msg.get("content") or ""
                if role == "system":
                    system_parts.append(content)
                else:
                    user_parts.append(content)
            return "\n\n".join(user_parts), "\n\n".join(system_parts) or None
        raise TypeError(f"Unsupported prompt type: {type(prompt).__name__}")


__all__ = ["ClaudeOauthClient"]
