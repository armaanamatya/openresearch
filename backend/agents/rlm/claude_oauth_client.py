"""rlm ``BaseLM`` subclass that drives the RLM root via Claude Agent SDK + OAuth.

The standard ``rlm.clients.anthropic.AnthropicClient`` requires a literal
``api_key`` constructor argument because it calls ``anthropic.Anthropic(api_key=...)``
directly — incompatible with Claude Code OAuth (no extractable bearer token).

``ClaudeOauthClient`` replaces that path: each ``completion()`` runs through the
Claude Agent SDK (``collect_agent_text``), which resolves auth itself (API key
*or* the logged-in ``claude`` CLI's OAuth credentials). All SDK calls are
thread-isolated via Workaround B (see ``backend/agents/rdr/agent.py``) so the
SDK's aclose() race cannot deadlock the rlm root loop.

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
    via Workaround B to contain the SDK's known ``aclose()`` deadlock.

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

    def completion(
        self, prompt: str | list[dict[str, Any]], model: str | None = None
    ) -> str:
        """Sync completion. The rlm root loop calls this synchronously."""
        text_prompt, system = self._render_prompt(prompt)
        resolved_model = model or self.model_name or "claude-sonnet-4-6"

        # Workaround B: thread-isolate the SDK call.
        from backend.agents.rdr.agent import _run_sdk_in_thread
        from pathlib import Path
        import tempfile

        # The SDK needs a cwd; use a throwaway temp dir for the root LLM call
        # since the RLM root just emits REPL code, not files.
        with tempfile.TemporaryDirectory(prefix="rlm-root-oauth-") as tmp:
            tmp_path = Path(tmp)
            # Combine system + user into a single prompt the SDK can consume.
            full_prompt = f"{system}\n\n{text_prompt}" if system else text_prompt
            try:
                text = _run_sdk_in_thread(
                    prompt=full_prompt,
                    code_dir=tmp_path,
                    model=resolved_model,
                    provider="anthropic",
                    runtime=None,  # collect_agent_text re-creates per call
                    max_turns=1,   # single-shot completion — no tool loop
                    timeout_s=self.timeout_s,
                )
            except TimeoutError as exc:
                logger.warning(
                    "ClaudeOauthClient.completion: SDK timeout — %s", exc
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
