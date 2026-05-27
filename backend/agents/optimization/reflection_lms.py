"""Reflection LM factories for GEPA Lane B.

Two backends, both returning a callable matching gepa's
``LanguageModel`` protocol (``__call__(prompt) -> str``):

1. **API-key (litellm-backed)** — the default. Calls litellm
   (same backend gepa uses internally) and threads each call
   through an :class:`LMCostTracker` so post-run cost is recoverable.
   Use this when an ``OPENAI_API_KEY`` / equivalent is set.

2. **claude-oauth callable** — for users with only the Claude Code
   subscription (no API balance). Routes through ``ClaudeOauthClient``
   → ``claude-agent-sdk``. Cost is $0 per the subscription but is
   subject to subscription rate limits.

Note: gepa 0.1.1 does **not** export a ``gepa.lm.LM`` class (the
context7 docs reference it, but the runtime package doesn't ship it).
Cost tracking therefore goes through our own
``LMCostTracker.wrap_callable`` rather than reading totals off a gepa
LM instance.

Selection happens in ``optimize_prompts_gepa.py`` via
``--reflection-backend {api-key,claude-oauth}`` (default: api-key).
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from backend.agents.optimization.cost_tracker import LMCostTracker

logger = logging.getLogger(__name__)


# Best-effort per-1K-token costs for common reflection models. Used by
# the api-key backend's cost estimator. Update as pricing changes.
# All values in USD per 1K tokens.
_PRICE_TABLE: dict[str, tuple[float, float]] = {  # (input, output)
    "openai/gpt-4.1": (0.0030, 0.0120),
    "openai/gpt-5": (0.0050, 0.0200),
    "openai/gpt-5-mini": (0.0005, 0.0020),
    "openai/gpt-4o": (0.0025, 0.0100),
    "anthropic/claude-sonnet-4-6": (0.0030, 0.0150),
    "anthropic/claude-haiku-4-5-20251001": (0.0008, 0.0040),
}


def _estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    in_rate, out_rate = _PRICE_TABLE.get(model, (0.0, 0.0))
    return tokens_in * in_rate / 1000 + tokens_out * out_rate / 1000


def make_api_key_reflection_lm(
    model: str, tracker: LMCostTracker | None = None
) -> Callable[..., str]:
    """Return a ``(prompt, **kwargs) -> str`` callable backed by litellm.

    If ``tracker`` is provided, every call is recorded with an estimated
    cost from the price table; this is how the optimize driver
    reconstructs total reflection-LM spend post-run.
    """

    def _call(prompt: Any, **_kwargs: Any) -> str:
        # Lazy import — litellm is heavy and gepa-installed-only.
        import litellm

        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt
        resp = litellm.completion(model=model, messages=messages)
        return resp.choices[0].message.content or ""

    if tracker is None:
        return _call

    from backend.agents.optimization.cost_tracker import wrap_callable

    return wrap_callable(
        _call,
        tracker=tracker,
        model=model,
        estimate_cost=lambda ti, to: _estimate_cost(model, ti, to),
    )


def make_claude_oauth_reflection_lm(
    model_name: str = "claude-sonnet-4-6",
) -> Callable[..., str]:
    """Return a ``(prompt, **kwargs) -> str`` callable backed by Claude OAuth.

    The callable converts gepa's prompt shape into the string-or-message-list
    shape ``ClaudeOauthClient.completion`` expects, invokes it, returns the
    model's text output. No API key required — auth resolves through
    claude-agent-sdk's normal subscription path.
    """
    from backend.agents.rlm.claude_oauth_client import ClaudeOauthClient

    client = ClaudeOauthClient(model_name=model_name)

    def claude_oauth_reflection_lm(prompt: Any, **kwargs: Any) -> str:
        try:
            return client.completion(prompt, model=kwargs.get("model"))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "claude_oauth_reflection_lm: completion failed — returning empty string: %s",
                exc,
            )
            # gepa tolerates an empty reflection (skips the mutation); failing
            # here would crash the entire optimize run.
            return ""

    return claude_oauth_reflection_lm


__all__ = [
    "make_api_key_reflection_lm",
    "make_claude_oauth_reflection_lm",
]
