"""One-time monkeypatches for the ``rlm`` library's backend layer.

Two patches are registered here:

1. ``apply_oauth_backend_patch`` — registers the ``anthropic-oauth`` backend
   that routes through ClaudeOauthClient (Claude Agent SDK + OAuth).  rlm's
   ``get_client`` is a hardcoded if/elif switch with no plugin mechanism; we
   wrap it once.

2. ``apply_anthropic_caching_patch`` — wraps ``AnthropicClient.completion``
   so that when a system message is present the request payload carries a
   ``cache_control: {type: "ephemeral"}`` block on the system content.
   Anthropic's prompt-caching API caches tokens on a min-TTL of 5 minutes,
   saving ~50% of input tokens for the long stable system prompt across the
   20+ iterations of a typical RLM run.

   Design decisions (locked, from lane spec):
   - D2: AnthropicClient (API key path) needs the patch; rlm provides no hook.
   - D3: OAuth path (ClaudeOauthClient / claude-agent-sdk) is NOT patched —
     the SDK manages caching internally.
   - D4: OpenAI / Featherless / Azure paths have no prompt-cache concept; untouched.
   - D5: Caching is a NO-OP regression — if the wrapper encounters any error
     building the cached system block it falls back to the plain-string path so
     the run never crashes due to a caching misconfiguration.

Both patches are idempotent — safe to call multiple times; subsequent calls
after the first are no-ops.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_APPLIED = False
_ANTHROPIC_CACHING_APPLIED = False


def apply_oauth_backend_patch() -> None:
    """Install the ``anthropic-oauth`` backend dispatch on rlm.clients.get_client.

    Idempotent. After the first call, subsequent calls are no-ops.
    """
    global _APPLIED
    if _APPLIED:
        return

    import rlm.clients

    _original_get_client = rlm.clients.get_client

    def _patched_get_client(backend: str, backend_kwargs: dict[str, Any]):
        if backend == "anthropic-oauth":
            from backend.agents.rlm.claude_oauth_client import ClaudeOauthClient
            return ClaudeOauthClient(**backend_kwargs)
        return _original_get_client(backend, backend_kwargs)

    rlm.clients.get_client = _patched_get_client
    # Also patch the symbol imported into rlm.core.rlm at module load time —
    # rlm.core.rlm did ``from rlm.clients import BaseLM, get_client`` so it
    # holds a reference to the original.
    try:
        import rlm.core.rlm
        rlm.core.rlm.get_client = _patched_get_client
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "apply_oauth_backend_patch: could not patch rlm.core.rlm.get_client (%s); "
            "the unpatched reference may still be used. Error: %s",
            type(exc).__name__, exc,
        )

    _APPLIED = True
    logger.info("rlm anthropic-oauth backend registered")


def _wrap_system_with_cache_control(system: str | list | None) -> list | str | None:
    """Convert a plain system string to a list[TextBlockParam] with cache_control.

    The Anthropic messages API accepts ``system`` as either a plain ``str`` or
    a ``list`` of content blocks.  Prompt caching requires the latter form: each
    block can carry a ``cache_control: {"type": "ephemeral"}`` marker that tells
    the API to cache all tokens up to that breakpoint.

    We place one cache breakpoint at the end of the system prompt.  This is the
    maximum cacheable prefix and covers the entire stable system text per run.

    Returns the original value unchanged if it is already a list or if it is
    ``None`` / empty — preserving prior behaviour for callers that already pass
    structured system blocks.

    D5 contract: this function must never raise.  All errors are swallowed and
    logged; the caller falls back to the original system value.
    """
    try:
        if not system:
            return system
        if isinstance(system, list):
            # Already structured — do not re-wrap; avoids double-encoding.
            return system
        # Plain string path — wrap as a single text block with cache_control.
        return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
    except Exception as exc:  # noqa: BLE001 — D5: never crash due to caching logic
        logger.warning(
            "_wrap_system_with_cache_control: failed to build cache block (%s: %s); "
            "falling back to plain system string",
            type(exc).__name__, exc,
        )
        return system


def apply_anthropic_caching_patch() -> None:
    """Wrap ``AnthropicClient.completion`` to inject prompt-caching headers.

    After this patch, every call to ``AnthropicClient.completion`` that
    produces a ``system`` message will send the system as a structured content
    block with ``cache_control: {type: "ephemeral"}``.  The Anthropic API
    caches the token prefix on a 5-minute TTL, saving ~50% of input tokens
    for repeat calls with the same stable system prompt.

    Only the ``anthropic`` backend (API-key path) is patched here.
    ``ClaudeOauthClient`` (OAuth path) is intentionally left untouched —
    the claude-agent-sdk manages caching internally (D3).

    Idempotent. After the first call, subsequent calls are no-ops.
    """
    global _ANTHROPIC_CACHING_APPLIED
    if _ANTHROPIC_CACHING_APPLIED:
        return

    try:
        import rlm.clients.anthropic as _rlm_anthro
    except ImportError as exc:  # pragma: no cover — rlm always installed in practice
        logger.warning(
            "apply_anthropic_caching_patch: rlm.clients.anthropic not importable (%s); "
            "prompt caching will not be active for the anthropic backend",
            exc,
        )
        return

    _original_completion = _rlm_anthro.AnthropicClient.completion

    def _cached_completion(
        self: Any,
        prompt: str | list[dict[str, Any]],
        model: str | None = None,
    ) -> str:
        """Drop-in replacement that injects cache_control on the system block.

        We call the original ``_prepare_messages`` to split out the system
        string, then wrap it with cache_control before building the API kwargs.
        Falls back to the original completion path on any error (D5).
        """
        try:
            messages, system = self._prepare_messages(prompt)

            resolved_model = model or self.model_name
            if not resolved_model:
                raise ValueError("Model name is required for Anthropic client.")

            cached_system = _wrap_system_with_cache_control(system)

            kwargs: dict[str, Any] = {
                "model": resolved_model,
                "max_tokens": self.max_tokens,
                "messages": messages,
            }
            if cached_system is not None:
                kwargs["system"] = cached_system

            response = self.client.messages.create(**kwargs)
            self._track_cost(response, resolved_model)
            return response.content[0].text

        except Exception as exc:  # noqa: BLE001 — D5: fall back to original on any error
            logger.warning(
                "_cached_completion: error in caching wrapper (%s: %s); "
                "retrying with original completion path",
                type(exc).__name__, exc,
            )
            return _original_completion(self, prompt, model)

    _rlm_anthro.AnthropicClient.completion = _cached_completion  # type: ignore[method-assign]
    _ANTHROPIC_CACHING_APPLIED = True
    logger.info("rlm AnthropicClient.completion wrapped with prompt-caching (cache_control: ephemeral)")


__all__ = ["apply_oauth_backend_patch", "apply_anthropic_caching_patch"]
