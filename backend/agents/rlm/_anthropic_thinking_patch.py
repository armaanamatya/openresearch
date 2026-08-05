"""Make the rlm ``AnthropicClient`` tolerate extended-THINKING responses.

rlm's ``AnthropicClient.{completion,acompletion}`` return ``response.content[0].text``,
which assumes the first content block is a text block. Claude with extended thinking
(e.g. Foundry ``claude-sonnet-5``) returns a ``ThinkingBlock`` as ``content[0]`` — which
has no ``.text`` attribute — so a ``sonnet-foundry``/``opus-foundry`` ROOT crashed at
iteration 0 with ``AttributeError: 'ThinkingBlock' object has no attribute 'text'``
(GCP smoke, 2026-08-01).

This patches both methods to concatenate the TEXT blocks and skip thinking/tool_use
blocks — a crash-free superset of ``content[0].text``. Idempotent; applied at import time
alongside the other rlm-library patches in ``run.py``.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)
_APPLIED = False


def disable_thinking_for(model: str | None) -> bool:
    """True for Claude models that DEFAULT to extended thinking (Foundry
    ``claude-sonnet-5`` / ``claude-opus-4-8``).

    On those, an unrequested thinking block appears on complex prompts and eats the
    ``max_tokens`` budget — truncating structured output (rubric JSON, root code) —
    so requests pass ``thinking={"type":"disabled"}``. Every other model (e.g. the
    ``claude-sonnet-4-6`` grader, gpt-5) is left byte-identical.
    """
    m = (model or "").lower()
    return ("sonnet-5" in m) or ("opus-4-8" in m)


def extract_text(response: Any) -> str:
    """Join every text block in an Anthropic response, skipping thinking/tool_use blocks.

    Robust to a plain-string ``content`` and to a response carrying no text block at all
    (returns ``""`` rather than raising) — a crash-free superset of ``content[0].text``.
    """
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content or []:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


def apply_anthropic_thinking_safe_patch() -> None:
    """Patch ``AnthropicClient.{completion,acompletion}`` to be thinking-block-safe.

    Idempotent. No-op (logs) if ``rlm`` isn't importable. The method bodies mirror the
    upstream ones exactly except for the final text extraction.
    """
    global _APPLIED
    if _APPLIED:
        return
    try:
        from rlm.clients.anthropic import AnthropicClient
    except Exception:  # noqa: BLE001 — rlm not importable in some tooling contexts
        logger.debug("anthropic thinking patch: rlm.clients.anthropic not importable")
        return

    def completion(self, prompt, model=None):  # noqa: ANN001, ANN201
        messages, system = self._prepare_messages(prompt)
        model = model or self.model_name
        if not model:
            raise ValueError("Model name is required for Anthropic client.")
        kwargs = {"model": model, "max_tokens": self.max_tokens, "messages": messages}
        if system:
            kwargs["system"] = system
        if disable_thinking_for(model):
            kwargs["thinking"] = {"type": "disabled"}
        response = self.client.messages.create(**kwargs)
        self._track_cost(response, model)
        return extract_text(response)

    async def acompletion(self, prompt, model=None):  # noqa: ANN001, ANN201
        messages, system = self._prepare_messages(prompt)
        model = model or self.model_name
        if not model:
            raise ValueError("Model name is required for Anthropic client.")
        kwargs = {"model": model, "max_tokens": self.max_tokens, "messages": messages}
        if system:
            kwargs["system"] = system
        if disable_thinking_for(model):
            kwargs["thinking"] = {"type": "disabled"}
        response = await self.async_client.messages.create(**kwargs)
        self._track_cost(response, model)
        return extract_text(response)

    AnthropicClient.completion = completion
    AnthropicClient.acompletion = acompletion
    _APPLIED = True
    logger.info("anthropic thinking-safe patch applied")


_SDK_APPLIED = False


def apply_anthropic_sdk_thinking_patch() -> None:
    """Disable extended thinking at the ``anthropic`` SDK boundary for thinking-default
    models — covers EVERY in-process client at once (rlm root, grader/rubric-gen
    ``AnthropicMessagesClient``, the navigation accelerator, ...), keyed on the REAL
    API model name in the request kwargs (``claude-sonnet-5``/``claude-opus-4-8``), so
    it can't be defeated by a client that names the model differently internally.

    Wraps ``Messages.create`` / ``AsyncMessages.create`` to inject
    ``thinking={"type":"disabled"}`` when the model defaults to thinking and the caller
    hasn't set it. Idempotent; no-op if ``anthropic`` isn't importable. Off for every
    other model (byte-identical).
    """
    global _SDK_APPLIED
    if _SDK_APPLIED:
        return
    try:
        from anthropic.resources.messages import AsyncMessages, Messages
    except Exception:  # noqa: BLE001 — anthropic not importable in some contexts
        logger.debug("anthropic sdk thinking patch: anthropic.resources.messages missing")
        return

    _orig_create = Messages.create

    def create(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN201
        if disable_thinking_for(kwargs.get("model")) and "thinking" not in kwargs:
            kwargs["thinking"] = {"type": "disabled"}
        return _orig_create(self, *args, **kwargs)

    _orig_acreate = AsyncMessages.create

    async def acreate(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN201
        if disable_thinking_for(kwargs.get("model")) and "thinking" not in kwargs:
            kwargs["thinking"] = {"type": "disabled"}
        return await _orig_acreate(self, *args, **kwargs)

    Messages.create = create
    AsyncMessages.create = acreate
    _SDK_APPLIED = True
    logger.info("anthropic sdk thinking-disable patch applied")


__all__ = [
    "apply_anthropic_thinking_safe_patch",
    "apply_anthropic_sdk_thinking_patch",
    "disable_thinking_for",
    "extract_text",
]
