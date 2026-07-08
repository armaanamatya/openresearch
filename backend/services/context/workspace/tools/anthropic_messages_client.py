"""Raw Anthropic Messages API LlmClient — the ANTHROPIC_API_KEY grader path.

Mirrors the ``LlmClient`` protocol (``complete(*, system, user) -> str``) used
by the PaperBench leaf scorer, but talks the **raw** Anthropic Messages API
(``anthropic.Anthropic().messages.create(...)``) instead of the bundled
claude-agent-sdk OAuth transport (``ClaudeLlmClient``).

Why this exists (spec 2026-06-16 §A5 — decoupled, sampler-capable grader
transport): the grader normally rides the *root model's* ``ctx.llm_client``, so
a root/CLI wedge takes grading down with it. This client gives the grader an
independent, pinned-Sonnet, ``temperature=0`` transport when an
``ANTHROPIC_API_KEY`` is available — distinct from the root. Unlike the SDK
OAuth path it CAN pin ``temperature=0`` (the Messages API exposes it), so its
samples are as deterministic as the API allows; the median-of-N then squeezes
out the residual nondeterminism.

Default model is Sonnet (``claude-sonnet-4-6`` — the same model the grader
runs, honouring the CLAUDE.md "grader stays Sonnet-quality" rule); override via
the constructor (``OPENRESEARCH_GRADER_MODEL`` is read by ``grader_transport``).
``import anthropic`` is lazy so this module never costs an import on the
non-Anthropic paths.
"""

from __future__ import annotations

from backend.services.context.workspace.tools._retry import with_429_backoff

# Same Sonnet id the OAuth grader path pins (rlm_query._DEFAULT_OAUTH_MODEL).
DEFAULT_GRADER_MODEL = "claude-sonnet-4-6"


def _zero_usage() -> dict[str, int]:
    """All-zeros usage in the CostLedgerEntry.from_usage shape."""
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "reasoning_tokens": 0,
    }


def _usage_from_message(usage: object) -> dict[str, int]:
    """Extract token counts from an Anthropic ``Message.usage`` object.

    Anthropic field names differ from OpenAI's Chat Completions:
    ``input_tokens`` / ``output_tokens`` (not prompt/completion) and
    ``cache_read_input_tokens`` / ``cache_creation_input_tokens`` at the top
    level (not nested in a details object). Robust to a missing usage object
    and to fields some responses omit.
    """
    if usage is None:
        return _zero_usage()

    def _int(name: str) -> int:
        return int(getattr(usage, name, 0) or 0)

    return {
        "input_tokens": _int("input_tokens"),
        "output_tokens": _int("output_tokens"),
        "cache_read_input_tokens": _int("cache_read_input_tokens"),
        "cache_creation_input_tokens": _int("cache_creation_input_tokens"),
        "reasoning_tokens": 0,
    }


def _text_from_content(content: object) -> str:
    """Concatenate the text blocks of a ``Message.content`` list.

    The grader prompts never use tools, so in practice this is a single text
    block; concatenating is robust if the API ever returns several.
    """
    parts = [getattr(b, "text", "") or "" for b in (content or [])]
    return "".join(parts)


class AnthropicMessagesClient:
    """LlmClient backed by the raw Anthropic Messages API (ANTHROPIC_API_KEY).

    ``max_tokens`` defaults to 4096 to match the OpenAI/Azure grader clients —
    the leaf scorer returns structured JSON that a few-hundred-token ceiling
    would truncate. ``temperature=0`` is pinned for grader determinism.
    """

    def __init__(
        self,
        model: str = DEFAULT_GRADER_MODEL,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 4096,
        timeout: float = 300.0,
    ) -> None:
        import anthropic

        # max_retries=6: the SDK natively retries 429/5xx with exponential
        # backoff and honours Retry-After; with_429_backoff below is the
        # belt-and-suspenders layer the sibling clients also carry.
        # A None api_key lets the SDK resolve ANTHROPIC_API_KEY from the env
        # (matching how grader_transport constructs it). base_url is omitted
        # unless truthy so the SDK resolves its own default (byte-identical
        # when unset — e.g. Foundry's Anthropic-compatible endpoint scoping).
        _client_kwargs: dict = {"api_key": api_key, "timeout": timeout, "max_retries": 6}
        if base_url:
            _client_kwargs["base_url"] = base_url
        self._client = anthropic.Anthropic(**_client_kwargs)
        self._model = model
        self._max_tokens = max_tokens
        # Mirror the sibling clients: callers (binding._ledger) read
        # ``_last_usage`` after a call for cost-ledger recording.
        self._last_usage: dict[str, int] = _zero_usage()
        # Latched once a model rejects ``temperature`` (see _messages_create):
        # Claude Opus 4.8 / Sonnet 5 (incl. the Foundry deployments) deprecate
        # it, so subsequent calls omit the param instead of re-probing.
        self._omit_temperature = False

    @with_429_backoff
    def complete(self, *, system: str, user: str) -> str:
        resp = self._messages_create(system=system, user=user, temperature=0)
        self._last_usage = _usage_from_message(getattr(resp, "usage", None))
        return _text_from_content(getattr(resp, "content", None))

    @with_429_backoff
    def complete_samples(
        self,
        *,
        system: str,
        user: str,
        n: int = 1,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> list[str]:
        """Return ``n`` completions via ``n`` ``temperature=0`` Messages calls.

        Optional grader-fidelity sampling path (spec 2026-06-16 §A5). The
        Anthropic Messages API has no native multi-sample (no ``n``) and no
        ``seed``, so this is sequential. ``temperature`` defaults to ``0``
        (honoured — the API exposes it, unlike the OAuth SDK path); ``seed`` is
        accepted-and-IGNORED (no API support). ``_last_usage`` reflects the
        LAST call.
        """
        eff_temp = 0 if temperature is None else temperature
        return [self._complete_once(system=system, user=user, temperature=eff_temp)
                for _ in range(n)]

    def _complete_once(self, *, system: str, user: str, temperature: float) -> str:
        resp = self._messages_create(system=system, user=user, temperature=temperature)
        self._last_usage = _usage_from_message(getattr(resp, "usage", None))
        return _text_from_content(getattr(resp, "content", None))

    def _messages_create(self, *, system: str, user: str, temperature: float):
        """Call ``messages.create``, tolerating models that deprecate ``temperature``.

        Claude Opus 4.8 / Sonnet 5 (incl. the Foundry deployments
        ``claude-opus-4-8`` / ``claude-sonnet-5``) reject ``temperature``
        ("temperature is deprecated for this model"). On that specific 400 we
        drop the param and retry — the model is near-deterministic by default
        and the caller's median-of-N still squeezes residual nondeterminism —
        and latch ``_omit_temperature`` so later calls skip the wasted probe.
        Models that accept ``temperature`` (the default ``claude-sonnet-4-6``
        grader) are byte-identical to before.
        """
        base = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if not self._omit_temperature:
            try:
                return self._client.messages.create(temperature=temperature, **base)
            except Exception as exc:  # noqa: BLE001 — inspect + re-raise non-temperature errors
                msg = str(exc).lower()
                if not ("temperature" in msg and "deprecat" in msg):
                    raise
                self._omit_temperature = True
        return self._client.messages.create(**base)


__all__ = ["AnthropicMessagesClient", "DEFAULT_GRADER_MODEL"]
