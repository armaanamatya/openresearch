"""Decoupled, sampler-capable grader transport (spec 2026-06-16 §A5).

Two universal entry points the grader calls instead of touching a client's
sampler API directly:

- ``sample_completions(client, ...)`` — return ``n`` completion strings from
  ANY ``LlmClient``. Uses ``client.complete_samples(...)`` when the client
  exposes it (OpenAI/Azure native-``n``, Anthropic/OAuth sequential), and
  otherwise falls back to ``n`` calls to plain ``complete``. This is the
  backwards-compatible floor — an old client with only ``complete`` still
  yields ``n`` samples.

- ``build_grader_client(fallback_client, fallback_label)`` — optionally swap
  the grader onto an INDEPENDENT, sampler-capable transport via
  ``REPROLAB_GRADER_BACKEND`` / ``REPROLAB_GRADER_MODEL``. Default (both unset)
  returns the fallback UNCHANGED — today's behaviour, the grader rides the root
  client. On any construction error it falls back too, never raising — grading
  must survive a misconfigured backend.

All imports of the concrete client classes are lazy so this module costs
nothing on the default path and never hard-depends on the OpenAI / Anthropic
SDKs being installed.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def _flag_value(name: str) -> str:
    """Return a normalized env value (stripped, lower-cased); '' when unset."""
    return os.environ.get(name, "").strip().lower()


def sample_completions(
    client: Any,
    *,
    system: str,
    user: str,
    n: int,
    temperature: float = 0,
    seed: int | None = None,
) -> list[str]:
    """Return ``n`` completion strings from ``client``, sampler-capable or not.

    When ``client`` has a ``complete_samples`` method (the optional §A5
    protocol extension — OpenAI/Azure native-``n``, Anthropic/OAuth sequential)
    it is used and passed ``temperature``/``seed``; each backend honours what
    it can. Otherwise this degrades to ``n`` sequential ``complete`` calls so
    EVERY client — including one that only implements the base
    ``complete(*, system, user) -> str`` protocol — produces ``n`` samples.

    The per-leaf median-of-``n`` in the grader is the universal denoiser on top
    of whatever determinism the chosen backend can offer.
    """
    fn = getattr(client, "complete_samples", None)
    if callable(fn):
        try:
            result = fn(system=system, user=user, n=n, temperature=temperature, seed=seed)
        except (TypeError, AttributeError):
            result = None
        # A real complete_samples returns list[str]; a MagicMock auto-attribute
        # (e.g. a test stub that only implements `complete`) returns a Mock, not a
        # list — fall through to plain `complete` so any loose stub still yields n
        # samples. This keeps `complete` the universal, backwards-compatible floor.
        if isinstance(result, list):
            return [str(x) for x in result]
    return [client.complete(system=system, user=user) for _ in range(n)]


def build_grader_client(
    fallback_client: Any, fallback_label: str = ""
) -> tuple[Any, str]:
    """Resolve the grader's transport, honouring ``REPROLAB_GRADER_BACKEND``.

    Returns ``(client, label)``.

    - Both ``REPROLAB_GRADER_BACKEND`` and ``REPROLAB_GRADER_MODEL`` unset/empty
      → ``(fallback_client, fallback_label)`` UNCHANGED. This is the default:
      the grader rides the root model's client exactly as today.
    - ``REPROLAB_GRADER_BACKEND`` in {anthropic, openai, azure} → construct the
      corresponding sampler-capable client from env (``REPROLAB_GRADER_MODEL``
      overrides the model id; anthropic defaults to Sonnet). The label is
      ``"grader:<backend>:<model>"``.
    - Any other backend value, or any construction error (missing SDK, missing
      credential), logs a warning and returns the fallback. We NEVER raise:
      grading must survive a misconfigured grader backend (and a root/CLI wedge
      is exactly when this decoupling matters most).
    """
    backend = _flag_value("REPROLAB_GRADER_BACKEND")
    model_override = os.environ.get("REPROLAB_GRADER_MODEL", "").strip() or None

    # Default path: nothing requested → ride the root client unchanged.
    if not backend and not model_override:
        return fallback_client, fallback_label

    # A model override with no explicit backend is ambiguous about transport;
    # honour it only on a known backend. Without a backend we can't know which
    # SDK to build, so fall through to the default (root client) — the model
    # override alone does not change the transport.
    if not backend:
        return fallback_client, fallback_label

    try:
        if backend in ("oauth", "claude-oauth", "anthropic-oauth"):
            # Route the grader through the Claude OAuth subscription (the
            # most-available transport — no API credit needed; fits A5's
            # "survives a root/CLI wedge" intent and unblocks grader calibration
            # in OAuth-only environments). model_override or default_oauth_model()
            # (Sonnet). Mirrors run.py::_build_llm_client's anthropic-oauth path.
            from backend.services.context.workspace.tools.rlm_query import ClaudeLlmClient

            client = ClaudeLlmClient(model=model_override)
            return client, f"grader:oauth:{model_override or 'claude-oauth'}"

        if backend == "anthropic":
            from backend.services.context.workspace.tools.anthropic_messages_client import (
                DEFAULT_GRADER_MODEL,
                AnthropicMessagesClient,
            )

            model = model_override or DEFAULT_GRADER_MODEL
            # api_key=None → SDK resolves ANTHROPIC_API_KEY from the env.
            client = AnthropicMessagesClient(model=model)
            return client, f"grader:anthropic:{model}"

        if backend == "openai":
            from backend.services.context.workspace.tools.openai_client import (
                OpenAILlmClient,
            )

            # model_override or the OpenAILlmClient default (gpt-4o-mini).
            # api_key=None → SDK resolves OPENAI_API_KEY from the env.
            if model_override:
                client = OpenAILlmClient(model=model_override)
            else:
                client = OpenAILlmClient()
            return client, f"grader:openai:{model_override or 'gpt-4o-mini'}"

        if backend == "azure":
            from backend.services.context.workspace.tools.azure_openai_client import (
                AzureOpenAILlmClient,
            )

            azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
            if not azure_endpoint:
                raise ValueError(
                    "REPROLAB_GRADER_BACKEND=azure requires AZURE_OPENAI_ENDPOINT"
                )
            azure_deployment = (
                os.environ.get("AZURE_OPENAI_DEPLOYMENT", "").strip() or None
            )
            api_key = os.environ.get("AZURE_OPENAI_API_KEY", "").strip() or None
            model = model_override or "gpt-4o"
            client = AzureOpenAILlmClient(
                model=model,
                api_key=api_key,
                azure_endpoint=azure_endpoint,
                azure_deployment=azure_deployment,
            )
            return client, f"grader:azure:{model}"

        logger.warning(
            "grader_transport: unknown REPROLAB_GRADER_BACKEND=%r — falling back "
            "to the root client. Supported: anthropic, openai, azure.",
            backend,
        )
        return fallback_client, fallback_label

    except Exception as exc:  # noqa: BLE001 — never let grader-backend setup kill grading
        logger.warning(
            "grader_transport: failed to build grader backend %r (%s) — falling "
            "back to the root client so grading still runs.",
            backend,
            exc,
        )
        return fallback_client, fallback_label


__all__ = ["sample_completions", "build_grader_client"]
