"""Register the rlm ``get_client`` for the ``anthropic-foundry`` backend literal.

The rlm ``AnthropicClient`` constructor takes no ``base_url``, so a foundry-anthropic
ROOT is reached only by teaching rlm's client factory to build an
``anthropic.Anthropic(api_key, base_url)`` for this backend — WITHOUT touching the
process-global ``ANTHROPIC_BASE_URL`` (which would hijack ``claude-oauth``).
Mirror of ``apply_oauth_backend_patch``; idempotent.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)
_APPLIED = False


def build_anthropic_foundry_client(backend_kwargs: dict[str, Any]):
    """Construct an rlm-compatible Anthropic client pinned to the Foundry endpoint."""
    from backend.agents.runtime.foundry_anthropic import (
        resolve_foundry_anthropic_credentials,
    )
    from rlm.clients.anthropic import AnthropicClient

    base_url, api_key, _models = resolve_foundry_anthropic_credentials()
    model_name = backend_kwargs.get("model_name", "claude-opus-4-8")
    max_tokens = backend_kwargs.get("max_tokens", 8192)
    client = AnthropicClient(api_key=api_key, model_name=model_name, max_tokens=max_tokens)
    # AnthropicClient built anthropic.Anthropic(api_key=…) with no base_url; re-point
    # its underlying sync+async SDK clients at the Foundry base_url in-place.
    import anthropic

    client.client = anthropic.Anthropic(api_key=api_key, base_url=base_url,
                                        timeout=getattr(client, "timeout", 300.0))
    client.async_client = anthropic.AsyncAnthropic(api_key=api_key, base_url=base_url,
                                                   timeout=getattr(client, "timeout", 300.0))
    return client


def apply_anthropic_foundry_backend_patch() -> None:
    """Teach rlm's client factory about the ``anthropic-foundry`` backend. Idempotent."""
    global _APPLIED
    if _APPLIED:
        return
    from rlm import clients as _rlm_clients

    _orig_get_client = _rlm_clients.get_client

    def _patched_get_client(backend: str, backend_kwargs: dict[str, Any] | None = None):
        if backend == "anthropic-foundry":
            return build_anthropic_foundry_client(backend_kwargs or {})
        return _orig_get_client(backend, backend_kwargs)

    _rlm_clients.get_client = _patched_get_client
    # Also patch the name the RLM class imported, if it bound it directly.
    try:
        import rlm.core.rlm as _rlm_core
        if hasattr(_rlm_core, "get_client"):
            _rlm_core.get_client = _patched_get_client
    except Exception:  # noqa: BLE001
        logger.debug("anthropic-foundry patch: rlm.core.rlm.get_client not rebound")
    _APPLIED = True
    logger.info("anthropic-foundry backend patch applied")


__all__ = ["apply_anthropic_foundry_backend_patch", "build_anthropic_foundry_client"]
