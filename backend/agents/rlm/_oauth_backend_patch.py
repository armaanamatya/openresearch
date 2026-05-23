"""One-time monkeypatch of ``rlm.clients.get_client`` to register the
``anthropic-oauth`` backend.

rlm's ``get_client`` is a hardcoded ``if/elif`` switch with no plugin
mechanism. Rather than fork rlm or run a heavier integration, we wrap the
function once: the wrapper dispatches ``anthropic-oauth`` to our custom
``ClaudeOauthClient`` and falls through to rlm's original implementation
for every other backend.

The patch is idempotent — calling ``apply_oauth_backend_patch()`` twice is
a no-op on the second call.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_APPLIED = False


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


__all__ = ["apply_oauth_backend_patch"]
