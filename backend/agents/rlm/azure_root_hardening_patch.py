"""Harden the rlm vendored AzureOpenAIClient against transient AOAI 429s.

The upstream ``rlm.clients.azure_openai.AzureOpenAIClient`` builds its
``openai.AzureOpenAI`` / ``AsyncAzureOpenAI`` clients with the SDK default
of ``max_retries=2`` — insufficient under AOAI throttling bursts.  It also
defaults ``api_version`` to a stale ``"2024-02-01"`` (vendored at line 40),
which the ``_inject_azure_kwargs`` call in ``models.py`` now supersedes before
the client is constructed, so the version fix is a belt-and-suspenders guard
here.

This module patches ``rlm.clients.azure_openai.AzureOpenAIClient`` at import
time by substituting a subclass that overrides ``__init__`` to rebuild
``self.client`` / ``self.async_client`` with ``max_retries=6``.  The lazy
``from rlm.clients.azure_openai import AzureOpenAIClient`` inside
``rlm.clients.__init__`` picks up the patched class on every call.

Mechanism choice — subclass (option b), not method-wrapping (option a):
- ``with_429_backoff`` in ``_retry.py`` is a synchronous decorator; wrapping
  ``acompletion`` (``async def``) would require a separate async wrapper.
  Rebuilding the underlying ``openai.AzureOpenAI`` / ``AsyncAzureOpenAI``
  clients with ``max_retries=6`` lets the SDK's built-in retry logic handle
  both the sync and async paths uniformly, with no decorator divergence.
- Subclassing avoids changing the method signatures, so any future rlm
  version upgrade automatically inherits the behaviour without needing to
  re-wire decorators.

Fail-soft contract: if the rlm client import shape changes in a future
package update (class renamed, constructor signature changed, etc.),
``apply_azure_root_hardening_patch()`` logs a WARNING and returns without
raising — the unpatched upstream class is then used as-is, which degrades to
the previous behaviour (weak retry) rather than crashing.

Import once from ``run.py`` BEFORE ``from rlm import RLM``.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_PATCH_ATTR = "_openresearch_azure_hardening_applied"


def apply_azure_root_hardening_patch() -> None:
    """Patch ``rlm.clients.azure_openai.AzureOpenAIClient`` with ``max_retries=6``.

    Idempotent: a second call is a no-op (guarded by ``_PATCH_ATTR``).
    """
    try:
        import rlm.clients.azure_openai as _azure_module

        original_cls = _azure_module.AzureOpenAIClient

        # Idempotency: already patched by a previous import.
        if getattr(original_cls, _PATCH_ATTR, False):
            return

        import openai as _openai

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "azure_root_hardening_patch: could not import rlm azure client or openai — "
            "patch not applied (degraded to upstream default max_retries=2). Reason: %s",
            exc,
        )
        return

    try:

        class _HardenedAzureOpenAIClient(original_cls):
            """Drop-in subclass that rebuilds the openai clients with max_retries=6.

            The parent's __init__ resolves constructor args against env vars (e.g.
            ``api_key`` defaults to ``AZURE_OPENAI_API_KEY``).  We mirror that same
            resolution so the rebuilt clients use exactly the same resolved values,
            then replace ``self.client`` / ``self.async_client`` with clients
            constructed with ``max_retries=6``.
            """

            def __init__(
                self,
                api_key: str | None = None,
                model_name: str | None = None,
                azure_endpoint: str | None = None,
                api_version: str | None = None,
                azure_deployment: str | None = None,
                **kwargs,
            ):
                # Mirror the parent's env-var resolution for the three fields that
                # default to env vars (api_key, azure_endpoint, api_version).
                # We capture the resolved values BEFORE super().__init__ runs so
                # that we can pass them directly to openai.AzureOpenAI without
                # relying on the already-modified internal state of the client.
                resolved_api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
                resolved_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
                resolved_version = (
                    api_version
                    or os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
                )
                resolved_deployment = azure_deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT")

                # Let the parent run normally (sets self.timeout, usage tracking, etc.)
                super().__init__(
                    api_key=api_key,
                    model_name=model_name,
                    azure_endpoint=azure_endpoint,
                    api_version=api_version,
                    azure_deployment=azure_deployment,
                    **kwargs,
                )

                # Rebuild the openai clients with max_retries=6, using the same
                # resolved values so the endpoint/key/version are byte-identical.
                self.client = _openai.AzureOpenAI(
                    api_key=resolved_api_key,
                    azure_endpoint=resolved_endpoint,
                    api_version=resolved_version,
                    azure_deployment=resolved_deployment,
                    timeout=self.timeout,
                    max_retries=6,
                )
                self.async_client = _openai.AsyncAzureOpenAI(
                    api_key=resolved_api_key,
                    azure_endpoint=resolved_endpoint,
                    api_version=resolved_version,
                    azure_deployment=resolved_deployment,
                    timeout=self.timeout,
                    max_retries=6,
                )

        _HardenedAzureOpenAIClient._openresearch_azure_hardening_applied = True  # type: ignore[attr-defined]

        _azure_module.AzureOpenAIClient = _HardenedAzureOpenAIClient  # type: ignore[attr-defined]
        logger.debug(
            "azure_root_hardening_patch: AzureOpenAIClient patched with max_retries=6"
        )

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "azure_root_hardening_patch: failed to build hardened subclass — "
            "patch not applied (degraded to upstream default max_retries=2). Reason: %s",
            exc,
        )


apply_azure_root_hardening_patch()
