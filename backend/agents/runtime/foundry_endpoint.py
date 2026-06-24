"""Canonical resolver for the Azure AI Foundry OpenAI-compatible endpoint.

One place that turns ``AZURE_FOUNDRY_ENDPOINT`` / ``AZURE_FOUNDRY_DEPLOYMENT`` /
``AZURE_FOUNDRY_API_KEY`` (read from ``os.environ`` first, then Settings/.env)
into a normalised ``(base_url, deployment, api_key)`` triple. Used identically by
the root model registry, the executor runtime, and the grader/verifier transport
so grok — or any model deployed to a ``*.services.ai.azure.com/openai/v1``
endpoint — is reachable the same way for every role.

This is a v1 *OpenAI-compatible* surface (Bearer auth, ``/openai/v1/chat/
completions``), distinct from the classic Azure OpenAI ``/openai/deployments/
{name}?api-version=`` path — so it rides the plain OpenAI SDK (``base_url`` +
``api_key``), not ``AsyncAzureOpenAI``.

Stdlib + ``backend.config`` only — no import of ``backend.agents.*`` so any layer
can use it without an import cycle.
"""

from __future__ import annotations

import os

# Canonical Foundry mode-alias vocabulary — the ONE place every tier (root
# registry, executor, factory, grader/verifier transport, navigation
# accelerator) reads to decide whether a requested mode/backend names the
# Azure AI Foundry OpenAI-compatible endpoint. Compare against a normalised
# (``.strip().lower()``) value. Keep it minimal: ``azure-foundry`` is the
# canonical key, the rest are documented convenience handles. (The root-tier
# ``models._MODEL_ALIASES`` table is a separate, broader vocabulary — it also
# maps grok-3/grok-4 etc. — and is intentionally NOT unified here.)
FOUNDRY_MODE_ALIASES: frozenset[str] = frozenset(
    {"azure-foundry", "foundry", "grok", "grok-4.3"}
)


def normalize_foundry_base_url(raw: str) -> str:
    """Normalise a Foundry endpoint to the ``/openai/v1`` base the OpenAI SDK appends to.

    Accepts the bare resource URL, the ``/openai`` or ``/openai/v1`` base, or the
    full ``/openai/v1/chat/completions`` path (whatever the operator pastes from
    the portal) and returns the canonical ``…/openai/v1`` base. ``""`` in → ``""``.
    """
    url = (raw or "").strip().rstrip("/")
    if not url:
        return ""
    if url.endswith("/chat/completions"):
        url = url[: -len("/chat/completions")].rstrip("/")
    if url.endswith("/openai/v1"):
        return url
    if url.endswith("/openai"):
        return url + "/v1"
    return url + "/openai/v1"


def _env_or_settings(env_var: str, settings_attr: str) -> str:
    """``os.environ`` first, then Settings-backed ``.env`` (fail-soft)."""
    direct = os.environ.get(env_var, "").strip()
    if direct:
        return direct
    try:
        from backend.config import get_settings

        return str(getattr(get_settings(), settings_attr, "") or "").strip()
    except Exception:  # noqa: BLE001 - settings import must never break resolution
        return ""


def resolve_foundry_credentials() -> tuple[str, str, str]:
    """Return ``(base_url, deployment, api_key)`` for the Foundry endpoint.

    ``base_url`` is normalised to ``…/openai/v1``. Any unset field is ``""`` —
    callers decide whether to fail fast or degrade.
    """
    endpoint = _env_or_settings("AZURE_FOUNDRY_ENDPOINT", "azure_foundry_endpoint")
    deployment = _env_or_settings("AZURE_FOUNDRY_DEPLOYMENT", "azure_foundry_deployment")
    api_key = _env_or_settings("AZURE_FOUNDRY_API_KEY", "azure_foundry_api_key")
    return normalize_foundry_base_url(endpoint), deployment, api_key


def has_foundry_credentials() -> bool:
    """True iff both a base_url and an api_key are resolvable (deployment may default)."""
    base_url, _deployment, api_key = resolve_foundry_credentials()
    return bool(base_url and api_key)


__all__ = [
    "FOUNDRY_MODE_ALIASES",
    "normalize_foundry_base_url",
    "resolve_foundry_credentials",
    "has_foundry_credentials",
]
