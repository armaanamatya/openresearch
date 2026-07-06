"""Anthropic-compatible Azure Foundry endpoint resolver.

Mirror of ``foundry_endpoint.py`` (the OpenAI-compat resolver) but for the
``…/anthropic/v1`` Messages-API surface that serves Claude Opus 4.8 + Sonnet 5.
Same resource + key as the OpenAI-compat Foundry; only the base-url path differs.
os.environ → Settings-backed .env, fail-soft; unset ⇒ "" (callers decide).
"""
from __future__ import annotations

from urllib.parse import urlparse

from backend.agents.runtime.foundry_endpoint import _env_or_settings

ANTHROPIC_FOUNDRY_MODES: frozenset[str] = frozenset(
    {"anthropic-foundry", "opus-foundry", "sonnet-foundry"}
)
_OPUS_DEFAULT = "claude-opus-4-8"
_SONNET_DEFAULT = "claude-sonnet-5"


def normalize_anthropic_base_url(raw: str) -> str:
    """Return the canonical ``…/anthropic/v1`` base (strip trailing ``/messages`` / slash).

    A bare host (no path component — e.g. an explicit
    ``AZURE_FOUNDRY_ANTHROPIC_ENDPOINT`` override pasted as just the resource
    URL) is completed to ``…/anthropic/v1``, mirroring
    ``foundry_endpoint.normalize_foundry_base_url``'s bare-host handling. A
    genuine custom base (a non-``/anthropic`` path already present) is trusted
    as-is.
    """
    url = (raw or "").strip().rstrip("/")
    if not url:
        return ""
    if url.endswith("/messages"):
        url = url[: -len("/messages")].rstrip("/")
    if url.endswith("/anthropic/v1"):
        return url
    if url.endswith("/anthropic"):
        return url + "/v1"
    parsed = urlparse(url)
    if not parsed.path or parsed.path == "/":
        return url + "/anthropic/v1"
    return url  # already a base (explicit override) — trust it


def _settings_endpoint() -> str:
    """The OpenAI-compat endpoint (host source when no explicit anthropic endpoint)."""
    return _env_or_settings("AZURE_FOUNDRY_ENDPOINT", "azure_foundry_endpoint")


def _derive_base_url() -> str:
    explicit = _env_or_settings("AZURE_FOUNDRY_ANTHROPIC_ENDPOINT",
                                "azure_foundry_anthropic_endpoint")
    if explicit:
        return normalize_anthropic_base_url(explicit)
    oai = _settings_endpoint()
    if not oai:
        return ""
    parsed = urlparse(oai)
    if not parsed.scheme or not parsed.netloc:
        return ""
    return f"{parsed.scheme}://{parsed.netloc}/anthropic/v1"


def resolve_foundry_anthropic_credentials() -> tuple[str, str, dict[str, str]]:
    """Return ``(base_url, api_key, {"opus":…, "sonnet":…})``. Any unset field is ``""``."""
    base_url = _derive_base_url()
    api_key = _env_or_settings("AZURE_FOUNDRY_API_KEY", "azure_foundry_api_key")
    models = {
        "opus": _env_or_settings("AZURE_FOUNDRY_ANTHROPIC_OPUS",
                                 "azure_foundry_anthropic_opus") or _OPUS_DEFAULT,
        "sonnet": _env_or_settings("AZURE_FOUNDRY_ANTHROPIC_SONNET",
                                   "azure_foundry_anthropic_sonnet") or _SONNET_DEFAULT,
    }
    return base_url, api_key, models


def has_foundry_anthropic_credentials() -> bool:
    base_url, api_key, _ = resolve_foundry_anthropic_credentials()
    return bool(base_url and api_key)
