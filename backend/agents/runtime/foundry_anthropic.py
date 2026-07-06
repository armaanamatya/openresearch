"""Anthropic-compatible Azure Foundry endpoint resolver.

Mirror of ``foundry_endpoint.py`` (the OpenAI-compat resolver) but for the
``…/anthropic`` Messages-API surface that serves Claude Opus 4.8 + Sonnet 5.
The Anthropic SDK / claude CLI append ``/v1/messages`` to ``base_url`` themselves,
so the canonical base ends at ``…/anthropic`` — a base ending in ``…/anthropic/v1``
makes the SDK POST to ``…/anthropic/v1/v1/messages`` → 404 ``api_not_supported``.
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
    """Return the canonical ``…/anthropic`` base for the Anthropic SDK.

    The Anthropic SDK / claude CLI append ``/v1/messages`` to ``base_url``
    themselves, so the base MUST end at ``…/anthropic`` — a trailing ``/v1``
    (or ``/v1/messages``) is stripped (else the SDK POSTs to
    ``…/anthropic/v1/v1/messages`` → 404). A bare host (no path — e.g. an
    explicit ``AZURE_FOUNDRY_ANTHROPIC_ENDPOINT`` pasted as just the resource
    URL, with or without a scheme) is completed to ``…/anthropic``. A genuine
    custom base (a non-``/anthropic`` path already present) is trusted as-is.
    Suffix matching (``/messages``, ``/anthropic/v1``, ``/anthropic``) is
    case-insensitive, but the original casing of the input is preserved in
    the returned string. Any query string or URL fragment is stripped since
    the canonical base carries neither.
    """
    url = (raw or "").strip()
    if not url:
        return ""
    # The canonical base carries neither a fragment nor a query string.
    url = url.split("#", 1)[0].split("?", 1)[0]
    if "://" not in url:
        url = f"https://{url}"
    url = url.rstrip("/")
    lower = url.lower()
    if lower.endswith("/messages"):
        url = url[: -len("/messages")].rstrip("/")
        lower = url.lower()
    if lower.endswith("/anthropic/v1"):
        return url[: -len("/v1")].rstrip("/")
    if lower.endswith("/anthropic"):
        return url
    parsed = urlparse(url)
    if not parsed.path or parsed.path == "/":
        return url + "/anthropic"
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
    if "://" not in oai:
        oai = f"https://{oai}"
    parsed = urlparse(oai)
    if not parsed.scheme or not parsed.netloc:
        return ""
    return f"{parsed.scheme}://{parsed.netloc}/anthropic"


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
