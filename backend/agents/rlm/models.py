"""Root-model registry for the RLM orchestrator.

Each entry in ROOT_MODELS describes one supported root model: which `rlms` ClientBackend
to use, its kwargs, a cheaper sub-call model for depth-1 routing, any per-model system-prompt
addendum, and whether the model has been paper-validated as an RLM root (arXiv 2512.24601).

Validated roots (paper §3.2, §4): GPT-5, Qwen3-Coder-480B-A35B.
Unvalidated (Claude Opus 4.1 appears only as a coding-agent baseline; Kimi K2.5 is untested
against the RLM benchmarks). Unvalidated roots emit a ``root_model_unvalidated`` warning
at run-start via ``resolve_root_model``.

OpenRouter model slugs are config-overridable via env vars so they can be updated without
touching source code.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, replace

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Env-var keys
# ---------------------------------------------------------------------------

_ENV_ROOT_MODEL = "OPENRESEARCH_RLM_ROOT_MODEL"

# OpenRouter slug overrides — fall back to spec-default slugs if unset.
_ENV_SLUG_QWEN_ROOT = "OPENRESEARCH_RLM_ROOT_SLUG_QWEN"
_ENV_SLUG_QWEN_SUB = "OPENRESEARCH_RLM_SUB_SLUG_QWEN"
_ENV_SLUG_KIMI_ROOT = "OPENRESEARCH_RLM_ROOT_SLUG_KIMI"
_ENV_SLUG_KIMI_SUB = "OPENRESEARCH_RLM_SUB_SLUG_KIMI"


# Default slugs used when env vars are absent.
_DEFAULT_QWEN_ROOT_SLUG = "qwen/qwen3-coder-480b-a35b"
_DEFAULT_QWEN_SUB_SLUG = "qwen/qwen3-72b"
_DEFAULT_KIMI_ROOT_SLUG = "moonshotai/kimi-k2.5"
_DEFAULT_KIMI_SUB_SLUG = "moonshotai/kimi-vl-a3b-thinking"


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RootModel:
    """Immutable descriptor for one supported RLM root model.

    Attributes:
        key: Registry lookup key (e.g. ``"gpt-5"``).
        rlm_backend: A valid ``rlms`` ``ClientBackend`` literal
            (``"openai"``, ``"anthropic"``, ``"openrouter"``, …).
        backend_kwargs: Passed verbatim to ``RLM(backend_kwargs=…)``.
            Must contain at least ``{"model_name": "<slug>"}``.
        sub_backend: Backend for cheaper depth-1 sub-calls
            (``other_backends=[sub_backend]``).
        sub_backend_kwargs: Kwargs for the sub-call client.
        prompt_addendum: Per-model text appended verbatim to the
            system prompt.  Empty string for most models.
        paper_validated: ``True`` only for models validated as RLM roots
            in the paper (GPT-5, Qwen3-Coder). Unvalidated roots trigger
            a ``root_model_unvalidated`` warning at run-start.
    """

    key: str
    rlm_backend: str
    backend_kwargs: dict = field(default_factory=dict)
    sub_backend: str = "openai"
    sub_backend_kwargs: dict = field(default_factory=dict)
    prompt_addendum: str = ""
    paper_validated: bool = False
    api_key_env: str | None = None
    """Env var holding this model's API key.

    When ``None``, the env var is derived from the backend type via
    ``_BACKEND_ENV_KEY``; set it for OpenAI-compatible hosts that use a
    non-standard key (e.g. Featherless uses ``FEATHERLESS_API_KEY`` rather
    than ``OPENAI_API_KEY`` even though the backend type is ``"openai"``).

    Applies to BOTH the root and sub-call backends — correct only when both
    run on the same host (true for every current entry). A model mixing
    providers across root and sub-call would need a separate per-backend key.
    """


# ---------------------------------------------------------------------------
# Qwen anti-over-subcalling addendum (paper Appendix C / mapping §5)
# ---------------------------------------------------------------------------

_QWEN_PROMPT_ADDENDUM = (
    "IMPORTANT: Be very careful about using `llm_query` as it incurs high runtime costs. "
    "Always batch as much information as reasonably possible into each call "
    "(aim for around ~200k characters per call). "
    "Only use `llm_query` when you genuinely cannot answer from the REPL state alone. "
    "Prefer to accumulate results as REPL variables and call primitives directly rather "
    "than issuing many small `llm_query` calls."
)

# ---------------------------------------------------------------------------
# Featherless — OpenAI-compatible inference host for Qwen3-Coder
# ---------------------------------------------------------------------------

FEATHERLESS_BASE_URL = "https://api.featherless.ai/v1"
FEATHERLESS_ROOT_MODEL = "Qwen/Qwen3-Coder-480B-A35B-Instruct"
FEATHERLESS_SUBCALL_MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct"

# Featherless's plan caps *input* context far below the model's native window
# (Qwen3-Coder is natively 128K+). The rlm engine sizes compaction off a
# model's context limit, so it must be told this cap — see
# register_featherless_context_limits.
FEATHERLESS_PLAN_CONTEXT_LIMIT = 49_152


def register_featherless_context_limits() -> None:
    """Teach the rlm library the Featherless plan's input-context cap.

    rlm sizes compaction off ``MODEL_CONTEXT_LIMITS``, which records each
    model's *native* window. For Qwen3-Coder that is 128K+, but the Featherless
    plan caps input context at ``FEATHERLESS_PLAN_CONTEXT_LIMIT``. Without this,
    rlm believes it has ample headroom, compaction never triggers, and the
    root's accumulated context eventually 400s the provider mid-run
    (``context_length_exceeded``). Registering the real cap — by exact model
    name, which ``get_context_limit`` matches first — makes compaction trigger
    in time. Idempotent; safe to call once per run.
    """
    try:
        from rlm.utils.token_utils import MODEL_CONTEXT_LIMITS
    except ImportError:  # pragma: no cover - rlm is always installed in practice
        return
    for model_name in (FEATHERLESS_ROOT_MODEL, FEATHERLESS_SUBCALL_MODEL):
        MODEL_CONTEXT_LIMITS[model_name] = FEATHERLESS_PLAN_CONTEXT_LIMIT


# ---------------------------------------------------------------------------
# Azure AI Foundry — generic OpenAI-compatible custom endpoint
# ---------------------------------------------------------------------------
# The ``azure-foundry`` root is fully env-driven: any model deployed to a
# ``*.services.ai.azure.com/openai/v1`` endpoint (e.g. Grok) is reachable by
# pointing AZURE_FOUNDRY_ENDPOINT / _DEPLOYMENT / _API_KEY at it — no code edit
# to swap the deployed model. base_url + model_name are injected at resolve time
# (see ``_inject_foundry_kwargs``); the registry entry stays secret- and
# value-free, like every other entry.
AZURE_FOUNDRY_KEY = "azure-foundry"


# ---------------------------------------------------------------------------
# Registry builder — deferred so env vars are read at call time, not import time
# ---------------------------------------------------------------------------


def _build_registry() -> dict[str, RootModel]:
    """Build the ROOT_MODELS registry, reading OpenRouter slugs from env.

    Called once at module level; the result is module-level ``ROOT_MODELS``.
    Env vars are read at this call so tests can monkeypatch them before import
    (or before calling ``_build_registry()`` directly in tests that patch after import).
    """
    qwen_root_slug = os.environ.get(_ENV_SLUG_QWEN_ROOT, _DEFAULT_QWEN_ROOT_SLUG)
    qwen_sub_slug = os.environ.get(_ENV_SLUG_QWEN_SUB, _DEFAULT_QWEN_SUB_SLUG)
    kimi_root_slug = os.environ.get(_ENV_SLUG_KIMI_ROOT, _DEFAULT_KIMI_ROOT_SLUG)
    kimi_sub_slug = os.environ.get(_ENV_SLUG_KIMI_SUB, _DEFAULT_KIMI_SUB_SLUG)

    return {
        "gpt-5": RootModel(
            key="gpt-5",
            rlm_backend="openai",
            backend_kwargs={"model_name": "gpt-5"},
            sub_backend="openai",
            sub_backend_kwargs={"model_name": "gpt-5-mini"},
            prompt_addendum="",
            paper_validated=True,
        ),
        "qwen3-coder": RootModel(
            key="qwen3-coder",
            rlm_backend="openrouter",
            backend_kwargs={"model_name": qwen_root_slug},
            sub_backend="openrouter",
            sub_backend_kwargs={"model_name": qwen_sub_slug},
            prompt_addendum=_QWEN_PROMPT_ADDENDUM,
            paper_validated=True,
        ),
        "kimi-k2.5": RootModel(
            key="kimi-k2.5",
            rlm_backend="openrouter",
            backend_kwargs={"model_name": kimi_root_slug},
            sub_backend="openrouter",
            sub_backend_kwargs={"model_name": kimi_sub_slug},
            prompt_addendum="",
            paper_validated=False,
        ),
        "claude": RootModel(
            key="claude",
            rlm_backend="anthropic",
            backend_kwargs={"model_name": "claude-opus-4-7"},
            sub_backend="anthropic",
            sub_backend_kwargs={"model_name": "claude-haiku-4-5-20251001"},
            prompt_addendum="",
            paper_validated=False,
        ),
        "claude-oauth": RootModel(
            key="claude-oauth",
            rlm_backend="anthropic-oauth",
            backend_kwargs={"model_name": "claude-sonnet-4-6"},
            sub_backend="anthropic-oauth",
            sub_backend_kwargs={"model_name": "claude-haiku-4-5-20251001"},
            prompt_addendum="",
            paper_validated=False,
            api_key_env=None,  # OAuth — no env-var key required
        ),
        "qwen3-coder-featherless": RootModel(
            key="qwen3-coder-featherless",
            rlm_backend="openai",
            backend_kwargs={"model_name": FEATHERLESS_ROOT_MODEL, "base_url": FEATHERLESS_BASE_URL},
            sub_backend="openai",
            sub_backend_kwargs={"model_name": FEATHERLESS_SUBCALL_MODEL, "base_url": FEATHERLESS_BASE_URL},
            prompt_addendum=_QWEN_PROMPT_ADDENDUM,
            paper_validated=True,
            api_key_env="FEATHERLESS_API_KEY",
        ),
        "azure-gpt-4o": RootModel(
            key="azure-gpt-4o",
            rlm_backend="azure_openai",
            # azure_endpoint and azure_deployment are env-driven — injected at
            # resolve time by _inject_api_key (extended for azure_openai backend).
            # model_name identifies the model; azure_deployment names the
            # Azure-side deployment (may differ from the model name).
            backend_kwargs={"model_name": "gpt-4o", "azure_deployment": "gpt-4o"},
            sub_backend="azure_openai",
            sub_backend_kwargs={"model_name": "gpt-4o", "azure_deployment": "gpt-4o"},
            prompt_addendum="",
            paper_validated=False,
            api_key_env="AZURE_OPENAI_API_KEY",
        ),
        "azure-foundry": RootModel(
            key="azure-foundry",
            rlm_backend="openai",
            # base_url + model_name are env-driven (AZURE_FOUNDRY_ENDPOINT /
            # AZURE_FOUNDRY_DEPLOYMENT), injected at resolve time by
            # _inject_foundry_kwargs. An OpenAI-compatible custom endpoint, so it
            # rides the same OpenAILlmClient path as Featherless.
            backend_kwargs={},
            sub_backend="openai",
            sub_backend_kwargs={},
            prompt_addendum="",
            paper_validated=False,
            api_key_env="AZURE_FOUNDRY_API_KEY",
        ),
    }


ROOT_MODELS: dict[str, RootModel] = _build_registry()

# Valid ClientBackend literals from rlm/core/types.py — used for contract validation.
# Also includes "anthropic-oauth" which is our custom patched backend (not an rlm
# core type, but valid in our extended registry).
_VALID_RLM_BACKENDS = frozenset(
    {
        "openai",
        "portkey",
        "openrouter",
        "vercel",
        "vllm",
        "litellm",
        "anthropic",
        "anthropic-oauth",
        "azure_openai",
        "gemini",
    }
)


# ---------------------------------------------------------------------------
# API-key injection — the registry stores no secrets; the key for each backend
# is read from the environment at resolve time and merged into backend_kwargs.
# ---------------------------------------------------------------------------

# Which env var holds the API key for each rlm ClientBackend.
_BACKEND_ENV_KEY: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "azure_openai": "AZURE_OPENAI_API_KEY",
}

_MODEL_LABELS: dict[str, str] = {
    "gpt-5": "GPT-5",
    "qwen3-coder": "Qwen3-Coder",
    "kimi-k2.5": "Kimi K2.5",
    "claude": "Claude API",
    "claude-oauth": "Claude OAuth",
    "qwen3-coder-featherless": "Qwen3-Coder (Featherless)",
    "azure-gpt-4o": "Azure GPT-4o",
    "azure-foundry": "Azure Foundry (custom endpoint)",
}


def _credential_value(env_var: str | None) -> str:
    """Read a credential from the real environment or Settings-backed .env.

    ``pydantic-settings`` reads ``.env`` without mutating ``os.environ``. The
    RLM registry historically read only ``os.environ``, so API keys present in
    ``.env`` could appear missing to both model resolution and the `/models`
    availability list. Keep the direct env lookup first, then fall back to the
    fields Settings already exposes for common providers.
    """
    if not env_var:
        return ""
    direct = os.environ.get(env_var, "")
    if direct:
        return direct
    try:
        from backend.config import get_settings

        settings = get_settings()
    except Exception:  # noqa: BLE001 - settings import must not break registry reads
        return ""
    attr = {
        "ANTHROPIC_API_KEY": "anthropic_api_key",
        "OPENAI_API_KEY": "openai_api_key",
        "OPENAI_ADMIN_KEY": "openai_admin_key",
        "AZURE_OPENAI_API_KEY": "azure_openai_api_key",
        "AZURE_FOUNDRY_API_KEY": "azure_foundry_api_key",
    }.get(env_var)
    return str(getattr(settings, attr, "") or "") if attr else ""


def _env_var_for(backend: str, api_key_env: str | None) -> str | None:
    """The env var holding the API key for *backend*.

    An explicit api_key_env (a RootModel property) wins; otherwise derive
    from the backend type via ``_BACKEND_ENV_KEY``.
    """
    return api_key_env or _BACKEND_ENV_KEY.get(backend)


def _inject_api_key(env_var: str | None, kwargs: dict) -> dict:
    """Return a copy of *kwargs* with ``api_key`` injected from *env_var*.

    rlm's ``AnthropicClient`` requires ``api_key`` as a constructor argument
    (it does not read the environment); ``OpenAIClient`` accepts it too. The
    registry deliberately stores ``backend_kwargs`` without secrets — the key
    is injected here, at resolve time, from the process environment.
    """
    out = dict(kwargs)
    if env_var:
        api_key = _credential_value(env_var)
        if api_key:
            out["api_key"] = api_key
    return out


def _inject_azure_kwargs(kwargs: dict, *, model_key: str) -> dict:
    """Return a copy of *kwargs* with Azure-specific env vars injected.

    ``rlm.clients.azure_openai.AzureOpenAIClient`` needs four values beyond
    the standard ``api_key``:

    - ``azure_endpoint`` (required): the Azure OpenAI resource URL, e.g.
      ``https://<resource>.openai.azure.com``.
    - ``azure_deployment`` (optional override): the deployment name.  Defaults
      to the registry's ``backend_kwargs["azure_deployment"]`` value but can be
      overridden per-environment via ``AZURE_OPENAI_DEPLOYMENT``.
    - ``api_version``: overrides the vendored rlm default (``2024-02-01``) with
      the repo-standard GA version.  Resolution order: existing kwarg value →
      ``AZURE_OPENAI_API_VERSION`` env var → ``DEFAULT_AZURE_OPENAI_API_VERSION``.

    Raises ``ValueError`` with an actionable message when ``azure_endpoint``
    is missing — the Azure backend cannot function without it.
    """
    from backend.services.context.workspace.tools.azure_openai_client import (
        DEFAULT_AZURE_OPENAI_API_VERSION,
    )

    out = dict(kwargs)
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    if not endpoint:
        raise ValueError(
            f"Root model {model_key!r} uses the 'azure_openai' backend but "
            "AZURE_OPENAI_ENDPOINT is not set. "
            "Set AZURE_OPENAI_ENDPOINT to your Azure OpenAI resource URL "
            "(e.g. https://<resource>.openai.azure.com) and optionally "
            "AZURE_OPENAI_DEPLOYMENT to your deployment name."
        )
    out["azure_endpoint"] = endpoint
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    if deployment:
        out["azure_deployment"] = deployment
    # Inject api_version only when not already present in the registry kwargs,
    # so a caller that sets it explicitly wins.
    if "api_version" not in out:
        out["api_version"] = (
            os.environ.get("AZURE_OPENAI_API_VERSION") or DEFAULT_AZURE_OPENAI_API_VERSION
        )
    return out


def _env_or_settings(env_var: str, settings_attr: str) -> str:
    """Read a config value from ``os.environ`` first, then Settings-backed .env.

    Mirrors ``_credential_value`` for non-secret Foundry config (endpoint /
    deployment): pydantic-settings reads ``.env`` without mutating
    ``os.environ``, so a value present only in ``.env`` must be fetched through
    Settings. Returns ``""`` when set in neither.
    """
    direct = os.environ.get(env_var, "").strip()
    if direct:
        return direct
    try:
        from backend.config import get_settings

        return str(getattr(get_settings(), settings_attr, "") or "").strip()
    except Exception:  # noqa: BLE001 - settings import must not break registry reads
        return ""


def _normalize_foundry_base_url(raw: str) -> str:
    """Normalise a Foundry endpoint to the ``/openai/v1`` base the OpenAI SDK appends to.

    Accepts the bare resource URL, the ``/openai`` or ``/openai/v1`` base, or the
    full ``/openai/v1/chat/completions`` path (whatever the operator pastes from
    the portal) and returns the canonical ``…/openai/v1`` base.
    """
    url = (raw or "").strip().rstrip("/")
    if url.endswith("/chat/completions"):
        url = url[: -len("/chat/completions")].rstrip("/")
    if url.endswith("/openai/v1"):
        return url
    if url.endswith("/openai"):
        return url + "/v1"
    return url + "/openai/v1"


def _inject_foundry_kwargs(kwargs: dict, *, model_key: str) -> dict:
    """Return a copy of *kwargs* with the env-driven Foundry base_url + model_name.

    Fully dynamic: ``AZURE_FOUNDRY_ENDPOINT`` → ``base_url`` (normalised),
    ``AZURE_FOUNDRY_DEPLOYMENT`` → ``model_name``, read from ``os.environ`` first
    then Settings (.env). Fails fast with an actionable message — never silently
    falls through to the plain-OpenAI branch — when either is missing.
    """
    out = dict(kwargs)
    endpoint = _env_or_settings("AZURE_FOUNDRY_ENDPOINT", "azure_foundry_endpoint")
    deployment = _env_or_settings("AZURE_FOUNDRY_DEPLOYMENT", "azure_foundry_deployment")
    missing = [
        name
        for name, val in (
            ("AZURE_FOUNDRY_ENDPOINT", endpoint),
            ("AZURE_FOUNDRY_DEPLOYMENT", deployment),
        )
        if not val
    ]
    if missing:
        raise ValueError(
            f"Root model {model_key!r} uses the Azure Foundry endpoint but "
            f"{' and '.join(missing)} {'is' if len(missing) == 1 else 'are'} not set. "
            "Set AZURE_FOUNDRY_ENDPOINT (e.g. "
            "https://<resource>.services.ai.azure.com/openai/v1) and "
            "AZURE_FOUNDRY_DEPLOYMENT (the deployed model name, e.g. grok-4.3)."
        )
    out["base_url"] = _normalize_foundry_base_url(endpoint)
    out["model_name"] = deployment
    return out


def _model_missing_credentials(entry: RootModel) -> list[str]:
    """Return safe, value-free credential labels missing for *entry*."""
    if entry.rlm_backend == "anthropic-oauth" or entry.sub_backend == "anthropic-oauth":
        try:
            from backend.agents.runtime.factory import has_provider_credentials

            if has_provider_credentials("anthropic"):
                return []
        except Exception:  # noqa: BLE001 - listing models must stay fail-soft
            pass
        return ["ANTHROPIC_API_KEY or Claude OAuth login"]

    required: list[str] = []
    for backend, env_var in (
        (entry.rlm_backend, _env_var_for(entry.rlm_backend, entry.api_key_env)),
        (entry.sub_backend, _env_var_for(entry.sub_backend, entry.api_key_env)),
    ):
        if env_var:
            required.append(env_var)
        if backend == "azure_openai":
            required.append("AZURE_OPENAI_ENDPOINT")
    if entry.key == AZURE_FOUNDRY_KEY:
        required.append("AZURE_FOUNDRY_ENDPOINT")

    missing: list[str] = []
    for env_var in dict.fromkeys(required):
        if env_var == "AZURE_OPENAI_ENDPOINT":
            value = os.environ.get(env_var, "")
        elif env_var == "AZURE_FOUNDRY_ENDPOINT":
            value = _env_or_settings(env_var, "azure_foundry_endpoint")
        else:
            value = _credential_value(env_var)
        if not value:
            missing.append(env_var)
    return missing


def list_root_model_choices() -> list[dict[str, object]]:
    """Public, secret-free model descriptors for the lab dropdown."""
    choices: list[dict[str, object]] = []
    for key, entry in ROOT_MODELS.items():
        missing = _model_missing_credentials(entry)
        choices.append(
            {
                "id": key,
                "label": _MODEL_LABELS.get(key, key),
                "provider": entry.rlm_backend.replace("_", "-"),
                "modelName": str(entry.backend_kwargs.get("model_name") or ""),
                "paperValidated": entry.paper_validated,
                "available": not missing,
                "missingCredentials": missing,
            }
        )
    return choices


# ---------------------------------------------------------------------------
# Model-name aliases
# ---------------------------------------------------------------------------
#
# The RLM registry keys (`claude`, `claude-oauth`, `gpt-5`, ...) are an internal
# vocabulary. End-users and the lab UI work in TWO other vocabularies:
#   (a) Anthropic SDK model names: `claude-sonnet-4-6`, `claude-opus-4-7`, ...
#   (b) Lab UI dropdown values: `sonnet`, `opus`
#
# Map all three vocabularies to a single registry key here so callers can pass
# whichever name they have at hand. The DEFAULT mapping prefers `claude-oauth`
# for Sonnet-class names — most local dev runs have OAuth available (no API key
# needed) and the OAuth client uses Sonnet by default. Users who explicitly want
# the API-key path can set OPENRESEARCH_RLM_ROOT_MODEL=claude to override.

_MODEL_ALIASES: dict[str, str] = {
    # Lab UI dropdown values
    "sonnet": "claude-oauth",
    "opus": "claude-oauth",      # OAuth supports Opus too; user can pin via env
    # Anthropic SDK names — Sonnet family
    "claude-sonnet-4-6": "claude-oauth",
    "claude-sonnet": "claude-oauth",
    "claude-sonnet-4-5": "claude-oauth",
    "claude-sonnet-4": "claude-oauth",
    # Anthropic SDK names — Opus family
    "claude-opus-4-7": "claude-oauth",
    "claude-opus": "claude-oauth",
    "claude-opus-4-1": "claude-oauth",
    # Anthropic SDK names — Haiku family
    "claude-haiku-4-5-20251001": "claude-oauth",
    "claude-haiku": "claude-oauth",
    # OpenAI common aliases
    "gpt5": "gpt-5",
    "gpt-4o": "gpt-5",
    "gpt-4o-mini": "gpt-5",
    # Qwen common aliases
    "qwen": "qwen3-coder",
    "qwen3": "qwen3-coder",
    "Qwen/Qwen3-Coder-480B-A35B-Instruct": "qwen3-coder-featherless",
    # Azure OpenAI aliases
    "azure": "azure-gpt-4o",
    "azure-openai": "azure-gpt-4o",
    "gpt-4o-azure": "azure-gpt-4o",
    # Azure AI Foundry (OpenAI-compatible custom endpoint) aliases. The actual
    # model served is whatever AZURE_FOUNDRY_DEPLOYMENT names — these are just
    # convenience handles so `--model grok` / `--model foundry` resolve here.
    "foundry": "azure-foundry",
    "azure-foundry-openai": "azure-foundry",
    "grok": "azure-foundry",
    "grok-3": "azure-foundry",
    "grok-4": "azure-foundry",
    "grok-4.3": "azure-foundry",
}


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


def resolve_root_model(name: str | None) -> RootModel:
    """Resolve *name* to a ``RootModel``.

    Resolution order:
    1. If *name* is provided (non-``None``, non-empty), look it up directly.
    2. If *name* is ``None`` (or empty), check ``OPENRESEARCH_RLM_ROOT_MODEL`` env var.
    3. If that is also unset, use the layered default:
       ``"gpt-5"`` when ``OPENAI_API_KEY`` is set, else ``"qwen3-coder"``.

    Raises:
        ValueError: If the resolved name does not match any key in the registry,
            or if the default selects an OpenRouter model but ``OPENROUTER_API_KEY``
            is absent (A1-H1 — fail loudly rather than silently pick an unreachable model).
    """
    if not name:
        name = os.environ.get(_ENV_ROOT_MODEL, "").strip() or None

    if not name:
        if os.environ.get("OPENAI_API_KEY"):
            name = "gpt-5"
        elif os.environ.get("FEATHERLESS_API_KEY"):
            name = "qwen3-coder-featherless"
        else:
            # Try Claude OAuth fallback before the OpenRouter-keyed default
            try:
                from backend.agents.runtime.factory import has_provider_credentials
                if has_provider_credentials("anthropic"):
                    name = "claude-oauth"
                else:
                    name = "qwen3-coder"
            except Exception:  # noqa: BLE001 — defensive: factory import failure
                name = "qwen3-coder"

    entry = ROOT_MODELS.get(name)
    if entry is None:
        # Try the alias map — handles lab UI dropdown values, Anthropic SDK
        # model names, and common nicknames.
        aliased = _MODEL_ALIASES.get(name)
        if aliased is not None:
            logger.info(
                "resolve_root_model: %r resolved via alias → %r",
                name, aliased,
            )
            name = aliased
            entry = ROOT_MODELS.get(name)
    if entry is None:
        valid = ", ".join(sorted(ROOT_MODELS))
        aliases_hint = ", ".join(sorted(_MODEL_ALIASES))
        raise ValueError(
            f"Unknown root model {name!r}. Valid registry keys: {valid}. "
            f"Aliases also accepted: {aliases_hint}. "
            f"Set {_ENV_ROOT_MODEL} or pass a valid --model argument."
        )

    # OAuth backends do not use an env-var key — verify SDK can resolve
    # credentials (API key OR Claude OAuth login) instead.
    if entry.rlm_backend == "anthropic-oauth" or entry.sub_backend == "anthropic-oauth":
        from backend.agents.runtime.factory import has_provider_credentials
        if not has_provider_credentials("anthropic"):
            raise ValueError(
                f"Root model {name!r} requires Claude credentials (ANTHROPIC_API_KEY "
                f"or `claude` CLI OAuth login) but neither was found. "
                f"Run `claude login` or set ANTHROPIC_API_KEY, or choose a different "
                f"model via {_ENV_ROOT_MODEL} / --model."
            )
        # Don't fall through to the env-var loop below for this branch.
        # Optional root-model pin (2026-06-12): REPROLAB_RLM_ROOT_MODEL_NAME
        # overrides the OAuth root's model id (e.g. an Opus id for a ratcheted
        # climb attempt where the sonnet root repeatedly lost the plot after
        # context resets) without a registry edit. Unset/empty = registry
        # default. OAuth serves any Claude id the subscription carries.
        _root_pin = os.environ.get("REPROLAB_RLM_ROOT_MODEL_NAME", "").strip()
        _pinned_bk = dict(entry.backend_kwargs)
        if _root_pin:
            _pinned_bk["model_name"] = _root_pin
            logger.info(
                "resolve_root_model: OAuth root model pinned to %r via "
                "REPROLAB_RLM_ROOT_MODEL_NAME", _root_pin,
            )
        return replace(
            entry,
            backend_kwargs=_pinned_bk,
            sub_backend_kwargs=dict(entry.sub_backend_kwargs),
        )

    # Resolve env var names for root and sub-call backends.  An explicit
    # api_key_env on the entry wins over the backend-type default (so
    # Featherless uses FEATHERLESS_API_KEY even though its backend is "openai").
    root_env = _env_var_for(entry.rlm_backend, entry.api_key_env)
    sub_env = _env_var_for(entry.sub_backend, entry.api_key_env)

    # Fail fast when a required API key is missing — catches both root and sub-call
    # backends.  A backend not in _BACKEND_ENV_KEY (and no explicit api_key_env)
    # is skipped (no key required).
    for _backend, _role, _env in (
        (entry.rlm_backend, "root", root_env),
        (entry.sub_backend, "sub-call", sub_env),
    ):
        if _env and not _credential_value(_env):
            raise ValueError(
                f"Root model {name!r} requires the {_backend!r} backend "
                f"({_role}) but {_env} is not set. "
                f"Set {_env} or choose a different model via "
                f"{_ENV_ROOT_MODEL} / --model."
            )

    # Inject the API key into backend_kwargs / sub_backend_kwargs — rlm's
    # AnthropicClient requires it as a constructor argument (it does not read
    # the environment). The registry itself stays secret-free.
    root_bk = _inject_api_key(root_env, entry.backend_kwargs)
    sub_bk = _inject_api_key(sub_env, entry.sub_backend_kwargs)

    # Azure OpenAI additionally needs azure_endpoint (required) and
    # azure_deployment (optional override) injected from env.
    if entry.rlm_backend == "azure_openai":
        root_bk = _inject_azure_kwargs(root_bk, model_key=name)
    if entry.sub_backend == "azure_openai":
        sub_bk = _inject_azure_kwargs(sub_bk, model_key=name)

    # Azure Foundry (OpenAI-compatible custom endpoint): inject the env-driven
    # base_url + model_name so the deployed model is swappable via .env alone.
    if name == AZURE_FOUNDRY_KEY:
        root_bk = _inject_foundry_kwargs(root_bk, model_key=name)
        sub_bk = _inject_foundry_kwargs(sub_bk, model_key=name)

    return replace(entry, backend_kwargs=root_bk, sub_backend_kwargs=sub_bk)
