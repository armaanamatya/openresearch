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

import os
from dataclasses import dataclass, field, replace


# ---------------------------------------------------------------------------
# Env-var keys
# ---------------------------------------------------------------------------

_ENV_ROOT_MODEL = "REPROLAB_RLM_ROOT_MODEL"

# OpenRouter slug overrides — fall back to spec-default slugs if unset.
_ENV_SLUG_QWEN_ROOT = "REPROLAB_RLM_ROOT_SLUG_QWEN"
_ENV_SLUG_QWEN_SUB = "REPROLAB_RLM_SUB_SLUG_QWEN"
_ENV_SLUG_KIMI_ROOT = "REPROLAB_RLM_ROOT_SLUG_KIMI"
_ENV_SLUG_KIMI_SUB = "REPROLAB_RLM_SUB_SLUG_KIMI"


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
}


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
        api_key = os.environ.get(env_var)
        if api_key:
            out["api_key"] = api_key
    return out


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


def resolve_root_model(name: str | None) -> RootModel:
    """Resolve *name* to a ``RootModel``.

    Resolution order:
    1. If *name* is provided (non-``None``, non-empty), look it up directly.
    2. If *name* is ``None`` (or empty), check ``REPROLAB_RLM_ROOT_MODEL`` env var.
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
        valid = ", ".join(sorted(ROOT_MODELS))
        raise ValueError(
            f"Unknown root model {name!r}. Valid keys: {valid}. "
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
        return replace(
            entry,
            backend_kwargs=dict(entry.backend_kwargs),
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
        if _env and not os.environ.get(_env):
            raise ValueError(
                f"Root model {name!r} requires the {_backend!r} backend "
                f"({_role}) but {_env} is not set. "
                f"Set {_env} or choose a different model via "
                f"{_ENV_ROOT_MODEL} / --model."
            )

    # Inject the API key into backend_kwargs / sub_backend_kwargs — rlm's
    # AnthropicClient requires it as a constructor argument (it does not read
    # the environment). The registry itself stays secret-free.
    return replace(
        entry,
        backend_kwargs=_inject_api_key(root_env, entry.backend_kwargs),
        sub_backend_kwargs=_inject_api_key(sub_env, entry.sub_backend_kwargs),
    )
