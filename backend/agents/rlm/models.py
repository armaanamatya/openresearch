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
    }


ROOT_MODELS: dict[str, RootModel] = _build_registry()

# Valid ClientBackend literals from rlm/core/types.py — used for contract validation.
_VALID_RLM_BACKENDS = frozenset(
    {
        "openai",
        "portkey",
        "openrouter",
        "vercel",
        "vllm",
        "litellm",
        "anthropic",
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


def _inject_api_key(backend: str, kwargs: dict) -> dict:
    """Return a copy of *kwargs* with ``api_key`` from the backend's env var.

    rlm's ``AnthropicClient`` requires ``api_key`` as a constructor argument
    (it does not read the environment); ``OpenAIClient`` accepts it too. The
    registry deliberately stores ``backend_kwargs`` without secrets — the key
    is injected here, at resolve time, from the process environment.
    """
    out = dict(kwargs)
    env_key = _BACKEND_ENV_KEY.get(backend)
    if env_key:
        api_key = os.environ.get(env_key)
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
        name = "gpt-5" if os.environ.get("OPENAI_API_KEY") else "qwen3-coder"

    entry = ROOT_MODELS.get(name)
    if entry is None:
        valid = ", ".join(sorted(ROOT_MODELS))
        raise ValueError(
            f"Unknown root model {name!r}. Valid keys: {valid}. "
            f"Set {_ENV_ROOT_MODEL} or pass a valid --model argument."
        )

    # A1-H1: if the resolved model uses OpenRouter, fail fast when the key is absent.
    if entry.rlm_backend == "openrouter" and not os.environ.get("OPENROUTER_API_KEY"):
        raise ValueError(
            f"Root model {name!r} requires the OpenRouter backend but "
            f"OPENROUTER_API_KEY is not set. "
            f"Set OPENROUTER_API_KEY or choose a different model via "
            f"{_ENV_ROOT_MODEL} / --model."
        )

    # Inject the API key into backend_kwargs / sub_backend_kwargs — rlm's
    # AnthropicClient requires it as a constructor argument (it does not read
    # the environment). The registry itself stays secret-free.
    return replace(
        entry,
        backend_kwargs=_inject_api_key(entry.rlm_backend, entry.backend_kwargs),
        sub_backend_kwargs=_inject_api_key(entry.sub_backend, entry.sub_backend_kwargs),
    )
