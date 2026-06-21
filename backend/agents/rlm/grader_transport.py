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
  ``OPENRESEARCH_GRADER_BACKEND`` / ``OPENRESEARCH_GRADER_MODEL``. Default (both unset)
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

from backend.agents.runtime.foundry_endpoint import FOUNDRY_MODE_ALIASES

logger = logging.getLogger(__name__)


def _flag_value(name: str) -> str:
    """Return a normalized env value (stripped, lower-cased); '' when unset."""
    return os.environ.get(name, "").strip().lower()


def resolve_anthropic_subrole_backend(settings: Any = None) -> str:
    """Pick ``"oauth"`` vs ``"anthropic"`` (API key) for a Claude sub-role.

    A per-role Claude pick (``--models verifier=sonnet``) must run on whichever
    Anthropic auth is actually present — an API key seamlessly substituting for
    OAuth and vice-versa — instead of hard-coding one transport. This mirrors
    the SDK/``factory.validate_provider_credentials`` auto-resolution so the
    decoupled verifier/grader CLIENT path behaves like the executor's
    ``make_runtime("anthropic")`` runtime (which the SDK already auto-resolves).

    Honours ``llm_auth_strategy`` (the production auth gate):

    * ``oauth_only`` → always ``"oauth"`` (the in-cluster default; a dead
      ``ANTHROPIC_API_KEY`` can't hijack the run — the documented trap);
    * ``api_only``   → always ``"anthropic"`` (key required);
    * ``auto`` (default) → prefer the key when present (consistent with
      ``validate_provider_credentials``'s auto branch), else OAuth.

    Returns a backend string ``build_transport_client`` understands; the actual
    client construction there still fails soft to the caller's fallback, so an
    unavailable transport never raises here.
    """
    try:
        from backend.config import get_settings
        s = settings if settings is not None else get_settings()
    except Exception:  # noqa: BLE001 — settings must never crash transport selection
        s = settings
    strategy = (getattr(s, "llm_auth_strategy", "auto") or "auto").strip().lower()
    if strategy == "oauth_only":
        return "oauth"
    if strategy == "api_only":
        return "anthropic"
    has_key = bool(
        os.environ.get("ANTHROPIC_API_KEY", "").strip()
        or getattr(s, "anthropic_api_key", "")
    )
    if has_key:
        return "anthropic"
    return "oauth"


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


def resolve_azure_deployment(model_override: str | None) -> str | None:
    """Single source of the Azure OpenAI ``model -> deployment`` rule.

    On Azure OpenAI the *deployment* routes the request (it becomes the URL
    path); the ``model`` arg is largely cosmetic. An explicit model override
    (e.g. ``OPENRESEARCH_VALIDATOR_MODEL=<deploymentB>``) must therefore override
    the deployment too, else two "different" same-provider roles would both hit
    ``AZURE_OPENAI_DEPLOYMENT`` and the separation would be a lie. Falls back to
    ``AZURE_OPENAI_DEPLOYMENT`` (stripped) and finally ``None``.

    Consumed by :func:`build_transport_client` (azure branch) and by
    ``run.py::_validator_separation_tier`` — previously each re-derived this rule
    and was kept in sync by comment only (ADR 2026-06-21, Decision 1).
    """
    return (
        model_override
        or os.environ.get("AZURE_OPENAI_DEPLOYMENT", "").strip()
        or None
    )


def build_transport_client(
    *,
    backend: str | None,
    model: str | None,
    fallback_client: Any,
    fallback_label: str = "",
    role_label: str = "grader",
) -> tuple[Any, str]:
    """Resolve an independent, sampler-capable LLM transport from explicit args.

    The backend-dispatch core shared by ``build_grader_client`` (grader role) and
    any future role (e.g. a verifier) that wants to swap onto a decoupled
    transport. ``backend`` / ``model`` are PARAMETERS here — no env reads — and
    the returned label is namespaced by ``role_label`` (e.g.
    ``"verifier:azure:gpt-4o"``).

    Returns ``(client, label)``.

    - ``backend`` and ``model`` both unset/empty → ``(fallback_client,
      fallback_label)`` UNCHANGED. The caller rides whatever transport it
      already had.
    - ``backend`` in {oauth/claude-oauth/anthropic-oauth, anthropic, openai,
      azure} → construct the corresponding sampler-capable client from env
      (``model`` overrides the model id; anthropic defaults to Sonnet). The
      label is ``"<role_label>:<backend>:<model>"``.
    - Any other backend value, or any construction error (missing SDK, missing
      credential), logs a warning and returns the fallback. We NEVER raise:
      the caller must survive a misconfigured backend (and a root/CLI wedge is
      exactly when this decoupling matters most).
    """
    backend = (backend or "").strip().lower()
    model_override = (model or "").strip() or None

    # Default path: nothing requested → ride the caller's client unchanged.
    if not backend and not model_override:
        return fallback_client, fallback_label

    # A model override with no explicit backend is ambiguous about transport;
    # honour it only on a known backend. Without a backend we can't know which
    # SDK to build, so fall through to the default (caller's client) — the model
    # override alone does not change the transport.
    if not backend:
        return fallback_client, fallback_label

    try:
        if backend in ("oauth", "claude-oauth", "anthropic-oauth"):
            # Route through the Claude OAuth subscription (the most-available
            # transport — no API credit needed; fits A5's "survives a root/CLI
            # wedge" intent and unblocks calibration in OAuth-only environments).
            # model_override or default_oauth_model() (Sonnet). Mirrors
            # run.py::_build_llm_client's anthropic-oauth path.
            from backend.services.context.workspace.tools.rlm_query import ClaudeLlmClient

            client = ClaudeLlmClient(model=model_override)
            return client, f"{role_label}:oauth:{model_override or 'claude-oauth'}"

        if backend == "anthropic":
            from backend.services.context.workspace.tools.anthropic_messages_client import (
                DEFAULT_GRADER_MODEL,
                AnthropicMessagesClient,
            )

            model = model_override or DEFAULT_GRADER_MODEL
            # api_key=None → SDK resolves ANTHROPIC_API_KEY from the env.
            client = AnthropicMessagesClient(model=model)
            return client, f"{role_label}:anthropic:{model}"

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
            return client, f"{role_label}:openai:{model_override or 'gpt-4o-mini'}"

        if backend == "azure":
            from backend.services.context.workspace.tools.azure_openai_client import (
                AzureOpenAILlmClient,
            )

            azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
            if not azure_endpoint:
                raise ValueError(
                    "transport backend=azure requires AZURE_OPENAI_ENDPOINT"
                )
            # On Azure OpenAI the DEPLOYMENT routes the request (it becomes the
            # URL path); ``model`` is largely cosmetic. So a ``model`` override
            # (e.g. OPENRESEARCH_VALIDATOR_MODEL=<deploymentB>) must override the
            # DEPLOYMENT too, otherwise executor(deploymentA) and a "different"
            # validator would both hit AZURE_OPENAI_DEPLOYMENT (deploymentA) and
            # the validator separation would be a lie. Single resolver shared with
            # run.py::_validator_separation_tier (ADR 2026-06-21, Decision 1).
            azure_deployment = resolve_azure_deployment(model_override)
            api_key = os.environ.get("AZURE_OPENAI_API_KEY", "").strip() or None
            model = model_override or "gpt-4o"
            client = AzureOpenAILlmClient(
                model=model,
                api_key=api_key,
                azure_endpoint=azure_endpoint,
                azure_deployment=azure_deployment,
            )
            return client, f"{role_label}:azure:{model}"

        if backend in FOUNDRY_MODE_ALIASES:
            # Azure AI Foundry OpenAI-compatible v1 endpoint (e.g. Grok): Bearer
            # auth, base_url=…/openai/v1, model=deployment — rides the plain
            # OpenAI SDK (OpenAILlmClient), not AzureOpenAILlmClient. All creds
            # come from the single canonical resolver. Missing creds → fall back
            # to the caller's client (this function NEVER raises).
            from backend.agents.runtime.foundry_endpoint import (
                resolve_foundry_credentials,
            )
            from backend.services.context.workspace.tools.openai_client import (
                OpenAILlmClient,
            )

            base_url, deployment, api_key = resolve_foundry_credentials()
            if not (base_url and api_key):
                logger.warning(
                    "grader_transport: %s backend=%r requires AZURE_FOUNDRY_ENDPOINT "
                    "+ AZURE_FOUNDRY_API_KEY — falling back to the caller's client.",
                    role_label,
                    backend,
                )
                return fallback_client, fallback_label
            model = model_override or deployment
            client = OpenAILlmClient(model=model, api_key=api_key, base_url=base_url)
            return client, f"{role_label}:azure-foundry:{model}"

        logger.warning(
            "grader_transport: unknown %s backend=%r — falling back to the "
            "caller's client. Supported: anthropic, openai, azure, azure-foundry.",
            role_label,
            backend,
        )
        return fallback_client, fallback_label

    except Exception as exc:  # noqa: BLE001 — never let transport setup kill the caller
        logger.warning(
            "grader_transport: failed to build %s backend %r (%s) — falling "
            "back to the caller's client so it still runs.",
            role_label,
            backend,
            exc,
        )
        return fallback_client, fallback_label


def build_grader_client(
    fallback_client: Any, fallback_label: str = ""
) -> tuple[Any, str]:
    """Resolve the grader's transport, honouring ``OPENRESEARCH_GRADER_BACKEND``.

    Thin env-reading wrapper over ``build_transport_client`` (role
    ``"grader"``). Reads ``OPENRESEARCH_GRADER_BACKEND`` /
    ``OPENRESEARCH_GRADER_MODEL`` and dispatches; observable behaviour is
    unchanged from before the extraction.

    Returns ``(client, label)``.

    - Both ``OPENRESEARCH_GRADER_BACKEND`` and ``OPENRESEARCH_GRADER_MODEL`` unset/empty
      → ``(fallback_client, fallback_label)`` UNCHANGED. This is the default:
      the grader rides the root model's client exactly as today.
    - ``OPENRESEARCH_GRADER_BACKEND`` in {anthropic, openai, azure} → construct the
      corresponding sampler-capable client from env (``OPENRESEARCH_GRADER_MODEL``
      overrides the model id; anthropic defaults to Sonnet). The label is
      ``"grader:<backend>:<model>"``.
    - Any other backend value, or any construction error (missing SDK, missing
      credential), logs a warning and returns the fallback. We NEVER raise:
      grading must survive a misconfigured grader backend (and a root/CLI wedge
      is exactly when this decoupling matters most).
    """
    return build_transport_client(
        backend=_flag_value("OPENRESEARCH_GRADER_BACKEND"),
        model=os.environ.get("OPENRESEARCH_GRADER_MODEL", "").strip() or None,
        fallback_client=fallback_client,
        fallback_label=fallback_label,
        role_label="grader",
    )


def build_validator_client(
    *, fallback_client: Any, fallback_label: str = ""
) -> tuple[Any, str]:
    """Resolve the external validator's transport — **FAIL-CLOSED** (§7.2).

    The Tier-2 adversarial validator (spec 2026-06-20) MUST NOT reuse the
    grader's *"we NEVER raise"* silent-fallback path: a validator that quietly
    degrades to the executor-family client provides ZERO independent grounding
    (it would judge the executor's work with the executor's own model lineage).
    So this builder is the deliberate inverse of ``build_grader_client``:

    * ``OPENRESEARCH_VALIDATOR_BACKEND`` unset/empty → the validator role was not
      overridden. Return ``(fallback_client, fallback_label)`` unchanged; the
      caller (run.py) sees the unchanged fallback, resolves ``separation`` as
      ``unavailable``, and the Tier-1 deterministic floor is the sole backstop.
    * ``OPENRESEARCH_VALIDATOR_BACKEND`` set → dispatch through
      ``build_transport_client`` with ``model=OPENRESEARCH_VALIDATOR_MODEL``. If
      that returns the **fallback** (a missing credential / unknown backend /
      construction error all fail soft to the fallback there), **RAISE
      ``ValueError``** — never silently fall through to the executor-family
      client. A misconfigured validator must fail loudly, not pretend to ground.

    For ``backend=azure`` the ``model`` arg overrides ``AZURE_OPENAI_DEPLOYMENT``
    (see ``build_transport_client``'s azure branch) so executor(deploymentA) and
    validator(deploymentB) hit DIFFERENT deployments — the same-provider "weak"
    panel is honestly distinct, not an alias.

    Supports the same backends as ``build_transport_client``: ``oauth`` /
    ``anthropic`` / ``openai`` / ``azure`` / ``azure-foundry`` / ``grok``.

    Returns ``(client, label)``.
    """
    backend = _flag_value("OPENRESEARCH_VALIDATOR_BACKEND")
    model = os.environ.get("OPENRESEARCH_VALIDATOR_MODEL", "").strip() or None

    # Validator role not overridden → ride the caller's client unchanged. The
    # caller treats this as ``unavailable`` (Tier-1 floor backs it).
    if not backend:
        return fallback_client, fallback_label

    client, label = build_transport_client(
        backend=backend,
        model=model,
        fallback_client=fallback_client,
        fallback_label=fallback_label,
        role_label="validator",
    )
    # FAIL-CLOSED: build_transport_client returns the *exact* fallback object on
    # a missing credential / unknown backend / construction error. Identity on
    # the client is the unambiguous signal that no independent transport was
    # built — an explicitly-requested validator that could not be constructed is
    # an error, never a silent degrade to the executor-family client.
    if client is fallback_client:
        raise ValueError(
            "OPENRESEARCH_VALIDATOR_BACKEND="
            f"{backend!r} could not construct an independent validator transport "
            "(missing credential, unknown backend, or construction error). The "
            "external validator is FAIL-CLOSED: it must not silently fall back to "
            "the executor-family client. Provide the backend's credentials "
            "(e.g. AZURE_OPENAI_* for backend=azure) or unset "
            "OPENRESEARCH_VALIDATOR_BACKEND to disable the validator."
        )
    return client, label


__all__ = [
    "sample_completions",
    "build_grader_client",
    "build_validator_client",
    "build_transport_client",
    "resolve_anthropic_subrole_backend",
]
