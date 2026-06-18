"""Per-role model resolution — one canonical chokepoint for the RLM harness.

The harness has five LLM roles: **planner** (the root reasoning loop),
**executor** (the `implement_baseline` code-writing sub-agent), **verifier**
(the `verify_against_rubric` judge), **grader** (the leaf scorer), and the
navigation accelerator (handled separately in ``accelerator.py``). Historically
each role read its own selection from a *different* env vocabulary
(``--model`` / ``OPENRESEARCH_EXECUTOR`` / ``OPENRESEARCH_GRADER_*`` /
``OPENRESEARCH_RUBRIC_VERIFIER_MODEL``). This module is the single resolver
that maps each role → a provider+model **descriptor**, from one precedence
ladder, so an operator can say e.g.::

    --models planner=opus,executor=gpt-4o-azure,verifier=sonnet,grader=o4-mini
    OPENRESEARCH_ROLE_MODELS='{"planner":"opus","executor":"gpt-4o-azure"}'

and mix Claude (Sonnet/Opus) and OpenAI (gpt-4/gpt-5, via Azure or direct)
freely across roles.

Design invariants (intentional, load-bearing):

* **Pure resolution, no client construction.** This module returns
  ``RoleSpec`` / ``RoleSelection`` descriptors only — it imports nothing heavy
  and builds no clients/runtimes. The consumers (``run.py``, ``factory.py``,
  ``grader_transport.py``) turn a descriptor into a client/runtime. That keeps
  this layer trivially unit-testable and free of import cycles.

* **De-collapsed Claude family.** ``models._MODEL_ALIASES`` collapses BOTH
  ``opus`` and ``sonnet`` → ``claude-oauth`` (whose *root* default is Sonnet).
  Per-role selection must distinguish them, so the vocabulary below maps
  ``opus`` → ``claude-opus-4-7`` and ``sonnet`` → ``claude-sonnet-4-6``
  explicitly. This is a deliberate divergence from the root alias table; the
  provider *classification* still matches ``models.ROOT_MODELS``.

* **Back-compat is byte-identical when unused.** When the unified surface is
  empty and no new verifier knob is set, every sub-role resolves to ``None``
  (= inherit today's behavior): executor → the existing
  ``_resolve_agent_runtime`` path, verifier → the planner's client, grader →
  the existing ``OPENRESEARCH_GRADER_*`` path. The resolver only *overrides*
  when an operator explicitly selects a role.

* **Fidelity is advisory.** The whole harness was validated on Claude
  sub-agents; an OpenAI/Azure pick for executor/verifier/grader is
  experimental. ``RoleSelection.fidelity_warnings(...)`` surfaces that (the
  caller emits a ``run_warning`` + stamps the report) but never blocks —
  mirrors the ``paper_validated`` flag on the root registry.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

# Canonical sub-role provider taxonomy. ``anthropic-oauth`` and ``anthropic``
# both build a ClaudeAgentRuntime / Claude client (the SDK resolves OAuth-vs-key
# itself); the distinction only matters to the client-consumer roles
# (verifier/grader) when picking an OAuth vs API-key transport.
PROVIDER_ANTHROPIC_OAUTH = "anthropic-oauth"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_OPENAI = "openai"
PROVIDER_AZURE = "azure"
# Azure AI Foundry OpenAI-compatible custom endpoint (e.g. Grok). A real
# sub-role provider — executor/grader/verifier can all run on it, key-only (no
# OAuth) — so a fully OAuth-free run is possible. Distinct from PROVIDER_AZURE
# (classic Azure OpenAI /openai/deployments path).
PROVIDER_AZURE_FOUNDRY = "azure-foundry"
# Passthrough stamp for a planner token that is a root-only registry key
# (qwen3-coder, kimi-k2.5, qwen3-coder-featherless, azure-foundry, …):
# resolve_root_model already validated it, so we stamp it as-is rather than
# rejecting it as an unknown sub-role token.
PROVIDER_ROOT = "root"

# Providers a sub-agent ROLE (executor/verifier/grader) can actually be built
# for. The planner may additionally be openrouter/featherless (root-only); those
# parse fine for stamping but are rejected if assigned to a sub-role.
SUBROLE_PROVIDERS: frozenset[str] = frozenset(
    {
        PROVIDER_ANTHROPIC_OAUTH,
        PROVIDER_ANTHROPIC,
        PROVIDER_OPENAI,
        PROVIDER_AZURE,
        PROVIDER_AZURE_FOUNDRY,
    }
)

# Claude family is the validated baseline for every role; anything else on a
# sub-role is experimental (fidelity warning).
_VALIDATED_SUBROLE_PROVIDERS: frozenset[str] = frozenset(
    {PROVIDER_ANTHROPIC_OAUTH, PROVIDER_ANTHROPIC}
)

ROLES: tuple[str, ...] = ("planner", "executor", "verifier", "grader")
# The three sub-roles a fidelity warning applies to (planner has its own
# paper_validated signal via the root registry).
_SUBROLES: frozenset[str] = frozenset({"executor", "verifier", "grader"})

# token -> (provider, concrete_model_or_None). ``None`` model = provider default
# (Azure: the deployment from AZURE_OPENAI_DEPLOYMENT). Tokens are lower-cased
# before lookup. Root-model keys (claude-oauth, azure-gpt-4o, gpt-5) are accepted
# too so the *planner* token resolves for stamping.
_ROLE_VOCAB: dict[str, tuple[str, str | None]] = {
    # --- Claude family (de-collapsed: opus != sonnet) ---
    "opus": (PROVIDER_ANTHROPIC_OAUTH, "claude-opus-4-7"),
    "claude-opus": (PROVIDER_ANTHROPIC_OAUTH, "claude-opus-4-7"),
    "claude-opus-4-7": (PROVIDER_ANTHROPIC_OAUTH, "claude-opus-4-7"),
    "claude-opus-4-1": (PROVIDER_ANTHROPIC_OAUTH, "claude-opus-4-1"),
    "sonnet": (PROVIDER_ANTHROPIC_OAUTH, "claude-sonnet-4-6"),
    "claude-sonnet": (PROVIDER_ANTHROPIC_OAUTH, "claude-sonnet-4-6"),
    "claude-sonnet-4-6": (PROVIDER_ANTHROPIC_OAUTH, "claude-sonnet-4-6"),
    "claude-sonnet-4-5": (PROVIDER_ANTHROPIC_OAUTH, "claude-sonnet-4-5"),
    "haiku": (PROVIDER_ANTHROPIC_OAUTH, "claude-haiku-4-5-20251001"),
    "claude-haiku": (PROVIDER_ANTHROPIC_OAUTH, "claude-haiku-4-5-20251001"),
    # Root-model keys (planner stamping / inherit).
    "claude-oauth": (PROVIDER_ANTHROPIC_OAUTH, "claude-sonnet-4-6"),
    "claude": (PROVIDER_ANTHROPIC, "claude-opus-4-7"),
    # --- OpenAI direct (OPENAI_API_KEY) ---
    "gpt-5": (PROVIDER_OPENAI, "gpt-5"),
    "gpt5": (PROVIDER_OPENAI, "gpt-5"),
    "gpt-4o": (PROVIDER_OPENAI, "gpt-4o"),
    "gpt-4o-mini": (PROVIDER_OPENAI, "gpt-4o-mini"),
    "gpt-4": (PROVIDER_OPENAI, "gpt-4"),
    "o4-mini": (PROVIDER_OPENAI, "o4-mini"),
    # --- Azure OpenAI (AZURE_OPENAI_* — model = deployment) ---
    "azure": (PROVIDER_AZURE, None),
    "azure-openai": (PROVIDER_AZURE, None),
    "gpt-4o-azure": (PROVIDER_AZURE, "gpt-4o"),
    "azure-gpt-4o": (PROVIDER_AZURE, "gpt-4o"),
    # --- Azure AI Foundry (OpenAI-compatible custom endpoint, e.g. Grok) ---
    # model None = use AZURE_FOUNDRY_DEPLOYMENT (dynamic, swappable via .env);
    # a real sub-role provider so executor/grader/verifier = grok all work.
    "azure-foundry": (PROVIDER_AZURE_FOUNDRY, None),
    "foundry": (PROVIDER_AZURE_FOUNDRY, None),
    "grok": (PROVIDER_AZURE_FOUNDRY, None),
    "grok-4.3": (PROVIDER_AZURE_FOUNDRY, None),
}


class RoleModelError(ValueError):
    """An unparseable role-model token or an unsupported sub-role provider."""


@dataclass(frozen=True)
class RoleSpec:
    """A resolved (provider, model) descriptor for one role.

    ``token`` is the operator-facing string ("opus", "gpt-4o-azure", ...).
    ``model`` is ``None`` only for the Azure provider default (= the deployment
    name, supplied at build time from ``AZURE_OPENAI_DEPLOYMENT``).
    """

    role: str
    token: str
    provider: str
    model: str | None

    @property
    def is_claude(self) -> bool:
        return self.provider in _VALIDATED_SUBROLE_PROVIDERS

    @property
    def stamp(self) -> str:
        """Compact ``provider:model`` identifier for ``final_report.models``."""
        return f"{self.provider}:{self.model or '<deployment>'}"


@dataclass(frozen=True)
class RoleSelection:
    """The resolved per-role picture for a run.

    ``planner`` is always set. The three sub-roles are ``None`` when the
    operator did not select them → the consumer inherits today's behavior.
    """

    planner: RoleSpec
    executor: RoleSpec | None
    verifier: RoleSpec | None
    grader: RoleSpec | None

    def get(self, role: str) -> RoleSpec | None:
        return getattr(self, role, None)

    @property
    def explicit_subroles(self) -> dict[str, RoleSpec]:
        """The sub-roles the operator actually overrode (non-inherit)."""
        return {
            role: spec
            for role in ("executor", "verifier", "grader")
            if (spec := self.get(role)) is not None
        }

    def stamp(self) -> dict[str, str | None]:
        """Per-role identifiers for ``final_report.models``.

        Sub-roles left to inherit stamp ``None``; ``run.py`` backfills those
        with the effective inherited model (planner's / the legacy grader's).
        """
        return {
            "planner": self.planner.stamp,
            "executor": self.executor.stamp if self.executor else None,
            "verifier": self.verifier.stamp if self.verifier else None,
            "grader": self.grader.stamp if self.grader else None,
        }

    def fidelity_warnings(self, *, fidelity_critical: bool) -> list[str]:
        """Advisory messages for non-Claude sub-role picks.

        Only emitted on a fidelity-critical run (a paper hint is present /
        SDAR-shaped). The harness was validated on Claude sub-agents, so an
        OpenAI/Azure executor/verifier/grader is experimental — surfaced, never
        blocked.
        """
        if not fidelity_critical:
            return []
        out: list[str] = []
        for role, spec in self.explicit_subroles.items():
            if not spec.is_claude:
                out.append(
                    f"role '{role}' is set to {spec.stamp} "
                    f"(token '{spec.token}'), which is not the paper-validated "
                    "Claude baseline — fidelity-critical results may drift."
                )
        return out


def supported_tokens() -> list[str]:
    """Sorted list of recognised tokens (for error messages / `--help`)."""
    return sorted(_ROLE_VOCAB)


def parse_model_spec(token: str, *, role: str) -> RoleSpec:
    """Resolve one ``token`` to a :class:`RoleSpec` for ``role``.

    Lenient for the planner (root-only providers like openrouter parse for
    stamping); strict for sub-roles (executor/verifier/grader must resolve to a
    provider we can build a runtime/client for). Raises :class:`RoleModelError`
    with the supported set on an unknown token.
    """
    key = (token or "").strip().lower()
    if not key:
        raise RoleModelError(f"empty model token for role '{role}'")
    entry = _ROLE_VOCAB.get(key)
    if entry is None:
        # Sub-roles must resolve to a buildable provider — strict.
        if role in _SUBROLES:
            raise RoleModelError(
                f"unknown model token {token!r} for role '{role}'. "
                f"Supported: {', '.join(supported_tokens())}"
            )
        # Planner: the token is the already-resolved root-model key
        # (resolve_root_model validated it upstream). Root-only keys —
        # qwen3-coder, kimi-k2.5, qwen3-coder-featherless, azure-foundry — are
        # absent from the sub-role vocab but parse fine here as a passthrough
        # for stamping (the real provider/model live in the RootModel; the
        # final report stamps planner from the resolved root label anyway).
        return RoleSpec(role=role, token=key, provider=PROVIDER_ROOT, model=key)
    provider, model = entry
    if role in _SUBROLES and provider not in SUBROLE_PROVIDERS:
        raise RoleModelError(
            f"token {token!r} (provider '{provider}') cannot drive sub-role "
            f"'{role}'; sub-roles support {sorted(SUBROLE_PROVIDERS)}"
        )
    return RoleSpec(role=role, token=key, provider=provider, model=model)


def planner_token_from_surface(raw: str | None) -> str | None:
    """Return the RAW planner token from the unified surface, or ``None``.

    ``run.py`` uses this to let ``--models planner=opus`` drive the actual root
    model (``resolve_root_model``) when ``--model`` is unset, so the planner
    stamp matches what ran. The token is returned verbatim (e.g. ``"opus"``) —
    ``resolve_root_model`` owns alias resolution + the root collapse, so this
    does not parse or validate it. A malformed surface yields ``None`` (the
    caller falls back to the default root).
    """
    try:
        return _parse_role_map(raw).get("planner") or None
    except RoleModelError:
        return None


def _parse_role_map(raw: str | None) -> dict[str, str]:
    """Parse a unified role→token map from JSON or ``k=v,k=v`` CLI form.

    Accepts ``{"planner":"opus","executor":"gpt-4o-azure"}`` or
    ``planner=opus,executor=gpt-4o-azure``. Unknown role keys are ignored
    (forward-compat); a malformed string raises :class:`RoleModelError`.
    """
    text = (raw or "").strip()
    if not text:
        return {}
    parsed: dict[str, str]
    if text.startswith("{"):
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RoleModelError(f"OPENRESEARCH_ROLE_MODELS is not valid JSON: {exc}") from exc
        if not isinstance(obj, dict):
            raise RoleModelError("role-model map must be a JSON object")
        parsed = {str(k).strip().lower(): str(v).strip() for k, v in obj.items()}
    else:
        parsed = {}
        for pair in text.split(","):
            pair = pair.strip()
            if not pair:
                continue
            if "=" not in pair:
                raise RoleModelError(
                    f"malformed --models entry {pair!r}; expected role=token"
                )
            role, _, tok = pair.partition("=")
            parsed[role.strip().lower()] = tok.strip()
    return {role: tok for role, tok in parsed.items() if role in ROLES and tok}


def resolve_role_models(
    *,
    planner_token: str,
    cli_models: str | None = None,
    role_models_json: str | None = None,
    grader_backend_env: str | None = None,
    grader_model_env: str | None = None,
    verifier_model_setting: str | None = None,
) -> RoleSelection:
    """Resolve all four roles from the unified surface + legacy feeders.

    Precedence per sub-role: (1) the unified map (``cli_models`` over
    ``role_models_json``), (2) a legacy per-role feeder where one exists,
    (3) ``None`` = inherit. The planner is always resolved from
    ``planner_token`` (the already resolved root-model key). All inputs are
    passed in explicitly (no ``os.environ`` reads) so this stays pure +
    testable; ``run.py`` is the one caller that reads env/settings and threads
    them here.

    **Executor has no legacy feeder by design.** The legacy
    ``OPENRESEARCH_EXECUTOR`` flag (azure/qwen/vllm/...) stays entirely on the
    existing ``executor.resolve_executor()`` path, which health-probes and
    *gracefully falls back* to Claude on missing creds. Routing it through this
    resolver would drive the run-time's fail-fast ``make_runtime`` branch and
    change that missing-creds behaviour. So ``executor`` is set ONLY from the
    unified surface; a legacy ``OPENRESEARCH_EXECUTOR`` run is stamped via
    ``ctx.agent_model`` instead.
    """
    unified = _parse_role_map(cli_models)
    if role_models_json:
        env_map = _parse_role_map(role_models_json)
        # CLI wins over the env JSON, key by key.
        unified = {**env_map, **unified}

    # Planner is single-sourced from planner_token (the resolved root-model key).
    # A `--models planner=X` pick is folded into the root model by run.py BEFORE
    # this call (so planner_token already reflects it); re-reading it from the
    # unified map here would risk a stamp that diverges from the root collapse
    # (e.g. opus→claude-oauth→Sonnet at the root, but opus→Opus if parsed here).
    planner = parse_model_spec(planner_token, role="planner")

    def _resolve_subrole(role: str, legacy_token: str | None) -> RoleSpec | None:
        if role in unified:
            return parse_model_spec(unified[role], role=role)
        if legacy_token:
            return parse_model_spec(legacy_token, role=role)
        return None

    # Executor: unified surface only (see the docstring — the legacy
    # OPENRESEARCH_EXECUTOR flag keeps its own graceful-fallback path).
    executor = _resolve_subrole("executor", None)

    # Verifier legacy feeder: an explicit rubric_verifier_model setting / env.
    verifier = _resolve_subrole("verifier", (verifier_model_setting or "").strip() or None)

    # Grader legacy feeder: only when the operator set GRADER_BACKEND to a
    # concrete provider (azure/openai). ``oauth``/``anthropic`` stay inherit
    # (today they ride the planner client); a bare GRADER_MODEL also stays in
    # the existing grader_transport path.
    grader_legacy: str | None = None
    gb = (grader_backend_env or "").strip().lower()
    if gb in {"azure", "azure-openai"}:
        grader_legacy = "gpt-4o-azure"
    elif gb == "openai":
        grader_legacy = (grader_model_env or "").strip() or "gpt-4o-mini"
    grader = _resolve_subrole("grader", grader_legacy)

    return RoleSelection(planner=planner, executor=executor, verifier=verifier, grader=grader)
