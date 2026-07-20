"""EnvironmentAdapter — the paper-agnostic provisioning contract (Phase 1a).

Part of the provisioning-seam refactor (see
``docs/history/plans/2026-07-01-phase-1a-1b-provisioning-seam-and-gates.md``):
``env_cache.py``'s ``EnvCacheManager`` hard-codes three SDAR-specific setup
methods (``ensure_alfworld`` / ``acquire_webshop`` / ``ensure_search_qa_index``).
This module defines the generalized shape every environment-provisioning adapter
implements — ``provision`` (required), ``smoke`` and ``health`` (optional, safe
defaults) — so a future paper-specific environment can be added as a new adapter
without touching ``EnvCacheManager`` at all.

``EnvSetupResult`` and ``_fail`` are moved **verbatim** from ``env_cache.py``
(same fields, same body) — ``env_cache.py`` itself is untouched by this unit and
will re-export these symbols from here in a later task.

Imports only stdlib + ``backend.agents.rlm.exclusion`` at module scope; the
``health()`` default lazily imports ``backend.agents.rlm.env_liveness`` inside
the method body so this package stays a light, focused dependency for adapters
that never need the liveness view.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from backend.agents.rlm.exclusion import (
    AXIS_ENVIRONMENT,
    KIND_ENV_SETUP_FAILED,
    Exclusion,
)

__all__ = [
    "EnvSetupResult",
    "SmokeResult",
    "HealthReport",
    "ProvisionCtx",
    "EnvironmentAdapter",
    "_fail",
]


@dataclass
class EnvSetupResult:
    """Outcome of provisioning one environment.

    Exactly one of (``ok=True`` with a path/url/env-vars) or (``ok=False`` with
    ``exclusion``) holds. ``data_path`` is set for ALFWorld, ``base_url`` for
    WebShop; ``Search-QA`` returns ``ok=True`` with ``env_vars`` carrying the
    retriever selection (``SEARCH_QA_INDEX_DIR`` + ``SEARCH_QA_RETRIEVER``) and no
    path/url. ``env_vars`` is a generic bag merged into the child environment by
    :meth:`as_env_vars` (alongside the ALFWorld/WebShop legacy keys).
    """

    env: str
    ok: bool
    data_path: str | None = None
    base_url: str | None = None
    exclusion: Exclusion | None = None
    detail: str = ""
    env_vars: dict[str, str] = field(default_factory=dict)

    def as_env_vars(self) -> dict[str, str]:
        """Cache locations to splice into a child run's environment (empty on fail)."""
        if not self.ok:
            return {}
        out: dict[str, str] = dict(self.env_vars)
        if self.data_path:
            out["ALFWORLD_DATA"] = self.data_path
        if self.base_url:
            out["WEBSHOP_URL"] = self.base_url
        return out


@dataclass(frozen=True)
class SmokeResult:
    """Cheap post-provision liveness check — the "served > 0" signal.

    Distinct from :class:`HealthReport`: ``smoke`` is a synchronous, immediate
    check an adapter can run right after ``provision`` (e.g. "are there game
    files on disk"); ``HealthReport`` aggregates what actually happened during
    a completed run (episodes served/unavailable from ``env_health.jsonl``).
    """

    ok: bool
    served: int | None = None
    detail: str = ""


@dataclass(frozen=True)
class HealthReport:
    """Runtime ``env_health.jsonl`` aggregate view for one environment."""

    env: str
    served: int = 0
    unavailable: int = 0
    detail: str = ""


@dataclass(frozen=True)
class ProvisionCtx:
    """Minimal provisioning context passed to every adapter method.

    Intentionally thin in Phase 1a — Phase 1c enriches it (budget, scope, run
    paths) as the ``ReproductionRun`` state machine is wired up.
    """

    display_name: str = ""
    code_dir: str | None = None  # for health() to read env_health.jsonl


def _fail(env: str, reason: str, evidence: str = "") -> EnvSetupResult:
    """Build a fail result carrying a VERIFIED env_setup_failed Exclusion."""
    return EnvSetupResult(
        env=env, ok=False, detail=reason,
        exclusion=Exclusion(
            item=env, axis=AXIS_ENVIRONMENT, kind=KIND_ENV_SETUP_FAILED,
            reason=reason, verified=True, evidence=evidence,
        ),
    )


def _canon(name: str) -> str:
    """Lowercase, separator-stripped canonical form for adapter alias matching.

    ``"Alf-World"`` / ``"alf world"`` / ``"ALFWorld"`` all canonicalize to
    ``"alfworld"``. Used only by the default :meth:`EnvironmentAdapter.applies`
    — every shipped adapter overrides ``applies`` with its own explicit alias
    set (matching ``EnvCacheManager.setup``'s historical alias tuples), so this
    is a convenience fallback for adapters that don't need bespoke aliasing.
    """
    return "".join(ch for ch in (name or "").strip().lower() if ch.isalnum())


class EnvironmentAdapter(ABC):
    """The paper-agnostic environment-provisioning contract.

    A concrete adapter owns everything needed to stand up ONE environment:
    idempotent ``provision`` (the heavy, possibly-failing setup step), a cheap
    ``smoke`` check right after provisioning, and a ``health`` aggregate read
    from a completed run's ``env_health.jsonl``. ``provision`` is the only
    required method; ``applies``/``smoke``/``health`` have safe, generic
    defaults so a minimal adapter needs only ``key`` + ``provision``.
    """

    key: str  # class attribute, e.g. "alfworld" — every concrete adapter sets this

    def applies(self, env_name: str) -> bool:
        """Best-effort default alias match; concrete adapters typically override."""
        return bool(getattr(self, "key", "")) and self.key in _canon(env_name)

    @abstractmethod
    def provision(self, ctx: ProvisionCtx) -> EnvSetupResult:
        """Idempotently stand up the environment. Never raises (fail-soft)."""
        raise NotImplementedError

    def smoke(self, ctx: ProvisionCtx) -> SmokeResult:
        """Cheap liveness check right after provisioning. Default: assume healthy."""
        return SmokeResult(ok=True)

    def health(self, ctx: ProvisionCtx) -> HealthReport:
        """Aggregate this adapter's env from the run's ``env_health.jsonl``.

        Lazy-imports ``env_liveness`` so this module stays a light dependency
        for adapters/callers that never touch the liveness view. Fail-soft:
        any error (missing/None ``code_dir``, malformed records, import
        failure) degrades to an empty :class:`HealthReport`, never raises.
        """
        try:
            from backend.agents.rlm.env_liveness import read_env_health

            data = read_env_health(ctx.code_dir)
            served = 0
            unavailable = 0
            for env_name, stats in (data or {}).items():
                if not self.applies(env_name):
                    continue
                stats = stats or {}
                served += int(stats.get("episodes_served", 0) or 0)
                unavailable += int(stats.get("episodes_unavailable", 0) or 0)
            return HealthReport(env=ctx.display_name, served=served, unavailable=unavailable)
        except Exception:  # noqa: BLE001 — health is advisory, never fatal
            return HealthReport(env=ctx.display_name)
