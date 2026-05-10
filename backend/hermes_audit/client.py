"""Hermes audit client — robust, self-learning, fallback-aware.

Public surface (unchanged): ``NousHermesClient(...).audit(...)`` returns
a ``HermesAuditReport``. Callers that already construct
``NousHermesClient(model=..., enabled=...)`` keep working as-is.

What's new under the hood:

* **Provider chain.** Tries Nous Hermes first, then Claude (direct
  Anthropic SDK), then OpenAI (direct OpenAI SDK). Each provider is a
  Protocol implementation in ``providers.py`` — new providers plug in
  by registration, never by editing this file's branching.
* **Self-learning order.** Persists per-provider success / failure
  counters to ``<runs_root>/.hermes_adapter_memory.json`` between runs.
  The next run starts with last-known-good provider first and skips
  providers that have failed ``MAX_CONSECUTIVE_FAILURES`` (3) times in
  a row until they recover.
* **Robust JSON extraction.** Three strategies (fenced block, balanced
  braces, prose-prefix-strip) tried in order. Common LLM output shapes
  parse cleanly; the rest raise loudly.
* **Observable fallbacks.** Every fallback attempt logs to stderr at
  WARNING; the final report carries ``provider`` so the lab UI can
  show which auditor produced it. Failures never silently substitute
  a fake "ok" — terminal status is ``unavailable`` only after the
  whole chain has been exhausted.
* **Settings-driven keys.** Providers source API keys via
  ``backend.config.Settings`` (which pydantic-settings loads from
  ``.env`` regardless of os.environ state). This supersedes the old
  os.environ-based config resolution: the values were always in .env,
  but never reached os.environ for processes that didn't ``source``
  it (Lab UI's spawned children, pytest, …) — Settings closes that gap.

Why no async: each audit is one short LLM call producing JSON. The
sync ``audit()`` contract keeps ``HermesAuditService`` simple and lets
us call it from both the sync setup paths and (via ``asyncio.to_thread``
if needed) any async context.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

from backend.hermes_audit.memory import (
    AdapterMemory,
    load_memory,
    save_memory,
)
from backend.hermes_audit.models import (
    HermesAuditReport,
    HermesAuditScope,
    HermesAuditStatus,
    HermesInterventionType,
)
from backend.hermes_audit.providers import (
    AuditProvider,
    ClaudeAuditProvider,
    ClaudeCodeSdkProvider,
    NousHermesProvider,
    OpenAIAuditProvider,
    extract_audit_json,
)


logger = logging.getLogger(__name__)


def _default_provider_chain(nous_model: str) -> list[AuditProvider]:
    """Default chain: Nous → Claude (API key) → Claude Code SDK
    (subscription) → OpenAI. Each provider is constructed with safe
    defaults; callers wanting different models pass an explicit
    ``providers=`` list to ``NousHermesClient``.

    Why this order:

    * ``nous_hermes`` is first because it's the project-native auditor
      and ships with its own model config; whoever installed
      ``hermes-agent`` did so deliberately.
    * ``claude`` (direct Anthropic API key) is second: lowest latency,
      no agent overhead, fully sync.
    * ``claude_code_sdk`` is third: when no Anthropic API key is set
      but the operator has a Claude Code subscription, audits run
      against that subscription instead of failing through to OpenAI.
      Slightly heavier (spins up the agent SDK), so we don't preempt
      an explicit API key configuration.
    * ``openai`` is last as the cross-provider fallback.
    """

    return [
        NousHermesProvider(model=nous_model),
        ClaudeAuditProvider(),
        ClaudeCodeSdkProvider(),
        OpenAIAuditProvider(),
    ]


class NousHermesClient:
    """Adapter that audits a payload via the best-available provider.

    The class name is preserved for backward compatibility — under the
    hood it now manages a chain of providers, not just Nous Hermes.
    """

    def __init__(
        self,
        *,
        model: str = "anthropic/claude-sonnet-4",
        enabled: bool = True,
        providers: Sequence[AuditProvider] | None = None,
        runs_root: str | Path | None = None,
    ) -> None:
        self.model = model
        self.enabled = enabled
        self._providers: list[AuditProvider] = list(
            providers if providers is not None else _default_provider_chain(model)
        )
        # ``runs_root`` is where the self-learning memory file lives. When
        # not provided we fall back to ``./runs`` so tests / local CLI
        # invocations work out of the box.
        self._runs_root = Path(runs_root) if runs_root is not None else Path("runs")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def audit(
        self,
        *,
        scope: HermesAuditScope,
        target: str,
        payload: dict[str, Any],
    ) -> HermesAuditReport:
        if not self.enabled:
            return _disabled_report(scope=scope, target=target)

        prompt = _build_prompt(scope=scope, target=target, payload=payload)
        memory = load_memory(self._runs_root)
        order = memory.preferred_order([p.name for p in self._providers])
        provider_by_name = {p.name: p for p in self._providers}

        last_error: str = ""
        last_provider_tried: str = ""

        for provider_name in order:
            provider = provider_by_name.get(provider_name)
            if provider is None:
                continue

            if not provider.is_available():
                memory.record_failure(provider.name, error="provider not available (precheck)")
                logger.warning("hermes-audit: %s skipped (precheck failed)", provider.name)
                continue

            last_provider_tried = provider.name
            try:
                response_text = provider.call(prompt)
                data = extract_audit_json(response_text)
            except Exception as exc:  # noqa: BLE001 — record failure, try next
                memory.record_failure(provider.name, error=f"{type(exc).__name__}: {exc}")
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "hermes-audit: %s failed (%s); trying next provider",
                    provider.name,
                    last_error,
                )
                continue

            data.setdefault("target", target)
            data.setdefault("scope", scope.value)
            data.setdefault("provider", provider.name)
            try:
                report = HermesAuditReport(**data)
            except Exception as exc:  # noqa: BLE001 — schema mismatch, try next
                memory.record_failure(
                    provider.name, error=f"schema_mismatch: {type(exc).__name__}: {exc}"
                )
                last_error = f"schema_mismatch: {type(exc).__name__}: {exc}"
                logger.warning(
                    "hermes-audit: %s returned non-conforming JSON (%s); trying next",
                    provider.name,
                    exc,
                )
                continue

            memory.record_success(provider.name)
            _persist_memory_quietly(self._runs_root, memory)
            return report

        # Whole chain exhausted — return an honest "unavailable" report.
        # Status is ``unavailable``, not ``system_error``: every provider
        # had a chance and none produced a parseable report. The report's
        # ``provider`` field reflects the LAST provider tried so operators
        # can see where the chain bottomed out.
        _persist_memory_quietly(self._runs_root, memory)
        return HermesAuditReport(
            target=target,
            scope=scope,
            status=HermesAuditStatus.unavailable,
            summary="All Hermes audit providers failed",
            recommended_intervention=HermesInterventionType.annotate,
            provider=last_provider_tried or "none",
            error_message=last_error or "no providers available",
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _disabled_report(*, scope: HermesAuditScope, target: str) -> HermesAuditReport:
    return HermesAuditReport(
        target=target,
        scope=scope,
        status=HermesAuditStatus.unavailable,
        summary="Nous Hermes audit disabled",
        recommended_intervention=HermesInterventionType.annotate,
        provider="disabled",
    )


def _build_prompt(
    *, scope: HermesAuditScope, target: str, payload: dict[str, Any]
) -> str:
    return (
        "You are auditing a research reproduction pipeline for unsupported claims.\n"
        f"Scope: {scope.value}\n"
        f"Target: {target}\n"
        "Return ONLY a single JSON object with these fields: target, scope, "
        "status (one of: grounded, caveat, unsupported), summary, findings, "
        "unsupported_claims, evidence_refs, recommended_intervention, "
        "corrective_note, confidence (one of: low, medium, high).\n"
        "Do not include prose before or after the JSON. Do not wrap in code fences.\n"
        f"Payload:\n```json\n{json.dumps(payload, indent=2)}\n```"
    )


def _persist_memory_quietly(runs_root: Path, memory: AdapterMemory) -> None:
    """Save memory; never let a memory-write failure break an audit."""

    try:
        save_memory(runs_root, memory)
    except OSError as exc:  # disk full, perms, etc.
        logger.warning("hermes-audit: could not persist adapter memory: %s", exc)


__all__ = ["NousHermesClient"]
