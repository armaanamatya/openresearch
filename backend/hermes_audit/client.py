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
    HermesAuditConfidence,
    HermesAuditReport,
    HermesAuditScope,
    HermesAuditStatus,
    HermesEvidenceRef,
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

            data = _normalize_audit_data(
                data,
                target=target,
                scope=scope,
                provider=provider.name,
            )
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
    intervention_values = ", ".join(item.value for item in HermesInterventionType)
    return (
        "You are auditing a research reproduction pipeline for unsupported claims.\n"
        f"Scope: {scope.value}\n"
        f"Target: {target}\n"
        "Return ONLY a single JSON object with these fields: target, scope, "
        "status (one of: grounded, caveat, unsupported), summary, findings, "
        "unsupported_claims, evidence_refs, recommended_intervention, "
        "corrective_note, confidence (one of: low, medium, high).\n"
        "Schema constraints:\n"
        "- findings must be an array of strings.\n"
        "- unsupported_claims must be an array of strings, not objects.\n"
        "- evidence_refs must be an array of objects with string fields: "
        "kind, path, snippet, description.\n"
        f"- recommended_intervention must be exactly one of: {intervention_values}.\n"
        "- Put detailed rationale in findings or corrective_note, not in "
        "recommended_intervention.\n"
        "Do not include prose before or after the JSON. Do not wrap in code fences.\n"
        f"Payload:\n```json\n{json.dumps(payload, indent=2)}\n```"
    )


def _normalize_audit_data(
    data: dict[str, Any],
    *,
    target: str,
    scope: HermesAuditScope,
    provider: str,
) -> dict[str, Any]:
    """Coerce common LLM JSON variants into HermesAuditReport's contract.

    The audit providers are intentionally allowed to be heterogeneous. In
    practice, Claude/Hermes often return semantically useful JSON with richer
    objects for ``unsupported_claims`` or free-form text for
    ``recommended_intervention``. Rejecting those responses forces a fallback
    and loses evidence. This normalizer keeps the strict public model while
    accepting common, safely lossy variants at the adapter boundary.
    """

    raw_response = dict(data)
    normalized = dict(data)
    normalized["target"] = _coerce_string(normalized.get("target") or target)
    normalized["scope"] = _coerce_scope(normalized.get("scope"), scope)
    normalized["provider"] = _coerce_string(normalized.get("provider") or provider)

    unsupported_claims = _coerce_string_list(normalized.get("unsupported_claims"))
    findings = _coerce_string_list(normalized.get("findings"))
    normalized["unsupported_claims"] = unsupported_claims
    normalized["findings"] = findings
    normalized["evidence_refs"] = _coerce_evidence_refs(normalized.get("evidence_refs"))

    normalized["status"] = _coerce_status(
        normalized.get("status"),
        has_unsupported_claims=bool(unsupported_claims),
    )
    normalized["confidence"] = _coerce_confidence(normalized.get("confidence"))
    normalized["summary"] = _coerce_string(normalized.get("summary"))

    corrective_note = _coerce_string(normalized.get("corrective_note"))
    raw_intervention = normalized.get("recommended_intervention")
    intervention = _coerce_intervention(raw_intervention)
    if intervention is None:
        intervention = _default_intervention(
            normalized["status"],
            has_unsupported_claims=bool(unsupported_claims),
        )
        raw_note = _coerce_string(raw_intervention)
        if raw_note:
            corrective_note = "\n".join(
                part
                for part in (
                    corrective_note,
                    f"Auditor recommendation: {raw_note}",
                )
                if part
            )
    normalized["recommended_intervention"] = intervention
    normalized["corrective_note"] = corrective_note
    normalized["raw_response"] = raw_response
    return normalized


def _coerce_scope(value: Any, fallback: HermesAuditScope) -> str:
    if isinstance(value, HermesAuditScope):
        return value.value
    raw = _coerce_string(value).lower()
    return raw if raw in {item.value for item in HermesAuditScope} else fallback.value


def _coerce_status(value: Any, *, has_unsupported_claims: bool) -> str:
    if isinstance(value, HermesAuditStatus):
        return value.value
    raw = _enumish(value)
    aliases = {
        "grounded": HermesAuditStatus.grounded.value,
        "ok": HermesAuditStatus.grounded.value,
        "supported": HermesAuditStatus.grounded.value,
        "verified": HermesAuditStatus.grounded.value,
        "caveat": HermesAuditStatus.caveat.value,
        "caveated": HermesAuditStatus.caveat.value,
        "warning": HermesAuditStatus.caveat.value,
        "partial": HermesAuditStatus.caveat.value,
        "unsupported": HermesAuditStatus.unsupported.value,
        "unsupported_claim": HermesAuditStatus.unsupported.value,
        "unsupported_claims": HermesAuditStatus.unsupported.value,
        "not_grounded": HermesAuditStatus.unsupported.value,
        "unavailable": HermesAuditStatus.unavailable.value,
        "system_error": HermesAuditStatus.system_error.value,
    }
    if raw in aliases:
        return aliases[raw]
    return HermesAuditStatus.unsupported.value if has_unsupported_claims else HermesAuditStatus.caveat.value


def _coerce_confidence(value: Any) -> str:
    if isinstance(value, HermesAuditConfidence):
        return value.value
    if isinstance(value, (int, float)):
        if value >= 0.75:
            return HermesAuditConfidence.high.value
        if value <= 0.4:
            return HermesAuditConfidence.low.value
        return HermesAuditConfidence.medium.value
    raw = _enumish(value)
    if raw in {item.value for item in HermesAuditConfidence}:
        return raw
    if raw in {"certain", "strong"}:
        return HermesAuditConfidence.high.value
    if raw in {"uncertain", "weak"}:
        return HermesAuditConfidence.low.value
    return HermesAuditConfidence.medium.value


def _coerce_intervention(value: Any) -> str | None:
    if isinstance(value, HermesInterventionType):
        return value.value
    raw = _enumish(value)
    if raw in {item.value for item in HermesInterventionType}:
        return raw
    return None


def _default_intervention(status: str, *, has_unsupported_claims: bool) -> str:
    if status == HermesAuditStatus.unsupported.value or has_unsupported_claims:
        return HermesInterventionType.request_evidence.value
    if status == HermesAuditStatus.system_error.value:
        return HermesInterventionType.escalate_human.value
    return HermesInterventionType.annotate.value


def _coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if not isinstance(value, list):
        value = [value]
    result: list[str] = []
    for item in value:
        text = _coerce_claim_like_string(item).strip()
        if text:
            result.append(text)
    return result


def _coerce_claim_like_string(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        claim = _coerce_string(value.get("claim") or value.get("summary") or value.get("title"))
        details: list[str] = []
        for key in (
            "reason",
            "rationale",
            "issue",
            "detail",
            "why",
            "missing_evidence",
            "evidence_gap",
            "recommendation",
        ):
            detail = _coerce_string(value.get(key))
            if detail:
                details.append(detail)
        if claim:
            return f"{claim} ({'; '.join(details)})" if details else claim
    return _coerce_string(value)


def _coerce_evidence_refs(value: Any) -> list[dict[str, str]]:
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    refs: list[dict[str, str]] = []
    for item in value:
        ref = _coerce_evidence_ref(item)
        if ref is not None:
            refs.append(ref)
    return refs


def _coerce_evidence_ref(value: Any) -> dict[str, str] | None:
    if isinstance(value, HermesEvidenceRef):
        return value.model_dump(mode="json")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        return {
            "kind": "audit_reference",
            "path": "",
            "snippet": "",
            "description": text,
        }
    if isinstance(value, dict):
        kind = _coerce_string(
            value.get("kind")
            or value.get("source_type")
            or value.get("type")
            or value.get("retrieved_via")
            or "audit_reference"
        )
        path = _coerce_string(
            value.get("path")
            or value.get("source_path")
            or value.get("locator")
            or value.get("source")
            or value.get("source_id")
            or value.get("chunk_id")
        )
        snippet = _coerce_string(
            value.get("snippet")
            or value.get("quote")
            or value.get("relevant_quote")
            or value.get("content")
        )
        description = _coerce_string(
            value.get("description")
            or value.get("summary")
            or value.get("detail")
            or value.get("evidence")
        )
        if not any((kind, path, snippet, description)):
            return None
        return {
            "kind": kind or "audit_reference",
            "path": path,
            "snippet": snippet,
            "description": description,
        }
    text = _coerce_string(value)
    if not text:
        return None
    return {
        "kind": "audit_reference",
        "path": "",
        "snippet": "",
        "description": text,
    }


def _coerce_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    try:
        return json.dumps(value, sort_keys=True)
    except TypeError:
        return str(value)


def _enumish(value: Any) -> str:
    return _coerce_string(value).lower().replace("-", "_").replace(" ", "_")


def _persist_memory_quietly(runs_root: Path, memory: AdapterMemory) -> None:
    """Save memory; never let a memory-write failure break an audit."""

    try:
        save_memory(runs_root, memory)
    except OSError as exc:  # disk full, perms, etc.
        logger.warning("hermes-audit: could not persist adapter memory: %s", exc)


__all__ = ["NousHermesClient"]
