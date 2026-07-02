"""Root-cause failure attribution — signature + infra/method scope.

Extends the classifier in :mod:`failure_classifier` (which only returns a
``(failure_class, suggested_fix)`` pair) with a stable per-root-cause
``signature`` and a coarse ``scope`` (``"infra"`` vs ``"method"``) so a
downstream cross-run memory store (Phase 1e Unit B, ``experience_memory.py``)
can safely decide WHICH failures are safe to share across papers and WHICH
must stay scoped to the originating paper:

* ``infra``   — environment / dependency / GPU / network shape problems.
  Paper-invariant: the fix ("add the package", "shrink the batch",
  "escalate the SKU") generalises across ANY paper run. Safe to promote
  into a GLOBAL memory store keyed by ``signature`` alone.
* ``method``  — a modelling / training-protocol / result-quality problem
  that is specific to how THIS paper's baseline was implemented. Must
  never leak into cross-paper infra memory (a "fix" for one paper's
  training-protocol bug is very unlikely to be a fix for another paper's).

An unrecognised/ambiguous ``failure_class`` conservatively defaults to
``method`` scope with reduced ``confidence`` — never ``infra`` — so a
failure the classifier can't confidently place never pollutes the global
store just because its text superficially resembles an infra failure.

This module is read-only w.r.t. ``failure_classifier`` (imports
``classify_failure``, never mutates it) and is fail-soft throughout:
``attribute_failure`` never raises. A malformed/empty ``result`` degrades
to a conservative low-confidence ``method`` attribution rather than
aborting the caller.

See ``docs/superpowers/plans/2026-07-01-phase-1e-experience-memory.md``,
Unit A.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from backend.agents.rlm.failure_classifier import classify_failure

# ---------------------------------------------------------------------------
# Scope table
# ---------------------------------------------------------------------------
# Paper-invariant, cross-paper failure classes: environment / dependency /
# GPU / network shape problems whose fix generalises across ANY paper run.
# Eligible for promotion into the GLOBAL infra memory store (Unit B).
_INFRA_CLASSES: frozenset[str] = frozenset({
    "missing_module", "requirements_not_found", "dockerfile_invalid", "cuda_shlib_load",
    "cuda_oom", "oom_killed", "network_flake", "runpod_capacity", "runpod_transient_500",
    "runpod_ssh_timeout", "disk_exhausted", "nccl_timeout", "cuda_device_assert",
})

# Paper-specific failure classes: a modelling / training-protocol / result-
# quality problem scoped to THIS paper's implementation. Stays in the
# existing per-paper lessons store — never global.
_METHOD_CLASSES: frozenset[str] = frozenset({
    "scope_shape_violation", "contract_violation", "silent_oom", "insufficient_train_steps",
    "insufficient_training", "degenerate_training", "incomplete_metrics", "code_bug",
    "fabrication_suspected", "result_quality",
})

# Any failure_class NOT in either table above — including "unknown" and any
# class added to failure_classifier.FAILURE_CLASSES that this table hasn't
# been updated for yet — defaults to "method" scope with reduced confidence.
# Conservative: an unrecognised failure must never pollute global infra
# memory just because it superficially looks infra-shaped.
_DEFAULT_SCOPE = "method"
_DEFAULT_CONFIDENCE = 0.5
_KNOWN_CONFIDENCE = 1.0

# ---------------------------------------------------------------------------
# Signature normalization — strip run-specific noise (digits / paths / hex)
# so the SAME root cause across DIFFERENT runs (different pids, tmp dirs,
# byte offsets, timestamps, ...) collapses onto ONE stable signature.
# ---------------------------------------------------------------------------
_HEX_RE = re.compile(r"0x[0-9a-fA-F]+")
_PATH_RE = re.compile(r"(?:/[\w.\-]+){2,}")
_DIGIT_RE = re.compile(r"\d+")
_WS_RE = re.compile(r"\s+")

_ERROR_TAIL_LIMIT = 2000
_ERROR_TAIL_KEYS: tuple[str, ...] = ("error", "logs", "stderr", "stdout")


@dataclass(frozen=True)
class FailureAttribution:
    """A root-cause attribution for a single failed ``run_experiment`` result."""

    signature: str
    root_cause: str
    scope: str
    confidence: float
    evidence_refs: tuple[str, ...] = ()


def _normalize(text: str) -> str:
    """Strip digits / file paths / hex so the SAME root cause across
    different runs normalizes to the SAME string (and therefore the same
    ``signature``), e.g. two different pids in an otherwise identical
    traceback tail must not mint two different signatures.
    """
    if not text:
        return ""
    normalized = text.lower()
    normalized = _HEX_RE.sub("<hex>", normalized)
    normalized = _PATH_RE.sub("<path>", normalized)
    normalized = _DIGIT_RE.sub("<n>", normalized)
    normalized = _WS_RE.sub(" ", normalized).strip()
    return normalized


def _error_tail(result: dict) -> str:
    """Best-effort short text describing the failure, used only to derive
    the ``signature`` — never persisted verbatim."""
    for key in _ERROR_TAIL_KEYS:
        value = result.get(key)
        if value:
            return str(value)[-_ERROR_TAIL_LIMIT:]
    return ""


def _resolve_failure_class(result: dict) -> str:
    """Prefer an explicit ``result["failure_class"]`` (set by a postflight
    guard, e.g. the dockerfile-shape / zero-metrics / stub-metrics guards);
    fall back to the text-based classifier."""
    explicit = result.get("failure_class")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    try:
        klass, _fix = classify_failure(result)
    except Exception:  # noqa: BLE001 — fail-soft, mirrors classify_failure itself
        return "unknown"
    return klass if isinstance(klass, str) and klass else "unknown"


def _scope_for(klass: str) -> tuple[str, float]:
    """Route a failure_class to ``(scope, confidence)``."""
    if klass in _INFRA_CLASSES:
        return "infra", _KNOWN_CONFIDENCE
    if klass in _METHOD_CLASSES:
        return "method", _KNOWN_CONFIDENCE
    return _DEFAULT_SCOPE, _DEFAULT_CONFIDENCE


def _signature(klass: str, error_tail: str) -> str:
    payload = f"{klass}:{_normalize(error_tail)}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def attribute_failure(
    result: dict,
    *,
    arxiv_id: str | None = None,
    evidence_refs: tuple[str, ...] = (),
) -> FailureAttribution:
    """Attribute a failed ``run_experiment`` result to a root cause + scope.

    Never raises: an unparseable/empty/non-dict ``result`` degrades to a
    conservative ``scope="method"`` low-confidence attribution rather than
    aborting the caller or — worse — mis-scoping a failure into the global
    infra memory store.

    ``arxiv_id`` is accepted for call-site symmetry with downstream memory
    consumers (Unit B routes ``ExperienceMemory.record(attribution,
    arxiv_id=...)``) but is not itself part of the attribution — the whole
    point of ``scope`` is to decide whether paper identity matters at all.
    """
    try:
        safe_result = result if isinstance(result, dict) else {}
        klass = _resolve_failure_class(safe_result)
        scope, confidence = _scope_for(klass)
        signature = _signature(klass, _error_tail(safe_result))
        try:
            refs = tuple(evidence_refs) if evidence_refs else ()
        except TypeError:
            refs = ()
        return FailureAttribution(
            signature=signature,
            root_cause=klass,
            scope=scope,
            confidence=confidence,
            evidence_refs=refs,
        )
    except Exception:  # noqa: BLE001 — fail-soft: attribution must never raise
        return FailureAttribution(
            signature=hashlib.sha1(b"unknown:").hexdigest()[:16],
            root_cause="unknown",
            scope=_DEFAULT_SCOPE,
            confidence=_DEFAULT_CONFIDENCE,
            evidence_refs=(),
        )


__all__ = ["FailureAttribution", "attribute_failure"]
