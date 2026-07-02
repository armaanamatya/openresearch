"""Deterministic post-hoc reader: run directory -> trust-scored AttemptAssessment.

Spec: docs/superpowers/specs/2026-07-01-reproduction-campaign-and-self-improving-harness-design.md
§8.1 (AttemptAssessment) + Codex resolution F4 (§20): validator absence quarantines.

PURE READER. No LLM calls, no subprocess execution, no writes -- every field is a
deterministic function of on-disk artifacts under ``run_dir``. Prose (agent
justifications, logs, transcripts) is never read into the assessment; only
structured, machine-checkable fields are consulted. Fail-soft on individual
artifact reads (a missing/unparseable file degrades to an honest default, never
raises out of :func:`assess_attempt`); fail-CLOSED on trust (ambiguity is
quarantined, never treated as clean).

``CampaignSpend`` is imported from the shared leaf module ``campaign_types``
(not from ``campaign_policy``) -- this module sits below ``campaign_policy``
in the campaign import DAG (``campaign_policy`` depends on it, never the
reverse), so it cannot import sibling campaign modules; ``campaign_types``
is a stdlib-only leaf both modules import from, so there is only ever one
``CampaignSpend`` definition.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from backend.agents.rlm import evidence_audit, external_validator, failure_attribution
from backend.agents.rlm.campaign_types import CampaignSpend
from backend.agents.rlm.rubric_gen import canary_tripped

__all__ = [
    "CampaignSpend",
    "ValidatorStatus",
    "ReportDigest",
    "AttemptAssessment",
    "assess_attempt",
    "rubric_sha256",
]

_FABRICATION_CLASS = "fabrication_suspected"
_ALL_MODELS_FAILED_CLASS = "all_models_failed"


# ---------------------------------------------------------------------------
# Frozen types (cross-unit contract; design context "Frozen types" block)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidatorStatus:
    status: str  # raw verdict status "clean"|"vetoed"|"unavailable", or "missing"
    fingerprint: str | None
    fresh: bool  # verdict.evidence_fingerprint == fingerprint of current code/metrics.json


@dataclass(frozen=True)
class ReportDigest:
    score: float | None
    target: float | None
    meets_target: bool | None
    implementation_verdict: str | None
    replication_verdict: str | None
    verdict: str | None
    stop_reason: str | None
    exclusions: tuple[str, ...]
    path: str
    # Additive (spec §12 locked decision 9): "declared exclusions with
    # verification status" + "claims-vs-measured deltas from the existing
    # rubric/two-axis machinery", both sourced from existing structured
    # artifacts (never prose). Defaulted so every existing constructor call /
    # round-trip keeps working untouched.
    per_claim: tuple[dict[str, Any], ...] = ()
    exclusions_detail: tuple[dict[str, Any], ...] = ()


def _report_digest_from_dict(d: Mapping[str, Any]) -> ReportDigest:
    return ReportDigest(
        score=d.get("score"),
        target=d.get("target"),
        meets_target=d.get("meets_target"),
        implementation_verdict=d.get("implementation_verdict"),
        replication_verdict=d.get("replication_verdict"),
        verdict=d.get("verdict"),
        stop_reason=d.get("stop_reason"),
        exclusions=tuple(d.get("exclusions") or ()),
        path=d.get("path", ""),
        per_claim=tuple(dict(c) for c in (d.get("per_claim") or ()) if isinstance(c, Mapping)),
        exclusions_detail=tuple(
            dict(c) for c in (d.get("exclusions_detail") or ()) if isinstance(c, Mapping)
        ),
    )


@dataclass(frozen=True)
class AttemptAssessment:
    attempt_n: int
    driver: str
    project_id: str
    directives_sha256: str
    final_report: ReportDigest | None  # None => report missing
    evidence_predicates: Mapping[str, bool]  # EvidenceAudit predicate name -> pass
    guard_flags: Mapping[str, bool]  # guard name -> tripped
    validator: ValidatorStatus
    leaf_pass_count: int | None
    leaf_vector_ref: str | None
    failure_class: str | None
    failure_signature: str | None
    failure_scope: str | None  # "infra"|"method"|None
    cost: CampaignSpend
    rubric_sha256_ok: bool | None  # None = no pin to check
    hard_quarantined: bool  # fabrication guard tripped OR rubric hash mismatch
    soft_quarantined: bool  # validator not clean/fresh
    quarantine_reasons: tuple[str, ...]

    @property
    def grade_usable_for_terminal(self) -> bool:
        """Spec §8.1+F4/F5: usable for a terminal decision (not hard, not soft)."""
        return not self.hard_quarantined and not self.soft_quarantined

    @property
    def usable_for_seeding(self) -> bool:
        """Spec §8.2 F5: usable to seed the next attempt's lineage (not hard)."""
        return not self.hard_quarantined

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict -- the campaign ledger row payload."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> AttemptAssessment:
        """Round-trip a :meth:`to_dict` payload (or an equivalent JSON-loaded dict)."""
        raw_report = d.get("final_report")
        raw_validator = d.get("validator") or {}
        raw_cost = d.get("cost") or {}
        return cls(
            attempt_n=d["attempt_n"],
            driver=d["driver"],
            project_id=d["project_id"],
            directives_sha256=d["directives_sha256"],
            final_report=(
                _report_digest_from_dict(raw_report) if isinstance(raw_report, Mapping) else None
            ),
            evidence_predicates=dict(d.get("evidence_predicates") or {}),
            guard_flags=dict(d.get("guard_flags") or {}),
            validator=ValidatorStatus(
                status=raw_validator.get("status", "missing"),
                fingerprint=raw_validator.get("fingerprint"),
                fresh=bool(raw_validator.get("fresh", False)),
            ),
            leaf_pass_count=d.get("leaf_pass_count"),
            leaf_vector_ref=d.get("leaf_vector_ref"),
            failure_class=d.get("failure_class"),
            failure_signature=d.get("failure_signature"),
            failure_scope=d.get("failure_scope"),
            cost=CampaignSpend(
                llm_usd=float(raw_cost.get("llm_usd", 0.0)),
                gpu_usd=float(raw_cost.get("gpu_usd", 0.0)),
                gpu_hours=float(raw_cost.get("gpu_hours", 0.0)),
                wall_s=float(raw_cost.get("wall_s", 0.0)),
            ),
            rubric_sha256_ok=d.get("rubric_sha256_ok"),
            hard_quarantined=bool(d.get("hard_quarantined", False)),
            soft_quarantined=bool(d.get("soft_quarantined", False)),
            quarantine_reasons=tuple(d.get("quarantine_reasons") or ()),
        )


def rubric_sha256(tree: Mapping[str, Any]) -> str:
    """Canonical-JSON sha256 of a rubric tree (design context "Determinism")."""
    canonical = json.dumps(tree, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Generic fail-soft artifact readers
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Every parseable dict-shaped line, in file order. Unparseable lines are
    silently skipped -- so ``rows[-1]`` is always the last PARSEABLE row."""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return []
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _safe_float(value: Any, *, default: float | None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_iso(ts: Any) -> datetime | None:
    if not isinstance(ts, str) or not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def _duration_s(start_raw: Any, end_raw: Any) -> float | None:
    start, end = _parse_iso(start_raw), _parse_iso(end_raw)
    if start is None or end is None:
        return None
    try:
        # Clamp a negative computed duration (clock skew / swapped timestamps)
        # to 0.0 -- mirrors backend/services/pricing/timing.py's max(0.0, ...).
        return max(0.0, (end - start).total_seconds())
    except TypeError:
        # naive/aware datetime mismatch between the two timestamps -- unusable
        return None


# ---------------------------------------------------------------------------
# 1. ReportDigest (final_report.json)
# ---------------------------------------------------------------------------


def _coerce_gap(gap: Any) -> str:
    if isinstance(gap, str):
        return gap
    if isinstance(gap, dict):
        item = gap.get("item")
        if isinstance(item, str) and item:
            return item
        return json.dumps(gap, sort_keys=True, separators=(",", ":"))
    return str(gap)


def _load_final_report(run_dir: Path) -> tuple[dict[str, Any] | None, Path]:
    path = run_dir / "final_report.json"
    data = _read_json(path)
    return (data if isinstance(data, dict) else None), path


# Keys kept from a ``reproducibility.per_claim`` / ``scope.exclusions`` entry.
# Each list keeps only structured, machine-checkable fields -- per_claim drops
# ``reason`` (LLM-authored rationale prose); exclusions_detail drops
# ``evidence`` (a raw OOM signature / capacity-arithmetic snippet) but keeps
# its own ``reason``, which is a short harness-generated one-liner, not prose.
_PER_CLAIM_KEYS: tuple[str, ...] = (
    "claim_id", "status", "credit", "eligible", "measured_mean", "ci_low", "ci_high",
)
_EXCLUSION_DETAIL_KEYS: tuple[str, ...] = ("axis", "item", "kind", "verified", "reason")


def _project_keys(item: Any, keys: tuple[str, ...]) -> dict[str, Any] | None:
    """A plain dict holding only ``keys`` from ``item``, or ``None`` when
    ``item`` is not itself a mapping (defensive against a malformed entry)."""
    if not isinstance(item, Mapping):
        return None
    return {k: item.get(k) for k in keys}


def _per_claim_from_report(data: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """``reproducibility.per_claim`` (``two_axis_report.py::_verdict_to_payload``,
    itself sourced from ``reproducibility_verdict.ClaimVerdict``), projected onto
    the campaign-report-relevant keys -- ``reason`` (LLM-authored rationale
    prose) is deliberately dropped, matching this module's "no prose" contract.
    """
    repro = data.get("reproducibility")
    if not isinstance(repro, Mapping):
        return ()
    raw = repro.get("per_claim")
    if not isinstance(raw, list):
        return ()
    return tuple(c for c in (_project_keys(entry, _PER_CLAIM_KEYS) for entry in raw) if c is not None)


def _exclusions_detail_from_metrics(run_dir: Path) -> tuple[dict[str, Any], ...]:
    """``scope.exclusions`` from ``code/metrics.json`` (``exclusion.Exclusion.
    to_gap()`` shape), projected onto the verification-status-relevant keys --
    the free-text ``evidence`` field (a raw OOM signature / capacity arithmetic
    snippet) is dropped."""
    metrics = _read_json(run_dir / "code" / "metrics.json")
    if not isinstance(metrics, Mapping):
        return ()
    scope = metrics.get("scope")
    if not isinstance(scope, Mapping):
        return ()
    raw = scope.get("exclusions")
    if not isinstance(raw, list):
        return ()
    return tuple(
        c for c in (_project_keys(entry, _EXCLUSION_DETAIL_KEYS) for entry in raw) if c is not None
    )


def _report_digest_from_data(data: dict[str, Any], path: Path, run_dir: Path) -> ReportDigest:
    rubric = data.get("rubric") if isinstance(data.get("rubric"), dict) else {}
    scope = data.get("scope") if isinstance(data.get("scope"), dict) else {}
    gaps = scope.get("gaps") if isinstance(scope.get("gaps"), list) else []
    raw_stop = data.get("stop_reason")
    if isinstance(raw_stop, dict):
        stop_reason = raw_stop.get("kind")
    elif raw_stop is not None:
        stop_reason = str(raw_stop)
    else:
        stop_reason = None
    return ReportDigest(
        score=rubric.get("overall_score"),
        target=rubric.get("target_score"),
        meets_target=rubric.get("meets_target"),
        implementation_verdict=data.get("implementation_verdict"),
        replication_verdict=data.get("replication_verdict"),
        verdict=data.get("verdict"),
        stop_reason=stop_reason,
        exclusions=tuple(_coerce_gap(g) for g in gaps),
        path=str(path),
        per_claim=_per_claim_from_report(data),
        exclusions_detail=_exclusions_detail_from_metrics(run_dir),
    )


# ---------------------------------------------------------------------------
# 2. Guard flags (experiment_runs.jsonl latest row + dashboard_events.jsonl)
# ---------------------------------------------------------------------------

# Sub-source disambiguation for a fabrication_suspected latest row (brief point 3):
# first matching substring, checked in this order, wins.
_FABRICATION_SUBSOURCE_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("all-zero", "constant"), "zero_metrics"),
    (("placeholder keys",), "stub_metrics"),
    (("peak VRAM",), "vram"),
    (("metric-semantics",), "metric_semantics"),
    (("eval-metric provenance",), "eval_provenance"),
)


def _fabrication_subsource(error_text: str) -> str:
    for substrings, subsource in _FABRICATION_SUBSOURCE_RULES:
        if any(s in error_text for s in substrings):
            return subsource
    return "fabrication"


def _latest_experiment_row(run_dir: Path) -> dict[str, Any] | None:
    rows = _read_jsonl(run_dir / "experiment_runs.jsonl")
    return rows[-1] if rows else None


def _run_warning_codes(run_dir: Path) -> set[str]:
    """Codes of every run_warning event, either JSONL shape (brief point 3)."""
    codes: set[str] = set()
    for row in _read_jsonl(run_dir / "dashboard_events.jsonl"):
        if row.get("event") != "run_warning":
            continue
        payload = row.get("data") if isinstance(row.get("data"), dict) else row
        code = payload.get("code")
        if isinstance(code, str):
            codes.add(code)
    return codes


def _canary_flag(run_dir: Path) -> bool:
    """Evaluator-lockdown canary (spec §10.4): False when
    ``rubric_evaluation.json`` is absent/unparseable OR the canary leaf isn't
    present in ``leaf_scores`` -- both cases fall through to
    :func:`rubric_gen.canary_tripped`'s own conservative default."""
    data = _read_json(run_dir / "rubric_evaluation.json")
    if not isinstance(data, dict):
        return False
    return canary_tripped(data)


def _guard_flags(latest_row: dict[str, Any] | None, run_dir: Path) -> dict[str, bool]:
    fclass = str(latest_row.get("failure_class") or "") if latest_row else ""
    warn_codes = _run_warning_codes(run_dir)
    return {
        "fabrication": fclass == _FABRICATION_CLASS,
        "all_models_failed": fclass == _ALL_MODELS_FAILED_CLASS,
        "env_unavailable": "env_unavailable" in warn_codes,
        "no_learning_signal": "no_learning_signal" in warn_codes,
        "canary_tripped": _canary_flag(run_dir),
    }


# ---------------------------------------------------------------------------
# 3-4. Evidence predicates + validator (composed, not reimplemented)
# ---------------------------------------------------------------------------

_EMPTY_PREDICATES: dict[str, bool] = {
    "backed_by_ledger": False,
    "provenance_present": False,
    "metrics_non_degenerate": False,
    "metric_keys_real": False,
    "rerun_agrees": False,
    "run_level_clean": False,
}


def _evidence_predicates(run_dir: Path) -> tuple[dict[str, bool], list[str]]:
    try:
        audit = evidence_audit.audit_evidence_from_dir(run_dir)
    except Exception as exc:
        # Fail-CLOSED: an audit that cannot run must never read as "clean".
        return dict(_EMPTY_PREDICATES), [f"evidence_audit_error:{type(exc).__name__}"]
    predicates = {
        "backed_by_ledger": audit.backed_by_ledger,
        "provenance_present": audit.provenance_present,
        "metrics_non_degenerate": audit.metrics_non_degenerate,
        "metric_keys_real": audit.metric_keys_real,
        "rerun_agrees": audit.rerun_agrees,
        "run_level_clean": audit.run_level_clean,
    }
    return predicates, []


def _validator_status(run_dir: Path) -> tuple[ValidatorStatus, list[str]]:
    verdict = external_validator.load_verdict(run_dir)  # None on missing OR corrupt file
    if verdict is None:
        return ValidatorStatus(status="missing", fingerprint=None, fresh=False), ["validator:missing"]

    metrics = _read_json(run_dir / "code" / "metrics.json")
    fresh = isinstance(metrics, dict) and (
        verdict.evidence_fingerprint == external_validator.evidence_fingerprint(metrics)
    )

    reasons: list[str] = []
    if verdict.status != "clean":
        reasons.append(f"validator:{verdict.status}")
    if not fresh:
        reasons.append("validator:stale")
    return ValidatorStatus(status=verdict.status, fingerprint=verdict.evidence_fingerprint, fresh=fresh), reasons


# ---------------------------------------------------------------------------
# 5. Rubric integrity (generated_rubric.json vs. the campaign-pinned hash)
# ---------------------------------------------------------------------------


def _rubric_integrity(run_dir: Path, pinned_rubric_sha256: str | None) -> tuple[bool | None, list[str]]:
    if pinned_rubric_sha256 is None:
        return None, []
    tree = _read_json(run_dir / "generated_rubric.json")
    if not isinstance(tree, dict) or rubric_sha256(tree) != pinned_rubric_sha256:
        return False, ["rubric_integrity_mismatch"]
    return True, []


# ---------------------------------------------------------------------------
# 6. Leaf vector (rubric_evaluation.json)
# ---------------------------------------------------------------------------


def _leaf_vector(run_dir: Path) -> tuple[int | None, str | None]:
    path = run_dir / "rubric_evaluation.json"
    data = _read_json(path)
    if not isinstance(data, dict):
        return None, None
    leaf_scores = data.get("leaf_scores")
    if not isinstance(leaf_scores, list):
        return None, None
    count = 0
    for item in leaf_scores:
        if not isinstance(item, dict):
            continue
        score = _safe_float(item.get("score"), default=None)
        if score is not None and score >= 0.5:
            count += 1
    return count, str(path)


# ---------------------------------------------------------------------------
# 7. Failure attribution (report_missing override, else the latest row)
# ---------------------------------------------------------------------------


def _failure_fields(
    report_present: bool,
    latest_row: dict[str, Any] | None,
    arxiv_id: str | None,
) -> tuple[str | None, str | None, str | None]:
    if not report_present:
        return "report_missing", "report_missing", "infra"
    if latest_row is None:
        return None, None, None
    attribution = failure_attribution.attribute_failure(latest_row, arxiv_id=arxiv_id)
    return attribution.root_cause, attribution.signature, attribution.scope


# ---------------------------------------------------------------------------
# 8. Cost reconstruction
# ---------------------------------------------------------------------------


def _sum_cost_usd(path: Path) -> float:
    return sum(_safe_float(row.get("cost_usd"), default=0.0) or 0.0 for row in _read_jsonl(path))


def _gpu_rate_and_count(gpu_plan: dict[str, Any]) -> tuple[float, float]:
    gpu_count = _safe_float(gpu_plan.get("gpu_count"), default=1.0) or 1.0
    rate = _safe_float(gpu_plan.get("total_usd_per_hr"), default=None)
    if rate is None:
        sku_rate = _safe_float(gpu_plan.get("sku_usd_per_hr"), default=None)
        rate = (sku_rate * gpu_count) if sku_rate is not None else 0.0
    return rate, gpu_count


def _gpu_hours(run_dir: Path, gpu_count: float) -> float:
    timing = _read_json(run_dir / "timing.json")
    if isinstance(timing, dict):
        from_timing = _safe_float(timing.get("gpu_hours"), default=None)
        if from_timing is not None:
            return from_timing
    total_wall = sum(
        _safe_float(row.get("wall_time_s"), default=0.0) or 0.0
        for row in _read_jsonl(run_dir / "experiment_runs.jsonl")
    )
    return total_wall * gpu_count / 3600.0


def _wall_seconds(run_dir: Path, final_report_raw: dict[str, Any] | None) -> float:
    demo = _read_json(run_dir / "demo_status.json")
    if isinstance(demo, dict):
        d = _duration_s(demo.get("startedAt"), demo.get("completedAt") or demo.get("updatedAt"))
        if d is not None:
            return d
    if isinstance(final_report_raw, dict):
        d = _duration_s(final_report_raw.get("started_at"), final_report_raw.get("completed_at"))
        if d is not None:
            return d
    return 0.0


def _cost(run_dir: Path, final_report_raw: dict[str, Any] | None) -> CampaignSpend:
    gpu_plan = _read_json(run_dir / "rlm_state" / "gpu_plan.json")
    rate, gpu_count = _gpu_rate_and_count(gpu_plan if isinstance(gpu_plan, dict) else {})
    gpu_hours = _gpu_hours(run_dir, gpu_count)
    return CampaignSpend(
        llm_usd=_sum_cost_usd(run_dir / "cost_ledger.jsonl"),
        gpu_usd=gpu_hours * rate,
        gpu_hours=gpu_hours,
        wall_s=_wall_seconds(run_dir, final_report_raw),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def assess_attempt(
    run_dir: Path,
    *,
    attempt_n: int,
    driver: str,
    project_id: str,
    directives_sha256: str,
    pinned_rubric_sha256: str | None,
    arxiv_id: str | None = None,
) -> AttemptAssessment:
    """Read every artifact under ``run_dir`` and derive one trust-scored row.

    Never raises: every artifact read is individually fail-soft (missing/
    unparseable -> an honest default), so a partially-written or crashed
    attempt still yields a usable (quarantined) assessment.
    """
    run_dir = Path(run_dir)
    quarantine_reasons: list[str] = []

    report_data, report_path = _load_final_report(run_dir)
    final_report = (
        _report_digest_from_data(report_data, report_path, run_dir) if report_data is not None else None
    )

    latest_row = _latest_experiment_row(run_dir)
    guard_flags = _guard_flags(latest_row, run_dir)
    if guard_flags["fabrication"]:
        subsource = _fabrication_subsource(str((latest_row or {}).get("error") or ""))
        quarantine_reasons.append(f"guard:fabrication:{subsource}")
    if guard_flags["all_models_failed"]:
        quarantine_reasons.append("guard:all_models_failed")
    if guard_flags["canary_tripped"]:
        quarantine_reasons.append("guard:canary_tripped")

    evidence_predicates, audit_reasons = _evidence_predicates(run_dir)
    quarantine_reasons.extend(audit_reasons)

    validator, validator_reasons = _validator_status(run_dir)
    quarantine_reasons.extend(validator_reasons)

    rubric_sha256_ok, rubric_reasons = _rubric_integrity(run_dir, pinned_rubric_sha256)
    quarantine_reasons.extend(rubric_reasons)

    leaf_pass_count, leaf_vector_ref = _leaf_vector(run_dir)
    failure_class, failure_signature, failure_scope = _failure_fields(
        report_data is not None, latest_row, arxiv_id
    )
    cost = _cost(run_dir, report_data)

    hard_quarantined = (
        guard_flags["fabrication"]
        or guard_flags["all_models_failed"]
        or guard_flags["canary_tripped"]
        or rubric_sha256_ok is False
    )
    soft_quarantined = validator.status != "clean" or not validator.fresh

    return AttemptAssessment(
        attempt_n=attempt_n,
        driver=driver,
        project_id=project_id,
        directives_sha256=directives_sha256,
        final_report=final_report,
        evidence_predicates=evidence_predicates,
        guard_flags=guard_flags,
        validator=validator,
        leaf_pass_count=leaf_pass_count,
        leaf_vector_ref=leaf_vector_ref,
        failure_class=failure_class,
        failure_signature=failure_signature,
        failure_scope=failure_scope,
        cost=cost,
        rubric_sha256_ok=rubric_sha256_ok,
        hard_quarantined=hard_quarantined,
        soft_quarantined=soft_quarantined,
        quarantine_reasons=tuple(quarantine_reasons),
    )
