"""Deterministic reproducibility scoring."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable

from backend.agents.schemas import Assumption, RiskLevel, VerificationReport, VerifierScore
from backend.services.scoring.model import DynamicThresholdAssessment, ReproducibilityScore


class ReproducibilityScoringService:
    """Builds the PRD's final reproducibility score from verifier artifacts."""

    def score(
        self,
        report: VerificationReport,
        *,
        assumptions: Iterable[Assumption | dict[str, Any]] = (),
        artifact_presence: dict[str, bool] | None = None,
    ) -> ReproducibilityScore:
        scores = tuple(report.verifier_scores)
        method = _named_score(scores, "method")
        environment = max(_named_score(scores, "environment"), _named_score(scores, "execution"))
        data = _named_score(scores, "data")
        metric = _named_score(scores, "metric")
        artifact = max(_named_score(scores, "artifact"), _named_score(scores, "diff"))

        if artifact_presence:
            artifact = max(artifact, _artifact_presence_score(artifact_presence))

        thresholds = tuple(_threshold_for(item) for item in scores)
        assumption_risk = _max_assumption_risk(assumptions)
        caveats = tuple(
            finding
            for verifier in scores
            for finding in verifier.findings
            if verifier.severity in {"medium", "high"} and finding
        )
        blocking = tuple(
            mismatch
            for verifier in scores
            for mismatch in verifier.mismatches
            if verifier.severity in {"high", "critical"} and mismatch
        )

        composite = (
            environment * 0.20
            + method * 0.25
            + data * 0.20
            + metric * 0.15
            + artifact * 0.20
        )
        if assumption_risk == "high":
            composite = min(composite, 84.0)
        elif assumption_risk == "critical":
            composite = min(composite, 69.0)
        if blocking:
            composite = min(composite, 59.0)

        return ReproducibilityScore(
            environment_recovered=environment,
            method_fidelity=method,
            data_pipeline_confidence=data,
            metric_validity=metric,
            artifact_completeness=artifact,
            composite=round(composite, 2),
            assumption_risk=assumption_risk,
            overall_status=report.status.value,
            dynamic_thresholds=thresholds,
            caveats=caveats,
            blocking_issues=blocking or tuple(report.decision_log_entry.splitlines()[:1] if report.decision_log_entry else ()),
            generated_at=datetime.now(timezone.utc),
        )


def _named_score(scores: tuple[VerifierScore, ...], needle: str) -> float:
    matches = [
        score.score * 100.0
        for score in scores
        if needle in score.verifier_name.lower()
    ]
    return max(matches) if matches else 0.0


def _threshold_for(score: VerifierScore) -> DynamicThresholdAssessment:
    severity = score.severity or "medium"
    evidence_count = len(score.evidence_refs)
    complexity = "high" if score.mismatches else "medium"
    if score.verifier_name.lower().startswith(("artifact", "environment")) and not score.mismatches:
        complexity = "low"
    risk = "critical" if severity == "critical" else "high" if severity == "high" else "medium"
    evidence_quality = "strong" if evidence_count >= 3 else "medium" if evidence_count else "weak"

    threshold = 70.0
    if complexity == "high":
        threshold += 12
    if risk in {"high", "critical"}:
        threshold += 10 if risk == "high" else 18
    if evidence_quality == "weak":
        threshold += 8
    elif evidence_quality == "strong":
        threshold -= 8
    threshold = max(50.0, min(95.0, threshold))

    actual = score.score * 100.0
    verdict = "verified" if actual >= threshold else "caveated" if actual >= threshold - 15 else "failed"
    return DynamicThresholdAssessment(
        verification_item=score.verifier_name,
        assessed_complexity=complexity,
        assessed_risk=risk,
        evidence_quality=evidence_quality,
        dynamic_threshold=threshold,
        actual_confidence=round(actual, 2),
        verdict=verdict,
        threshold_reasoning=(
            f"Threshold derived from complexity={complexity}, risk={risk}, "
            f"evidence_quality={evidence_quality}, evidence_refs={evidence_count}."
        ),
    )


def _max_assumption_risk(
    assumptions: Iterable[Assumption | dict[str, Any]],
) -> str:
    order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    max_risk = "low"
    for assumption in assumptions:
        raw = assumption.risk if isinstance(assumption, Assumption) else assumption.get("risk", "low")
        value = raw.value if isinstance(raw, RiskLevel) else str(raw)
        if order.get(value, 0) > order[max_risk]:
            max_risk = value
    return max_risk


def _artifact_presence_score(presence: dict[str, bool]) -> float:
    if not presence:
        return 0.0
    return sum(1 for value in presence.values() if value) / len(presence) * 100.0


__all__ = ["ReproducibilityScoringService"]
