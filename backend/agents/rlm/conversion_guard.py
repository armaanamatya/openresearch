"""Pure detector: a final report whose provenance fields DENY measured evidence
the rubric block proves was graded (the Adam projection-failure shape).

The contradiction is intra-report: the authoritative provenance fields
(baseline_metrics / experiment_run_id / primitive_trace) say "no measured result"
while the grader demonstrably scored populated on-disk metrics. Returns a structured
issue dict (or None). Stdlib only; no I/O; fail-open (ambiguous -> None)."""
from __future__ import annotations
from typing import Any


def _provenance_empty(report: dict[str, Any]) -> bool:
    return (
        not report.get("baseline_metrics")
        and report.get("experiment_run_id") in (None, "")
        and not report.get("primitive_trace")
    )


def detect_projection_incoherence(
    report: dict[str, Any], rubric: dict[str, Any], metrics_on_disk: dict[str, Any] | None,
) -> dict[str, Any] | None:
    graded_measured = bool(rubric) and (
        rubric.get("evidence_cites_metrics") is True
        or (rubric.get("overall_score") not in (None, 0, 0.0) and bool(metrics_on_disk))
    )
    if graded_measured and bool(metrics_on_disk) and _provenance_empty(report):
        return {
            "kind": "empty_provenance_with_graded_evidence",
            "detail": "report provenance is empty but the grader scored populated metrics.json",
        }
    return None
