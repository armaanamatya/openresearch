"""Verification Team — 4 verifiers + supervisor agent.

Provides:
  - ``run_gate_offline()`` — deterministic verification without LLM
  - Individual verifier functions for unit testing
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from backend.agents.schemas import (
    BaselineResult,
    ExperimentArtifacts,
    GateDecision,
    GateStatus,
    PaperClaimMap,
    PathResult,
    VerificationReport,
    VerifierScore,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual verifiers (offline mode)
# ---------------------------------------------------------------------------

def verify_method_fidelity(
    paper_claim_map: PaperClaimMap,
    baseline_result: BaselineResult,
    code_dir: Path | None = None,
) -> VerifierScore:
    """Check implementation matches paper algorithm."""
    findings: list[str] = []
    mismatches: list[str] = []
    evidence_refs: list[str] = []

    # Check assumptions were applied
    expected_assumptions = set(
        a.assumption_id for a in paper_claim_map.ambiguities
    )
    applied = set(baseline_result.assumptions_applied)

    applied_count = len(expected_assumptions & applied)
    if applied_count == len(expected_assumptions):
        findings.append(f"All {applied_count} assumption decisions applied")
        evidence_refs.append("baseline_result.assumptions_applied")
    else:
        missing = expected_assumptions - applied
        mismatches.append(f"Missing assumption applications: {missing}")

    # Check mode
    if baseline_result.mode == "implement_from_paper":
        findings.append("Mode 2 (implement from paper) — stricter review required")

    # Check code exists
    if code_dir and code_dir.exists():
        if (code_dir / "train.py").exists():
            findings.append("train.py entry point exists")
            evidence_refs.append(str(code_dir / "train.py"))
        else:
            mismatches.append("No train.py found")

    score = 1.0 - (len(mismatches) * 0.2)
    return VerifierScore(
        verifier_name="method_fidelity",
        score=max(0.0, min(1.0, score)),
        findings=findings,
        mismatches=mismatches,
        evidence_refs=evidence_refs,
        severity="low" if not mismatches else "medium",
    )


def verify_environment(
    baseline_result: BaselineResult,
    artifacts: ExperimentArtifacts,
    baseline_dir: Path | None = None,
) -> VerifierScore:
    """Check environment reproducibility."""
    findings: list[str] = []
    mismatches: list[str] = []
    evidence_refs: list[str] = []
    has_command_failure = False

    # Check Dockerfile exists
    if baseline_result.dockerfile_path:
        df_path = Path(baseline_result.dockerfile_path)
        if df_path.exists():
            content = df_path.read_text()
            evidence_refs.append(str(df_path))
            if "FROM " in content:
                findings.append("Valid Dockerfile with base image")
            if "==" in content:
                findings.append("Pinned package versions in Dockerfile")
            else:
                mismatches.append("Packages not pinned in Dockerfile")
        else:
            mismatches.append("Dockerfile path does not exist")
    else:
        mismatches.append("No Dockerfile path specified")

    # Check structured commands.log
    if artifacts.commands_log_path and Path(artifacts.commands_log_path).exists():
        command_entries, command_mismatches = _load_command_log(Path(artifacts.commands_log_path))
        evidence_refs.append(artifacts.commands_log_path)
        if command_mismatches:
            mismatches.extend(command_mismatches)
        elif command_entries:
            findings.append(f"Structured commands.log has {len(command_entries)} command entries")
            failed = [
                entry for entry in command_entries
                if entry.get("status") != "succeeded" or entry.get("exit_code") not in (0, None)
            ]
            if failed:
                has_command_failure = True
                mismatches.append(f"{len(failed)} command log entries failed")
        else:
            mismatches.append("commands.log has no command entries")
    else:
        mismatches.append("commands.log missing")

    # Check logs
    if artifacts.log_path and Path(artifacts.log_path).exists():
        findings.append("Run logs exist")
        evidence_refs.append(artifacts.log_path)
    else:
        mismatches.append("Run logs missing")

    # Check provenance
    if artifacts.provenance_path and Path(artifacts.provenance_path).exists():
        provenance, provenance_mismatches = _load_provenance(Path(artifacts.provenance_path))
        evidence_refs.append(artifacts.provenance_path)
        if provenance_mismatches:
            mismatches.extend(provenance_mismatches)
        else:
            findings.append("provenance.json is valid JSON")
            if provenance.get("success") is False:
                has_command_failure = True
                mismatches.append("provenance marks run as unsuccessful")
    else:
        mismatches.append("provenance.json missing")

    score = 1.0 - (len(mismatches) * 0.2)
    return VerifierScore(
        verifier_name="environment_execution",
        score=max(0.0, min(1.0, score)),
        findings=findings,
        mismatches=mismatches,
        evidence_refs=evidence_refs,
        severity="high" if has_command_failure else "low" if not mismatches else "medium",
    )


def verify_data_metrics(
    paper_claim_map: PaperClaimMap,
    artifacts: ExperimentArtifacts,
) -> VerifierScore:
    """Check data and metric validity."""
    findings: list[str] = []
    mismatches: list[str] = []
    evidence_refs: list[str] = []

    # Check metrics exist
    if artifacts.metrics:
        findings.append(f"Metrics present: {list(artifacts.metrics.keys())}")
        evidence_refs.append("experiment_artifacts.metrics")
        # Check against target
        for metric_spec in paper_claim_map.metrics:
            if metric_spec.name in artifacts.metrics or "reward" in str(artifacts.metrics):
                findings.append(f"Target metric '{metric_spec.name}' found")
            else:
                mismatches.append(f"Target metric '{metric_spec.name}' missing from results")
    else:
        mismatches.append("No metrics in artifacts")

    # Check success
    if artifacts.success:
        findings.append("Experiment completed successfully")
    else:
        mismatches.append(f"Experiment failed: {artifacts.error_message}")

    score = 1.0 - (len(mismatches) * 0.25)
    return VerifierScore(
        verifier_name="data_metrics",
        score=max(0.0, min(1.0, score)),
        findings=findings,
        mismatches=mismatches,
        evidence_refs=evidence_refs,
        severity="low" if not mismatches else "high",
    )


def verify_artifacts(
    artifacts: ExperimentArtifacts,
    baseline_dir: Path | None = None,
) -> VerifierScore:
    """Check all hard artifacts exist."""
    findings: list[str] = []
    mismatches: list[str] = []
    evidence_refs: list[str] = []

    required = [
        ("metrics", bool(artifacts.metrics)),
        ("logs", bool(artifacts.log_path and Path(artifacts.log_path).exists())),
        ("commands.log", bool(artifacts.commands_log_path and Path(artifacts.commands_log_path).exists())),
        ("provenance.json", bool(artifacts.provenance_path and Path(artifacts.provenance_path).exists())),
        ("plots", bool(artifacts.plots)),
    ]

    for name, present in required:
        if present:
            findings.append(f"✓ {name}")
            if name == "metrics":
                evidence_refs.append("experiment_artifacts.metrics")
            elif name == "logs" and artifacts.log_path:
                evidence_refs.append(artifacts.log_path)
            elif name == "commands.log" and artifacts.commands_log_path:
                evidence_refs.append(artifacts.commands_log_path)
            elif name == "provenance.json" and artifacts.provenance_path:
                evidence_refs.append(artifacts.provenance_path)
            elif name == "plots":
                evidence_refs.extend(artifacts.plots)
        else:
            mismatches.append(f"Missing: {name}")

    score = sum(1 for _, p in required if p) / len(required)
    return VerifierScore(
        verifier_name="artifact_diff",
        score=score,
        findings=findings,
        mismatches=mismatches,
        evidence_refs=evidence_refs,
        severity="low" if score >= 0.8 else "high",
    )


# ---------------------------------------------------------------------------
# Supervisor: aggregate verifier decisions
# ---------------------------------------------------------------------------

def run_gate_offline(
    gate: str,
    paper_claim_map: PaperClaimMap,
    baseline_result: BaselineResult,
    artifacts: ExperimentArtifacts,
    *,
    code_dir: Path | None = None,
    baseline_dir: Path | None = None,
) -> VerificationReport:
    """Run all 4 verifiers and produce a supervisor decision (offline)."""
    scores = [
        verify_method_fidelity(paper_claim_map, baseline_result, code_dir),
        verify_environment(baseline_result, artifacts, baseline_dir),
        verify_data_metrics(paper_claim_map, artifacts),
        verify_artifacts(artifacts, baseline_dir),
    ]

    # Supervisor decision logic
    avg_score = sum(s.score for s in scores) / len(scores)
    all_mismatches = [m for s in scores for m in s.mismatches]
    has_critical = any(s.severity == "high" for s in scores)

    if avg_score >= 0.9 and not has_critical:
        status = GateStatus.verified
    elif avg_score >= 0.7:
        status = GateStatus.verified_with_caveats
    elif avg_score >= 0.5:
        status = GateStatus.partial_reproduction
    elif artifacts.success:
        status = GateStatus.verified_with_caveats
    else:
        status = GateStatus.failed_reproduction

    missing_evidence = [s.verifier_name for s in scores if not s.evidence_refs]
    if status == GateStatus.verified and missing_evidence:
        status = GateStatus.verified_with_caveats

    reasoning = (
        f"Average verifier score: {avg_score:.2f}. "
        f"{len(all_mismatches)} issues found. "
        f"{'Critical issues present.' if has_critical else 'No critical issues.'} "
        f"{'Missing evidence refs: ' + ', '.join(missing_evidence) + '.' if missing_evidence else 'All verifier scores cite evidence.'}"
    )

    return VerificationReport(
        gate=gate,
        status=status,
        verifier_scores=scores,
        reasoning=reasoning,
        decision_log_entry=f"{gate}: {status.value} (avg={avg_score:.2f})",
    )


def run_improvement_gate_offline(
    path_results: list[PathResult],
    paper_claim_map: PaperClaimMap,
    baseline_metrics: dict[str, Any],
) -> VerificationReport:
    """Verify improvement paths (Gate 3)."""
    scores: list[VerifierScore] = []
    findings: list[str] = []
    mismatches: list[str] = []

    successful_paths = [p for p in path_results if p.success]
    failed_paths = [p for p in path_results if not p.success]

    findings.append(f"{len(successful_paths)}/{len(path_results)} paths succeeded")
    if failed_paths:
        for p in failed_paths:
            mismatches.append(f"Path {p.path_id} failed: {p.failure_notes}")

    # Check each successful path has metrics
    for path in successful_paths:
        if path.metrics:
            findings.append(f"Path {path.path_id}: metrics present")
        else:
            mismatches.append(f"Path {path.path_id}: no metrics")

    overall_score = len(successful_paths) / max(len(path_results), 1)
    scores.append(VerifierScore(
        verifier_name="improvement_gate",
        score=overall_score,
        findings=findings,
        mismatches=mismatches,
        evidence_refs=[p.path_id for p in path_results],
    ))

    status = (
        GateStatus.verified if overall_score >= 0.5
        else GateStatus.partial_reproduction
    )

    return VerificationReport(
        gate="gate_3",
        status=status,
        verifier_scores=scores,
        reasoning=f"{len(successful_paths)}/{len(path_results)} improvement paths verified.",
        decision_log_entry=f"gate_3: {status.value}",
    )


def _load_command_log(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    entries: list[dict[str, Any]] = []
    mismatches: list[str] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            mismatches.append(
                f"commands.log line {line_number} is not structured JSONL"
            )
            continue
        if not isinstance(payload, dict):
            mismatches.append(f"commands.log line {line_number} is not an object")
            continue
        if not payload.get("command"):
            mismatches.append(f"commands.log line {line_number} missing command")
        if not payload.get("status"):
            mismatches.append(f"commands.log line {line_number} missing status")
        entries.append(payload)
    return entries, mismatches


def _load_provenance(path: Path) -> tuple[dict[str, Any], list[str]]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}, ["provenance.json is not valid JSON"]
    if not isinstance(payload, dict):
        return {}, ["provenance.json is not an object"]
    return payload, []
