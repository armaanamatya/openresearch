"""Offline, fail-closed A/B evidence gate for scheduler authority.

This module is deliberately stdlib-only and reads completed run artifacts.  It
does **not** import the campaign loop or mutate a run.  Its output is evidence
for an operator review, never permission to change a feature-flag default.

The input is a JSON manifest (schema version 1)::

    {
      "schema_version": 1,
      "pairs": [{
        "pair_id": "sdar-seed-1",
        "shadow": "/abs/runs/sdar-shadow-1",
        "authoritative": "/abs/runs/sdar-authority-1",
        "calibrations": {
          "shadow": "grader_calibration.json",
          "authoritative": "grader_calibration.json"
        },
        "verified_gpu_costs": {
          "shadow": {
            "gpu_usd": 12.30,
            "source": "provider_bill_export",
            "tokens_total_path": "tokens_total.json",
            "gpu_evidence_path": "billing_export.json"
          },
          "authoritative": {"...": "..."}
        }
      }]
    }

Relative run paths resolve against the manifest directory.  Calibration and
cost-proof paths resolve against the arm's run directory.  A calibration is
either a direct record produced by ``scripts/calibrate_grader.py`` or its
append-only ledger (the latter requires ``{"path": ..., "run_id": ...}``
when the run-directory name is not the record id).

The gate is intentionally stricter than the shadow reader:

* at least three distinct, terminal pairs with the same paper/rubric/terminal
  deterministic evidence;
* full ASHA-advisory coverage in both arms and a persisted
  ``decision["asha_authority_audit"]`` on every authoritative decision;
* calibration K >= 5 and sample grader sigma <= 0.02 in each arm, with an
  authoritative score degradation no larger than that pair's sigma;
* only literal ``training_diverged`` true-kills;
* at least one *applied* non-continue authority action and a positive GPU-$
  saving whose cost records are bound to the arm's run id, token record, and a
  source-specific node-observation or provider-export record.  A
  ``cost_ledger.jsonl`` claim is never proof.

An authority run that conservatively records ``applied: false`` is healthy but
cannot pass this adoption gate: it did not exercise authoritative control.

There is deliberately no accepted ``applied: true`` record in the present
runtime.  The campaign does not yet persist a provenance-validated defining
metric at a paper-pinned optimizer-step/checkpoint lineage.  The gate rejects
locally-authored audit prose as a substitute for that receipt.  This lets it
produce the paired-run and grader-sigma evidence now while making an authority
default flip impossible until the missing deterministic evidence producer is
implemented.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from backend.agents.rlm.asha_shadow_report import analyze_shadow_rows


SCHEMA_VERSION = 1
MIN_PAIRS = 3
MIN_CALIBRATION_K = 5
MAX_GRADER_SIGMA = 0.02
_TERMINAL_KINDS = frozenset({"REPRODUCED", "CONTRADICTED", "INFEASIBLE", "EXHAUSTED"})
_ACTIONS = frozenset({"promote", "freeze", "kill", "continue"})
_GPU_COST_SOURCES = frozenset({"node_observation", "provider_bill_export"})
_BREAKAGE_CLASS = "training_diverged"
_DECISION_KEYS = frozenset(
    {"kind", "rule", "stop_reason", "next_plan", "champion_attempt_n"}
)


@dataclass(frozen=True)
class GateFailure:
    """One immutable reason the manifest cannot support an authority flip."""

    code: str
    detail: str
    pair_id: str | None = None


@dataclass(frozen=True)
class PairEvidence:
    """Auditable per-pair summary; ``None`` fields identify failed extraction."""

    pair_id: str
    shadow_run: str | None
    authoritative_run: str | None
    terminal_kind: str | None
    shadow_score: float | None
    authoritative_score: float | None
    pair_sigma: float | None
    score_delta: float | None
    shadow_advisory_coverage: float | None
    authoritative_advisory_coverage: float | None
    applied_action_count: int
    verified_gpu_saving_usd: float | None


@dataclass(frozen=True)
class AuthorityGateReport:
    """Pure result.  ``eligible`` still requires explicit operator sign-off."""

    eligible_for_operator_review: bool
    pair_count: int
    applied_action_count: int
    total_verified_gpu_savings_usd: float | None
    pairs: tuple[PairEvidence, ...]
    failures: tuple[GateFailure, ...]

    @property
    def failure_codes(self) -> tuple[str, ...]:
        return tuple(sorted({failure.code for failure in self.failures}))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["failure_codes"] = list(self.failure_codes)
        return payload


@dataclass(frozen=True)
class _ArmArtifacts:
    run_dir: Path
    state: Mapping[str, Any]
    rows: tuple[Mapping[str, Any], ...]
    terminal_kind: str
    paper_ref: str
    rubric_sha256: str
    score: float
    advisory_coverage: float
    advisory_actions: tuple[tuple[str, str], ...]
    audits: tuple[Mapping[str, Any], ...]
    calibration_sigma: float


def evaluate_manifest(
    manifest: Mapping[str, Any], *, manifest_dir: Path | str = "."
) -> AuthorityGateReport:
    """Evaluate an already-decoded manifest without changing files or flags.

    Malformed/missing evidence returns a rejected report rather than treating
    absent values as zero.  This makes the result convenient for automation
    while preserving fail-closed semantics.
    """
    failures: list[GateFailure] = []
    pair_reports: list[PairEvidence] = []
    manifest_dir = Path(manifest_dir)

    if not isinstance(manifest, Mapping):
        return _invalid_manifest_report("manifest must be a JSON object")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        return _invalid_manifest_report(
            f"schema_version must equal {SCHEMA_VERSION}, got {manifest.get('schema_version')!r}"
        )
    pairs = manifest.get("pairs")
    if not isinstance(pairs, list):
        return _invalid_manifest_report("pairs must be a JSON array")
    if len(pairs) < MIN_PAIRS:
        failures.append(
            GateFailure(
                "minimum_pair_count",
                f"need at least {MIN_PAIRS} complete pairs, found {len(pairs)}",
            )
        )

    seen_pair_ids: set[str] = set()
    seen_runs: set[Path] = set()
    total_actions = 0
    total_saving = 0.0
    all_costs_verified = True

    for index, raw_pair in enumerate(pairs):
        pair_id = _pair_id(raw_pair, index, failures)
        if pair_id in seen_pair_ids:
            failures.append(GateFailure("duplicate_pair_id", "pair_id must be unique", pair_id))
        seen_pair_ids.add(pair_id)
        report, pair_failures, applied_actions, saving, run_paths = _evaluate_pair(
            raw_pair, pair_id=pair_id, manifest_dir=manifest_dir
        )
        pair_reports.append(report)
        failures.extend(pair_failures)
        total_actions += applied_actions
        if saving is None:
            all_costs_verified = False
        else:
            total_saving += saving
        for run_path in run_paths:
            if run_path in seen_runs:
                failures.append(
                    GateFailure(
                        "reused_run",
                        "each run directory may appear in exactly one arm of one pair",
                        pair_id,
                    )
                )
            seen_runs.add(run_path)

    if total_actions < 1:
        failures.append(
            GateFailure(
                "meaningful_authority_action",
                "need at least one audit with applied=true and a non-continue action",
            )
        )
    if not all_costs_verified or total_saving <= 0.0:
        failures.append(
            GateFailure(
                "positive_verified_gpu_saving",
                "verified authoritative GPU spend must be lower than shadow spend in aggregate",
            )
        )

    return AuthorityGateReport(
        eligible_for_operator_review=not failures,
        pair_count=len(pairs),
        applied_action_count=total_actions,
        total_verified_gpu_savings_usd=total_saving if all_costs_verified else None,
        pairs=tuple(pair_reports),
        failures=tuple(failures),
    )


def evaluate_manifest_path(path: Path | str) -> AuthorityGateReport:
    """Load and evaluate a JSON manifest.  Invalid JSON is a hard input error."""
    manifest_path = Path(path)
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read manifest: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"manifest is not valid JSON: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("manifest must be a JSON object")
    return evaluate_manifest(raw, manifest_dir=manifest_path.parent)


def _invalid_manifest_report(detail: str) -> AuthorityGateReport:
    return AuthorityGateReport(
        eligible_for_operator_review=False,
        pair_count=0,
        applied_action_count=0,
        total_verified_gpu_savings_usd=None,
        pairs=(),
        failures=(GateFailure("malformed_manifest", detail),),
    )


def _pair_id(raw_pair: Any, index: int, failures: list[GateFailure]) -> str:
    if not isinstance(raw_pair, Mapping):
        fallback = f"<pair-{index + 1}>"
        failures.append(GateFailure("malformed_pair", "pair must be an object", fallback))
        return fallback
    pair_id = raw_pair.get("pair_id")
    if not isinstance(pair_id, str) or not pair_id.strip():
        fallback = f"<pair-{index + 1}>"
        failures.append(GateFailure("malformed_pair", "pair_id must be a non-empty string", fallback))
        return fallback
    return pair_id


def _evaluate_pair(
    raw_pair: Any, *, pair_id: str, manifest_dir: Path
) -> tuple[PairEvidence, list[GateFailure], int, float | None, tuple[Path, ...]]:
    failures: list[GateFailure] = []
    empty = _empty_pair(pair_id)
    if not isinstance(raw_pair, Mapping):
        return empty, failures, 0, None, ()

    shadow_path = _run_path(raw_pair, "shadow", manifest_dir, pair_id, failures)
    authority_path = _run_path(raw_pair, "authoritative", manifest_dir, pair_id, failures)
    run_paths = tuple(path for path in (shadow_path, authority_path) if path is not None)
    if shadow_path is None or authority_path is None:
        return empty, failures, 0, None, run_paths
    if shadow_path == authority_path:
        failures.append(GateFailure("arm_path_mismatch", "shadow and authoritative runs must differ", pair_id))
        return _pair_with_paths(empty, shadow_path, authority_path), failures, 0, None, run_paths

    shadow = _load_arm(shadow_path, pair_id=pair_id, arm="shadow", calibration_spec=_arm_spec(raw_pair, "calibrations", "shadow"), require_audit=False, failures=failures)
    authority = _load_arm(authority_path, pair_id=pair_id, arm="authoritative", calibration_spec=_arm_spec(raw_pair, "calibrations", "authoritative"), require_audit=True, failures=failures)
    if shadow is None or authority is None:
        return _pair_with_paths(empty, shadow_path, authority_path), failures, 0, None, run_paths

    _validate_pair_match(shadow, authority, pair_id, failures)
    applied_actions = _validate_authority_audits(authority, pair_id, failures)
    pair_sigma = max(shadow.calibration_sigma, authority.calibration_sigma)
    score_delta = authority.score - shadow.score
    if score_delta < -pair_sigma - 1e-12:
        failures.append(
            GateFailure(
                "score_degradation",
                f"authoritative score delta {score_delta:.6f} is below -pair sigma {-pair_sigma:.6f}",
                pair_id,
            )
        )

    saving = _read_verified_saving(raw_pair, shadow.run_dir, authority.run_dir, pair_id, failures)
    return (
        PairEvidence(
            pair_id=pair_id,
            shadow_run=str(shadow.run_dir),
            authoritative_run=str(authority.run_dir),
            terminal_kind=shadow.terminal_kind if shadow.terminal_kind == authority.terminal_kind else None,
            shadow_score=shadow.score,
            authoritative_score=authority.score,
            pair_sigma=pair_sigma,
            score_delta=score_delta,
            shadow_advisory_coverage=shadow.advisory_coverage,
            authoritative_advisory_coverage=authority.advisory_coverage,
            applied_action_count=applied_actions,
            verified_gpu_saving_usd=saving,
        ),
        failures,
        applied_actions,
        saving,
        run_paths,
    )


def _empty_pair(pair_id: str) -> PairEvidence:
    return PairEvidence(
        pair_id=pair_id,
        shadow_run=None,
        authoritative_run=None,
        terminal_kind=None,
        shadow_score=None,
        authoritative_score=None,
        pair_sigma=None,
        score_delta=None,
        shadow_advisory_coverage=None,
        authoritative_advisory_coverage=None,
        applied_action_count=0,
        verified_gpu_saving_usd=None,
    )


def _pair_with_paths(pair: PairEvidence, shadow: Path, authority: Path) -> PairEvidence:
    return PairEvidence(
        **{**asdict(pair), "shadow_run": str(shadow), "authoritative_run": str(authority)}
    )


def _run_path(
    pair: Mapping[str, Any], arm: str, manifest_dir: Path, pair_id: str, failures: list[GateFailure]
) -> Path | None:
    value = pair.get(arm)
    if not isinstance(value, str) or not value.strip():
        failures.append(GateFailure("malformed_pair", f"{arm} must be a non-empty path string", pair_id))
        return None
    path = Path(value)
    if not path.is_absolute():
        path = manifest_dir / path
    path = path.resolve()
    if not path.is_dir():
        failures.append(GateFailure("missing_run_directory", f"{arm} run directory does not exist: {path}", pair_id))
        return None
    return path


def _arm_spec(pair: Mapping[str, Any], container: str, arm: str) -> Any:
    """Read a per-arm field; singular aliases keep hand-written manifests usable."""
    value = pair.get(container)
    if value is None and container == "calibrations":
        value = pair.get("calibration")
    if not isinstance(value, Mapping):
        return None
    return value.get(arm)


def _load_arm(
    run_dir: Path,
    *,
    pair_id: str,
    arm: str,
    calibration_spec: Any,
    require_audit: bool,
    failures: list[GateFailure],
) -> _ArmArtifacts | None:
    state = _read_json_mapping(run_dir / "campaign" / "campaign.json", "campaign_state", pair_id, failures)
    rows = _read_jsonl(run_dir / "campaign" / "attempts.jsonl", pair_id, failures)
    report = _read_json_mapping(run_dir / "final_report.json", "final_report", pair_id, failures)
    if state is None or rows is None or report is None:
        return None

    terminal = state.get("terminal")
    kind = terminal.get("kind") if isinstance(terminal, Mapping) else None
    paper_ref = state.get("paper_ref")
    rubric_sha256 = state.get("rubric_sha256")
    if state.get("state") != "terminal" or not isinstance(kind, str) or kind not in _TERMINAL_KINDS:
        failures.append(GateFailure("nonterminal_or_invalid_evidence", f"{arm} campaign must have a recognized terminal kind", pair_id))
        return None
    if not isinstance(paper_ref, str) or not paper_ref.strip() or not isinstance(rubric_sha256, str) or not rubric_sha256.strip():
        failures.append(GateFailure("campaign_identity_missing", f"{arm} campaign must persist paper_ref and rubric_sha256", pair_id))
        return None
    score = _extract_score(report)
    if score is None:
        failures.append(GateFailure("missing_final_score", f"{arm} final_report lacks finite rubric.overall_score", pair_id))
        return None

    decided = [row for row in rows if row.get("status") == "decided"]
    if not decided:
        failures.append(GateFailure("missing_decisions", f"{arm} run has no decided rows", pair_id))
        return None
    advisory_actions, audits = _validate_decisions(
        decided, rows, pair_id=pair_id, arm=arm, require_audit=require_audit, failures=failures
    )
    # Use the shared shadow-report reducer as a second, independently-simple
    # coverage/accounting view.  Gate validation above remains strict because the
    # reader intentionally tolerates malformed live data for operator diagnostics.
    shadow_report = analyze_shadow_rows(rows)
    coverage = shadow_report.with_advisory / shadow_report.total_decided if shadow_report.total_decided else 0.0
    if coverage != 1.0:
        code = "shadow_advisory_coverage" if arm == "shadow" else "authoritative_advisory_coverage"
        failures.append(GateFailure(code, f"{arm} requires advisory on every decided row ({shadow_report.with_advisory}/{shadow_report.total_decided})", pair_id))

    _validate_advisory_kills(advisory_actions, rows, pair_id, arm, failures)
    sigma = _read_calibration(run_dir, calibration_spec, pair_id, arm, failures)
    if sigma is None:
        return None
    return _ArmArtifacts(
        run_dir=run_dir,
        state=state,
        rows=tuple(rows),
        terminal_kind=kind,
        paper_ref=paper_ref,
        rubric_sha256=rubric_sha256,
        score=score,
        advisory_coverage=coverage,
        advisory_actions=tuple(advisory_actions),
        audits=tuple(audits),
        calibration_sigma=sigma,
    )


def _read_json_mapping(path: Path, label: str, pair_id: str, failures: list[GateFailure]) -> Mapping[str, Any] | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(GateFailure(f"missing_or_malformed_{label}", f"{path}: {exc}", pair_id))
        return None
    if not isinstance(raw, Mapping):
        failures.append(GateFailure(f"missing_or_malformed_{label}", f"{path} must contain an object", pair_id))
        return None
    return raw


def _read_jsonl(path: Path, pair_id: str, failures: list[GateFailure]) -> list[Mapping[str, Any]] | None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        failures.append(GateFailure("missing_campaign_ledger", f"{path}: {exc}", pair_id))
        return None
    rows: list[Mapping[str, Any]] = []
    for line_no, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            failures.append(GateFailure("malformed_campaign_ledger", f"{path}:{line_no}: {exc.msg}", pair_id))
            return None
        if not isinstance(row, Mapping):
            failures.append(GateFailure("malformed_campaign_ledger", f"{path}:{line_no} must be an object", pair_id))
            return None
        rows.append(row)
    return rows


def _extract_score(report: Mapping[str, Any]) -> float | None:
    rubric = report.get("rubric")
    value = rubric.get("overall_score") if isinstance(rubric, Mapping) else None
    return _finite_number(value)


def _validate_decisions(
    decided: Sequence[Mapping[str, Any]],
    all_rows: Sequence[Mapping[str, Any]],
    *,
    pair_id: str,
    arm: str,
    require_audit: bool,
    failures: list[GateFailure],
) -> tuple[list[tuple[str, str]], list[Mapping[str, Any]]]:
    actions: list[tuple[str, str]] = []
    audits: list[Mapping[str, Any]] = []
    for row in decided:
        decision = row.get("decision")
        if not isinstance(decision, Mapping):
            failures.append(GateFailure("malformed_decision", f"{arm} decided row lacks decision object", pair_id))
            continue
        _validate_base_decision(decision, pair_id, arm, failures)

        advisory = decision.get("asha_advisory")
        row_actions: list[tuple[str, str]] = []
        if isinstance(advisory, Mapping):
            decisions = advisory.get("decisions")
            if not isinstance(decisions, list):
                failures.append(GateFailure("malformed_advisory", f"{arm} advisory decisions must be an array", pair_id))
            else:
                for branch in decisions:
                    if not isinstance(branch, Mapping):
                        failures.append(GateFailure("malformed_advisory", f"{arm} advisory branch must be an object", pair_id))
                        continue
                    branch_id = branch.get("branch_id")
                    action = branch.get("action")
                    reason = branch.get("reason")
                    if not isinstance(branch_id, str) or not branch_id or action not in _ACTIONS or not isinstance(reason, str) or not reason:
                        failures.append(GateFailure("malformed_advisory", f"{arm} advisory branch has invalid id/action/reason", pair_id))
                        continue
                    row_actions.append((branch_id, action))
                    actions.append((branch_id, action))

        # Audit presence is independent of the shadow advisory shape: an
        # incomplete advisory must not accidentally exempt an authority arm
        # from its durable audit obligation.
        audit = decision.get("asha_authority_audit")
        if require_audit:
            if not isinstance(audit, Mapping):
                failures.append(GateFailure("authority_audit_missing", "every authoritative decided row must persist asha_authority_audit", pair_id))
            else:
                if audit.get("applied") is True:
                    _validate_applied_audit_link(audit, row_actions, all_rows, pair_id, failures)
                if _validate_audit_shape(audit, pair_id, failures):
                    audits.append(audit)
        elif audit is not None:
            failures.append(GateFailure("shadow_arm_contaminated", "shadow arm must not carry authority audit markers", pair_id))
    return actions, audits


def _validate_base_decision(
    decision: Mapping[str, Any], pair_id: str, arm: str, failures: list[GateFailure]
) -> None:
    """Make the gate reject an audit pasted onto an incomplete/no-op decision."""
    missing = _DECISION_KEYS.difference(decision)
    if missing:
        failures.append(
            GateFailure(
                "decision_contract_missing",
                f"{arm} decision is missing required Decision.to_dict keys: {sorted(missing)!r}",
                pair_id,
            )
        )
        return
    if decision.get("kind") == "CONTINUE":
        next_plan = decision.get("next_plan")
        rung = next_plan.get("scope_rung") if isinstance(next_plan, Mapping) else None
        if type(rung) is not int or rung < 0:
            failures.append(
                GateFailure(
                    "decision_contract_missing",
                    f"{arm} CONTINUE decision requires non-negative next_plan.scope_rung",
                    pair_id,
                )
            )


def _validate_applied_audit_link(
    audit: Mapping[str, Any],
    row_actions: Sequence[tuple[str, str]],
    all_rows: Sequence[Mapping[str, Any]],
    pair_id: str,
    failures: list[GateFailure],
) -> None:
    """Require the eventual action receipt to point at a real proposed branch.

    The current campaign never reaches this as an accepted action, but checking
    the linkage now prevents a hand-written audit from looking like a real
    promote/freeze/kill once a receipt producer is introduced.
    """
    branch_id = audit.get("source_branch_id")
    action = audit.get("action")
    attempt_n = audit.get("source_attempt_n")
    if not isinstance(branch_id, str) or not branch_id or type(attempt_n) is not int:
        failures.append(
            GateFailure(
                "unlinked_authority_audit",
                "applied audit requires source_branch_id and integer source_attempt_n",
                pair_id,
            )
        )
        return
    if (branch_id, action) not in row_actions:
        failures.append(
            GateFailure(
                "unlinked_authority_audit",
                "applied audit must exactly match an advisory action in the same decided row",
                pair_id,
            )
        )
    assessment = next(
        (
            row.get("assessment")
            for row in all_rows
            if row.get("status") == "assessed" and row.get("attempt_n") == attempt_n
        ),
        None,
    )
    if not isinstance(assessment, Mapping):
        failures.append(
            GateFailure(
                "unlinked_authority_audit",
                "applied audit source_attempt_n must have a durable assessed row",
                pair_id,
            )
        )
    if action == "kill" and (
        not isinstance(assessment, Mapping)
        or assessment.get("failure_class") != _BREAKAGE_CLASS
    ):
        failures.append(
            GateFailure(
                "invalid_true_kill",
                "applied kill must link to an assessed training_diverged receipt",
                pair_id,
            )
        )


def _validate_audit_shape(audit: Mapping[str, Any], pair_id: str, failures: list[GateFailure]) -> bool:
    enabled = audit.get("enabled")
    applied = audit.get("applied")
    action = audit.get("action")
    basis = audit.get("deterministic_evidence_basis")
    source_branch_id = audit.get("source_branch_id")
    if enabled is not True or type(applied) is not bool or action not in _ACTIONS:
        failures.append(GateFailure("malformed_authority_audit", "audit requires enabled=true, boolean applied, and a valid action", pair_id))
        return False
    if not isinstance(basis, str) or not basis.strip():
        failures.append(GateFailure("malformed_authority_audit", "audit requires a deterministic_evidence_basis string", pair_id))
        return False
    if source_branch_id is not None and (not isinstance(source_branch_id, str) or not source_branch_id.strip()):
        failures.append(GateFailure("malformed_authority_audit", "source_branch_id must be a non-empty string when supplied", pair_id))
        return False
    if applied and action == "continue":
        failures.append(GateFailure("malformed_authority_audit", "applied audit cannot have continue action", pair_id))
        return False
    if not applied and action != "continue":
        failures.append(GateFailure("malformed_authority_audit", "unapplied audit must record continue action", pair_id))
        return False
    if action == "kill" and audit.get("failure_class") != _BREAKAGE_CLASS:
        failures.append(GateFailure("invalid_true_kill", "kill audit requires literal failure_class='training_diverged'", pair_id))
        return False
    if applied:
        # A free-text label is not evidence.  In particular, accepting a
        # manifest-authored ``deterministic_evidence_basis`` here would make an
        # authority action forgeable without the missing optimizer-step,
        # checkpoint, and provenance receipts.  Keep this refusal local to the
        # gate rather than silently treating a grade-derived shadow decision as
        # live control.  The future receipt validator must replace this branch
        # atomically with the campaign-side durable producer.
        failures.append(
            GateFailure(
                "authoritative_evidence_receipt_unavailable",
                "applied authority requires a provenance-validated deterministic "
                "metric plus paper-step/checkpoint lineage; the current runtime "
                "does not persist that receipt",
                pair_id,
            )
        )
        return False
    return True


def _validate_advisory_kills(
    actions: Sequence[tuple[str, str]], rows: Sequence[Mapping[str, Any]], pair_id: str, arm: str, failures: list[GateFailure]
) -> None:
    failure_by_attempt: dict[str, Any] = {}
    for row in rows:
        if row.get("status") != "assessed":
            continue
        assessment = row.get("assessment")
        attempt_n = row.get("attempt_n")
        if isinstance(assessment, Mapping) and attempt_n is not None:
            failure_by_attempt[str(attempt_n)] = assessment.get("failure_class")
    for branch_id, action in actions:
        if action == "kill" and failure_by_attempt.get(branch_id) != _BREAKAGE_CLASS:
            failures.append(GateFailure("invalid_true_kill", f"{arm} advisory killed branch {branch_id!r} without literal training_diverged evidence", pair_id))


def _validate_pair_match(shadow: _ArmArtifacts, authority: _ArmArtifacts, pair_id: str, failures: list[GateFailure]) -> None:
    if shadow.paper_ref != authority.paper_ref or shadow.rubric_sha256 != authority.rubric_sha256:
        failures.append(GateFailure("pair_identity_mismatch", "paired arms must share paper_ref and rubric_sha256", pair_id))
    if shadow.terminal_kind != authority.terminal_kind:
        failures.append(GateFailure("terminal_evidence_mismatch", "paired arms must have the same deterministic terminal kind", pair_id))


def _validate_authority_audits(authority: _ArmArtifacts, pair_id: str, failures: list[GateFailure]) -> int:
    applied = 0
    for audit in authority.audits:
        if audit.get("applied") is True:
            action = audit.get("action")
            if action in _ACTIONS and action != "continue":
                applied += 1
    return applied


def _read_calibration(run_dir: Path, spec: Any, pair_id: str, arm: str, failures: list[GateFailure]) -> float | None:
    path_value: Any = spec
    record_id: str | None = None
    if isinstance(spec, Mapping):
        path_value = spec.get("path")
        candidate_id = spec.get("run_id")
        if candidate_id is not None:
            record_id = candidate_id if isinstance(candidate_id, str) and candidate_id else None
            if record_id is None:
                failures.append(GateFailure("malformed_grader_calibration", f"{arm} calibration run_id must be a non-empty string", pair_id))
                return None
    if not isinstance(path_value, str) or not path_value.strip():
        failures.append(GateFailure("missing_grader_calibration", f"{arm} requires a calibration path", pair_id))
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = run_dir / path
    raw = _read_json_mapping(path, "grader_calibration", pair_id, failures)
    if raw is None:
        return None
    record = _calibration_record(raw, run_dir, record_id, pair_id, arm, failures)
    if record is None:
        return None
    k = record.get("k")
    overall = record.get("overall")
    if type(k) is not int or k < MIN_CALIBRATION_K or not isinstance(overall, Mapping):
        failures.append(GateFailure("grader_calibration_k", f"{arm} needs calibration k >= {MIN_CALIBRATION_K}", pair_id))
        return None
    n = overall.get("n")
    scores = overall.get("scores")
    stored_sigma = _finite_number(overall.get("stdev"))
    if type(n) is not int or n != k or n < MIN_CALIBRATION_K or not isinstance(scores, list) or len(scores) != n or stored_sigma is None:
        failures.append(GateFailure("malformed_grader_calibration", f"{arm} calibration needs matching k/n/raw scores/stdev", pair_id))
        return None
    numeric_scores = [_finite_number(score) for score in scores]
    if any(score is None for score in numeric_scores):
        failures.append(GateFailure("malformed_grader_calibration", f"{arm} calibration scores must be finite numbers", pair_id))
        return None
    values = [float(score) for score in numeric_scores if score is not None]
    computed_sigma = statistics.stdev(values)
    if not math.isclose(stored_sigma, computed_sigma, rel_tol=0.0, abs_tol=1e-12):
        failures.append(GateFailure("malformed_grader_calibration", f"{arm} persisted grader sigma does not match raw scores", pair_id))
        return None
    for key, expected in (("mean", statistics.fmean(values)), ("min", min(values)), ("max", max(values))):
        stored = _finite_number(overall.get(key))
        if stored is None or not math.isclose(stored, expected, rel_tol=0.0, abs_tol=1e-12):
            failures.append(GateFailure("malformed_grader_calibration", f"{arm} persisted {key} does not match raw scores", pair_id))
            return None
    if computed_sigma > MAX_GRADER_SIGMA + 1e-12:
        failures.append(GateFailure("grader_sigma", f"{arm} grader sigma {computed_sigma:.6f} exceeds {MAX_GRADER_SIGMA:.2f}", pair_id))
    return computed_sigma


def _calibration_record(raw: Mapping[str, Any], run_dir: Path, record_id: str | None, pair_id: str, arm: str, failures: list[GateFailure]) -> Mapping[str, Any] | None:
    records = raw.get("records")
    if records is None:
        # A standalone summary is valid only when its producer bound it to the
        # arm it calibrates.  Otherwise a manifest can reuse arbitrary raw
        # scores from a different paper/run to satisfy the sigma gate.
        if raw.get("run_id") != run_dir.name:
            failures.append(
                GateFailure(
                    "unbound_grader_calibration",
                    f"{arm} direct calibration must persist run_id={run_dir.name!r}",
                    pair_id,
                )
            )
            return None
        return raw
    if not isinstance(records, list):
        failures.append(GateFailure("malformed_grader_calibration", f"{arm} calibration ledger records must be an array", pair_id))
        return None
    expected_id = record_id or run_dir.name
    matching = [record for record in records if isinstance(record, Mapping) and record.get("run_id") == expected_id]
    if len(matching) != 1:
        failures.append(GateFailure("missing_grader_calibration", f"{arm} calibration ledger needs exactly one record for {expected_id!r}", pair_id))
        return None
    return matching[0]


def _read_verified_saving(pair: Mapping[str, Any], shadow_dir: Path, authority_dir: Path, pair_id: str, failures: list[GateFailure]) -> float | None:
    costs = pair.get("verified_gpu_costs")
    if costs is None:
        costs = pair.get("verified_gpu_cost")
    if not isinstance(costs, Mapping):
        failures.append(GateFailure("verified_gpu_cost", "pair requires verified_gpu_costs for both arms", pair_id))
        return None
    shadow = _read_verified_cost(costs.get("shadow"), shadow_dir, pair_id, "shadow", failures)
    authority = _read_verified_cost(costs.get("authoritative"), authority_dir, pair_id, "authoritative", failures)
    if shadow is None or authority is None:
        return None
    return shadow - authority


def _read_verified_cost(raw: Any, run_dir: Path, pair_id: str, arm: str, failures: list[GateFailure]) -> float | None:
    if not isinstance(raw, Mapping):
        failures.append(GateFailure("verified_gpu_cost", f"{arm} verified GPU cost must be an object", pair_id))
        return None
    cost = _finite_number(raw.get("gpu_usd"))
    source = raw.get("source")
    if (
        cost is None
        or cost < 0.0
        or source not in _GPU_COST_SOURCES
        or raw.get("run_id") != run_dir.name
    ):
        failures.append(GateFailure("verified_gpu_cost", f"{arm} cost needs finite non-negative gpu_usd and supported non-ledger source", pair_id))
        return None
    token_path = _evidence_path(raw.get("tokens_total_path"), run_dir)
    gpu_path = _evidence_path(raw.get("gpu_evidence_path"), run_dir)
    if token_path is None or gpu_path is None or not _nonempty_file(token_path) or not _nonempty_file(gpu_path):
        failures.append(GateFailure("verified_gpu_cost", f"{arm} cost needs existing tokens_total and GPU-proof artifacts", pair_id))
        return None
    if "cost_ledger" in gpu_path.name.lower() or "cost_ledger" in token_path.name.lower():
        failures.append(GateFailure("verified_gpu_cost", f"{arm} cost ledger is not verified GPU cost evidence", pair_id))
        return None
    try:
        token_payload = json.loads(token_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        failures.append(GateFailure("verified_gpu_cost", f"{arm} tokens_total artifact must be parseable JSON", pair_id))
        return None
    if not isinstance(token_payload, Mapping):
        failures.append(GateFailure("verified_gpu_cost", f"{arm} tokens_total artifact must be an object", pair_id))
        return None
    if token_payload.get("run_id") != run_dir.name:
        failures.append(
            GateFailure(
                "verified_gpu_cost",
                f"{arm} tokens_total must bind run_id={run_dir.name!r}",
                pair_id,
            )
        )
        return None
    gpu_payload = _read_json_mapping(gpu_path, "gpu_cost_evidence", pair_id, failures)
    if gpu_payload is None:
        return None
    if gpu_payload.get("schema_version") != 1 or gpu_payload.get("source") != source:
        failures.append(
            GateFailure(
                "verified_gpu_cost",
                f"{arm} GPU proof must declare schema_version=1 and source={source!r}",
                pair_id,
            )
        )
        return None
    records = gpu_payload.get("records")
    record_id = raw.get("evidence_record_id")
    if not isinstance(records, list) or not isinstance(record_id, str) or not record_id:
        failures.append(
            GateFailure(
                "verified_gpu_cost",
                f"{arm} cost must name one source-specific evidence record",
                pair_id,
            )
        )
        return None
    matching = [
        record
        for record in records
        if isinstance(record, Mapping)
        and record.get("record_id") == record_id
        and record.get("run_id") == run_dir.name
    ]
    if len(matching) != 1:
        failures.append(
            GateFailure(
                "verified_gpu_cost",
                f"{arm} GPU proof needs exactly one record for this arm and evidence_record_id",
                pair_id,
            )
        )
        return None
    evidence_cost = _source_record_cost(matching[0], source)
    if evidence_cost is None or not math.isclose(cost, evidence_cost, rel_tol=0.0, abs_tol=1e-9):
        failures.append(
            GateFailure(
                "verified_gpu_cost",
                f"{arm} manifest gpu_usd must exactly match its source-specific evidence record",
                pair_id,
            )
        )
        return None
    return cost


def _source_record_cost(record: Mapping[str, Any], source: str) -> float | None:
    """Validate the minimum arithmetic/identity contract for one cost source.

    This verifies only a durable, source-shaped exported observation.  It does
    not treat the local campaign ledger as a provider bill, and deliberately
    requires the future cloud collector to write a record per run/arm.
    """
    if source == "provider_bill_export":
        provider_id = record.get("provider_record_id")
        sku = record.get("gpu_sku")
        cost = _finite_number(record.get("cost_usd"))
        if not isinstance(provider_id, str) or not provider_id or not isinstance(sku, str) or not sku:
            return None
        return cost if cost is not None and cost >= 0.0 else None
    if source == "node_observation":
        node_uid = record.get("node_uid")
        sku = record.get("gpu_sku")
        seconds = _finite_number(record.get("gpu_seconds"))
        rate = _finite_number(record.get("rate_usd_per_gpu_hour"))
        if (
            not isinstance(node_uid, str)
            or not node_uid
            or not isinstance(sku, str)
            or not sku
            or seconds is None
            or rate is None
            or seconds < 0.0
            or rate < 0.0
        ):
            return None
        return seconds / 3600.0 * rate
    return None


def _evidence_path(value: Any, run_dir: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    return path if path.is_absolute() else run_dir / path


def _nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def render_report(report: AuthorityGateReport) -> str:
    """Render a compact human review summary; JSON remains the canonical output."""
    verdict = "ELIGIBLE FOR OPERATOR REVIEW" if report.eligible_for_operator_review else "REJECTED"
    lines = [
        f"ASHA authority A/B evidence gate: {verdict}",
        f"pairs: {report.pair_count}  applied actions: {report.applied_action_count}  "
        f"verified GPU saving: {report.total_verified_gpu_savings_usd if report.total_verified_gpu_savings_usd is not None else 'unverified'}",
    ]
    for pair in report.pairs:
        lines.append(
            f"  {pair.pair_id}: terminal={pair.terminal_kind or '?'} "
            f"score_delta={pair.score_delta if pair.score_delta is not None else '?'} "
            f"sigma={pair.pair_sigma if pair.pair_sigma is not None else '?'} "
            f"saving={pair.verified_gpu_saving_usd if pair.verified_gpu_saving_usd is not None else '?'}"
        )
    if report.failures:
        lines.append("failures:")
        lines.extend(
            f"  - {failure.code}{f' [{failure.pair_id}]' if failure.pair_id else ''}: {failure.detail}"
            for failure in report.failures
        )
    else:
        lines.append("Operator sign-off is still required; this tool never changes a default.")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail-closed offline A/B evidence gate for ASHA scheduler authority."
    )
    parser.add_argument("manifest", type=Path, help="JSON manifest of shadow/authoritative run pairs")
    parser.add_argument("--json", action="store_true", help="emit machine-readable gate result")
    args = parser.parse_args(argv)
    try:
        report = evaluate_manifest_path(args.manifest)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(render_report(report))
    return 0 if report.eligible_for_operator_review else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
