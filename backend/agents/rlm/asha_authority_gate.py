"""Offline, fail-closed A/B evidence gate for scheduler authority.

This module is deliberately stdlib-only and reads completed run artifacts.  It
does **not** import the campaign loop or mutate a run.  Its output is evidence
for an operator review, never permission to change a feature-flag default.

The input is a JSON manifest (schema version 1)::

    {
      "schema_version": 1,
      "controller_event_store": "/abs/controller/events.db",
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
* every applied action must bind a paper-step/checkpoint/metric receipt to the
  exact audit SHA **and** to a completed BranchPromoted/FrozenPoolEviction/
  BranchTrueKilled event in the controller-owned EventStore outside the run.

An authority run that conservatively records ``applied: false`` is healthy but
cannot pass this adoption gate: it did not exercise authoritative control.

The current campaign runtime does not yet produce the complete receipt plus
controller-event transition, so it correctly records ``applied:false``.  The
gate nevertheless validates the full contract now, rather than accepting
locally-authored audit prose or mutable run-local evidence as a substitute.
This makes an authority default flip impossible until the controller producer
is implemented and the paired evidence exists.

Trust boundary: ``controller_event_store`` is an operator-supplied, trusted
manifest field and must be ACL/IAM-isolated from both worker run directories.
The gate verifies that its path is outside each arm, but filesystem ownership
and cloud IAM are deployment controls that must keep workers from writing it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import statistics
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from backend.agents.rlm.asha_shadow_report import analyze_shadow_rows
from backend.agents.rlm.asha_scheduler import BranchObservation, RungConfig, asha_decide
from backend.agents.rlm.scheduler_evidence import PaperStepLadder, load_verified_receipt


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
_DECISION_EVIDENCE_KEYS = frozenset({
    "schema_version", "campaign_id", "ladder_sha256", "metric_id", "direction",
    "rung", "config", "cohort", "selected_branch_id", "action",
})
_DECISION_CONFIG_KEYS = frozenset({"gpu_usd_budget", "a100_cap", "eta", "noise_floor"})
_DECISION_COHORT_KEYS = frozenset({
    "branch_id", "receipt_path", "receipt_sha256", "branch_type", "is_safety_bracket", "gpu_usd",
})


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
    seen_campaign_ids: set[str] = set()
    seen_action_proofs: set[str] = set()
    # A rung receipt is a one-time authority input.  Hashing an altered audit
    # must never make the same metric/checkpoint evidence eligible for a second
    # promotion/freeze/kill elsewhere in the manifest.
    seen_receipt_hashes: set[str] = set()
    controller_event_store = _controller_event_store_path(manifest, manifest_dir, failures)
    total_actions = 0
    total_saving = 0.0
    all_costs_verified = True

    for index, raw_pair in enumerate(pairs):
        pair_id = _pair_id(raw_pair, index, failures)
        if pair_id in seen_pair_ids:
            failures.append(GateFailure("duplicate_pair_id", "pair_id must be unique", pair_id))
        seen_pair_ids.add(pair_id)
        report, pair_failures, applied_actions, saving, run_paths = _evaluate_pair(
            raw_pair,
            pair_id=pair_id,
            manifest_dir=manifest_dir,
            controller_event_store=controller_event_store,
            seen_action_proofs=seen_action_proofs,
            seen_receipt_hashes=seen_receipt_hashes,
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
            campaign_id = _campaign_identity_from_run(run_path)
            if campaign_id is not None:
                if campaign_id in seen_campaign_ids:
                    failures.append(
                        GateFailure(
                            "reused_campaign_identity",
                            "each A/B arm must have a distinct controller campaign identity",
                            pair_id,
                        )
                    )
                seen_campaign_ids.add(campaign_id)

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
    raw_pair: Any,
    *,
    pair_id: str,
    manifest_dir: Path,
    controller_event_store: Path | None,
    seen_action_proofs: set[str],
    seen_receipt_hashes: set[str],
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

    if controller_event_store is not None and any(
        _path_is_within(controller_event_store, run_path)
        for run_path in (shadow_path, authority_path)
    ):
        failures.append(
            GateFailure(
                "controller_action_event_invalid",
                "controller_event_store must live outside both worker-owned pair arms",
                pair_id,
            )
        )
        controller_event_store = None

    shadow = _load_arm(shadow_path, pair_id=pair_id, arm="shadow", calibration_spec=_arm_spec(raw_pair, "calibrations", "shadow"), require_audit=False, failures=failures)
    authority = _load_arm(
        authority_path,
        pair_id=pair_id,
        arm="authoritative",
        calibration_spec=_arm_spec(raw_pair, "calibrations", "authoritative"),
        require_audit=True,
        controller_event_store=controller_event_store,
        seen_action_proofs=seen_action_proofs,
        seen_receipt_hashes=seen_receipt_hashes,
        failures=failures,
    )
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


def _controller_event_store_path(
    manifest: Mapping[str, Any], manifest_dir: Path, failures: list[GateFailure]
) -> Path | None:
    """Resolve the controller-owned SQLite event store without creating it.

    This path is intentionally a manifest-level operator input, rather than a
    run-local artifact: workers can write a run directory but must not have
    write access to the controller's event-store database.  We defer a missing
    store failure until an arm actually claims ``applied:true`` so shadow-only
    analyses remain usable.
    """
    value = manifest.get("controller_event_store")
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        failures.append(GateFailure("controller_event_store_invalid", "controller_event_store must be a non-empty path", None))
        return None
    path = Path(value)
    if not path.is_absolute():
        path = manifest_dir / path
    try:
        if path.is_symlink() or not path.is_file():
            raise OSError("not a regular file")
        return path.resolve(strict=True)
    except OSError as exc:
        failures.append(GateFailure("controller_event_store_invalid", f"cannot read controller event store: {exc}", None))
        return None


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _campaign_identity_from_run(run_dir: Path) -> str | None:
    try:
        state = json.loads((run_dir / "campaign" / "campaign.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = state.get("project_id") if isinstance(state, Mapping) else None
    return value if isinstance(value, str) and value.strip() else None


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
    controller_event_store: Path | None = None,
    seen_action_proofs: set[str] | None = None,
    seen_receipt_hashes: set[str] | None = None,
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
    campaign_id = state.get("project_id")
    if not isinstance(campaign_id, str) or not campaign_id.strip():
        failures.append(GateFailure("campaign_identity_missing", f"{arm} campaign must persist project_id", pair_id))
        return None
    advisory_actions, audits = _validate_decisions(
        decided,
        rows,
        run_dir=run_dir,
        campaign_id=campaign_id,
        controller_event_store=controller_event_store,
        seen_action_proofs=seen_action_proofs if seen_action_proofs is not None else set(),
        seen_receipt_hashes=seen_receipt_hashes if seen_receipt_hashes is not None else set(),
        pair_id=pair_id,
        arm=arm,
        require_audit=require_audit,
        failures=failures,
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
    run_dir: Path,
    campaign_id: str,
    controller_event_store: Path | None,
    seen_action_proofs: set[str],
    seen_receipt_hashes: set[str],
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
                valid_link = True
                if audit.get("applied") is True:
                    valid_link = _validate_applied_audit_link(
                        audit,
                        row_actions,
                        all_rows,
                        decision,
                        run_dir,
                        campaign_id,
                        controller_event_store,
                        seen_action_proofs,
                        seen_receipt_hashes,
                        pair_id,
                        failures,
                    )
                valid_shape = _validate_audit_shape(audit, pair_id, failures)
                if valid_link and valid_shape:
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
    decision: Mapping[str, Any],
    run_dir: Path,
    campaign_id: str,
    controller_event_store: Path | None,
    seen_action_proofs: set[str],
    seen_receipt_hashes: set[str],
    pair_id: str,
    failures: list[GateFailure],
) -> bool:
    """Require the eventual action receipt to point at a real proposed branch.

    The current campaign never reaches this as an accepted action, but checking
    the linkage now prevents a hand-written audit from looking like a real
    promote/freeze/kill once a receipt producer is introduced.
    """
    branch_id = audit.get("source_branch_id")
    action = audit.get("action")
    attempt_n = audit.get("source_attempt_n")
    valid = True
    source_valid = isinstance(branch_id, str) and bool(branch_id) and type(attempt_n) is int
    if not source_valid:
        failures.append(
            GateFailure(
                "unlinked_authority_audit",
                "applied audit requires source_branch_id and integer source_attempt_n",
                pair_id,
            )
        )
        valid = False
    if source_valid and (branch_id, action) not in row_actions:
        failures.append(
            GateFailure(
                "unlinked_authority_audit",
                "applied audit must exactly match an advisory action in the same decided row",
                pair_id,
            )
        )
        valid = False
    assessment: Any = None
    if source_valid:
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
            valid = False
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
        valid = False
    receipt_bundle = _load_audit_receipt(
        audit, run_dir, campaign_id, pair_id, failures
    )
    if receipt_bundle is None:
        return False
    receipt, ladder = receipt_bundle
    receipt_sha256 = str(audit.get("scheduler_receipt_sha256") or "")
    if receipt_sha256 in seen_receipt_hashes:
        failures.append(
            GateFailure(
                "reused_authority_receipt",
                "a verified scheduler receipt may authorize only one action across the manifest",
                pair_id,
            )
        )
        valid = False
    if source_valid and (receipt.branch_id != branch_id or receipt.attempt_n != attempt_n):
        failures.append(
            GateFailure(
                "unlinked_authority_receipt",
                "applied audit receipt must bind the same source branch and attempt",
                pair_id,
            )
        )
        valid = False
    if action == "freeze" and not receipt.checkpoint_path:
        failures.append(
            GateFailure(
                "unlinked_authority_receipt",
                "freeze action requires a verified resumable checkpoint receipt",
                pair_id,
            )
        )
        valid = False
    if action == "kill" and receipt.termination_cause != _BREAKAGE_CLASS:
        failures.append(
            GateFailure(
                "invalid_true_kill",
                "applied kill receipt requires literal termination_cause='training_diverged'",
                pair_id,
            )
        )
        valid = False
    decision_evidence_sha256 = _load_decision_evidence(
        audit,
        run_dir=run_dir,
        campaign_id=campaign_id,
        ladder=ladder,
        source_receipt=receipt,
        action=str(action),
        all_rows=all_rows,
        decision=decision,
        pair_id=pair_id,
        failures=failures,
    )
    if decision_evidence_sha256 is None:
        valid = False
    if not _controller_event_proves_action(
        controller_event_store,
        run_dir=run_dir,
        campaign_id=campaign_id,
        receipt=receipt,
        audit=audit,
        decision_evidence_sha256=decision_evidence_sha256,
        action=str(action),
        decision=decision,
        pair_id=pair_id,
        failures=failures,
    ):
        valid = False
    proof_id = _authority_proof_id(audit)
    if valid and proof_id in seen_action_proofs:
        failures.append(
            GateFailure(
                "reused_authority_action_evidence",
                "an applied receipt/audit proof may appear in only one decision across the manifest",
                pair_id,
            )
        )
        valid = False
    if valid:
        seen_action_proofs.add(proof_id)
        seen_receipt_hashes.add(receipt_sha256)
    return valid


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
    return True


def _load_audit_receipt(
    audit: Mapping[str, Any],
    run_dir: Path,
    campaign_id: str,
    pair_id: str,
    failures: list[GateFailure],
) -> Any | None:
    """Load the precise receipt an applied audit claims, fail closed.

    The A/B manifest is intentionally not allowed to nominate this evidence.
    It has to live in the authoritative arm's scheduler-receipt directory and
    be attested by that arm's campaign ledger, exactly like the runtime input.
    """
    receipt_path = audit.get("scheduler_receipt_path")
    ladder_path = audit.get("scheduler_ladder_path")
    receipt_sha256 = audit.get("scheduler_receipt_sha256")
    if (
        not isinstance(receipt_path, str)
        or not receipt_path.strip()
        or not isinstance(ladder_path, str)
        or not ladder_path.strip()
        or not isinstance(receipt_sha256, str)
        or len(receipt_sha256) != 64
    ):
        failures.append(
            GateFailure(
                "authoritative_evidence_receipt_missing",
                "applied audit requires scheduler receipt path, ladder path, and SHA-256",
                pair_id,
            )
        )
        return None
    ladder_file = _safe_arm_file(ladder_path, run_dir)
    if ladder_file is None:
        failures.append(
            GateFailure(
                "authoritative_evidence_receipt_missing",
                "scheduler_ladder_path must be a non-symlink relative file inside the run",
                pair_id,
            )
        )
        return None
    try:
        raw_ladder = json.loads(ladder_file.read_text(encoding="utf-8"))
        if not isinstance(raw_ladder, Mapping):
            raise ValueError("ladder must be an object")
        ladder = PaperStepLadder.from_mapping(raw_ladder)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        failures.append(GateFailure("authoritative_evidence_receipt_missing", f"invalid scheduler ladder: {exc}", pair_id))
        return None
    raw_receipt = _safe_arm_file(receipt_path, run_dir)
    if raw_receipt is None:
        failures.append(
            GateFailure(
                "authoritative_evidence_receipt_missing",
                "scheduler_receipt_path must be a non-symlink relative file inside the run",
                pair_id,
            )
        )
        return None
    try:
        actual_sha256 = hashlib.sha256(raw_receipt.read_bytes()).hexdigest()
    except OSError as exc:
        failures.append(GateFailure("authoritative_evidence_receipt_missing", f"cannot read receipt: {exc}", pair_id))
        return None
    if actual_sha256 != receipt_sha256:
        failures.append(
            GateFailure("authoritative_evidence_receipt_mismatch", "audit receipt SHA-256 does not match the receipt file", pair_id)
        )
        return None
    receipt = load_verified_receipt(
        raw_receipt,
        ladder=ladder,
        run_dir=run_dir,
        expected_campaign_id=campaign_id,
    )
    if receipt is None:
        failures.append(
            GateFailure(
                "authoritative_evidence_receipt_invalid",
                "applied audit receipt fails paper-step, checkpoint, metric, bundle, or ledger verification",
                pair_id,
            )
        )
        return None
    return receipt, ladder


def _load_decision_evidence(
    audit: Mapping[str, Any],
    *,
    run_dir: Path,
    campaign_id: str,
    ladder: PaperStepLadder,
    source_receipt: Any,
    action: str,
    all_rows: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
    pair_id: str,
    failures: list[GateFailure],
) -> str | None:
    """Verify the controller's canonical, grade-free ASHA decision input.

    A valid rung artifact alone cannot establish *why* a branch was selected.
    The decision evidence therefore lists every same-rung verified receipt,
    explicit ASHA width/noise inputs, a deterministic branch-id order, and the
    claimed action.  We recompute the pure scheduler output here; no report,
    grade, curve predictor, or prose field is consulted.
    """
    path_value = audit.get("scheduler_decision_evidence_path")
    expected_sha = audit.get("scheduler_decision_evidence_sha256")
    if (
        not isinstance(path_value, str)
        or not path_value.strip()
        or not isinstance(expected_sha, str)
        or len(expected_sha) != 64
    ):
        failures.append(GateFailure(
            "authoritative_decision_evidence_missing",
            "applied audit requires canonical scheduler decision evidence path and SHA-256",
            pair_id,
        ))
        return None
    path = _safe_arm_file(path_value, run_dir)
    if path is None:
        failures.append(GateFailure(
            "authoritative_decision_evidence_missing",
            "scheduler_decision_evidence_path must be a non-symlink relative file inside the run",
            pair_id,
        ))
        return None
    try:
        content = path.read_bytes()
    except OSError as exc:
        failures.append(GateFailure("authoritative_decision_evidence_missing", f"cannot read decision evidence: {exc}", pair_id))
        return None
    actual_sha = hashlib.sha256(content).hexdigest()
    if actual_sha != expected_sha:
        failures.append(GateFailure(
            "authoritative_decision_evidence_mismatch",
            "audit decision-evidence SHA-256 does not match its file",
            pair_id,
        ))
        return None
    try:
        raw = json.loads(content)
    except json.JSONDecodeError as exc:
        failures.append(GateFailure("authoritative_decision_evidence_invalid", f"invalid decision evidence JSON: {exc}", pair_id))
        return None
    if not isinstance(raw, Mapping):
        failures.append(GateFailure("authoritative_decision_evidence_invalid", "decision evidence must be an object", pair_id))
        return None
    # The canonical object is deliberately closed-world: a grade, curve
    # prediction, or arbitrary policy hint cannot silently become a scheduler
    # input.  The pure recomputation below reads only verified receipts plus the
    # two independent meter inputs.
    if frozenset(raw) != _DECISION_EVIDENCE_KEYS:
        failures.append(GateFailure(
            "authoritative_decision_evidence_invalid",
            "decision evidence has missing or non-deterministic fields",
            pair_id,
        ))
        return None
    if (
        raw.get("schema_version") != 1
        or raw.get("campaign_id") != campaign_id
        or raw.get("ladder_sha256") != ladder.sha256
        or raw.get("metric_id") != ladder.metric_id
        or raw.get("direction") != ladder.direction
        or raw.get("selected_branch_id") != source_receipt.branch_id
        or raw.get("action") != action
    ):
        failures.append(GateFailure(
            "authoritative_decision_evidence_invalid",
            "decision evidence campaign, ladder, metric, branch, or action binding is invalid",
            pair_id,
        ))
        return None
    try:
        rung = raw["rung"]
        if type(rung) is not int or rung != ladder.rung_steps.index(source_receipt.to_step):
            raise ValueError("rung does not match source receipt step")
        config_raw = raw["config"]
        cohort_raw = raw["cohort"]
        if not isinstance(config_raw, Mapping) or not isinstance(cohort_raw, list) or not cohort_raw:
            raise ValueError("config/cohort missing")
        if frozenset(config_raw) != _DECISION_CONFIG_KEYS:
            raise ValueError("config has missing or non-deterministic fields")
        _verify_decision_config_against_advisory(
            decision=decision,
            rung=rung,
            config=config_raw,
        )
        raw_gpu_budget = config_raw.get("gpu_usd_budget")
        gpu_budget = None if raw_gpu_budget is None else _finite_nonnegative(raw_gpu_budget)
        if raw_gpu_budget is not None and gpu_budget is None:
            raise ValueError("gpu_usd_budget invalid")
        a100_cap = config_raw.get("a100_cap")
        if a100_cap is not None and (type(a100_cap) is not int or a100_cap <= 0):
            raise ValueError("a100_cap invalid")
        eta = _finite_positive(config_raw.get("eta", 3.0))
        noise_floor = _finite_nonnegative(config_raw.get("noise_floor", 0.0))
        if eta is None or noise_floor is None:
            raise ValueError("eta/noise_floor invalid")
        observations: list[BranchObservation] = []
        branches: list[str] = []
        cohort_receipts: set[tuple[str, str]] = set()
        source_seen = 0
        for member in cohort_raw:
            if not isinstance(member, Mapping):
                raise ValueError("cohort member must be an object")
            if frozenset(member) != _DECISION_COHORT_KEYS:
                raise ValueError("cohort member has missing or non-deterministic fields")
            branch_id = member.get("branch_id")
            receipt_path = member.get("receipt_path")
            receipt_sha = member.get("receipt_sha256")
            branch_type = member.get("branch_type", "faithful")
            safety = member.get("is_safety_bracket", False)
            gpu_usd = _finite_nonnegative(member.get("gpu_usd", 0.0))
            if (
                not isinstance(branch_id, str) or not branch_id
                or not isinstance(receipt_path, str) or not receipt_path
                or not isinstance(receipt_sha, str) or len(receipt_sha) != 64
                or branch_type not in {"faithful", "ambiguity", "discovery"}
                or type(safety) is not bool or gpu_usd is None
            ):
                raise ValueError("cohort member fields invalid")
            member_path = _safe_arm_file(receipt_path, run_dir)
            if member_path is None or hashlib.sha256(member_path.read_bytes()).hexdigest() != receipt_sha:
                raise ValueError("cohort receipt path/SHA invalid")
            member_receipt = load_verified_receipt(
                member_path,
                ladder=ladder,
                run_dir=run_dir,
                expected_campaign_id=campaign_id,
            )
            if member_receipt is None or member_receipt.branch_id != branch_id:
                raise ValueError("cohort receipt binding invalid")
            if member_receipt.to_step != source_receipt.to_step:
                raise ValueError("cohort crosses fidelity rungs")
            _verify_member_against_campaign_records(
                member_receipt=member_receipt,
                branch_type=branch_type,
                is_safety_bracket=safety,
                gpu_usd=gpu_usd,
                all_rows=all_rows,
            )
            if receipt_sha == audit.get("scheduler_receipt_sha256"):
                if member_receipt != source_receipt:
                    raise ValueError("source receipt content mismatch")
                source_seen += 1
            branches.append(branch_id)
            cohort_receipts.add((branch_id, receipt_sha))
            observations.append(BranchObservation(
                branch_id=branch_id,
                branch_type=branch_type,
                score=member_receipt.metric_value,
                gpu_usd=gpu_usd,
                broken=member_receipt.termination_cause == _BREAKAGE_CLASS,
                is_safety_bracket=safety,
            ))
        if branches != sorted(branches) or len(set(branches)) != len(branches) or source_seen != 1:
            raise ValueError("cohort must be unique/sorted and include source receipt once")
        available_receipts = _verified_same_rung_receipts(
            run_dir=run_dir,
            campaign_id=campaign_id,
            ladder=ladder,
            to_step=source_receipt.to_step,
        )
        if cohort_receipts != available_receipts:
            raise ValueError("cohort does not exactly inventory all verified same-rung receipts")
        decisions = asha_decide(
            observations,
            RungConfig(
                rung=rung,
                gpu_usd_budget=gpu_budget,
                a100_cap=a100_cap,
                eta=eta,
                higher_is_better=ladder.direction == "maximize",
                noise_floor=noise_floor,
            ),
        )
        selected = next((d for d in decisions if d.branch_id == source_receipt.branch_id), None)
        if selected is None or selected.action != action:
            raise ValueError("pure ASHA result does not select claimed action")
    except (KeyError, OSError, TypeError, ValueError) as exc:
        failures.append(GateFailure(
            "authoritative_decision_evidence_invalid",
            f"decision evidence is not a verified deterministic ASHA input: {exc}",
            pair_id,
        ))
        return None
    return actual_sha


def _verify_decision_config_against_advisory(
    *,
    decision: Mapping[str, Any],
    rung: int,
    config: Mapping[str, Any],
) -> None:
    """Bind width inputs to the durable advisory in the decided ledger row.

    The decision artifact may make only the pure scheduler inputs explicit; it
    cannot invent a cheaper budget or a larger A100 cap than the controller
    recorded when it made that decision.
    """
    advisory = decision.get("asha_advisory")
    if not isinstance(advisory, Mapping) or advisory.get("rung") != rung:
        raise ValueError("decision artifact rung is not bound to its durable ASHA advisory")
    width = advisory.get("width_meter")
    if not isinstance(width, Mapping):
        raise ValueError("decision artifact lacks durable ASHA width meter")
    for key in ("gpu_usd_budget", "a100_cap", "eta", "noise_floor"):
        if config.get(key) != width.get(key):
            raise ValueError(f"decision artifact {key} differs from durable ASHA width meter")


def _campaign_attempt_records(
    rows: Sequence[Mapping[str, Any]],
    attempt_n: int,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Return exactly one assessed and launched row for an authority receipt."""
    assessments = [
        row.get("assessment") for row in rows
        if row.get("status") == "assessed" and row.get("attempt_n") == attempt_n
        and isinstance(row.get("assessment"), Mapping)
    ]
    launched = [
        row for row in rows
        if row.get("status") == "launched" and row.get("attempt_n") == attempt_n
    ]
    if len(assessments) != 1 or len(launched) != 1:
        raise ValueError("every authority cohort receipt requires exactly one assessed and launched row")
    return assessments[0], launched[0]


def _row_scheduler_metadata(record: Mapping[str, Any]) -> tuple[str, bool]:
    """Read durable scheduler metadata with the legacy faithful/non-safety default."""
    branch_type = record.get("branch_type", "faithful")
    safety = record.get("is_safety_bracket", False)
    if branch_type not in {"faithful", "ambiguity", "discovery"} or type(safety) is not bool:
        raise ValueError("invalid durable branch metadata")
    if safety and branch_type != "faithful":
        raise ValueError("only faithful durable branches may carry a safety bracket")
    return str(branch_type), safety


def _verified_assessment_gpu_usd(assessment: Mapping[str, Any]) -> float:
    cost = assessment.get("cost")
    if not isinstance(cost, Mapping):
        raise ValueError("authority cohort assessment lacks deterministic cost")
    gpu_usd = _finite_nonnegative(cost.get("gpu_usd"))
    if gpu_usd is None:
        raise ValueError("authority cohort assessment gpu_usd is invalid")
    return gpu_usd


def _verify_member_against_campaign_records(
    *,
    member_receipt: Any,
    branch_type: str,
    is_safety_bracket: bool,
    gpu_usd: float,
    all_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Prevent a decision JSON from rewriting branch or width metadata."""
    # The live adapter defines a branch id as the durable attempt id.  Preserve
    # that invariant until a controller-owned branch queue supplies a different
    # immutable mapping; do not let the worker invent one in decision evidence.
    if member_receipt.branch_id != str(member_receipt.attempt_n):
        raise ValueError("receipt branch_id is not the durable attempt branch id")
    assessment, launched = _campaign_attempt_records(all_rows, member_receipt.attempt_n)
    assessed_metadata = _row_scheduler_metadata(assessment)
    launched_metadata = _row_scheduler_metadata(launched)
    if assessed_metadata != launched_metadata:
        raise ValueError("assessment and launch branch metadata disagree")
    if assessed_metadata != (branch_type, is_safety_bracket):
        raise ValueError("cohort branch metadata differs from assessment/launch records")
    if gpu_usd != _verified_assessment_gpu_usd(assessment):
        raise ValueError("cohort gpu_usd differs from deterministic assessment cost")


def _verified_same_rung_receipts(
    *,
    run_dir: Path,
    campaign_id: str,
    ladder: PaperStepLadder,
    to_step: int,
) -> set[tuple[str, str]]:
    """Inventory every valid same-rung receipt under the campaign-owned root.

    The decision writer is not permitted to choose its own comparison set.  A
    malformed receipt in this authority namespace also fails closed: otherwise
    a worker could hide a stronger branch by corrupting its receipt.
    """
    root = run_dir / "campaign" / "scheduler_receipts"
    try:
        if not root.is_dir() or _has_symlink_component(root, run_dir):
            raise ValueError("scheduler receipt root is missing or symlinked")
        inventory: set[tuple[str, str]] = set()
        for path in sorted(root.rglob("*")):
            if _has_symlink_component(path, run_dir):
                raise ValueError("scheduler receipt path is invalid")
            if path.is_dir():
                continue
            if not path.is_file():
                raise ValueError("scheduler receipt path is invalid")
            content = path.read_bytes()
            receipt = load_verified_receipt(
                path,
                ladder=ladder,
                run_dir=run_dir,
                expected_campaign_id=campaign_id,
            )
            if receipt is None:
                raise ValueError("scheduler receipt is invalid")
            if receipt.to_step == to_step:
                key = (receipt.branch_id, hashlib.sha256(content).hexdigest())
                if key in inventory:
                    raise ValueError("duplicate verified receipt identity")
                inventory.add(key)
        if not inventory:
            raise ValueError("no verified receipts at decision rung")
        return inventory
    except (OSError, ValueError) as exc:
        raise ValueError(f"same-rung receipt inventory invalid: {exc}") from exc


def _finite_nonnegative(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    value = float(value)
    return value if value >= 0.0 else None


def _finite_positive(value: Any) -> float | None:
    parsed = _finite_nonnegative(value)
    return parsed if parsed is not None and parsed > 0.0 else None


def _safe_arm_file(value: str, run_dir: Path) -> Path | None:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        return None
    candidate = run_dir / path
    try:
        if _has_symlink_component(candidate, run_dir) or not candidate.is_file():
            return None
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(run_dir.resolve())
    except (OSError, ValueError):
        return None
    return candidate


def _has_symlink_component(path: Path, root: Path) -> bool:
    """Reject a symlink at *any* point below a trusted run root."""
    root = root.resolve()
    current = path
    while True:
        if current.is_symlink():
            return True
        if current == root:
            return False
        parent = current.parent
        try:
            parent.relative_to(root)
        except ValueError:
            return True
        if parent == current:
            return True
        current = parent


_CONTROLLER_EVENT_BY_ACTION = {
    "promote": "branch_promoted",
    "freeze": "frozen_pool_eviction",
    "kill": "branch_true_killed",
}
_CONTROLLER_EVENT_SOURCE = "agents.rlm.reproduction_campaign"


def _audit_sha256(audit: Mapping[str, Any]) -> str:
    """Hash the exact durable audit object, without accepting prose as evidence."""
    encoded = json.dumps(dict(audit), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _authority_proof_id(audit: Mapping[str, Any]) -> str:
    """Stable identity for one applied receipt/audit transition."""
    receipt_sha256 = audit.get("scheduler_receipt_sha256")
    return f"{receipt_sha256}:{_audit_sha256(audit)}"


def _controller_event_proves_action(
    store_path: Path | None,
    *,
    run_dir: Path,
    campaign_id: str,
    receipt: Any,
    audit: Mapping[str, Any],
    decision_evidence_sha256: str | None,
    action: str,
    decision: Mapping[str, Any],
    pair_id: str,
    failures: list[GateFailure],
) -> bool:
    """Require an independently stored branch-event for applied authority.

    The campaign ledger belongs to the run and is useful for resumability, but
    it is not the authority boundary.  A controller EventStore located outside
    the worker-owned run directory records the completed transition only after
    the controller applies it.  This reader opens SQLite read-only and never
    creates tables or writes a checkpoint.
    """
    expected_type = _CONTROLLER_EVENT_BY_ACTION.get(action)
    if store_path is None or expected_type is None or decision_evidence_sha256 is None:
        failures.append(
            GateFailure(
                "controller_action_event_missing",
                "applied authority requires a controller-owned branch-tree event store and known action",
                pair_id,
            )
        )
        return False
    try:
        store_path.relative_to(run_dir.resolve())
    except ValueError:
        pass
    else:
        failures.append(
            GateFailure(
                "controller_action_event_invalid",
                "controller_event_store must live outside the worker-owned run directory",
                pair_id,
            )
        )
        return False
    try:
        uri = f"file:{store_path.as_posix()}?mode=ro"
        with sqlite3.connect(uri, uri=True) as conn:
            rows = conn.execute(
                """
                SELECT aggregate_type, event_type, schema_version, payload_json, metadata_json
                FROM event_store_events
                WHERE aggregate_id = ? AND event_type = ?
                """,
                (f"branch-tree:{campaign_id}", expected_type),
            ).fetchall()
    except (sqlite3.Error, OSError) as exc:
        failures.append(
            GateFailure("controller_action_event_invalid", f"cannot read controller event store: {exc}", pair_id)
        )
        return False

    receipt_sha256 = audit.get("scheduler_receipt_sha256")
    audit_sha256 = _audit_sha256(audit)
    try:
        receipt_rung = None
        # The evidence loader already verified this membership; repeat the
        # simple rung derivation here so the controller event cannot promote a
        # receipt at one fidelity while the live decision advances another.
        ladder_path = _safe_arm_file(str(audit.get("scheduler_ladder_path") or ""), run_dir)
        if ladder_path is None:
            raise ValueError("missing ladder")
        ladder_raw = json.loads(ladder_path.read_text(encoding="utf-8"))
        ladder = PaperStepLadder.from_mapping(ladder_raw)
        receipt_rung = ladder.rung_steps.index(receipt.to_step)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        failures.append(GateFailure(
            "controller_action_event_invalid",
            f"cannot bind controller event to receipt rung: {exc}",
            pair_id,
        ))
        return False
    if action == "promote":
        next_plan = decision.get("next_plan")
        expected_scope = receipt_rung + 1
        if (
            decision.get("kind") != "CONTINUE"
            or not isinstance(next_plan, Mapping)
            or next_plan.get("scope_rung") != expected_scope
        ):
            failures.append(GateFailure(
                "controller_action_event_invalid",
                "promotion must bind receipt rung to CONTINUE next_plan.scope_rung",
                pair_id,
            ))
            return False
    for aggregate_type, event_type, schema_version, payload_raw, metadata_raw in rows:
        if aggregate_type != "branch_tree" or event_type != expected_type or schema_version != 1:
            continue
        try:
            payload = json.loads(payload_raw)
            metadata = json.loads(metadata_raw)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, Mapping) or not isinstance(metadata, Mapping):
            continue
        if metadata.get("source") != _CONTROLLER_EVENT_SOURCE:
            continue
        if (
            payload.get("branch_id") != receipt.branch_id
            or payload.get("receipt_sha256") != receipt_sha256
            or payload.get("authority_audit_sha256") != audit_sha256
            or payload.get("decision_evidence_sha256") != decision_evidence_sha256
        ):
            continue
        if action == "promote" and (
            payload.get("from_rung") != receipt_rung
            or payload.get("to_rung") != receipt_rung + 1
        ):
            continue
        if action in {"freeze", "kill"} and payload.get("rung") != receipt_rung:
            continue
        if action == "freeze" and payload.get("ckpt_uri") != receipt.checkpoint_path:
            continue
        if action == "kill" and payload.get("termination_cause") != _BREAKAGE_CLASS:
            continue
        return True

    failures.append(
        GateFailure(
            "controller_action_event_missing",
            "no controller branch-tree event binds this applied action to its receipt and audit hashes",
            pair_id,
        )
    )
    return False


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
