"""Durable, receipt-gated authority runtime for reproduction branch trees.

The shadow adapter is deliberately *not* used here.  This module owns the
smallest authoritative state machine that can safely turn a verified
``BranchRungReceipt`` into a queue/frozen-pool/true-kill transition:

* fidelity is a :class:`~scheduler_evidence.PaperStepLadder` in optimizer
  steps; it is never inferred from campaign scope or elapsed time;
* width is supplied independently as provider-verified per-branch GPU dollars
  plus a hard physical-GPU cap;
* a successful write-ahead action is durably recorded before the matching
  branch-lineage event is offered to the caller;
* only a receipt whose literal ``termination_cause`` is ``training_diverged``
  may become a true kill.  Every other non-promotion is a reversible freeze.

The runtime is intentionally stdlib-only and does no cloud or LLM I/O.  The
campaign controller owns submitting queued branches and the event-store
adapter owns appending the returned event facts.  A missing receipt, missing
provider cost, malformed state, or event delivery failure therefore cannot be
silently converted into a speculative launch.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from backend.agents.rlm.asha_scheduler import (
    BranchObservation,
    BranchType,
    RungConfig,
    asha_decide,
)
from backend.agents.rlm.scheduler_evidence import (
    BranchRungReceipt,
    PaperStepLadder,
    load_verified_receipt,
)


SCHEMA_VERSION = 1
TREE_STATE_NAME = "scheduler_tree_state.json"
TREE_WAL_NAME = "scheduler_tree_actions.jsonl"
_BRANCH_STATES = frozenset({"queued", "awaiting_receipt", "frozen", "killed", "completed"})
_TRUE_KILL_CAUSE = "training_diverged"

BranchState = Literal["queued", "awaiting_receipt", "frozen", "killed", "completed"]
EventSink = Callable[[str, Mapping[str, Any]], None]


class SchedulerRuntimeError(RuntimeError):
    """A malformed or unavailable authority input; callers must fail closed."""


@dataclass(frozen=True)
class SchedulerBranch:
    """One durable branch projection record.

    ``receipt_path`` is relative to the campaign run directory.  It is stored
    for audit/recovery only; every authority use re-verifies the receipt and
    its artifacts rather than trusting this projection field.
    """

    branch_id: str
    branch_type: BranchType
    hypothesis_fingerprint: str
    rung: int
    state: BranchState = "queued"
    parent_branch_id: str | None = None
    receipt_path: str | None = None
    receipt_sha256: str | None = None
    checkpoint_path: str | None = None
    termination_cause: str | None = None
    attempt_n: int | None = None

    def __post_init__(self) -> None:
        if not self.branch_id or not self.hypothesis_fingerprint:
            raise ValueError("branch_id and hypothesis_fingerprint are required")
        if self.branch_type not in {"faithful", "ambiguity", "discovery"}:
            raise ValueError("unknown branch_type")
        if type(self.rung) is not int or self.rung < 0:
            raise ValueError("rung must be a non-negative integer")
        if self.state not in _BRANCH_STATES:
            raise ValueError("unknown branch state")
        if self.attempt_n is not None and (type(self.attempt_n) is not int or self.attempt_n < 1):
            raise ValueError("attempt_n must be a positive integer when present")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "SchedulerBranch":
        return cls(
            branch_id=_string(raw.get("branch_id"), "branch_id"),
            branch_type=_branch_type(raw.get("branch_type", "faithful")),
            hypothesis_fingerprint=_string(raw.get("hypothesis_fingerprint"), "hypothesis_fingerprint"),
            rung=_nonnegative_int(raw.get("rung"), "rung"),
            state=_branch_state(raw.get("state", "queued")),
            parent_branch_id=_optional_string(raw.get("parent_branch_id"), "parent_branch_id"),
            receipt_path=_optional_string(raw.get("receipt_path"), "receipt_path"),
            receipt_sha256=_optional_sha(raw.get("receipt_sha256"), "receipt_sha256"),
            checkpoint_path=_optional_string(raw.get("checkpoint_path"), "checkpoint_path"),
            termination_cause=_optional_string(raw.get("termination_cause"), "termination_cause"),
            attempt_n=_optional_positive_int(raw.get("attempt_n"), "attempt_n"),
        )


@dataclass(frozen=True)
class AuthorityAction:
    """One completed, deterministic branch transition."""

    branch_id: str
    action: Literal["promote", "freeze", "kill", "complete"]
    reason: str
    rung: int
    receipt_sha256: str
    decision_evidence_sha256: str
    authority_audit_sha256: str
    checkpoint_path: str | None = None
    to_rung: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AuthorityResult:
    """Stable outcome of applying one same-rung cohort."""

    actions: tuple[AuthorityAction, ...]
    decision_evidence: Mapping[str, Any]
    decision_evidence_sha256: str
    authority_audit_sha256: str

    @property
    def selected_branch_id(self) -> str | None:
        for action in self.actions:
            if action.action in {"promote", "complete"}:
                return action.branch_id
        return None

    def authority_audit(self) -> dict[str, Any]:
        """Additive audit object suitable for a controller ``decided`` row."""
        return {
            "enabled": True,
            "applied": True,
            "action": "tree_transition",
            "selected_branch_id": self.selected_branch_id,
            "decision_evidence_sha256": self.decision_evidence_sha256,
            "authority_audit_sha256": self.authority_audit_sha256,
            "actions": [action.to_dict() for action in self.actions],
        }


@dataclass
class _Projection:
    campaign_id: str
    ladder_sha256: str
    branches: dict[str, SchedulerBranch] = field(default_factory=dict)
    used_receipts: set[str] = field(default_factory=set)
    applied_seq: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "campaign_id": self.campaign_id,
            "ladder_sha256": self.ladder_sha256,
            "branches": {key: asdict(value) for key, value in sorted(self.branches.items())},
            "used_receipts": sorted(self.used_receipts),
            "applied_seq": self.applied_seq,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "_Projection":
        if raw.get("schema_version") != SCHEMA_VERSION:
            raise SchedulerRuntimeError("unsupported scheduler tree schema")
        branches_raw = raw.get("branches")
        if not isinstance(branches_raw, Mapping):
            raise SchedulerRuntimeError("scheduler tree branches must be an object")
        branches = {
            _string(key, "branch id"): SchedulerBranch.from_mapping(value)
            for key, value in branches_raw.items()
            if isinstance(value, Mapping)
        }
        if len(branches) != len(branches_raw):
            raise SchedulerRuntimeError("scheduler tree branch record is malformed")
        used = raw.get("used_receipts", [])
        if not isinstance(used, list) or any(not _is_sha256(item) for item in used):
            raise SchedulerRuntimeError("scheduler tree receipt registry is malformed")
        projection = cls(
            campaign_id=_string(raw.get("campaign_id"), "campaign_id"),
            ladder_sha256=_sha(raw.get("ladder_sha256"), "ladder_sha256"),
            branches=branches,
            used_receipts=set(used),
            applied_seq=_nonnegative_int(raw.get("applied_seq", 0), "applied_seq"),
        )
        return projection


class SchedulerTreeRuntime:
    """Write-ahead durable branch-tree projection for one campaign.

    The caller may instantiate this only after the controller has resolved the
    paper-owned ladder.  ``event_sink`` receives facts *after* their matching
    WAL action is fsync'd.  An event error raises a ``SchedulerRuntimeError``
    after the mutation, so a controller can reconcile the durable WAL rather
    than pretending the transition never happened.
    """

    def __init__(
        self,
        run_dir: Path | str,
        *,
        campaign_id: str,
        ladder: PaperStepLadder,
        event_sink: EventSink | None = None,
    ) -> None:
        self.run_dir = Path(run_dir).resolve()
        self.campaign_dir = self.run_dir / "campaign"
        self.state_path = self.campaign_dir / TREE_STATE_NAME
        self.wal_path = self.campaign_dir / TREE_WAL_NAME
        self.campaign_id = _string(campaign_id, "campaign_id")
        self.ladder = ladder
        self.event_sink = event_sink
        self._projection = self._load_projection()

    @property
    def branches(self) -> Mapping[str, SchedulerBranch]:
        return dict(self._projection.branches)

    def register_branch(
        self,
        *,
        branch_id: str,
        branch_type: BranchType,
        hypothesis_fingerprint: str,
        parent_branch_id: str | None = None,
        rung: int = 0,
    ) -> bool:
        """Durably register one branch, returning ``False`` for F10 dedup.

        The caller supplies the *existing* campaign novelty fingerprint.  The
        runtime never derives a scheduler-specific substitute.
        """
        branch_id = _string(branch_id, "branch_id")
        hypothesis_fingerprint = _string(hypothesis_fingerprint, "hypothesis_fingerprint")
        if branch_id in self._projection.branches:
            existing = self._projection.branches[branch_id]
            if existing.hypothesis_fingerprint != hypothesis_fingerprint:
                raise SchedulerRuntimeError("branch id already belongs to another F10 fingerprint")
            return False
        for existing in self._projection.branches.values():
            if existing.hypothesis_fingerprint == hypothesis_fingerprint:
                self._commit(
                    "dedup_hit",
                    {"hypothesis_fingerprint": hypothesis_fingerprint, "existing_branch_id": existing.branch_id},
                    event=("dedup_hit", {
                        "hypothesis_fingerprint": hypothesis_fingerprint,
                        "existing_branch_id": existing.branch_id,
                    }),
                )
                return False
        if parent_branch_id is not None and parent_branch_id not in self._projection.branches:
            raise SchedulerRuntimeError("parent branch is unknown")
        branch = SchedulerBranch(
            branch_id=branch_id,
            branch_type=_branch_type(branch_type),
            hypothesis_fingerprint=hypothesis_fingerprint,
            parent_branch_id=parent_branch_id,
            rung=_nonnegative_int(rung, "rung"),
        )
        self._commit(
            "branch_registered",
            {"branch": asdict(branch)},
            event=("branch_spawned", {
                "branch_id": branch.branch_id,
                "branch_type": branch.branch_type,
                "parent_branch_id": branch.parent_branch_id,
                "rung": branch.rung,
                "hypothesis_fingerprint": branch.hypothesis_fingerprint,
            }),
        )
        return True

    def record_receipt(self, branch_id: str, receipt_path: Path | str) -> BranchRungReceipt:
        """Verify and durably attach a controller-attested rung receipt.

        The branch becomes ``awaiting_receipt`` only after all artifact and
        campaign-ledger checks pass.  This is deliberately a no-launch state:
        a receipt may be recorded long before a complete cohort is available.
        """
        branch = self._branch(branch_id)
        if branch.state not in {"queued", "awaiting_receipt"}:
            raise SchedulerRuntimeError("only queued branches may record a rung receipt")
        receipt = load_verified_receipt(
            receipt_path,
            ladder=self.ladder,
            run_dir=self.run_dir,
            expected_campaign_id=self.campaign_id,
        )
        if receipt is None:
            raise SchedulerRuntimeError("scheduler receipt failed deterministic verification")
        if receipt.branch_id != branch.branch_id or receipt.to_step != self.ladder.rung_steps[branch.rung]:
            raise SchedulerRuntimeError("receipt does not bind this branch and rung")
        if receipt.attempt_n < 1:
            raise SchedulerRuntimeError("receipt attempt number is invalid")
        receipt_abs = _safe_receipt_path(receipt_path, self.run_dir)
        receipt_sha256 = _sha256_bytes(receipt_abs.read_bytes())
        if receipt_sha256 in self._projection.used_receipts:
            raise SchedulerRuntimeError("receipt has already produced an authority transition")
        updated = SchedulerBranch(
            **{**asdict(branch), "state": "awaiting_receipt", "receipt_path": str(receipt_abs.relative_to(self.run_dir)),
               "receipt_sha256": receipt_sha256, "checkpoint_path": receipt.checkpoint_path,
               "termination_cause": receipt.termination_cause, "attempt_n": receipt.attempt_n}
        )
        self._commit(
            "receipt_recorded",
            {"branch": asdict(updated)},
            event=("rung_climbed", {
                "branch_id": updated.branch_id,
                "rung": updated.rung,
                "step_budget": receipt.to_step,
                "measured_score": receipt.metric_value,
                "gpu_usd": None,
                "resume_from_ckpt": receipt.checkpoint_path,
                "receipt_sha256": receipt_sha256,
            }),
        )
        return receipt

    def decide_ready_rung(
        self,
        *,
        rung: int,
        provider_gpu_usd_by_branch: Mapping[str, float],
        gpu_usd_budget: float,
        a100_cap: int,
        eta: float = 3.0,
        noise_floor: float = 0.0,
    ) -> AuthorityResult:
        """Apply deterministic ASHA to every verified receipt at ``rung``.

        Provider costs are intentionally required.  A campaign ledger's GPU
        column, LLM spend, wall-clock time, or an LLM grade cannot stand in
        for the width meter.
        """
        rung = _nonnegative_int(rung, "rung")
        if rung >= len(self.ladder.rung_steps):
            raise SchedulerRuntimeError("rung is outside the paper step ladder")
        if type(a100_cap) is not int or a100_cap < 1:
            raise SchedulerRuntimeError("a100_cap must be a positive integer")
        if not _finite_nonnegative(gpu_usd_budget):
            raise SchedulerRuntimeError("gpu_usd_budget must be finite and non-negative")
        if not _finite_nonnegative(noise_floor) or not isinstance(eta, (int, float)) or not math.isfinite(float(eta)) or eta <= 0:
            raise SchedulerRuntimeError("scheduler width configuration is malformed")

        ready = [branch for branch in self._projection.branches.values()
                 if branch.rung == rung and branch.state == "awaiting_receipt"]
        if not ready:
            raise SchedulerRuntimeError("no verified branch receipts are ready at this rung")
        receipts: dict[str, BranchRungReceipt] = {}
        observations: list[BranchObservation] = []
        for branch in sorted(ready, key=lambda item: item.branch_id):
            receipt = self._reload_branch_receipt(branch)
            usd = _provider_cost(provider_gpu_usd_by_branch, branch.branch_id)
            receipts[branch.branch_id] = receipt
            observations.append(BranchObservation(
                branch_id=branch.branch_id,
                branch_type=branch.branch_type,
                score=receipt.metric_value,
                gpu_usd=usd,
                broken=receipt.termination_cause == _TRUE_KILL_CAUSE,
                is_safety_bracket=False,
            ))

        config = RungConfig(
            rung=rung,
            gpu_usd_budget=float(gpu_usd_budget),
            a100_cap=a100_cap,
            eta=float(eta),
            higher_is_better=self.ladder.direction == "maximize",
            noise_floor=float(noise_floor),
        )
        decisions = asha_decide(observations, config)
        if len(decisions) != len(observations) or {item.branch_id for item in decisions} != {item.branch_id for item in observations}:
            raise SchedulerRuntimeError("ASHA produced an incomplete cohort decision")

        evidence = {
            "schema_version": SCHEMA_VERSION,
            "campaign_id": self.campaign_id,
            "ladder_sha256": self.ladder.sha256,
            "metric_id": self.ladder.metric_id,
            "direction": self.ladder.direction,
            "rung": rung,
            "config": {
                "gpu_usd_budget": float(gpu_usd_budget), "a100_cap": a100_cap,
                "eta": float(eta), "noise_floor": float(noise_floor),
            },
            "cohort": [
                {
                    "branch_id": branch.branch_id,
                    "receipt_path": branch.receipt_path,
                    "receipt_sha256": branch.receipt_sha256,
                    "branch_type": branch.branch_type,
                    "is_safety_bracket": False,
                    "gpu_usd": _provider_cost(provider_gpu_usd_by_branch, branch.branch_id),
                }
                for branch in sorted(ready, key=lambda item: item.branch_id)
            ],
            "selected_branch_id": None,
            "action": "tree_transition",
        }
        decision_evidence_sha256 = _canonical_sha256(evidence)
        actions: list[AuthorityAction] = []
        for decision in decisions:
            branch = self._branch(decision.branch_id)
            receipt = receipts[branch.branch_id]
            receipt_sha256 = branch.receipt_sha256
            if receipt_sha256 is None:
                raise SchedulerRuntimeError("ready branch lost receipt hash")
            action, updated, event = self._transition_for_decision(branch, receipt, decision.action, decision.reason)
            actions.append(AuthorityAction(
                branch_id=branch.branch_id,
                action=action,
                reason=decision.reason,
                rung=rung,
                receipt_sha256=receipt_sha256,
                decision_evidence_sha256=decision_evidence_sha256,
                authority_audit_sha256="",
                checkpoint_path=updated.checkpoint_path,
                to_rung=updated.rung if action == "promote" else None,
            ))

        selected = next((item.branch_id for item in actions if item.action in {"promote", "complete"}), None)
        evidence["selected_branch_id"] = selected
        # Selected branch is part of the authoritative evidence, so recompute
        # the hash before committing any transition.
        decision_evidence_sha256 = _canonical_sha256(evidence)
        audit_actions = [
            {**action.to_dict(), "decision_evidence_sha256": decision_evidence_sha256,
             "authority_audit_sha256": ""}
            for action in actions
        ]
        audit_seed = {
            "schema_version": SCHEMA_VERSION,
            "campaign_id": self.campaign_id,
            "ladder_sha256": self.ladder.sha256,
            "decision_evidence_sha256": decision_evidence_sha256,
            "actions": [
                {key: value for key, value in action.items() if key != "authority_audit_sha256"}
                for action in audit_actions
            ],
        }
        audit_sha256 = _canonical_sha256(audit_seed)
        finalized = tuple(
            AuthorityAction(**{**action.to_dict(), "decision_evidence_sha256": decision_evidence_sha256,
                               "authority_audit_sha256": audit_sha256})
            for action in actions
        )

        for action in finalized:
            updated = self._branch_transition_target(action)
            event_kind, event_payload = self._event_for_action(action, updated, tuple(sorted(item.branch_id for item in ready)))
            self._commit(
                "authority_transition",
                {"branch": asdict(updated), "receipt_sha256": action.receipt_sha256,
                 "decision_evidence_sha256": decision_evidence_sha256,
                 "authority_audit_sha256": audit_sha256, "action": action.to_dict()},
                event=(event_kind, event_payload),
            )
        return AuthorityResult(
            actions=finalized,
            decision_evidence=evidence,
            decision_evidence_sha256=decision_evidence_sha256,
            authority_audit_sha256=audit_sha256,
        )

    def _transition_for_decision(
        self,
        branch: SchedulerBranch,
        receipt: BranchRungReceipt,
        action: str,
        reason: str,
    ) -> tuple[Literal["promote", "freeze", "kill", "complete"], SchedulerBranch, tuple[str, dict[str, Any]]]:
        if action == "kill":
            if receipt.termination_cause != _TRUE_KILL_CAUSE:
                raise SchedulerRuntimeError("true kill requires literal training_diverged receipt cause")
            updated = SchedulerBranch(**{**asdict(branch), "state": "killed", "termination_cause": _TRUE_KILL_CAUSE})
            return "kill", updated, ("branch_true_killed", {})
        if action == "freeze":
            if not receipt.checkpoint_path:
                raise SchedulerRuntimeError("freeze requires a verified resumable checkpoint")
            updated = SchedulerBranch(**{**asdict(branch), "state": "frozen"})
            return "freeze", updated, ("frozen_pool_eviction", {})
        if action != "promote":
            raise SchedulerRuntimeError("unknown ASHA action")
        if branch.rung + 1 >= len(self.ladder.rung_steps):
            updated = SchedulerBranch(**{**asdict(branch), "state": "completed"})
            return "complete", updated, ("rung_complete", {})
        updated = SchedulerBranch(**{**asdict(branch), "state": "queued", "rung": branch.rung + 1})
        return "promote", updated, ("branch_promoted", {})

    def _branch_transition_target(self, action: AuthorityAction) -> SchedulerBranch:
        branch = self._branch(action.branch_id)
        if action.action == "kill":
            return SchedulerBranch(**{**asdict(branch), "state": "killed", "termination_cause": _TRUE_KILL_CAUSE})
        if action.action == "freeze":
            return SchedulerBranch(**{**asdict(branch), "state": "frozen"})
        if action.action == "complete":
            return SchedulerBranch(**{**asdict(branch), "state": "completed"})
        return SchedulerBranch(**{**asdict(branch), "state": "queued", "rung": branch.rung + 1})

    def _event_for_action(
        self,
        action: AuthorityAction,
        updated: SchedulerBranch,
        cohort_ids: tuple[str, ...],
    ) -> tuple[str, dict[str, Any]]:
        common = {
            "receipt_sha256": action.receipt_sha256,
            "authority_audit_sha256": action.authority_audit_sha256,
            "decision_evidence_sha256": action.decision_evidence_sha256,
        }
        if action.action == "promote":
            return "branch_promoted", {**common, "branch_id": action.branch_id, "from_rung": action.rung,
                                        "to_rung": updated.rung, "cohort_ids": cohort_ids}
        if action.action == "freeze":
            return "frozen_pool_eviction", {**common, "branch_id": action.branch_id, "rung": action.rung,
                                              "ckpt_uri": action.checkpoint_path, "reason": action.reason}
        if action.action == "kill":
            return "branch_true_killed", {**common, "branch_id": action.branch_id, "rung": action.rung,
                                            "termination_cause": _TRUE_KILL_CAUSE}
        return "rung_climbed", {**common, "branch_id": action.branch_id, "rung": action.rung,
                                  "step_budget": self.ladder.rung_steps[action.rung],
                                  "measured_score": None, "gpu_usd": None,
                                  "resume_from_ckpt": action.checkpoint_path}

    def _reload_branch_receipt(self, branch: SchedulerBranch) -> BranchRungReceipt:
        if not branch.receipt_path or not branch.receipt_sha256:
            raise SchedulerRuntimeError("branch is missing a receipt binding")
        path = _safe_receipt_path(branch.receipt_path, self.run_dir)
        if _sha256_bytes(path.read_bytes()) != branch.receipt_sha256:
            raise SchedulerRuntimeError("branch receipt changed after controller attestation")
        receipt = load_verified_receipt(path, ladder=self.ladder, run_dir=self.run_dir, expected_campaign_id=self.campaign_id)
        if receipt is None:
            raise SchedulerRuntimeError("branch receipt no longer verifies")
        if receipt.branch_id != branch.branch_id or receipt.to_step != self.ladder.rung_steps[branch.rung]:
            raise SchedulerRuntimeError("branch receipt does not match current rung")
        return receipt

    def _branch(self, branch_id: str) -> SchedulerBranch:
        branch_id = _string(branch_id, "branch_id")
        try:
            return self._projection.branches[branch_id]
        except KeyError as exc:
            raise SchedulerRuntimeError("unknown branch") from exc

    def _load_projection(self) -> _Projection:
        self.campaign_dir.mkdir(parents=True, exist_ok=True)
        if self.state_path.exists():
            try:
                projection = _Projection.from_mapping(json.loads(self.state_path.read_text(encoding="utf-8")))
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                raise SchedulerRuntimeError("scheduler tree state is malformed") from exc
        else:
            projection = _Projection(campaign_id=self.campaign_id, ladder_sha256=self.ladder.sha256)
        if projection.campaign_id != self.campaign_id or projection.ladder_sha256 != self.ladder.sha256:
            raise SchedulerRuntimeError("scheduler tree belongs to another campaign or step ladder")
        for record in self._read_wal():
            if record["seq"] > projection.applied_seq:
                self._apply_record(projection, record)
        self._write_snapshot(projection)
        return projection

    def _read_wal(self) -> list[dict[str, Any]]:
        if not self.wal_path.exists():
            return []
        records: list[dict[str, Any]] = []
        try:
            for line in self.wal_path.read_text(encoding="utf-8").splitlines():
                raw = json.loads(line)
                if not isinstance(raw, dict) or type(raw.get("seq")) is not int or not isinstance(raw.get("op"), str):
                    raise SchedulerRuntimeError("scheduler action WAL is malformed")
                records.append(raw)
        except (OSError, json.JSONDecodeError) as exc:
            raise SchedulerRuntimeError("scheduler action WAL is malformed") from exc
        expected = 1
        for record in records:
            if record["seq"] != expected:
                raise SchedulerRuntimeError("scheduler action WAL sequence is not contiguous")
            expected += 1
        return records

    def _commit(self, op: str, payload: Mapping[str, Any], *, event: tuple[str, Mapping[str, Any]] | None) -> None:
        seq = self._projection.applied_seq + 1
        record = {"schema_version": SCHEMA_VERSION, "seq": seq, "op": op, "payload": dict(payload)}
        encoded = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        try:
            with self.wal_path.open("a", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise SchedulerRuntimeError("cannot durably append scheduler action") from exc
        self._apply_record(self._projection, record)
        self._write_snapshot(self._projection)
        if event is not None and self.event_sink is not None:
            kind, payload = event
            try:
                self.event_sink(kind, payload)
            except Exception as exc:  # event delivery must be reconciled from the durable WAL
                raise SchedulerRuntimeError(f"scheduler transition is durable but event emission failed: {kind}") from exc

    @staticmethod
    def _apply_record(projection: _Projection, record: Mapping[str, Any]) -> None:
        payload = record.get("payload")
        if not isinstance(payload, Mapping):
            raise SchedulerRuntimeError("scheduler action payload is malformed")
        op = record.get("op")
        if op in {"branch_registered", "receipt_recorded", "authority_transition"}:
            raw_branch = payload.get("branch")
            if not isinstance(raw_branch, Mapping):
                raise SchedulerRuntimeError("scheduler branch action is malformed")
            branch = SchedulerBranch.from_mapping(raw_branch)
            projection.branches[branch.branch_id] = branch
            if op == "authority_transition":
                receipt_sha = _sha(payload.get("receipt_sha256"), "receipt_sha256")
                projection.used_receipts.add(receipt_sha)
        elif op == "dedup_hit":
            _string(payload.get("hypothesis_fingerprint"), "hypothesis_fingerprint")
            _string(payload.get("existing_branch_id"), "existing_branch_id")
        else:
            raise SchedulerRuntimeError("unknown scheduler action")
        projection.applied_seq = _nonnegative_int(record.get("seq"), "seq")

    def _write_snapshot(self, projection: _Projection) -> None:
        data = json.dumps(projection.to_dict(), sort_keys=True, separators=(",", ":"))
        tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.state_path)
        except OSError as exc:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise SchedulerRuntimeError("cannot write scheduler tree snapshot") from exc


def _safe_receipt_path(value: Path | str, run_dir: Path) -> Path:
    raw = Path(value)
    path = raw if raw.is_absolute() else run_dir / raw
    try:
        resolved = path.resolve(strict=True)
        allowed = (run_dir / "campaign" / "scheduler_receipts").resolve()
        resolved.relative_to(allowed)
    except (OSError, ValueError) as exc:
        raise SchedulerRuntimeError("receipt path is outside the scheduler receipt store") from exc
    if path.is_symlink():
        raise SchedulerRuntimeError("scheduler receipt path may not be a symlink")
    return resolved


def _provider_cost(costs: Mapping[str, float], branch_id: str) -> float:
    if branch_id not in costs:
        raise SchedulerRuntimeError("provider GPU cost is missing for a ready branch")
    value = costs[branch_id]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
        raise SchedulerRuntimeError("provider GPU cost must be finite and non-negative")
    return float(value)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _sha(value: Any, name: str) -> str:
    if not _is_sha256(value):
        raise SchedulerRuntimeError(f"{name} must be a SHA-256 digest")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise SchedulerRuntimeError(f"{name} must be a non-empty string")
    return value


def _optional_string(value: Any, name: str) -> str | None:
    return None if value is None else _string(value, name)


def _optional_sha(value: Any, name: str) -> str | None:
    return None if value is None else _sha(value, name)


def _nonnegative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise SchedulerRuntimeError(f"{name} must be a non-negative integer")
    return value


def _optional_positive_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 1:
        raise SchedulerRuntimeError(f"{name} must be a positive integer")
    return value


def _branch_type(value: Any) -> BranchType:
    if value not in {"faithful", "ambiguity", "discovery"}:
        raise SchedulerRuntimeError("branch_type is invalid")
    return value


def _branch_state(value: Any) -> BranchState:
    if value not in _BRANCH_STATES:
        raise SchedulerRuntimeError("branch state is invalid")
    return value


def _finite_nonnegative(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and value >= 0
