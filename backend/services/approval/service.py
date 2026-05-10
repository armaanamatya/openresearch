"""SQLite-backed approval checkpoint service."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from backend.persistence.database import Database
from backend.services.approval.model import (
    ApprovalAction,
    ApprovalEvaluation,
    ApprovalPolicy,
    ApprovalRequest,
    ApprovalRisk,
    ApprovalState,
)


def approval_id_for(*, project_id: str, action: str, label: str, details: str) -> str:
    h = hashlib.sha256()
    h.update(f"approval:{project_id}:{action}:{label}:".encode())
    h.update(details.encode())
    return f"appr_{h.hexdigest()[:20]}"


class ApprovalService:
    """Evaluates policy and persists explicit human approval checkpoints."""

    def __init__(self, db: Database, policy: ApprovalPolicy | None = None) -> None:
        self._db = db
        self.policy = policy or ApprovalPolicy()
        self._ensure_table()

    def _ensure_table(self) -> None:
        self._db.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS approval_requests (
                approval_id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                action TEXT NOT NULL,
                label TEXT NOT NULL,
                details TEXT NOT NULL,
                state TEXT NOT NULL,
                risk TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                resolved_at TEXT,
                resolved_by TEXT NOT NULL DEFAULT '',
                resolution_note TEXT NOT NULL DEFAULT ''
            );

            CREATE INDEX IF NOT EXISTS idx_approvals_project_state
                ON approval_requests(project_id, state);
            CREATE INDEX IF NOT EXISTS idx_approvals_action
                ON approval_requests(action);
            """
        )
        self._db.connection.commit()

    def evaluate(
        self,
        *,
        action: ApprovalAction,
        dataset_size_gb: float | None = None,
        runtime_minutes: float | None = None,
        gpu_cost_usd: float | None = None,
        repo_trust_level: str = "",
        license_state: str = "",
        network_stage: str = "",
        assumption_risk: str = "",
        external_data: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> ApprovalEvaluation:
        """Return whether an action must pause for human approval."""

        reason = ""
        risk: ApprovalRisk = "medium"
        requires = False

        if action == "dataset_download":
            size = float(dataset_size_gb or 0.0)
            requires = size > self.policy.max_dataset_download_gb_without_approval
            reason = (
                f"Dataset download is {size:.2f} GB; policy allows "
                f"{self.policy.max_dataset_download_gb_without_approval:.2f} GB without approval."
            )
            risk = "high" if size >= 10 else "medium"
        elif action == "long_run":
            minutes = float(runtime_minutes or 0.0)
            requires = minutes > self.policy.max_runtime_minutes_without_approval
            reason = (
                f"Run estimate is {minutes:.1f} minutes; policy allows "
                f"{self.policy.max_runtime_minutes_without_approval} minutes without approval."
            )
            risk = "high" if minutes >= 120 else "medium"
        elif action == "gpu_spend":
            cost = float(gpu_cost_usd or 0.0)
            requires = cost > self.policy.max_gpu_cost_without_approval_usd
            reason = (
                f"GPU spend estimate is ${cost:.2f}; policy allows "
                f"${self.policy.max_gpu_cost_without_approval_usd:.2f} without approval."
            )
            risk = "high" if cost >= 10 else "medium"
        elif action == "unofficial_repo":
            requires = not self.policy.allow_unofficial_repos or repo_trust_level not in {
                "primary",
                "strong_secondary",
            }
            reason = f"Repository trust level is {repo_trust_level or 'unknown'}."
            risk = "high" if repo_trust_level in {"weak", "unknown", ""} else "medium"
        elif action == "unknown_license":
            requires = self.policy.require_approval_for_unknown_license and (
                license_state in {"", "unknown", "requires_user_confirmation"}
            )
            reason = f"License state is {license_state or 'unknown'}."
            risk = "high"
        elif action == "sandbox_network":
            stage = network_stage or "run"
            allowed = (
                self.policy.allow_network_during_build
                if stage == "build"
                else self.policy.allow_network_during_run
            )
            requires = not allowed
            reason = f"Network requested during {stage}; policy allowed={allowed}."
            risk = "high" if stage != "build" else "medium"
        elif action == "high_risk_assumption":
            requires = assumption_risk in {"high", "critical"}
            reason = f"Assumption risk is {assumption_risk or 'unspecified'}."
            risk = "critical" if assumption_risk == "critical" else "high"
        elif action == "substitute_dataset":
            requires = True
            reason = "Substitute datasets change the reproduction contract."
            risk = "critical"
        elif action == "external_upload":
            requires = True
            reason = "External artifact upload can expose code, data, or credentials."
            risk = "high"
        elif action == "untrusted_code":
            requires = True
            reason = "Running untrusted code requires explicit approval."
            risk = "high"

        if external_data and not self.policy.allow_external_data_for_improvements:
            requires = True
            reason = f"{reason} External data for improvements is disallowed without approval."
            risk = "high"

        return ApprovalEvaluation(
            action=action,
            requires_approval=requires,
            reason=reason,
            risk=risk,
            policy_snapshot=self.policy,
            metadata=metadata or {},
        )

    def request_if_needed(
        self,
        *,
        project_id: str,
        label: str,
        evaluation: ApprovalEvaluation,
    ) -> ApprovalRequest | None:
        if not evaluation.requires_approval:
            return None
        return self.create_request(
            project_id=project_id,
            action=evaluation.action,
            label=label,
            details=evaluation.reason,
            risk=evaluation.risk,
            metadata=evaluation.metadata,
        )

    def create_request(
        self,
        *,
        project_id: str,
        action: ApprovalAction,
        label: str,
        details: str,
        risk: ApprovalRisk = "medium",
        metadata: dict[str, Any] | None = None,
    ) -> ApprovalRequest:
        now = datetime.now(timezone.utc)
        request = ApprovalRequest(
            approval_id=approval_id_for(
                project_id=project_id,
                action=action,
                label=label,
                details=details,
            ),
            project_id=project_id,
            action=action,
            label=label,
            details=details,
            risk=risk,
            metadata=metadata or {},
            created_at=now,
        )
        self._upsert(request)
        return request

    def resolve(
        self,
        approval_id: str,
        *,
        state: ApprovalState,
        resolved_by: str = "",
        note: str = "",
    ) -> ApprovalRequest:
        existing = self.get(approval_id)
        if existing is None:
            raise KeyError(f"Unknown approval request: {approval_id}")
        resolved = existing.model_copy(
            update={
                "state": state,
                "resolved_at": datetime.now(timezone.utc),
                "resolved_by": resolved_by,
                "resolution_note": note,
            }
        )
        self._upsert(resolved)
        return resolved

    def get(self, approval_id: str) -> ApprovalRequest | None:
        row = self._db.connection.execute(
            "SELECT * FROM approval_requests WHERE approval_id = ?",
            (approval_id,),
        ).fetchone()
        return _request_from_row(row) if row is not None else None

    def list_requests(
        self,
        *,
        project_id: str | None = None,
        state: ApprovalState | None = None,
    ) -> tuple[ApprovalRequest, ...]:
        clauses: list[str] = []
        params: list[Any] = []
        if project_id is not None:
            clauses.append("project_id = ?")
            params.append(project_id)
        if state is not None:
            clauses.append("state = ?")
            params.append(state)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._db.connection.execute(
            f"SELECT * FROM approval_requests {where} ORDER BY created_at DESC",
            tuple(params),
        ).fetchall()
        return tuple(_request_from_row(row) for row in rows)

    def _upsert(self, request: ApprovalRequest) -> None:
        self._db.connection.execute(
            """
            INSERT OR REPLACE INTO approval_requests
                (approval_id, project_id, action, label, details, state, risk,
                 metadata_json, created_at, resolved_at, resolved_by, resolution_note)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request.approval_id,
                request.project_id,
                request.action,
                request.label,
                request.details,
                request.state,
                request.risk,
                json.dumps(request.metadata, sort_keys=True),
                request.created_at.isoformat(),
                request.resolved_at.isoformat() if request.resolved_at else None,
                request.resolved_by,
                request.resolution_note,
            ),
        )
        self._db.connection.commit()


def _request_from_row(row: Any) -> ApprovalRequest:
    return ApprovalRequest(
        approval_id=row["approval_id"],
        project_id=row["project_id"],
        action=row["action"],
        label=row["label"],
        details=row["details"],
        state=row["state"],
        risk=row["risk"],
        metadata=json.loads(row["metadata_json"] or "{}"),
        created_at=datetime.fromisoformat(row["created_at"]),
        resolved_at=(
            datetime.fromisoformat(row["resolved_at"])
            if row["resolved_at"]
            else None
        ),
        resolved_by=row["resolved_by"],
        resolution_note=row["resolution_note"],
    )


__all__ = ["ApprovalService", "approval_id_for"]
