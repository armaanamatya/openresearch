"""Repository for verifications."""

from __future__ import annotations

import json

from backend.persistence.database import Database
from backend.schemas.verifications import VerificationRecord


class VerificationRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def create(self, record: VerificationRecord) -> None:
        self._db.connection.execute(
            """INSERT INTO verifications
               (verification_id, run_id, verifier_type, status,
                method_fidelity_score, environment_recovery_score,
                data_pipeline_confidence, artifact_completeness_score,
                caveats, blocking_issues, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.verification_id,
                record.run_id,
                record.verifier_type,
                record.status.value if hasattr(record.status, "value") else record.status,
                record.method_fidelity_score,
                record.environment_recovery_score,
                record.data_pipeline_confidence,
                record.artifact_completeness_score,
                json.dumps(record.caveats),
                json.dumps(record.blocking_issues),
                record.created_at.isoformat(),
            ),
        )
        self._db.connection.commit()

    def get_by_id(self, verification_id: str) -> VerificationRecord | None:
        row = self._db.connection.execute(
            "SELECT * FROM verifications WHERE verification_id = ?", (verification_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def list_by_run(self, run_id: str) -> list[VerificationRecord]:
        rows = self._db.connection.execute(
            "SELECT * FROM verifications WHERE run_id = ?", (run_id,)
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    @staticmethod
    def _row_to_record(row) -> VerificationRecord:
        return VerificationRecord(
            verification_id=row["verification_id"],
            run_id=row["run_id"],
            verifier_type=row["verifier_type"],
            status=row["status"],
            method_fidelity_score=row["method_fidelity_score"],
            environment_recovery_score=row["environment_recovery_score"],
            data_pipeline_confidence=row["data_pipeline_confidence"],
            artifact_completeness_score=row["artifact_completeness_score"],
            caveats=json.loads(row["caveats"]),
            blocking_issues=json.loads(row["blocking_issues"]),
            created_at=row["created_at"],
        )
