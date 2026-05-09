"""Repository for artifacts."""

from __future__ import annotations

from backend.persistence.database import Database
from backend.schemas.artifacts import Artifact


class ArtifactRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def create(self, artifact: Artifact) -> None:
        self._db.connection.execute(
            """INSERT INTO artifacts
               (artifact_id, artifact_type, run_id, file_path, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (
                artifact.artifact_id,
                artifact.artifact_type.value if hasattr(artifact.artifact_type, "value") else artifact.artifact_type,
                artifact.run_id,
                artifact.file_path,
                artifact.created_at.isoformat(),
            ),
        )
        self._db.connection.commit()

    def list_by_run(self, run_id: str) -> list[Artifact]:
        rows = self._db.connection.execute(
            "SELECT * FROM artifacts WHERE run_id = ?", (run_id,)
        ).fetchall()
        return [
            Artifact(
                artifact_id=r["artifact_id"],
                artifact_type=r["artifact_type"],
                run_id=r["run_id"],
                file_path=r["file_path"],
                created_at=r["created_at"],
            )
            for r in rows
        ]
