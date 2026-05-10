"""Failure diagnosis and taxonomy service."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from backend.persistence.database import Database
from backend.services.context.memory import CrossProjectMemoryService
from backend.services.diagnostics.model import FailureEvent, FailureKind


def failure_id_for(*, project_id: str, stage: str, command: str, stderr_tail: str) -> str:
    h = hashlib.sha256()
    h.update(f"failure:{project_id}:{stage}:{command}:".encode())
    h.update(stderr_tail.encode())
    return f"fail_{h.hexdigest()[:20]}"


class FailureDiagnosisService:
    """Classifies and stores failures using the PRD failure taxonomy."""

    def __init__(self, db: Database) -> None:
        self._db = db
        self._ensure_table()

    def _ensure_table(self) -> None:
        self._db.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS failure_events (
                failure_id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                kind TEXT NOT NULL,
                command TEXT NOT NULL DEFAULT '',
                exit_code INTEGER,
                retryable INTEGER NOT NULL,
                suspected_cause TEXT NOT NULL DEFAULT '',
                recommended_next_step TEXT NOT NULL DEFAULT '',
                stdout_tail TEXT NOT NULL DEFAULT '',
                stderr_tail TEXT NOT NULL DEFAULT '',
                artifact_refs_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_failure_events_project
                ON failure_events(project_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_failure_events_kind
                ON failure_events(kind);
            """
        )
        self._db.connection.commit()

    def diagnose(
        self,
        *,
        project_id: str,
        stage: str,
        command: str = "",
        exit_code: int | None = None,
        stdout: str = "",
        stderr: str = "",
        timed_out: bool = False,
        cause_kind: str = "",
        artifact_refs: tuple[str, ...] | list[str] = (),
    ) -> FailureEvent:
        stdout_tail = _tail(stdout)
        stderr_tail = _tail(stderr)
        kind, retryable, suspected, next_step = _classify(
            stage=stage,
            command=command,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            cause_kind=cause_kind,
        )
        event = FailureEvent(
            failure_id=failure_id_for(
                project_id=project_id,
                stage=stage,
                command=command,
                stderr_tail=stderr_tail,
            ),
            project_id=project_id,
            stage=stage,
            kind=kind,
            command=command,
            exit_code=exit_code,
            retryable=retryable,
            suspected_cause=suspected,
            recommended_next_step=next_step,
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
            artifact_refs=tuple(artifact_refs),
            created_at=datetime.now(timezone.utc),
        )
        self.upsert(event)
        return event

    def promote_to_memory(
        self,
        memory: CrossProjectMemoryService,
        event: FailureEvent,
        *,
        title: str | None = None,
        confidence: float = 0.7,
    ) -> str:
        record = memory.remember_failure_mode(
            source_project_id=event.project_id,
            title=title or f"{event.kind} during {event.stage}",
            summary=(
                f"{event.suspected_cause} Recommended next step: "
                f"{event.recommended_next_step}"
            ),
            failure_kind=event.kind,
            evidence_refs=event.artifact_refs,
            confidence=confidence,
        )
        return record.id

    def upsert(self, event: FailureEvent) -> None:
        self._db.connection.execute(
            """
            INSERT OR REPLACE INTO failure_events
                (failure_id, project_id, stage, kind, command, exit_code, retryable,
                 suspected_cause, recommended_next_step, stdout_tail, stderr_tail,
                 artifact_refs_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.failure_id,
                event.project_id,
                event.stage,
                event.kind,
                event.command,
                event.exit_code,
                int(event.retryable),
                event.suspected_cause,
                event.recommended_next_step,
                event.stdout_tail,
                event.stderr_tail,
                json.dumps(list(event.artifact_refs), sort_keys=True),
                event.created_at.isoformat(),
            ),
        )
        self._db.connection.commit()

    def list_events(
        self,
        *,
        project_id: str | None = None,
        kind: FailureKind | None = None,
    ) -> tuple[FailureEvent, ...]:
        clauses: list[str] = []
        params: list[Any] = []
        if project_id is not None:
            clauses.append("project_id = ?")
            params.append(project_id)
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._db.connection.execute(
            f"SELECT * FROM failure_events {where} ORDER BY created_at DESC",
            tuple(params),
        ).fetchall()
        return tuple(_event_from_row(row) for row in rows)


def _classify(
    *,
    stage: str,
    command: str,
    stdout: str,
    stderr: str,
    timed_out: bool,
    cause_kind: str,
) -> tuple[FailureKind, bool, str, str]:
    haystack = f"{stage}\n{command}\n{stdout}\n{stderr}\n{cause_kind}".lower()

    if timed_out or "exec_timeout" in haystack or "timed out" in haystack:
        return (
            "timeout",
            True,
            "The command exceeded its configured wall-clock budget.",
            "Resume from the checkpoint with a larger timeout or a smaller smoke-test run.",
        )
    if "no space left on device" in haystack or "disk quota" in haystack:
        return (
            "out_of_disk",
            False,
            "The sandbox or host ran out of disk space.",
            "Clear old run artifacts/images or increase the sandbox disk allocation.",
        )
    if "cuda out of memory" in haystack or "out of memory" in haystack or "oom" in haystack:
        return (
            "out_of_memory",
            True,
            "The run exceeded available memory.",
            "Reduce batch size/model size or request a larger sandbox/GPU.",
        )
    if "docker build" in haystack or "build_failed" in haystack:
        return (
            "failed_docker_build",
            True,
            "The Docker environment failed to build.",
            "Inspect build logs, pin failing packages, and retry from the environment gate.",
        )
    if "no matching distribution" in haystack or "resolutionimpossible" in haystack:
        return (
            "failed_dependency_resolution",
            True,
            "Python dependency constraints are incompatible or unavailable.",
            "Run dependency forensics and pin a compatible package set.",
        )
    if "modulenotfounderror" in haystack or "importerror" in haystack:
        return (
            "failed_install",
            True,
            "The runtime is missing an import required by the code.",
            "Add the package to the environment spec and rebuild the sandbox.",
        )
    if "license" in haystack and "blocked" in haystack:
        return (
            "blocked_license",
            False,
            "License state blocks autonomous use.",
            "Ask for human approval or choose a compliant artifact.",
        )
    if "credential" in haystack or "permission denied" in haystack or "401" in haystack or "403" in haystack:
        return (
            "blocked_credentials",
            False,
            "The action requires credentials or access rights.",
            "Request credentials or mark the reproduction blocked.",
        )
    if "dataset" in haystack and ("not found" in haystack or "unavailable" in haystack):
        return (
            "blocked_unavailable_dataset",
            False,
            "The required dataset is unavailable from the configured source.",
            "Request user-provided data or approval for a substitute dataset.",
        )
    if "download" in haystack or "connection" in haystack or "could not resolve host" in haystack:
        return (
            "failed_data_download",
            True,
            "A network or dataset download step failed.",
            "Retry with approved network access or cache the dataset manually.",
        )
    if "metric" in haystack:
        return (
            "failed_metric_validation",
            False,
            "Metric generation or validation failed.",
            "Inspect evaluation scripts and compare metric definitions against the contract.",
        )
    if "plot" in haystack or "matplotlib" in haystack:
        return (
            "failed_plot_generation",
            True,
            "Plot generation failed after or during evaluation.",
            "Regenerate plots from metrics or install the plotting dependency.",
        )
    if "eval" in command.lower() or "evaluate" in command.lower():
        return (
            "failed_evaluation",
            True,
            "The evaluation command failed.",
            "Inspect evaluation logs and rerun from the evaluation stage.",
        )
    if "train" in command.lower():
        return (
            "failed_training",
            True,
            "The training command failed.",
            "Inspect training logs, reduce the run budget, or adjust the environment.",
        )
    if "pytest" in haystack or "smoke" in haystack or "test" in haystack:
        return (
            "failed_smoke_test",
            True,
            "The smoke-test command failed.",
            "Fix the smallest failing command before running the full experiment.",
        )
    return (
        "failed_install",
        True,
        "The failure did not match a narrower taxonomy bucket.",
        "Inspect logs and classify manually if this repeats.",
    )


def _tail(text: str, limit: int = 4000) -> str:
    return text[-limit:] if len(text) > limit else text


def _event_from_row(row: Any) -> FailureEvent:
    return FailureEvent(
        failure_id=row["failure_id"],
        project_id=row["project_id"],
        stage=row["stage"],
        kind=row["kind"],
        command=row["command"],
        exit_code=row["exit_code"],
        retryable=bool(row["retryable"]),
        suspected_cause=row["suspected_cause"],
        recommended_next_step=row["recommended_next_step"],
        stdout_tail=row["stdout_tail"],
        stderr_tail=row["stderr_tail"],
        artifact_refs=tuple(json.loads(row["artifact_refs_json"] or "[]")),
        created_at=datetime.fromisoformat(row["created_at"]),
    )


__all__ = ["FailureDiagnosisService", "failure_id_for"]
