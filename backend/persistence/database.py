"""SQLite database connection and initialization."""

from __future__ import annotations

import sqlite3
from pathlib import Path


class Database:
    def __init__(self, url: str) -> None:
        # Extract path from sqlite:///path
        self._path = url.replace("sqlite:///", "")
        self._conn: sqlite3.Connection | None = None

    @property
    def connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self._path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=OFF")
        return self._conn

    def initialize(self) -> None:
        """Create all tables. Idempotent."""
        conn = self.connection
        conn.executescript(_SCHEMA)
        conn.commit()

    def list_tables(self) -> list[str]:
        cursor = self.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        return [row[0] for row in cursor.fetchall()]

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_tasks (
    task_id TEXT PRIMARY KEY,
    agent_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'created',
    parent_task_id TEXT,
    failure_substatus TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (parent_task_id) REFERENCES agent_tasks(task_id)
);

CREATE TABLE IF NOT EXISTS agent_messages (
    message_id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    content TEXT NOT NULL,
    structured_outputs TEXT NOT NULL DEFAULT '{}',
    timestamp TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    run_type TEXT NOT NULL,
    status TEXT NOT NULL,
    task_id TEXT NOT NULL,
    parent_run_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (task_id) REFERENCES agent_tasks(task_id),
    FOREIGN KEY (parent_run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    artifact_type TEXT NOT NULL,
    run_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS verifications (
    verification_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    verifier_type TEXT NOT NULL,
    status TEXT NOT NULL,
    method_fidelity_score REAL,
    environment_recovery_score REAL,
    data_pipeline_confidence REAL,
    artifact_completeness_score REAL,
    caveats TEXT NOT NULL DEFAULT '[]',
    blocking_issues TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
"""
