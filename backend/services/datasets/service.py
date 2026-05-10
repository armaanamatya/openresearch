"""SQLite-backed dataset cache."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.persistence.database import Database
from backend.services.datasets.model import DatasetCacheEntry, DatasetCacheStatus


def dataset_id_for(*, name: str, source_url: str = "", version: str = "", checksum: str = "") -> str:
    h = hashlib.sha256()
    h.update(f"dataset:{name.strip().lower()}:{source_url}:{version}:{checksum}".encode())
    return f"ds_{h.hexdigest()[:20]}"


class DatasetCacheService:
    """Tracks reusable local datasets without owning dataset-specific loaders."""

    def __init__(self, db: Database, *, cache_root: str | Path = "datasets") -> None:
        self._db = db
        self.cache_root = Path(cache_root)
        self._ensure_table()

    def _ensure_table(self) -> None:
        self._db.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS dataset_cache (
                dataset_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                source_url TEXT NOT NULL DEFAULT '',
                version TEXT NOT NULL DEFAULT '',
                checksum TEXT NOT NULL DEFAULT '',
                size_bytes INTEGER,
                local_path TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL,
                source_project_id TEXT NOT NULL DEFAULT '',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                failure_reason TEXT NOT NULL DEFAULT ''
            );

            CREATE INDEX IF NOT EXISTS idx_dataset_cache_name
                ON dataset_cache(name, version, status);
            CREATE INDEX IF NOT EXISTS idx_dataset_cache_project
                ON dataset_cache(source_project_id);
            """
        )
        self._db.connection.commit()

    def plan(
        self,
        *,
        name: str,
        source_url: str = "",
        version: str = "",
        checksum: str = "",
        size_bytes: int | None = None,
        source_project_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> DatasetCacheEntry:
        now = datetime.now(timezone.utc)
        dataset_id = dataset_id_for(
            name=name,
            source_url=source_url,
            version=version,
            checksum=checksum,
        )
        existing = self.get(dataset_id)
        if existing is not None:
            return existing
        local_path = self.cache_root / _safe_dataset_dir(name, version or checksum or "default")
        entry = DatasetCacheEntry(
            dataset_id=dataset_id,
            name=name,
            source_url=source_url,
            version=version,
            checksum=checksum,
            size_bytes=size_bytes,
            local_path=str(local_path),
            status="planned",
            source_project_id=source_project_id,
            metadata=metadata or {},
            created_at=now,
            updated_at=now,
        )
        self.upsert(entry)
        return entry

    def mark_downloading(self, dataset_id: str) -> DatasetCacheEntry:
        return self._transition(dataset_id, "downloading")

    def mark_available(
        self,
        dataset_id: str,
        *,
        local_path: str | Path | None = None,
        checksum: str | None = None,
        size_bytes: int | None = None,
    ) -> DatasetCacheEntry:
        entry = self._require(dataset_id)
        updates: dict[str, Any] = {
            "status": "available",
            "updated_at": datetime.now(timezone.utc),
            "failure_reason": "",
        }
        if local_path is not None:
            updates["local_path"] = str(local_path)
        if checksum is not None:
            updates["checksum"] = checksum
        if size_bytes is not None:
            updates["size_bytes"] = size_bytes
        updated = entry.model_copy(update=updates)
        self.upsert(updated)
        return updated

    def mark_failed(self, dataset_id: str, reason: str) -> DatasetCacheEntry:
        entry = self._require(dataset_id).model_copy(
            update={
                "status": "failed",
                "failure_reason": reason,
                "updated_at": datetime.now(timezone.utc),
            }
        )
        self.upsert(entry)
        return entry

    def mark_blocked(self, dataset_id: str, reason: str) -> DatasetCacheEntry:
        entry = self._require(dataset_id).model_copy(
            update={
                "status": "blocked",
                "failure_reason": reason,
                "updated_at": datetime.now(timezone.utc),
            }
        )
        self.upsert(entry)
        return entry

    def find_reusable(
        self,
        *,
        name: str,
        version: str = "",
        checksum: str = "",
    ) -> DatasetCacheEntry | None:
        clauses = ["LOWER(name) = LOWER(?)", "status = 'available'"]
        params: list[Any] = [name]
        if version:
            clauses.append("version = ?")
            params.append(version)
        if checksum:
            clauses.append("checksum = ?")
            params.append(checksum)
        row = self._db.connection.execute(
            f"""
            SELECT * FROM dataset_cache
            WHERE {' AND '.join(clauses)}
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            tuple(params),
        ).fetchone()
        return _entry_from_row(row) if row is not None else None

    def list_entries(
        self,
        *,
        status: DatasetCacheStatus | None = None,
        source_project_id: str | None = None,
    ) -> tuple[DatasetCacheEntry, ...]:
        clauses: list[str] = []
        params: list[Any] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if source_project_id is not None:
            clauses.append("source_project_id = ?")
            params.append(source_project_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._db.connection.execute(
            f"SELECT * FROM dataset_cache {where} ORDER BY updated_at DESC",
            tuple(params),
        ).fetchall()
        return tuple(_entry_from_row(row) for row in rows)

    def get(self, dataset_id: str) -> DatasetCacheEntry | None:
        row = self._db.connection.execute(
            "SELECT * FROM dataset_cache WHERE dataset_id = ?",
            (dataset_id,),
        ).fetchone()
        return _entry_from_row(row) if row is not None else None

    def upsert(self, entry: DatasetCacheEntry) -> None:
        self._db.connection.execute(
            """
            INSERT OR REPLACE INTO dataset_cache
                (dataset_id, name, source_url, version, checksum, size_bytes,
                 local_path, status, source_project_id, metadata_json, created_at,
                 updated_at, failure_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.dataset_id,
                entry.name,
                entry.source_url,
                entry.version,
                entry.checksum,
                entry.size_bytes,
                entry.local_path,
                entry.status,
                entry.source_project_id,
                json.dumps(entry.metadata, sort_keys=True),
                entry.created_at.isoformat(),
                entry.updated_at.isoformat(),
                entry.failure_reason,
            ),
        )
        self._db.connection.commit()

    def _transition(self, dataset_id: str, status: DatasetCacheStatus) -> DatasetCacheEntry:
        entry = self._require(dataset_id).model_copy(
            update={"status": status, "updated_at": datetime.now(timezone.utc)}
        )
        self.upsert(entry)
        return entry

    def _require(self, dataset_id: str) -> DatasetCacheEntry:
        entry = self.get(dataset_id)
        if entry is None:
            raise KeyError(f"Unknown dataset cache entry: {dataset_id}")
        return entry


def _entry_from_row(row: Any) -> DatasetCacheEntry:
    return DatasetCacheEntry(
        dataset_id=row["dataset_id"],
        name=row["name"],
        source_url=row["source_url"],
        version=row["version"],
        checksum=row["checksum"],
        size_bytes=row["size_bytes"],
        local_path=row["local_path"],
        status=row["status"],
        source_project_id=row["source_project_id"],
        metadata=json.loads(row["metadata_json"] or "{}"),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        failure_reason=row["failure_reason"],
    )


def _safe_dataset_dir(name: str, version: str) -> str:
    raw = f"{name}-{version}".lower()
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in raw)
    return safe.strip("-")[:96] or "dataset"


__all__ = ["DatasetCacheService", "dataset_id_for"]
