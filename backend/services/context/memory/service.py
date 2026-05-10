"""SQLite-backed cross-project memory service."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from backend.persistence.database import Database
from backend.services.context.memory.model import MemoryKind, MemoryRecord, MemorySearchResult


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def memory_id_for(
    *,
    kind: str,
    source_project_id: str,
    title: str,
    summary: str,
) -> str:
    h = hashlib.sha256()
    h.update(f"memory:{kind}:{source_project_id}:{title}:".encode())
    h.update(summary.encode())
    return f"mem_{h.hexdigest()[:20]}"


class CrossProjectMemoryService:
    """Reusable memory across paper reproductions.

    This is intentionally explicit. Only verified or useful records should be
    written by callers; the service stores and searches them, but does not
    decide promotion policy.
    """

    def __init__(self, db: Database) -> None:
        self._db = db
        self._ensure_table()

    def _ensure_table(self) -> None:
        self._db.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS cross_project_memory (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                source_project_id TEXT NOT NULL,
                paper_id TEXT NOT NULL DEFAULT '',
                title TEXT NOT NULL,
                summary TEXT NOT NULL,
                tags_json TEXT NOT NULL DEFAULT '[]',
                metrics_json TEXT NOT NULL DEFAULT '{}',
                evidence_refs_json TEXT NOT NULL DEFAULT '[]',
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_memory_kind
                ON cross_project_memory(kind);
            CREATE INDEX IF NOT EXISTS idx_memory_source_project
                ON cross_project_memory(source_project_id);
            CREATE INDEX IF NOT EXISTS idx_memory_paper
                ON cross_project_memory(paper_id);
            """
        )
        self._db.connection.commit()

    def remember(
        self,
        *,
        kind: MemoryKind,
        source_project_id: str,
        title: str,
        summary: str,
        paper_id: str = "",
        tags: tuple[str, ...] | list[str] = (),
        metrics: dict[str, Any] | None = None,
        evidence_refs: tuple[str, ...] | list[str] = (),
        confidence: float = 0.5,
    ) -> MemoryRecord:
        record = MemoryRecord(
            id=memory_id_for(
                kind=kind,
                source_project_id=source_project_id,
                title=title,
                summary=summary,
            ),
            kind=kind,
            source_project_id=source_project_id,
            paper_id=paper_id,
            title=title,
            summary=summary,
            tags=tuple(tags),
            metrics=metrics or {},
            evidence_refs=tuple(evidence_refs),
            confidence=confidence,
            created_at=datetime.now(timezone.utc),
        )
        self.upsert(record)
        return record

    def remember_environment_recipe(
        self,
        *,
        source_project_id: str,
        title: str,
        summary: str,
        paper_id: str = "",
        packages: dict[str, Any] | None = None,
        evidence_refs: tuple[str, ...] | list[str] = (),
        confidence: float = 0.8,
    ) -> MemoryRecord:
        tags = ["environment"]
        if packages:
            tags.extend(sorted(packages))
        return self.remember(
            kind="environment_recipe",
            source_project_id=source_project_id,
            paper_id=paper_id,
            title=title,
            summary=summary,
            tags=tuple(tags),
            metrics={"packages": packages or {}},
            evidence_refs=evidence_refs,
            confidence=confidence,
        )

    def remember_failure_mode(
        self,
        *,
        source_project_id: str,
        title: str,
        summary: str,
        failure_kind: str,
        paper_id: str = "",
        evidence_refs: tuple[str, ...] | list[str] = (),
        confidence: float = 0.7,
    ) -> MemoryRecord:
        return self.remember(
            kind="failure_mode",
            source_project_id=source_project_id,
            paper_id=paper_id,
            title=title,
            summary=summary,
            tags=("failure", failure_kind),
            metrics={"failure_kind": failure_kind},
            evidence_refs=evidence_refs,
            confidence=confidence,
        )

    def upsert(self, record: MemoryRecord) -> None:
        self._db.connection.execute(
            """
            INSERT OR REPLACE INTO cross_project_memory
                (id, kind, source_project_id, paper_id, title, summary, tags_json,
                 metrics_json, evidence_refs_json, confidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.kind,
                record.source_project_id,
                record.paper_id,
                record.title,
                record.summary,
                json.dumps(list(record.tags), sort_keys=True),
                json.dumps(record.metrics, sort_keys=True),
                json.dumps(list(record.evidence_refs), sort_keys=True),
                record.confidence,
                record.created_at.isoformat(),
            ),
        )
        self._db.connection.commit()

    def get(self, record_id: str) -> MemoryRecord | None:
        row = self._db.connection.execute(
            "SELECT * FROM cross_project_memory WHERE id = ?", (record_id,)
        ).fetchone()
        return _record_from_row(row) if row is not None else None

    def list_records(
        self,
        *,
        kind: MemoryKind | None = None,
        source_project_id: str | None = None,
        paper_id: str | None = None,
    ) -> tuple[MemoryRecord, ...]:
        clauses: list[str] = []
        params: list[Any] = []
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        if source_project_id is not None:
            clauses.append("source_project_id = ?")
            params.append(source_project_id)
        if paper_id is not None:
            clauses.append("paper_id = ?")
            params.append(paper_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._db.connection.execute(
            f"SELECT * FROM cross_project_memory {where} ORDER BY created_at DESC",
            tuple(params),
        ).fetchall()
        return tuple(_record_from_row(row) for row in rows)

    def search(
        self,
        query: str,
        *,
        kind: MemoryKind | None = None,
        limit: int = 5,
    ) -> tuple[MemorySearchResult, ...]:
        terms = _tokenize(query)
        if not terms:
            return ()
        candidates = self.list_records(kind=kind)
        results: list[MemorySearchResult] = []
        query_counts = Counter(terms)
        for record in candidates:
            haystack = " ".join([record.title, record.summary, *record.tags])
            hay_counts = Counter(_tokenize(haystack))
            matched = tuple(sorted(set(query_counts) & set(hay_counts)))
            if not matched:
                continue
            score = sum(query_counts[t] * hay_counts[t] for t in matched)
            score *= 0.5 + record.confidence
            results.append(
                MemorySearchResult(
                    record=record,
                    score=round(score, 6),
                    matched_terms=matched,
                )
            )
        results.sort(key=lambda item: (-item.score, item.record.title, item.record.id))
        return tuple(results[:limit])


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]


def _record_from_row(row: Any) -> MemoryRecord:
    return MemoryRecord(
        id=row["id"],
        kind=row["kind"],
        source_project_id=row["source_project_id"],
        paper_id=row["paper_id"],
        title=row["title"],
        summary=row["summary"],
        tags=tuple(json.loads(row["tags_json"] or "[]")),
        metrics=json.loads(row["metrics_json"] or "{}"),
        evidence_refs=tuple(json.loads(row["evidence_refs_json"] or "[]")),
        confidence=row["confidence"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


__all__ = ["CrossProjectMemoryService", "memory_id_for"]
