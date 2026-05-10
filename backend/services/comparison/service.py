"""Deterministic multi-paper comparison service."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Iterable

from backend.persistence.database import Database
from backend.services.comparison.model import (
    ComparableGroup,
    ComparisonReport,
    IncomparableRun,
    PaperRunSummary,
)


class MultiPaperComparisonService:
    """Groups reproduced papers by comparable dataset/split/metric contracts."""

    def __init__(self, db: Database | None = None) -> None:
        self._db = db
        if self._db is not None:
            self._ensure_table()

    def _ensure_table(self) -> None:
        self._db.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS comparison_reports (
                comparison_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            """
        )
        self._db.connection.commit()

    def compare(self, runs: Iterable[PaperRunSummary]) -> ComparisonReport:
        run_list = tuple(runs)
        comparable: dict[str, list[PaperRunSummary]] = defaultdict(list)
        incomparable: list[IncomparableRun] = []

        for run in run_list:
            reason = _incomparability_reason(run)
            if reason is not None:
                incomparable.append(IncomparableRun(project_id=run.project_id, reason=reason))
                continue
            comparable[_group_key(run)].append(run)

        groups = tuple(
            _build_group(key, values)
            for key, values in sorted(comparable.items(), key=lambda item: item[0])
        )
        shared_assumptions = _shared_assumptions(run_list)
        recommendations = _recommendations(groups, incomparable)
        report = ComparisonReport(
            comparison_id=_comparison_id(run_list),
            created_at=datetime.now(timezone.utc),
            status="complete" if not incomparable else "partial",
            groups=groups,
            incomparable_runs=tuple(incomparable),
            shared_assumptions=shared_assumptions,
            recommendations=recommendations,
        )
        if self._db is not None:
            self.persist(report)
        return report

    def persist(self, report: ComparisonReport) -> None:
        if self._db is None:
            raise RuntimeError("Cannot persist comparison report without a Database")
        self._db.connection.execute(
            """
            INSERT OR REPLACE INTO comparison_reports
                (comparison_id, created_at, status, payload_json)
            VALUES (?, ?, ?, ?)
            """,
            (
                report.comparison_id,
                report.created_at.isoformat(),
                report.status,
                report.model_dump_json(),
            ),
        )
        self._db.connection.commit()

    def get(self, comparison_id: str) -> ComparisonReport | None:
        if self._db is None:
            raise RuntimeError("Cannot read comparison report without a Database")
        row = self._db.connection.execute(
            "SELECT payload_json FROM comparison_reports WHERE comparison_id = ?",
            (comparison_id,),
        ).fetchone()
        if row is None:
            return None
        return ComparisonReport.model_validate_json(row["payload_json"])

    def list_reports(self, *, limit: int = 20) -> tuple[ComparisonReport, ...]:
        if self._db is None:
            raise RuntimeError("Cannot read comparison reports without a Database")
        rows = self._db.connection.execute(
            """
            SELECT payload_json FROM comparison_reports
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return tuple(ComparisonReport.model_validate_json(row["payload_json"]) for row in rows)


def _group_key(run: PaperRunSummary) -> str:
    return "::".join([
        run.dataset.strip().lower(),
        run.split.strip().lower(),
        run.metric_name.strip().lower(),
    ])


def _incomparability_reason(run: PaperRunSummary) -> str | None:
    if not run.dataset:
        return "missing dataset"
    if not run.metric_name:
        return "missing metric_name"
    if run.metric_value is None:
        return "missing metric_value"
    if run.reduced_run:
        return "reduced run must be compared separately"
    if run.status and run.status not in {"verified", "verified_with_caveats", "partial_reproduction"}:
        return f"status is not comparable: {run.status}"
    return None


def _build_group(key: str, runs: list[PaperRunSummary]) -> ComparableGroup:
    dataset, split, metric = key.split("::", 2)
    best = max(runs, key=lambda run: run.metric_value if run.metric_value is not None else float("-inf"))
    notes: list[str] = []
    if len(runs) == 1:
        notes.append("Only one comparable run in this group.")
    if any(run.status == "verified_with_caveats" for run in runs):
        notes.append("At least one run has verification caveats.")
    return ComparableGroup(
        group_key=key,
        dataset=dataset,
        split=split,
        metric_name=metric,
        runs=tuple(sorted(runs, key=lambda run: run.project_id)),
        best_project_id=best.project_id,
        best_metric_value=best.metric_value,
        notes=tuple(notes),
    )


def _shared_assumptions(runs: tuple[PaperRunSummary, ...]) -> tuple[str, ...]:
    counts: Counter[str] = Counter()
    for run in runs:
        counts.update(run.assumptions)
    if len(runs) < 2:
        return ()
    return tuple(sorted(item for item, count in counts.items() if count >= 2))


def _recommendations(
    groups: tuple[ComparableGroup, ...],
    incomparable: list[IncomparableRun],
) -> tuple[str, ...]:
    recommendations: list[str] = []
    for group in groups:
        if len(group.runs) >= 2 and group.best_project_id:
            recommendations.append(
                f"Use {group.best_project_id} as the current best run for "
                f"{group.dataset}/{group.metric_name}."
            )
    if incomparable:
        recommendations.append("Normalize dataset, metric, and run-budget contracts before comparison.")
    if not recommendations:
        recommendations.append("Add another verified run on the same dataset and metric.")
    return tuple(recommendations)


def _comparison_id(runs: tuple[PaperRunSummary, ...]) -> str:
    h = hashlib.sha256()
    for run in sorted(runs, key=lambda item: item.project_id):
        h.update(
            json.dumps(
                {
                    "project_id": run.project_id,
                    "paper_id": run.paper_id,
                    "dataset": run.dataset,
                    "split": run.split,
                    "metric_name": run.metric_name,
                    "metric_value": run.metric_value,
                },
                sort_keys=True,
            ).encode()
        )
    return f"cmp_{h.hexdigest()[:20]}"


__all__ = ["MultiPaperComparisonService"]
