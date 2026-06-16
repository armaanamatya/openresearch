"""SQLite-backed evaluation results store.

Tracks all eval runs for comparison across agent versions over time.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from backend.evals.schemas import (
    ABTestResult,
    EloRating,
    InnovationScore,
    ReproductionScore,
)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS eval_runs (
    run_id TEXT PRIMARY KEY,
    mode TEXT NOT NULL,
    version TEXT NOT NULL,
    paper_id TEXT NOT NULL,
    timestamp REAL NOT NULL,
    data_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reproduction_scores (
    run_id TEXT PRIMARY KEY,
    version TEXT NOT NULL,
    paper_id TEXT NOT NULL,
    build_success INTEGER NOT NULL,
    run_success INTEGER NOT NULL,
    metric_match REAL NOT NULL,
    fidelity_score REAL NOT NULL,
    assumption_accuracy REAL NOT NULL,
    step_count INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    wall_time_s REAL NOT NULL,
    composite REAL NOT NULL,
    timestamp REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS innovation_scores (
    run_id TEXT PRIMARY KEY,
    version TEXT NOT NULL,
    paper_id TEXT NOT NULL,
    mean_hypothesis_quality REAL NOT NULL,
    integrity_pass_rate REAL NOT NULL,
    research_map_composite REAL,
    timestamp REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS elo_ratings (
    version TEXT PRIMARY KEY,
    rating REAL NOT NULL,
    matches_played INTEGER NOT NULL,
    wins INTEGER NOT NULL,
    losses INTEGER NOT NULL,
    draws INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS ab_tests (
    test_id TEXT PRIMARY KEY,
    version_a TEXT NOT NULL,
    version_b TEXT NOT NULL,
    metric TEXT NOT NULL,
    n_a INTEGER NOT NULL,
    n_b INTEGER NOT NULL,
    p_a_better REAL NOT NULL,
    is_significant INTEGER NOT NULL,
    winner TEXT,
    timestamp REAL NOT NULL,
    data_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_repro_version ON reproduction_scores(version);
CREATE INDEX IF NOT EXISTS idx_repro_paper ON reproduction_scores(paper_id);
CREATE INDEX IF NOT EXISTS idx_innov_version ON innovation_scores(version);
"""


class EvalStore:
    """SQLite store for evaluation results."""

    def __init__(self, db_path: str | Path = "evals.db"):
        self._path = str(db_path)
        self._conn = sqlite3.connect(self._path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    # --- Write ---

    def save_reproduction(self, score: ReproductionScore, run_id: str | None = None) -> str:
        run_id = run_id or str(uuid.uuid4())[:12]
        self._conn.execute(
            """INSERT OR REPLACE INTO reproduction_scores
               (run_id, version, paper_id, build_success, run_success,
                metric_match, fidelity_score, assumption_accuracy,
                step_count, cost_usd, wall_time_s, composite, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id, score.version, score.paper_id,
                int(score.build_success), int(score.run_success),
                score.metric_match, score.fidelity_score,
                score.assumption_accuracy, score.step_count,
                score.cost_usd, score.wall_time_s,
                score.composite_score(), score.timestamp,
            ),
        )
        self._conn.commit()
        return run_id

    def save_innovation(self, score: InnovationScore, run_id: str | None = None) -> str:
        run_id = run_id or str(uuid.uuid4())[:12]
        rm_composite = (
            score.research_map_score.composite_score()
            if score.research_map_score else None
        )
        self._conn.execute(
            """INSERT OR REPLACE INTO innovation_scores
               (run_id, version, paper_id, mean_hypothesis_quality,
                integrity_pass_rate, research_map_composite, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id, score.version, score.paper_id,
                score.mean_hypothesis_quality(),
                score.integrity_pass_rate(),
                rm_composite, score.timestamp,
            ),
        )
        self._conn.commit()
        return run_id

    def save_elo_rating(self, rating: EloRating) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO elo_ratings
               (version, rating, matches_played, wins, losses, draws)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                rating.version, rating.rating,
                rating.matches_played, rating.wins,
                rating.losses, rating.draws,
            ),
        )
        self._conn.commit()

    def save_ab_test(self, result: ABTestResult) -> str:
        test_id = str(uuid.uuid4())[:12]
        self._conn.execute(
            """INSERT INTO ab_tests
               (test_id, version_a, version_b, metric, n_a, n_b,
                p_a_better, is_significant, winner, timestamp, data_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                test_id, result.version_a, result.version_b,
                result.metric, result.n_a, result.n_b,
                result.p_a_better, int(result.is_significant),
                result.winner, time.time(),
                result.model_dump_json(),
            ),
        )
        self._conn.commit()
        return test_id

    # --- Read ---

    def get_reproduction_scores(
        self, version: str | None = None, paper_id: str | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM reproduction_scores WHERE 1=1"
        params: list[Any] = []
        if version:
            query += " AND version = ?"
            params.append(version)
        if paper_id:
            query += " AND paper_id = ?"
            params.append(paper_id)
        query += " ORDER BY timestamp DESC"
        rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_innovation_scores(
        self, version: str | None = None, paper_id: str | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM innovation_scores WHERE 1=1"
        params: list[Any] = []
        if version:
            query += " AND version = ?"
            params.append(version)
        if paper_id:
            query += " AND paper_id = ?"
            params.append(paper_id)
        query += " ORDER BY timestamp DESC"
        rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_elo_ratings(self) -> list[EloRating]:
        rows = self._conn.execute(
            "SELECT * FROM elo_ratings ORDER BY rating DESC"
        ).fetchall()
        return [
            EloRating(
                version=r["version"], rating=r["rating"],
                matches_played=r["matches_played"],
                wins=r["wins"], losses=r["losses"], draws=r["draws"],
            )
            for r in rows
        ]

    def get_ab_tests(
        self, version_a: str | None = None, version_b: str | None = None,
    ) -> list[ABTestResult]:
        query = "SELECT data_json FROM ab_tests WHERE 1=1"
        params: list[Any] = []
        if version_a:
            query += " AND version_a = ?"
            params.append(version_a)
        if version_b:
            query += " AND version_b = ?"
            params.append(version_b)
        query += " ORDER BY timestamp DESC"
        rows = self._conn.execute(query, params).fetchall()
        return [ABTestResult.model_validate_json(r["data_json"]) for r in rows]

    def get_version_summary(self, version: str) -> dict[str, Any]:
        """Aggregate stats for a version across all papers."""
        repro = self.get_reproduction_scores(version=version)
        innov = self.get_innovation_scores(version=version)
        return {
            "version": version,
            "reproduction_runs": len(repro),
            "mean_composite": (
                sum(r["composite"] for r in repro) / len(repro) if repro else 0
            ),
            "mean_metric_match": (
                sum(r["metric_match"] for r in repro) / len(repro) if repro else 0
            ),
            "build_success_rate": (
                sum(r["build_success"] for r in repro) / len(repro) if repro else 0
            ),
            "innovation_runs": len(innov),
            "mean_hypothesis_quality": (
                sum(r["mean_hypothesis_quality"] for r in innov) / len(innov)
                if innov else 0
            ),
            "mean_integrity_pass_rate": (
                sum(r["integrity_pass_rate"] for r in innov) / len(innov)
                if innov else 0
            ),
        }
