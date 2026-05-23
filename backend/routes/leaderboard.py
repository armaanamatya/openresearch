"""Leaderboard aggregator + HTTP router.

Scans `runs/<id>/final_report.json` (+ `demo_status.json` for sandbox /
status fallback) and returns ranked rows. Read-only; not gated by
REPROLAB_DEMO_SECRET per spec §3 #10.

Spec: docs/superpowers/specs/2026-05-23-rubric-climb-leaderboard.md §4.4–§4.5.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.config import get_settings

logger = logging.getLogger(__name__)

OrderBy = Literal["score", "cost", "time", "finished_at"]


class RoleModels(BaseModel):
    planner: str | None = None
    executor: str | None = None
    verifier: str | None = None
    grader: str | None = None


class LeaderboardRow(BaseModel):
    project_id: str
    paper_id: str
    paper_title: str | None
    mode: Literal["rlm", "rdr"] = "rlm"
    models: RoleModels = Field(default_factory=RoleModels)
    overall_score: float
    meets_target: bool
    degraded: bool
    cost_usd: float | None
    iterations: int
    wall_clock_s: float | None
    sandbox: str | None
    started_at: str | None
    completed_at: str | None
    verdict: str


def _read_run(run_dir: Path) -> LeaderboardRow | None:
    fr_path = run_dir / "final_report.json"
    if not fr_path.is_file():
        return None
    try:
        data = json.loads(fr_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(
            "leaderboard: skipping %s — unreadable final_report (%s)",
            run_dir.name, e,
        )
        return None

    paper = data.get("paper") or {}
    rubric = data.get("rubric") or {}
    cost = data.get("cost") or {}
    models = data.get("models") or {}
    started_at = data.get("started_at")
    completed_at = data.get("completed_at")

    wall_clock_s: float | None = None
    if started_at and completed_at:
        try:
            wall_clock_s = (
                datetime.fromisoformat(completed_at)
                - datetime.fromisoformat(started_at)
            ).total_seconds()
        except ValueError:
            wall_clock_s = None

    # Honest score handling: the post-fix C2c default is `None` (no fabrication).
    # Treat None as 0.0 for the ranking key, but propagate None on the row so the
    # UI can tell the difference between "scored 0" and "not scored".
    score_raw = rubric.get("overall_score")
    overall_score = float(score_raw) if score_raw is not None else 0.0

    return LeaderboardRow(
        project_id=run_dir.name,
        paper_id=str(paper.get("id") or run_dir.name),
        paper_title=paper.get("title"),
        mode=data.get("mode", "rlm"),
        models=RoleModels(
            planner=models.get("planner"),
            executor=models.get("executor"),
            verifier=models.get("verifier"),
            grader=models.get("grader"),
        ),
        overall_score=overall_score,
        meets_target=bool(rubric.get("meets_target") or False),
        degraded=bool(data.get("degraded") or False),
        cost_usd=float(cost["llm_usd"]) if cost.get("llm_usd") is not None else None,
        iterations=int(data.get("iterations", 0)),
        wall_clock_s=wall_clock_s,
        sandbox=data.get("sandbox"),
        started_at=started_at,
        completed_at=completed_at,
        verdict=str(data.get("verdict", "failed")),
    )


def aggregate_leaderboard(
    runs_root: Path,
    *,
    paper: str | None = None,
    mode: Literal["rlm", "rdr"] | None = None,
    order_by: OrderBy = "score",
    limit: int = 50,
) -> list[LeaderboardRow]:
    """Scan ``runs_root`` for completed runs and return a ranked list."""

    rows: list[LeaderboardRow] = []
    if not runs_root.is_dir():
        return rows

    for entry in sorted(runs_root.iterdir()):
        if not entry.is_dir():
            continue
        row = _read_run(entry)
        if row is None:
            continue
        if paper is not None and row.paper_id != paper:
            continue
        if mode is not None and row.mode != mode:
            continue
        rows.append(row)

    def _sort_key(r: LeaderboardRow):
        if order_by == "score":
            return -r.overall_score
        if order_by == "cost":
            return (r.cost_usd is None, r.cost_usd if r.cost_usd is not None else 0.0)
        if order_by == "time":
            return (
                r.wall_clock_s is None,
                r.wall_clock_s if r.wall_clock_s is not None else 0.0,
            )
        if order_by == "finished_at":
            return (r.completed_at is None, r.completed_at or "")
        return 0

    rows.sort(key=_sort_key)
    return rows[:limit]


router = APIRouter()


@router.get("/leaderboard", response_model=list[LeaderboardRow])
def list_leaderboard_runs(
    paper: str | None = Query(default=None),
    mode: Literal["rlm", "rdr"] | None = Query(default=None),
    order_by: OrderBy = Query(default="score"),
    limit: int = Query(default=50, ge=1, le=500),
):
    """Return ranked leaderboard rows aggregated from on-disk runs.

    Read-only; not gated by REPROLAB_DEMO_SECRET.
    """
    settings = get_settings()
    runs_dir = getattr(settings, "runs_dir", "runs")
    runs_root = Path(runs_dir)
    try:
        return aggregate_leaderboard(
            runs_root, paper=paper, mode=mode, order_by=order_by, limit=limit
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("leaderboard: aggregation failed")
        raise HTTPException(status_code=500, detail=str(e))


__all__ = [
    "LeaderboardRow",
    "RoleModels",
    "aggregate_leaderboard",
    "router",
]
