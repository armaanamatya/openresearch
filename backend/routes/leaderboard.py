"""Leaderboard aggregator + HTTP router.

Scans `runs/<id>/final_report.json` (+ `demo_status.json` for sandbox /
status fallback) and returns ranked rows. Read-only; not gated by
REPROLAB_DEMO_SECRET per spec §3 #10.

Spec: docs/superpowers/specs/2026-05-23-rubric-climb-leaderboard.md §4.4–§4.5.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.config import get_settings
from backend.services.events.leaderboard_cache import evict_missing, get_or_load

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
    title: str | None = None
    mode: Literal["rlm", "rdr"] = "rlm"
    models: RoleModels = Field(default_factory=RoleModels)
    overall_score: float | None
    # β3: compute_adjusted_score — floor-anchored score on clipped runs.
    # Equals overall_score on max-mode runs (always-emit semantic).
    # None on very old reports written before this field was added.
    compute_adjusted_score: float | None = None
    # β4: execution_mode — "efficient" | "max" | None (legacy).
    execution_mode: str | None = None
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
    data = get_or_load(run_dir.name, fr_path)
    if data is None:
        logger.warning(
            "leaderboard: skipping %s — unreadable final_report",
            run_dir.name,
        )
        return None

    # Defensive: legacy/test final_report.json files (e.g. prj_verify_offline_report)
    # had rubric as a list-of-areas instead of the {overall_score, meets_target,
    # areas} dict. `data.get("rubric") or {}` keeps the list (truthy), then the
    # subsequent `.get()` blows up with `'list' object has no attribute 'get'` —
    # 500ing the whole leaderboard. Coerce to {} when the shape is wrong so a
    # malformed row gets a None score but doesn't fail the entire aggregation.
    def _as_dict(v):
        return v if isinstance(v, dict) else {}
    paper = _as_dict(data.get("paper"))
    rubric = _as_dict(data.get("rubric"))
    cost = _as_dict(data.get("cost"))
    models = _as_dict(data.get("models"))
    started_at = data.get("started_at")
    completed_at = data.get("completed_at")
    # β4: read execution_mode from demo_status.json (executionMode field) first,
    # then fall back to the run's direct demo_status read if it's in the data.
    # demo_status.json isn't loaded here — read it lazily from disk.
    _execution_mode: str | None = None
    _demo_status_path = run_dir / "demo_status.json"
    if _demo_status_path.is_file():
        try:
            import json as _json_m
            _ds = _json_m.loads(_demo_status_path.read_text(encoding="utf-8"))
            _execution_mode = _ds.get("executionMode") or _ds.get("execution_mode")
        except Exception:
            pass

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
    # Propagate None on the row so the UI can tell the difference between
    # "scored 0" and "not scored". The aggregator's sort_key coerces None to
    # the lowest rank below.
    score_raw = rubric.get("overall_score")
    overall_score: float | None = float(score_raw) if score_raw is not None else None

    # β3: compute_adjusted_score — floor-anchored score on clipped runs.
    # Falls back to overall_score for old reports without the field.
    _adj_raw = rubric.get("compute_adjusted_score")
    compute_adjusted_score: float | None = (
        float(_adj_raw) if _adj_raw is not None else overall_score
    )

    return LeaderboardRow(
        project_id=run_dir.name,
        paper_id=str(paper.get("id") or run_dir.name),
        paper_title=paper.get("title"),
        title=data.get("title"),
        mode=data.get("mode", "rlm"),
        models=RoleModels(
            planner=models.get("planner"),
            executor=models.get("executor"),
            verifier=models.get("verifier"),
            grader=models.get("grader"),
        ),
        overall_score=overall_score,
        compute_adjusted_score=compute_adjusted_score,
        execution_mode=_execution_mode,
        meets_target=bool(rubric.get("meets_target") or False),
        degraded=bool(data.get("degraded") or rubric.get("degraded") or False),
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

    # Drop cache entries for run directories that have been deleted since the
    # last request (D6 — lazy eviction).
    evict_missing(runs_root)

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
            # β3: default sort by compute_adjusted_score so efficient and max
            # runs are comparable. Falls back to overall_score when not present.
            adj = r.compute_adjusted_score if r.compute_adjusted_score is not None else r.overall_score
            return (adj is None, -(adj or 0.0))
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
    # Reuse the existing REPROLAB_RUNS_ROOT setting (bound on Settings.runs_root)
    # rather than introducing a parallel REPROLAB_RUNS_DIR.
    runs_root = settings.runs_root if settings.runs_root is not None else Path("runs")
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
