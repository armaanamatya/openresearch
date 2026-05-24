"""Worker reports HTTP router.

GET /runs/{project_id}/reports → {workers: [...], summary: {...}}

Returns 200 with empty arrays when the reports directory doesn't exist.
Not gated by REPROLAB_DEMO_SECRET (read-only introspection data).
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Query

from backend.agents.worker_reports import (
    get_or_build_summary,
    read_worker_reports,
)
from backend.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter()


def _runs_root() -> Path:
    import os
    s = get_settings()
    env_val = os.environ.get("REPROLAB_RUNS_ROOT")
    if s.runs_root is not None:
        return Path(s.runs_root)
    if env_val:
        return Path(env_val)
    return Path(__file__).resolve().parents[2] / "runs"


@router.get("/runs/{project_id}/reports")
def get_worker_reports(
    project_id: str,
    limit: int = Query(default=100, ge=1, le=500),
):
    """Return worker reports and summary for a run.

    200 with empty arrays on missing dir; never 500.
    """
    run_dir = _runs_root() / project_id
    if not run_dir.is_dir():
        return {"workers": [], "summary": {}}

    try:
        workers = read_worker_reports(run_dir)[:limit]
        summary = get_or_build_summary(run_dir)
    except Exception as exc:  # noqa: BLE001
        logger.warning("reports: error reading reports for %s: %s", project_id, exc)
        return {"workers": [], "summary": {}}

    return {"workers": workers, "summary": summary}


__all__ = ["router"]
