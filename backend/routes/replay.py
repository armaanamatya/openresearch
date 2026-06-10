"""Run replay HTTP router.

GET /runs/{project_id}/replay-events → {events: [...], metadata: {count, earliestTs, latestTs}}

Returns the persisted dashboard events (``dashboard_events.jsonl``) in append order so the
lab UI can replay a completed run's timeline (iterations, primitive calls, rubric climb,
candidates) with no compute — the replay driver paces them by their ``timestamp`` deltas.

200 with an empty list when the run/dir/file doesn't exist; never 500. Not gated by
REPROLAB_DEMO_SECRET (read-only introspection, same posture as the reports router).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, Query

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


def _read_events(path: Path, limit: int) -> list[dict]:
    """Read append-only JSONL events in order, tolerating a torn final line."""
    events: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                # A concurrent writer may leave a partial last line; skip it.
                continue
    if len(events) > limit:
        # Preserve order; keep the most recent `limit` (the replay tail).
        events = events[-limit:]
    return events


def _event_ts(e: dict) -> str | None:
    ts = e.get("timestamp")
    if ts:
        return ts
    data = e.get("data")
    if isinstance(data, dict):
        return data.get("timestamp")
    return None


@router.get("/runs/{project_id}/replay-events")
def get_replay_events(
    project_id: str,
    limit: int = Query(default=100000, ge=1, le=100000),
):
    """Return persisted dashboard events in append order for UI timeline replay.

    200 with an empty list on missing dir/file; never 500.
    """
    events_path = _runs_root() / project_id / "dashboard_events.jsonl"
    empty = {"events": [], "metadata": {"count": 0, "earliestTs": None, "latestTs": None}}
    if not events_path.is_file():
        return empty

    try:
        events = _read_events(events_path, limit)
    except Exception as exc:  # noqa: BLE001
        logger.warning("replay: error reading events for %s: %s", project_id, exc)
        return empty

    timestamps = [t for t in (_event_ts(e) for e in events) if t]
    return {
        "events": events,
        "metadata": {
            "count": len(events),
            "earliestTs": timestamps[0] if timestamps else None,
            "latestTs": timestamps[-1] if timestamps else None,
        },
    }


__all__ = ["router"]
