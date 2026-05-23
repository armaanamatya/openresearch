"""Chat-steering endpoint: POST /runs/{project_id}/messages.

Appends a user message to runs/<id>/user_messages.jsonl and emits a
`user_message` dashboard event so the SSE stream picks it up.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


def _runs_root() -> Path:
    """Mirror the logic in app.py create_app for resolving runs_root."""
    import os as _os
    from backend.config import get_settings as _gs
    s = _gs()
    env_val = _os.environ.get("REPROLAB_RUNS_ROOT")
    if s.runs_root is not None:
        return Path(s.runs_root)
    if env_val:
        return Path(env_val)
    return Path(__file__).resolve().parents[2] / "runs"


class UserMessageIn(BaseModel):
    role: Literal["user"]
    content: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@router.post("/runs/{project_id}/messages", status_code=202)
async def post_message(project_id: str, body: UserMessageIn) -> dict:
    """Append a user message to the run's user_messages.jsonl.

    Validates that the run directory exists (404 otherwise) and that
    content is non-empty (400 otherwise). Returns {"ok": true} on success.
    Emits a `user_message` dashboard event so the SSE stream surfaces it.
    """
    if not body.content or not body.content.strip():
        raise HTTPException(status_code=400, detail="content must be non-empty")

    run_dir = _runs_root() / project_id
    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail="Run not found")

    ts = _now_iso()
    message_entry = {"role": "user", "content": body.content, "ts": ts}
    dashboard_entry = {"event": "user_message", "timestamp": ts, **message_entry}

    messages_path = run_dir / "user_messages.jsonl"
    dashboard_path = run_dir / "dashboard_events.jsonl"

    # Atomic append: open(..., 'a') is safe for single-line JSONL appends
    # (POSIX guarantees atomicity for small writes below PIPE_BUF).
    with messages_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(message_entry, default=str) + "\n")

    with dashboard_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(dashboard_entry, default=str) + "\n")

    return {"ok": True}
