"""Chat-steering endpoints: POST /runs/{project_id}/messages and its campaign
sibling POST /runs/{project_id}/campaign/messages.

Appends a user message to runs/<id>/user_messages.jsonl and emits a
`user_message` dashboard event so the SSE stream picks it up. The campaign
route (F13, spec §10.6) mirrors the same demo-gate + 404/400 semantics but
writes to the campaign's own steering channel
(runs/<id>/campaign/user_messages.jsonl) so a per-attempt archive can never
carry away cross-attempt operator state.
"""

from __future__ import annotations

import json
import hmac
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from backend.config import get_settings

router = APIRouter()


def _runs_root() -> Path:
    """Mirror the logic in app.py create_app for resolving runs_root."""
    import os as _os
    from backend.config import get_settings as _gs
    s = _gs()
    env_val = _os.environ.get("OPENRESEARCH_RUNS_ROOT")
    if s.runs_root is not None:
        return Path(s.runs_root)
    if env_val:
        return Path(env_val)
    return Path(__file__).resolve().parents[2] / "runs"


class UserMessageIn(BaseModel):
    role: Literal["user"]
    content: str


class CampaignMessageIn(BaseModel):
    op: Literal["set_mode", "note"]
    mode: Literal["unattended", "checkpoint"] | None = None
    content: str = ""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _enforce_demo_gate(provided_secret: str | None, configured_secret: str) -> None:
    if not configured_secret:
        return
    if not provided_secret or not hmac.compare_digest(provided_secret, configured_secret):
        raise HTTPException(status_code=401, detail="A valid demo access secret is required.")


@router.post("/runs/{project_id}/messages", status_code=202)
async def post_message(
    project_id: str,
    body: UserMessageIn,
    x_demo_secret: str | None = Header(default=None),
) -> dict:
    """Append a user message to the run's user_messages.jsonl.

    Validates that the run directory exists (404 otherwise) and that
    content is non-empty (400 otherwise). Returns {"ok": true} on success.
    Emits a `user_message` dashboard event so the SSE stream surfaces it.
    """
    _enforce_demo_gate(x_demo_secret, get_settings().demo_secret)
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


@router.post("/runs/{project_id}/campaign/messages", status_code=202)
async def post_campaign_message(
    project_id: str,
    body: CampaignMessageIn,
    x_demo_secret: str | None = Header(default=None),
) -> dict:
    """Append an operator steering message to the campaign's own channel.

    Sibling of ``post_message`` (F13, spec §10.6): same demo gate, same
    404-on-missing-run-dir semantics, same validate-before-404 ordering.
    Unlike the run-level channel, ``campaign/`` is created on demand here —
    an operator may pre-steer before the campaign process has started — and
    this route never reads or writes ``campaign.json``; the campaign process
    consumes ``campaign/user_messages.jsonl`` via its own poll against the
    checkpointed ``steering_cursor``.
    """
    _enforce_demo_gate(x_demo_secret, get_settings().demo_secret)

    if body.op == "set_mode" and body.mode is None:
        raise HTTPException(status_code=400, detail="mode is required for op=set_mode")
    if body.op == "note" and not body.content.strip():
        raise HTTPException(status_code=400, detail="content must be non-empty for op=note")

    run_dir = _runs_root() / project_id
    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail="Run not found")

    campaign_dir = run_dir / "campaign"
    campaign_dir.mkdir(parents=True, exist_ok=True)

    ts = _now_iso()
    message_id = uuid4().hex
    message_entry: dict = {"id": message_id, "ts": ts, "op": body.op}
    if body.op == "set_mode":
        message_entry["mode"] = body.mode
    else:
        message_entry["content"] = body.content
    dashboard_entry = {"event": "campaign_user_message", "timestamp": ts, **message_entry}

    messages_path = campaign_dir / "user_messages.jsonl"
    dashboard_path = run_dir / "dashboard_events.jsonl"

    with messages_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(message_entry, default=str) + "\n")

    with dashboard_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(dashboard_entry, default=str) + "\n")

    return {"ok": True, "id": message_id}
