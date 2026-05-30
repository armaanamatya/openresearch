"""Phase 7b — SSE stream stops polling when the client disconnects.

The raw StreamingResponse generator (live_runs.py:696) previously looped on run
status only, with no is_disconnected() check — a closed browser tab left the
per-viewer file-poll loop running until the run itself ended (FM-005 resource leak).
"""
from __future__ import annotations

import json

import pytest

from backend.services.events.live_runs import FileLiveRunService


class _FakeRequest:
    """Duck-typed Starlette Request: disconnects after `connected_polls` checks."""

    def __init__(self, connected_polls: int = 0) -> None:
        self._remaining = connected_polls

    async def is_disconnected(self) -> bool:
        if self._remaining <= 0:
            return True
        self._remaining -= 1
        return False


def _make_running_run(tmp_path):
    run = tmp_path / "prj_x"
    run.mkdir()
    (run / "demo_status.json").write_text(
        json.dumps({
            "projectId": "prj_x", "status": "running", "outputDir": str(run),
            "runMode": "rlm", "startedAt": "2026-05-30T00:00:00Z",
        })
    )
    return run


@pytest.mark.asyncio
async def test_stream_stops_on_disconnect(tmp_path):
    _make_running_run(tmp_path)
    svc = FileLiveRunService(runs_root=tmp_path)
    req = _FakeRequest(connected_polls=0)  # disconnected on the first loop check

    events = []
    async for chunk in svc.stream_events("prj_x", request=req):
        events.append(chunk)
        if len(events) > 50:  # safety: must terminate well before this
            pytest.fail("stream did not stop on disconnect")

    # We got the initial run_state etc., then the loop exited promptly.
    assert any("run_state" in e for e in events)


@pytest.mark.asyncio
async def test_stream_without_request_still_works(tmp_path):
    """Backward compat: a completed run streams and terminates with no request."""
    run = tmp_path / "prj_done"
    run.mkdir()
    (run / "demo_status.json").write_text(
        json.dumps({
            "projectId": "prj_done", "status": "completed",
            "outputDir": str(run), "runMode": "rlm",
        })
    )
    svc = FileLiveRunService(runs_root=tmp_path)
    events = [chunk async for chunk in svc.stream_events("prj_done")]
    assert any("run_state" in e for e in events)
