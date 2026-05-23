"""Tests for two SSE-streaming I/O efficiency fixes.

D1: JSON-serialize once per loop iteration.
    - model_dump_json() is called once per loop tick; the result is reused
      for change detection AND for the SSE payload (via json.loads).

D2: Byte-offset reader for dashboard events.
    - _read_dashboard_events(project_id, byte_offset=N) seeks to byte N
      and reads only new lines instead of reading the entire file every
      poll cycle.
    - The public API uses a byte_offset int cursor (low-disruption; the
      stream_events loop now passes a byte cursor, not a line count).

Resume-safety (D3): An SSE client reconnecting with byte_offset=0 sees
all events from the start of the file.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_events(path: Path, count: int, *, start: int = 0) -> int:
    """Write `count` JSON events to `path`, return the total byte length after write."""
    with path.open("a", encoding="utf-8") as fh:
        for i in range(start, start + count):
            fh.write(json.dumps({"event": "test", "idx": i}) + "\n")
    return path.stat().st_size


# ---------------------------------------------------------------------------
# D2: Byte-offset reader
# ---------------------------------------------------------------------------


class TestByteOffsetReader:
    """_read_dashboard_events honours a byte-offset cursor instead of line-offset."""

    def test_returns_all_events_from_offset_zero(self, tmp_path: Path) -> None:
        """Reading from byte_offset=0 returns every event in the file."""
        events_path = tmp_path / "dashboard_events.jsonl"
        _write_events(events_path, 100)

        # Create a service pointing at this tmp dir as if it were a run dir.
        from backend.services.events.live_runs import FileLiveRunService

        svc = FileLiveRunService(runs_root=tmp_path.parent)
        project_id = tmp_path.name

        events, new_offset = svc._read_dashboard_events(project_id, byte_offset=0)
        assert len(events) == 100, f"Expected 100 events, got {len(events)}"
        assert new_offset > 0, "new_offset should advance past the file content"

    def test_returns_only_new_events_from_nonzero_byte_offset(self, tmp_path: Path) -> None:
        """Reading from mid-file byte_offset skips already-consumed bytes."""
        events_path = tmp_path / "dashboard_events.jsonl"
        # Write 50 events and capture the byte offset after them.
        mid_offset = _write_events(events_path, 50)
        # Write 50 more events.
        _write_events(events_path, 50, start=50)

        from backend.services.events.live_runs import FileLiveRunService

        svc = FileLiveRunService(runs_root=tmp_path.parent)
        project_id = tmp_path.name

        events, new_offset = svc._read_dashboard_events(project_id, byte_offset=mid_offset)
        assert len(events) == 50, f"Expected 50 new events, got {len(events)}"
        # All returned events should be from the second batch (idx >= 50).
        for ev in events:
            assert ev["idx"] >= 50, f"Got pre-offset event: {ev}"
        assert new_offset > mid_offset, "new_offset should advance past the newly-read bytes"

    def test_no_full_file_read_text_when_seeking(self, tmp_path: Path) -> None:
        """read_text() must NOT be called when byte_offset > 0 (byte-seek path)."""
        events_path = tmp_path / "dashboard_events.jsonl"
        mid_offset = _write_events(events_path, 100)
        _write_events(events_path, 10, start=100)

        from backend.services.events.live_runs import FileLiveRunService

        svc = FileLiveRunService(runs_root=tmp_path.parent)
        project_id = tmp_path.name

        # Patch Path.read_text to raise if called — the implementation must
        # use seek+read for byte_offset > 0 and never fall back to read_text.
        with patch.object(Path, "read_text", side_effect=AssertionError("read_text called — should use seek")):
            events, new_offset = svc._read_dashboard_events(project_id, byte_offset=mid_offset)

        assert len(events) == 10, f"Expected 10 events, got {len(events)}"

    def test_resume_from_offset_zero_sees_all_events(self, tmp_path: Path) -> None:
        """Reconnect from byte_offset=0 (resume safety) still returns full history."""
        events_path = tmp_path / "dashboard_events.jsonl"
        _write_events(events_path, 25)

        from backend.services.events.live_runs import FileLiveRunService

        svc = FileLiveRunService(runs_root=tmp_path.parent)
        project_id = tmp_path.name

        events, _ = svc._read_dashboard_events(project_id, byte_offset=0)
        assert len(events) == 25

    def test_returns_new_byte_offset(self, tmp_path: Path) -> None:
        """_read_dashboard_events returns (events, new_byte_offset) tuple."""
        events_path = tmp_path / "dashboard_events.jsonl"
        _write_events(events_path, 10)
        file_size = events_path.stat().st_size

        from backend.services.events.live_runs import FileLiveRunService

        svc = FileLiveRunService(runs_root=tmp_path.parent)
        project_id = tmp_path.name

        result = svc._read_dashboard_events(project_id, byte_offset=0)
        # The new API must return a (list, int) tuple so the caller can advance
        # its cursor to the end of the file.
        assert isinstance(result, tuple), (
            "_read_dashboard_events must return (events, new_byte_offset) tuple"
        )
        events, new_offset = result
        assert len(events) == 10
        assert new_offset == file_size, (
            f"new_offset {new_offset} != file size {file_size}"
        )

    def test_empty_file_returns_zero_offset(self, tmp_path: Path) -> None:
        """Non-existent file returns ([], 0)."""
        from backend.services.events.live_runs import FileLiveRunService

        svc = FileLiveRunService(runs_root=tmp_path.parent)
        project_id = tmp_path.name

        result = svc._read_dashboard_events(project_id, byte_offset=0)
        assert isinstance(result, tuple)
        events, offset = result
        assert events == []
        assert offset == 0


# ---------------------------------------------------------------------------
# D1: JSON-serialization caching
# ---------------------------------------------------------------------------


class TestJsonSerializationCaching:
    """stream_events calls model_dump_json() once per tick, not twice.

    Before D1 fix:
      - initial setup: model_dump(mode="json") + model_dump_json()  → 2 serializations
      - each loop tick that detects a change:
          model_dump_json() for comparison + model_dump(mode="json") for payload → 2 serializations

    After D1 fix:
      - initial setup: model_dump_json() once; json.loads(json_str) for payload → 1 serialization
      - each loop tick: model_dump_json() once; json.loads(cached) for payload → 1 serialization
    """

    @pytest.mark.asyncio
    async def test_no_model_dump_dict_call_on_initial_state(
        self, tmp_path: Path
    ) -> None:
        """Initial state should NOT call model_dump(mode='json') separately —
        it must derive the SSE payload from the JSON string produced by model_dump_json()."""
        from backend.services.events.live_runs import FileLiveRunService, LiveRunState

        runs_root = tmp_path / "runs"
        runs_root.mkdir()
        project_id = "prj_test_json_initial"
        run_dir = runs_root / project_id
        run_dir.mkdir()

        status = {
            "projectId": project_id,
            "outputDir": str(run_dir),
            "runMode": "rlm",
            "status": "completed",
            "startedAt": "2026-01-01T00:00:00+00:00",
            "updatedAt": "2026-01-01T00:00:00+00:00",
        }
        (run_dir / "demo_status.json").write_text(json.dumps(status), encoding="utf-8")

        svc = FileLiveRunService(runs_root=runs_root)

        model_dump_calls = [0]
        original_model_dump = LiveRunState.model_dump

        def _counting_model_dump(self, *args, **kwargs):
            model_dump_calls[0] += 1
            return original_model_dump(self, *args, **kwargs)

        with patch.object(LiveRunState, "model_dump", _counting_model_dump):
            events = []
            async for ev in svc.stream_events(project_id):
                events.append(ev)

        # After D1 fix: model_dump(mode="json") should NOT be called at all for the
        # initial SSE payload — the implementation should use json.loads(json_str).
        assert model_dump_calls[0] == 0, (
            f"model_dump() called {model_dump_calls[0]} times — "
            "after D1 fix, the SSE payload should use json.loads(model_dump_json()) "
            "not model_dump(mode='json') (eliminates a redundant serialize+deserialize)"
        )

    @pytest.mark.asyncio
    async def test_no_model_dump_dict_call_in_loop_on_change(
        self, tmp_path: Path
    ) -> None:
        """In the poll loop when state changes, model_dump(mode='json') must NOT be
        called — the SSE payload must use json.loads(cached_json_string) instead."""
        import asyncio
        from backend.services.events.live_runs import FileLiveRunService, LiveRunState

        runs_root = tmp_path / "runs"
        runs_root.mkdir()
        project_id = "prj_test_json_loop"
        run_dir = runs_root / project_id
        run_dir.mkdir()

        status_running = {
            "projectId": project_id,
            "outputDir": str(run_dir),
            "runMode": "rlm",
            "status": "running",
            "startedAt": "2026-01-01T00:00:00+00:00",
            "updatedAt": "2026-01-01T00:00:00+00:00",
        }
        status_completed = {**status_running, "status": "completed"}

        status_path = run_dir / "demo_status.json"
        status_path.write_text(json.dumps(status_running), encoding="utf-8")

        svc = FileLiveRunService(runs_root=runs_root)

        model_dump_calls = [0]
        tick = [0]

        async def _fast_sleep(t):
            tick[0] += 1
            if tick[0] >= 1:
                status_path.write_text(json.dumps(status_completed), encoding="utf-8")

        original_model_dump = LiveRunState.model_dump

        def _counting_model_dump(self, *args, **kwargs):
            model_dump_calls[0] += 1
            return original_model_dump(self, *args, **kwargs)

        with (
            patch("asyncio.sleep", side_effect=_fast_sleep),
            patch.object(LiveRunState, "model_dump", _counting_model_dump),
        ):
            events = []
            async for ev in svc.stream_events(project_id):
                events.append(ev)

        # After D1 fix: model_dump(mode="json") should never be called.
        # The loop detects change via model_dump_json() string comparison and
        # builds the SSE payload via json.loads(cached_json).
        assert model_dump_calls[0] == 0, (
            f"model_dump() called {model_dump_calls[0]} times in stream_events — "
            "after D1 fix, must be 0; use json.loads(model_dump_json()) for the payload"
        )
