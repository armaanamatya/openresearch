"""WS1-H1: the cost-summary daemon must not republish a stale non-terminal
demo_status snapshot over a terminal one. Flag-gated
(OPENRESEARCH_DEMO_STATUS_TERMINAL_GUARD), default-OFF byte-identical.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from backend.agents.rlm import run as run_mod


def _write_status(path: Path, status: str, **extra) -> None:
    path.write_text(json.dumps({"status": status, "projectId": "x", **extra}))


def test_guard_off_by_default(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OPENRESEARCH_DEMO_STATUS_TERMINAL_GUARD", raising=False)
    sp = tmp_path / "demo_status.json"
    _write_status(sp, "completed")
    assert run_mod._should_skip_stale_republish(sp) is False  # OFF ⇒ never skips


def test_guard_on_skips_when_terminal(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OPENRESEARCH_DEMO_STATUS_TERMINAL_GUARD", "1")
    sp = tmp_path / "demo_status.json"
    for terminal in ("completed", "failed", "stopped", "killed", "interrupted"):
        _write_status(sp, terminal)
        assert run_mod._should_skip_stale_republish(sp) is True, terminal


def test_guard_on_allows_when_running(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OPENRESEARCH_DEMO_STATUS_TERMINAL_GUARD", "1")
    sp = tmp_path / "demo_status.json"
    _write_status(sp, "running")
    assert run_mod._should_skip_stale_republish(sp) is False  # a live run still writes


def test_guard_on_missing_file_is_false(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OPENRESEARCH_DEMO_STATUS_TERMINAL_GUARD", "1")
    assert run_mod._should_skip_stale_republish(tmp_path / "nope.json") is False


def test_loop_preserves_a_concurrent_terminal_write(tmp_path: Path, monkeypatch) -> None:
    """End-to-end: a terminal write lands mid-iteration; ON ⇒ it survives."""
    monkeypatch.setenv("OPENRESEARCH_DEMO_STATUS_TERMINAL_GUARD", "1")
    sp = tmp_path / "demo_status.json"
    _write_status(sp, "running")
    stop = threading.Event()

    def _finalize_lands_terminal(project_dir, cur_iter):
        # Simulate _finalize's terminal write landing after the daemon read
        # `existing` but before its os.replace.
        _write_status(sp, "completed", verdict="reproduced")
        stop.set()
        return {"llm_usd": 0.0}

    monkeypatch.setattr(run_mod, "_compute_cost_summary", _finalize_lands_terminal)
    run_mod._update_cost_summary_loop(tmp_path, stop, lambda: 0, interval_s=0.01)

    data = json.loads(sp.read_text())
    assert data["status"] == "completed"        # stale "running" never republished
    assert data.get("verdict") == "reproduced"  # terminal payload intact
