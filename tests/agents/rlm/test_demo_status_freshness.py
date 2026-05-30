"""Phase 2 — demo_status.json::updatedAt advances during a live run (FM-005)."""
from __future__ import annotations

import json


def test_stamp_demo_status_advances_updatedat(tmp_path):
    from backend.agents.rlm.run import _stamp_demo_status_updated

    p = tmp_path / "demo_status.json"
    p.write_text(json.dumps({"status": "running", "updatedAt": "2026-05-30T00:00:00+00:00"}))

    _stamp_demo_status_updated(tmp_path)

    after = json.loads(p.read_text())["updatedAt"]
    assert after != "2026-05-30T00:00:00+00:00"
    # Other fields are preserved.
    assert json.loads(p.read_text())["status"] == "running"


def test_stamp_demo_status_missing_file_is_fail_soft(tmp_path):
    from backend.agents.rlm.run import _stamp_demo_status_updated

    # No demo_status.json present — must not raise.
    _stamp_demo_status_updated(tmp_path)


def test_stamp_demo_status_corrupt_file_is_fail_soft(tmp_path):
    from backend.agents.rlm.run import _stamp_demo_status_updated

    (tmp_path / "demo_status.json").write_text("{ not json")
    _stamp_demo_status_updated(tmp_path)  # must not raise
