"""E3 — loud-fail-soft sweep: the degradations_taken ledger (2026-06-16).

_collect_degradations aggregates the run's coded degradation run_warning events
(the channel they already emit on) into [{code, count, last_message}]. Advisory /
improvement warnings are excluded; a clean run returns [] so the report omits the
field (byte-for-byte today).
"""

from __future__ import annotations

import json

from backend.agents.rlm.report import _collect_degradations


def _events(run_dir, lines):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "dashboard_events.jsonl").write_text(
        "\n".join(json.dumps(line) for line in lines), encoding="utf-8"
    )


def test_no_events_returns_empty(tmp_path):
    assert _collect_degradations(tmp_path) == []


def test_only_degradation_codes_collected_and_deduped(tmp_path):
    _events(tmp_path, [
        {"event": "run_warning", "payload": {"code": "primitive_timeout", "message": "m1"}},
        {"event": "run_warning", "payload": {"code": "primitive_timeout", "message": "m2"}},
        {"event": "run_warning", "payload": {"code": "cells_manifest_restored", "message": "restored"}},
        # advisory (not a degrade) → excluded from the ledger
        {"event": "run_warning", "payload": {"code": "iteration_boundary_recommended", "message": "advice"}},
        # not a warning at all
        {"event": "rubric_score", "payload": {"overall_score": 0.7}},
    ])
    out = {r["code"]: r for r in _collect_degradations(tmp_path)}
    assert set(out) == {"primitive_timeout", "cells_manifest_restored"}
    assert out["primitive_timeout"]["count"] == 2
    assert out["primitive_timeout"]["last_message"] == "m2"  # newest message wins
    assert "iteration_boundary_recommended" not in out  # advisory excluded


def test_malformed_lines_are_failsoft(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "dashboard_events.jsonl").write_text("not json\n{bad\n", encoding="utf-8")
    assert _collect_degradations(tmp_path) == []
