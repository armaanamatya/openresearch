"""2026-05-30: archive_run_artifacts must move worker_reports.jsonl + reports/.

Previously these were left in place, so old + new worker rows commingled across
attempts and per-run failure analysis read stale rows as the current attempt's.
"""
from __future__ import annotations

from pathlib import Path

from backend.services.runs.archive import archive_run_artifacts


def test_worker_reports_and_reports_dir_are_archived(tmp_path: Path) -> None:
    runs_root = tmp_path
    run = runs_root / "prj_test"
    run.mkdir()
    # trigger + the artifacts under test
    (run / "final_report.json").write_text("{}")
    (run / "worker_reports.jsonl").write_text('{"agent_id":"x","status":"failed"}\n')
    (run / "reports").mkdir()
    (run / "reports" / "summary_report.json").write_text("{}")
    # a paper-level artifact that must NOT move
    (run / "generated_rubric.json").write_text("{}")

    res = archive_run_artifacts("prj_test", runs_root)
    assert res is not None
    attempt = Path(res["attempt_dir"])

    # moved out of the run root
    assert not (run / "worker_reports.jsonl").exists()
    assert not (run / "reports").exists()
    # landed under attempts/<ts>/
    assert (attempt / "worker_reports.jsonl").exists()
    assert (attempt / "reports" / "summary_report.json").exists()
    assert "worker_reports.jsonl" in res["moved"]
    assert "reports/" in res["moved"]
    # paper-level artifact preserved
    assert (run / "generated_rubric.json").exists()
