from pathlib import Path
from backend.agents.rlm.archive_completeness import check_bes_archive, REQUIRED_ARTIFACTS


def test_incomplete_archive_is_rejected(tmp_path):
    (tmp_path / "final_report.json").write_text("{}")
    res = check_bes_archive(tmp_path)
    assert res.complete is False
    assert "dashboard_events.jsonl" in res.missing


def test_complete_archive_passes(tmp_path):
    for name in REQUIRED_ARTIFACTS:
        (tmp_path / name).write_text("{}")
    (tmp_path / "candidates").mkdir()
    res = check_bes_archive(tmp_path)
    assert res.complete is True
    assert res.missing == []
