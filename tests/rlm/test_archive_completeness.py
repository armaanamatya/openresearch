from pathlib import Path
from backend.agents.rlm.archive_completeness import check_bes_archive, REQUIRED_ARTIFACTS


def test_incomplete_archive_is_rejected(tmp_path):
    (tmp_path / "final_report.json").write_text("{}")
    res = check_bes_archive(tmp_path)
    assert res.complete is False
    assert "dashboard_events.jsonl" in res.missing


def test_complete_archive_passes(tmp_path):
    # Curated archive layout: all files at the top level (legacy / curated format).
    for name in REQUIRED_ARTIFACTS:
        (tmp_path / name).write_text("{}")
    (tmp_path / "candidates").mkdir()
    res = check_bes_archive(tmp_path)
    assert res.complete is True
    assert res.missing == []


def test_live_run_layout_passes(tmp_path):
    # final_report / dashboard_events / experiment_runs / rubric_evaluation / generated_rubric at top level
    for name in ("final_report.json", "dashboard_events.jsonl", "experiment_runs.jsonl",
                 "rubric_evaluation.json", "generated_rubric.json"):
        (tmp_path / name).write_text("{}")
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "metrics.json").write_text("{}")        # metrics under code/
    (tmp_path / "rlm_state").mkdir()
    (tmp_path / "rlm_state" / "bes_candidates.json").write_text("{}")  # under rlm_state/
    (tmp_path / "candidates").mkdir()
    res = check_bes_archive(tmp_path)
    assert res.complete is True, f"missing: {res.missing}"


def test_missing_metrics_anywhere_is_incomplete(tmp_path):
    for name in ("final_report.json", "dashboard_events.jsonl", "experiment_runs.jsonl",
                 "rubric_evaluation.json", "generated_rubric.json"):
        (tmp_path / name).write_text("{}")
    (tmp_path / "rlm_state").mkdir()
    (tmp_path / "rlm_state" / "bes_candidates.json").write_text("{}")
    (tmp_path / "candidates").mkdir()
    res = check_bes_archive(tmp_path)
    assert res.complete is False
    assert any("metrics.json" in m for m in res.missing)
