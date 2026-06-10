"""Hard-stop salvage + report merge + per-attempt sidecar isolation.

Regression suite for the 2026-06-09 All-CNN scoreless failure:

* the watchdog/SIGTERM path shipped ``rubric.overall_score=None`` +
  ``verdict="failed"`` although the run had recorded a real 0.49 rubric score
  (``_hard_stop_with_report`` bypassed the best-of-run floor);
* ``write_final_report_rlm``'s rubric_evaluation merge treated the default
  ``overall_score=None`` as "already present" and never filled it;
* ``rubric_evaluation.json``/``rubric_tree.json`` were not archived per
  attempt, so 0-iteration attempts inherited the previous attempt's leaves.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.agents.rlm.report import RLMFinalReport, write_final_report_rlm
from backend.agents.rlm.run import _salvage_partial_report
from backend.services.runs.attempt_isolation import maybe_archive_prior_attempt


def _write_events(project_dir, scores):
    lines = [
        json.dumps({"event": "rubric_score", "iteration": i, "score": s, "target": 0.6})
        for i, s in enumerate(scores)
    ]
    (project_dir / "dashboard_events.jsonl").write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------- salvage

def test_salvage_floors_score_and_reconciles_verdict(tmp_path):
    _write_events(tmp_path, [0.4908, 0.4706])  # the real All-CNN sequence
    report = RLMFinalReport(verdict="failed", reproduction_summary="watchdog", iterations=2)
    score = _salvage_partial_report(
        report, tmp_path, stop_kind="wall_clock_watchdog", stop_detail="50400s deadline"
    )
    assert score == pytest.approx(0.4908)
    assert report.rubric["overall_score"] == pytest.approx(0.4908)
    assert report.rubric["best_of_run"] is True
    assert report.verdict == "partial"
    assert report.stop_reason == {
        "kind": "wall_clock_watchdog", "detail": "50400s deadline",
    }
    assert "[salvage]" in report.reproduction_summary


def test_salvage_low_score_stays_failed(tmp_path):
    _write_events(tmp_path, [0.05])
    report = RLMFinalReport(verdict="failed", reproduction_summary="sigterm", iterations=0)
    score = _salvage_partial_report(report, tmp_path, stop_kind="sigterm", stop_detail="x")
    assert score == pytest.approx(0.05)
    assert report.verdict == "failed"  # evidence ceiling below partial threshold


def test_salvage_never_upgrades_past_partial(tmp_path):
    _write_events(tmp_path, [0.91])
    report = RLMFinalReport(verdict="failed", reproduction_summary="w", iterations=1)
    _salvage_partial_report(report, tmp_path, stop_kind="wall_clock_watchdog", stop_detail="x")
    assert report.verdict == "partial"  # hard-stopped runs never claim "reproduced"


def test_salvage_without_events_ships_bare_report(tmp_path):
    report = RLMFinalReport(verdict="failed", reproduction_summary="w", iterations=0)
    score = _salvage_partial_report(report, tmp_path, stop_kind="sigterm", stop_detail="x")
    assert score is None
    assert report.verdict == "failed"
    assert report.rubric["overall_score"] is None
    assert report.stop_reason == {"kind": "sigterm", "detail": "x"}  # still recorded


# ------------------------------------------------------------------- merge

def test_merge_fills_none_score_from_rubric_evaluation(tmp_path):
    (tmp_path / "rubric_evaluation.json").write_text(json.dumps({
        "overall_score": 0.4706,
        "target_score": 0.6,
        "meets_target": False,
        "leaf_scores": [{"id": "leaf1", "score": 0.85}],
        "weak_leaves": [{"id": "leaf2", "score": 0.0}],
    }))
    report = RLMFinalReport(verdict="partial", reproduction_summary="w", iterations=2)
    json_path, _ = write_final_report_rlm(report, tmp_path)
    written = json.loads(json_path.read_text())
    assert written["rubric"]["overall_score"] == pytest.approx(0.4706)
    assert written["rubric"]["target_score"] == pytest.approx(0.6)
    assert written["rubric"]["meets_target"] is False
    assert written["rubric"]["leaf_scores"] == [{"id": "leaf1", "score": 0.85}]


def test_merge_never_overwrites_a_real_score(tmp_path):
    (tmp_path / "rubric_evaluation.json").write_text(json.dumps({"overall_score": 0.2}))
    report = RLMFinalReport(verdict="partial", reproduction_summary="w", iterations=1)
    report.rubric = dict(report.rubric, overall_score=0.83)
    json_path, _ = write_final_report_rlm(report, tmp_path)
    written = json.loads(json_path.read_text())
    assert written["rubric"]["overall_score"] == pytest.approx(0.83)


# ----------------------------------------------------------------- archive

def test_archiver_moves_rubric_and_telemetry_sidecars(tmp_path):
    project = tmp_path / "proj_x"
    project.mkdir()
    sidecars = (
        "final_report.json",   # trigger file
        "rubric_evaluation.json",
        "rubric_tree.json",
        "timing.json",
        "tokens_total.json",
        "worker_reports.jsonl",
        "environment_spec.json",
    )
    for name in sidecars:
        (project / name).write_text("{}")
    result = maybe_archive_prior_attempt("proj_x", tmp_path)
    assert result is not None
    for name in sidecars:
        assert not (project / name).exists(), f"{name} leaked into the fresh attempt"
    attempt_dirs = list((project / "attempts").iterdir())
    assert len(attempt_dirs) == 1
    for name in sidecars:
        assert (attempt_dirs[0] / name).is_file()


def test_archiver_keeps_paper_stable_artifacts(tmp_path):
    project = tmp_path / "proj_y"
    project.mkdir()
    (project / "final_report.json").write_text("{}")
    (project / "generated_rubric.json").write_text("{}")
    (project / "parsed_full_text.txt").write_text("paper")
    maybe_archive_prior_attempt("proj_y", tmp_path)
    assert (project / "generated_rubric.json").exists()
    assert (project / "parsed_full_text.txt").exists()


def test_cli_path_archiver_moves_sidecars_too(tmp_path):
    """The CLI path (`archive_run_artifacts`) is a SECOND archiver implementation.

    2026-06-09 live re-run: attempt_isolation had just been taught to move
    rubric_evaluation.json, but the CLI archiver hadn't — the stale eval stayed
    at the project root, where the report merge would have fabricated the
    previous attempt's score onto a pre-verify failure. Both archivers now
    share PER_ATTEMPT_SIDECARS.
    """
    from backend.services.runs.archive import archive_run_artifacts
    from backend.services.runs.attempt_isolation import PER_ATTEMPT_SIDECARS

    project = tmp_path / "proj_cli"
    project.mkdir()
    (project / "final_report.json").write_text("{}")  # trigger
    for name in PER_ATTEMPT_SIDECARS:
        (project / name).write_text("{}")
    result = archive_run_artifacts("proj_cli", tmp_path)
    assert result is not None
    for name in PER_ATTEMPT_SIDECARS:
        assert not (project / name).exists(), f"{name} leaked (CLI archive path)"
        assert (Path(result["attempt_dir"]) / name).is_file()


def test_both_archiver_manifests_carry_the_shared_sidecars():
    """Drift-proof: the shared tuple must be embedded in BOTH manifests."""
    from backend.services.runs import archive, attempt_isolation

    shared = set(attempt_isolation.PER_ATTEMPT_SIDECARS)
    assert shared <= set(archive._TOP_LEVEL_FILES)
    assert shared <= set(attempt_isolation._ARCHIVE_FILES)
