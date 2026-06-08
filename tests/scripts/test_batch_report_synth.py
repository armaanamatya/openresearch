"""Unit guardrails for batch_reproduce's terminal-report synthesis.

Closes the "no final_report.json on hard kill" gap: a reproduce child that is
SIGKILLed (kill -9 / OOM-killer / the scheduler's own SIGTERM→SIGKILL
escalation) cannot run any in-process finalizer, so ``final_report.json`` may
never be written. The batch scheduler synthesizes a minimal terminal report
after such a child exits so a killed run still leaves a scoreable artifact.

``scripts/`` is not a package, so batch_reproduce is loaded via importlib by
file path (mirrors tests/test_serve_local_llm.py). No real subprocess is run.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib

_BATCH_PATH = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "batch_reproduce.py"
_bspec = importlib.util.spec_from_file_location("batch_reproduce", _BATCH_PATH)
batch = importlib.util.module_from_spec(_bspec)
assert _bspec and _bspec.loader
_bspec.loader.exec_module(batch)


def test_synthesizes_failed_report_when_missing(tmp_path: pathlib.Path) -> None:
    """No final_report.json -> a verdict='failed' terminal report is written."""
    runs_root = tmp_path / "runs"
    project_id = "prj_killed"
    project_dir = runs_root / project_id
    project_dir.mkdir(parents=True)
    report_path = project_dir / "final_report.json"
    assert not report_path.exists()

    batch._ensure_terminal_report(runs_root, project_id)

    assert report_path.exists()
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["verdict"] == "failed"
    assert "synthesized by the batch scheduler" in data["reproduction_summary"]
    # No fabricated metrics: the rubric score stays an honest null.
    assert data["rubric"]["overall_score"] is None
    assert data["baseline_metrics"] == {}
    # Markdown sibling is written alongside the JSON.
    assert (project_dir / "final_report.md").exists()


def test_iterations_counted_from_experiment_runs(tmp_path: pathlib.Path) -> None:
    """Best-effort iteration count = non-blank lines in experiment_runs.jsonl."""
    runs_root = tmp_path / "runs"
    project_id = "prj_partial"
    project_dir = runs_root / project_id
    project_dir.mkdir(parents=True)
    # Two real result lines, plus a blank line that must NOT be counted.
    (project_dir / "experiment_runs.jsonl").write_text(
        json.dumps({"success": True, "metrics": {"acc": 0.5}}) + "\n"
        + "\n"
        + json.dumps({"success": False, "metrics": {}}) + "\n",
        encoding="utf-8",
    )

    batch._ensure_terminal_report(runs_root, project_id)

    data = json.loads((project_dir / "final_report.json").read_text(encoding="utf-8"))
    assert data["verdict"] == "failed"
    assert data["iterations"] == 2


def test_missing_experiment_runs_yields_zero_iterations(tmp_path: pathlib.Path) -> None:
    """No experiment_runs.jsonl -> iterations defaults to 0 (never raises)."""
    runs_root = tmp_path / "runs"
    project_id = "prj_no_runs"
    (runs_root / project_id).mkdir(parents=True)

    batch._ensure_terminal_report(runs_root, project_id)

    data = json.loads((runs_root / project_id / "final_report.json").read_text(encoding="utf-8"))
    assert data["iterations"] == 0


def test_does_not_overwrite_existing_report(tmp_path: pathlib.Path) -> None:
    """A pre-existing final_report.json is left byte-for-byte untouched."""
    runs_root = tmp_path / "runs"
    project_id = "prj_completed"
    project_dir = runs_root / project_id
    project_dir.mkdir(parents=True)
    report_path = project_dir / "final_report.json"
    sentinel = json.dumps({"verdict": "reproduced", "marker": "real-report"}, indent=2)
    report_path.write_text(sentinel, encoding="utf-8")

    batch._ensure_terminal_report(runs_root, project_id)

    # Unchanged: the real report is authoritative, never clobbered.
    assert report_path.read_text(encoding="utf-8") == sentinel


def test_never_raises_on_synthesis_failure(tmp_path: pathlib.Path, monkeypatch) -> None:
    """If write_final_report_rlm blows up, the helper swallows it (batch loop safe)."""
    import backend.agents.rlm.report as report_mod

    def _boom(*_a, **_k):  # noqa: ANN002, ANN003
        raise RuntimeError("disk full")

    monkeypatch.setattr(report_mod, "write_final_report_rlm", _boom)

    runs_root = tmp_path / "runs"
    project_id = "prj_boom"
    (runs_root / project_id).mkdir(parents=True)

    # Must not propagate — a synthesis failure can never crash the reap loop.
    batch._ensure_terminal_report(runs_root, project_id)
    assert not (runs_root / project_id / "final_report.json").exists()
