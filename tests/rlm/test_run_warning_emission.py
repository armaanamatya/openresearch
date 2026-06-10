"""run_experiment must emit a dashboard_event with code='iteration_boundary_recommended'
whenever its outcome is repairable or partial_evidence."""
import json
from pathlib import Path
from backend.agents.rlm.primitives import _emit_iteration_boundary_warning


def test_emits_warning_event_for_repairable(tmp_path: Path):
    events_file = tmp_path / "dashboard_events.jsonl"
    _emit_iteration_boundary_warning(
        run_dir=tmp_path,
        outcome="repairable",
        brief="preflight blocked: load_dataset('imdb')",
    )
    assert events_file.exists()
    events = [json.loads(l) for l in events_file.read_text().splitlines()]
    assert len(events) == 1
    assert events[0]["event"] == "run_warning"
    assert events[0]["code"] == "iteration_boundary_recommended"
    assert "preflight blocked" in events[0]["message"]


def test_emits_warning_event_for_partial_evidence(tmp_path: Path):
    _emit_iteration_boundary_warning(
        run_dir=tmp_path,
        outcome="partial_evidence",
        brief="wall_clock_s=21600",
    )
    events = [json.loads(l) for l in (tmp_path / "dashboard_events.jsonl").read_text().splitlines()]
    assert events[0]["code"] == "iteration_boundary_recommended"


def test_does_not_emit_for_ok_outcome(tmp_path: Path):
    _emit_iteration_boundary_warning(
        run_dir=tmp_path,
        outcome="ok",
        brief="success",
    )
    assert not (tmp_path / "dashboard_events.jsonl").exists()
