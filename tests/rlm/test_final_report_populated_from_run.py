"""Verify _finalize populates the new RLMFinalReport fields from the run.

Spec: 2026-05-23-rubric-climb-leaderboard §4.5.
"""

import json
from datetime import datetime
from pathlib import Path

from backend.agents.rlm.run import _finalize


class _FakeCtx:
    """Minimal RunContext stub — _finalize uses .project_id and .agent_model."""

    def __init__(self, project_id, agent_model):
        self.project_id = project_id
        self.agent_model = agent_model


def test_finalize_writes_planner_executor_and_timestamps(tmp_path: Path):
    # _finalize reads started_at from demo_status.json (already on disk at finalize time).
    started_iso = "2026-05-23T04:10:09+00:00"
    (tmp_path / "demo_status.json").write_text(json.dumps({
        "projectId": "prj_test_finalize",
        "status": "running",
        "startedAt": started_iso,
    }))

    ctx = _FakeCtx("prj_test_finalize", "claude-sonnet-4-6")
    emitted: list[dict] = []
    _finalize(
        project_dir=tmp_path,
        ctx=ctx,
        emit=emitted.append,
        result_obj=None,
        iterations=2,
        run_failed=False,
        llm_model="gpt-5",
        corpus_sentinels=None,
        tools_label="real",
    )

    fr_path = tmp_path / "final_report.json"
    assert fr_path.exists()
    payload = json.loads(fr_path.read_text())

    assert payload["mode"] == "rlm"
    assert payload["models"]["planner"] == "gpt-5"
    assert payload["models"]["executor"] == "claude-sonnet-4-6"
    assert payload["models"]["verifier"] is None
    assert payload["models"]["grader"] is None
    assert payload["started_at"] == started_iso
    started = datetime.fromisoformat(started_iso)
    completed = datetime.fromisoformat(payload["completed_at"])
    assert completed >= started


def test_finalize_handles_missing_demo_status(tmp_path: Path):
    """If demo_status.json is absent, started_at is None and the run still finalizes."""
    ctx = _FakeCtx("prj_test_no_demo", "claude-sonnet-4-6")
    _finalize(
        project_dir=tmp_path,
        ctx=ctx,
        emit=[].append,
        result_obj=None,
        iterations=0,
        run_failed=True,
        llm_model="gpt-5",
        corpus_sentinels=None,
        tools_label="real",
    )
    payload = json.loads((tmp_path / "final_report.json").read_text())
    assert payload["started_at"] is None
    assert payload["completed_at"] is not None
