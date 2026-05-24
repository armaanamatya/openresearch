from __future__ import annotations

import json
from pathlib import Path

from backend.services.events.live_runs import FileLiveRunService


def test_live_run_payload_includes_worker_reports(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "prj_worker_reports"
    run_dir.mkdir(parents=True)
    status = {
        "projectId": "prj_worker_reports",
        "outputDir": str(run_dir),
        "runMode": "rlm",
        "llmProvider": "anthropic",
        "status": "completed",
        "startedAt": "2026-05-24T00:00:00+00:00",
        "updatedAt": "2026-05-24T00:00:01+00:00",
    }
    (run_dir / "demo_status.json").write_text(json.dumps(status), encoding="utf-8")
    (run_dir / "worker_reports.jsonl").write_text(
        json.dumps({
            "report_id": "wr-1",
            "agent_id": "baseline-implementation",
            "implemented": ["Added train.py"],
            "commands": [{"command": "python train.py", "exit_code": 0}],
            "issues": [],
            "procedures_followed": True,
        }) + "\n",
        encoding="utf-8",
    )

    service = FileLiveRunService(runs_root=runs_root, repo_root=tmp_path)
    state = service._load_run("prj_worker_reports")

    assert state is not None
    assert state.payload["workerReports"][0]["agent_id"] == "baseline-implementation"
