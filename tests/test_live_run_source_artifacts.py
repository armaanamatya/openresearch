from __future__ import annotations

import json
from pathlib import Path

from backend.services.events.live_runs import FileLiveRunService, StartRunRequest


def test_live_run_prepares_fixture_pdf_and_benchmark_bundle(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "demo_paper.pdf").write_bytes(b"%PDF-1.4\nfixture\n%%EOF\n")
    runs_root = tmp_path / "runs"
    service = FileLiveRunService(runs_root=runs_root, repo_root=repo_root)
    output_dir = runs_root / "ui_demo_test"

    source_pdf, benchmark = service._prepare_source_artifacts(  # noqa: SLF001
        StartRunRequest(mode="sdk"),
        "ui_demo_test",
        output_dir,
        None,
    )

    code_dir = output_dir / "code"
    assert (code_dir / "paper.pdf").read_bytes().startswith(b"%PDF-1.4")
    assert (code_dir / "final_benchmark_report.md").exists()
    assert (code_dir / "logs" / "paperbench_eval.log").exists()
    assert service._final_report_path("ui_demo_test") == code_dir / "final_benchmark_report.md"  # noqa: SLF001
    assert source_pdf["codePath"].endswith("code/paper.pdf")
    assert benchmark["overallScore"] == 91.4

    comparison = json.loads((code_dir / "paperbench_comparison.json").read_text())
    assert comparison["paperbench_task_id"] == "reprolab-demo/ppo-cartpole-v1"
    assert comparison["result"]["status"] == "reproduced_with_caveats"


def test_live_run_copies_uploaded_pdf_to_generated_code_root(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    runs_root = tmp_path / "runs"
    uploaded = tmp_path / "uploaded.pdf"
    uploaded.write_bytes(b"%PDF-1.4\nuploaded\n%%EOF\n")
    service = FileLiveRunService(runs_root=runs_root, repo_root=repo_root)
    output_dir = runs_root / "prj_upload"

    source_pdf, benchmark = service._prepare_source_artifacts(  # noqa: SLF001
        StartRunRequest(mode="sdk"),
        "prj_upload",
        output_dir,
        {"path": str(uploaded), "fileName": "paper.pdf"},
    )

    assert (output_dir / "code" / "paper.pdf").read_bytes() == uploaded.read_bytes()
    assert (output_dir / "raw_paper.pdf").read_bytes() == uploaded.read_bytes()
    assert source_pdf["fileName"] == "paper.pdf"
    assert benchmark["verdict"] == "pending_pipeline_result"
