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
        StartRunRequest(mode="rlm"),
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
        StartRunRequest(mode="rlm"),
        "prj_upload",
        output_dir,
        {"path": str(uploaded), "fileName": "paper.pdf"},
    )

    assert (output_dir / "code" / "paper.pdf").read_bytes() == uploaded.read_bytes()
    assert (output_dir / "raw_paper.pdf").read_bytes() == uploaded.read_bytes()
    assert source_pdf["fileName"] == "paper.pdf"
    assert benchmark["verdict"] == "pending_pipeline_result"


def test_non_demo_paper_run_does_not_assert_cartpole_benchmark(tmp_path: Path) -> None:
    """P2 data-integrity fix: a real (non-demo) paper run's demo_status.json,
    reprolab_manifest.json, and paperbench_comparison.json must never assert
    the canned ReproLab CartPole/PPO/mean_reward benchmark identity — that
    would be a false claim about what the run actually targeted (e.g. a real
    arxiv paper like 2605.00365). Absent/null is required instead of the
    hardcoded literal, even (especially) when the run fails before scoring.
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    runs_root = tmp_path / "runs"
    uploaded = tmp_path / "arxiv_2605.00365.pdf"
    uploaded.write_bytes(b"%PDF-1.4\nreal-paper\n%%EOF\n")
    service = FileLiveRunService(runs_root=runs_root, repo_root=repo_root)
    project_id = "prj_real_paper"
    output_dir = runs_root / project_id

    _source_pdf, benchmark = service._prepare_source_artifacts(  # noqa: SLF001
        StartRunRequest(mode="rlm"),
        project_id,
        output_dir,
        {"path": str(uploaded), "fileName": "arxiv_2605.00365.pdf"},
    )

    # demo_status.json's benchmark block (via BenchmarkSummary validation)
    assert benchmark["paperbenchTaskId"] is None
    assert benchmark["targetMetric"] is None
    assert benchmark["targetValue"] is None
    assert benchmark["benchmarkName"] is None
    for value in benchmark.values():
        assert value != "reprolab-demo/ppo-cartpole-v1"
        assert value != "mean_reward"

    code_dir = output_dir / "code"
    manifest = json.loads((code_dir / "reprolab_manifest.json").read_text())
    manifest_json = json.dumps(manifest)
    assert "ppo-cartpole" not in manifest_json
    assert "reprolab-demo" not in manifest_json
    assert manifest["benchmark"]["paperbenchTaskId"] is None
    assert manifest["benchmark"]["targetMetric"] is None

    comparison = json.loads((code_dir / "paperbench_comparison.json").read_text())
    comparison_json = json.dumps(comparison)
    assert "ppo-cartpole" not in comparison_json
    assert "CartPole" not in comparison_json
    assert comparison["paperbench_task_id"] is None
    assert comparison["claim"]["metric"] is None
    assert comparison["claim"]["environment"] is None
    assert comparison["result"]["metric"] is None

    log_text = (code_dir / "logs" / "paperbench_eval.log").read_text()
    assert "ppo-cartpole" not in log_text

    report_text = (code_dir / "final_benchmark_report.md").read_text()
    assert "ppo-cartpole" not in report_text
    assert "pending" in report_text


def test_demo_paper_still_gets_the_canned_benchmark_values(tmp_path: Path) -> None:
    """The built-in ReproLab canned demo (no uploaded PDF) keeps today's
    values — this fix only stops the false assertion for real paper runs."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "demo_paper.pdf").write_bytes(b"%PDF-1.4\nfixture\n%%EOF\n")
    runs_root = tmp_path / "runs"
    service = FileLiveRunService(runs_root=runs_root, repo_root=repo_root)
    project_id = "ui_demo_still_canned"
    output_dir = runs_root / project_id

    _source_pdf, benchmark = service._prepare_source_artifacts(  # noqa: SLF001
        StartRunRequest(mode="rlm"),
        project_id,
        output_dir,
        None,
    )

    assert benchmark["paperbenchTaskId"] == "reprolab-demo/ppo-cartpole-v1"
    assert benchmark["targetMetric"] == "mean_reward"
    assert benchmark["targetValue"] == 475.0
    assert benchmark["benchmarkName"] == "PaperBench-style final benchmark"
    assert benchmark["overallScore"] == 91.4
