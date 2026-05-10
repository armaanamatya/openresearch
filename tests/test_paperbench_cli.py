"""Tests for the `reprolab paperbench` CLI subcommand.

These tests exercise list/summary/run (dry mode)/status against a synthetic
mini bundle in tmp_path, so they require no LLM credentials and no network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.cli import main as cli_main


MINI_RUBRIC = {
    "id": "root",
    "requirements": "Replicate the paper.",
    "weight": 1,
    "task_category": "Root",
    "sub_tasks": [
        {
            "id": "code",
            "requirements": "Code is implemented.",
            "weight": 6,
            "task_category": "Code Development",
            "sub_tasks": [
                {
                    "id": "code-a",
                    "requirements": "Module A.",
                    "weight": 1,
                    "task_category": "Code Development",
                    "sub_tasks": [],
                },
                {
                    "id": "code-b",
                    "requirements": "Module B.",
                    "weight": 1,
                    "task_category": "Code Development",
                    "sub_tasks": [],
                },
            ],
        },
        {
            "id": "exec",
            "requirements": "reproduce.sh runs.",
            "weight": 2,
            "task_category": "Execution",
            "sub_tasks": [],
        },
        {
            "id": "result",
            "requirements": "Numbers match.",
            "weight": 2,
            "task_category": "Result Match",
            "sub_tasks": [],
        },
    ],
}


@pytest.fixture
def mini_bundle(tmp_path: Path) -> Path:
    bundles_root = tmp_path / "third_party" / "paperbench"
    bundle = bundles_root / "mini"
    bundle.mkdir(parents=True)
    (bundle / "config.yaml").write_text("id: mini\ntitle: \"Mini\"\n", encoding="utf-8")
    (bundle / "paper.md").write_text("paper", encoding="utf-8")
    (bundle / "addendum.md").write_text("addendum", encoding="utf-8")
    (bundle / "task_instructions.md").write_text("instructions", encoding="utf-8")
    (bundle / "rubric.json").write_text(json.dumps(MINI_RUBRIC), encoding="utf-8")
    (bundle / "blacklist.txt").write_text("https://example.com/banned\n", encoding="utf-8")
    return bundles_root


def test_paperbench_list_emits_bundle_metadata(
    mini_bundle: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = cli_main(["paperbench", "list", "--bundles-root", str(mini_bundle)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["bundles"][0]["paper_id"] == "mini"
    assert payload["bundles"][0]["metadata"]["title"] == "Mini"


def test_paperbench_summary_reports_rubric_and_ceiling(
    mini_bundle: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = cli_main(["paperbench", "summary", "--paper-id", "mini", "--bundles-root", str(mini_bundle)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["paper_id"] == "mini"
    weights = payload["rubric_summary"]["task_category_weights"]
    assert weights["Code Development"]["weight"] == pytest.approx(0.6)
    assert weights["Execution"]["weight"] == pytest.approx(0.2)
    assert weights["Result Match"]["weight"] == pytest.approx(0.2)
    assert payload["code_development_ceiling"] == pytest.approx(0.6)


def test_paperbench_run_dry_mode_persists_status_and_submission(
    mini_bundle: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs_root = tmp_path / "runs"
    rc = cli_main(
        [
            "--runs-root",
            str(runs_root),
            "paperbench",
            "run",
            "--no-pipeline",
            "--paper-id",
            "mini",
            "--bundles-root",
            str(mini_bundle),
        ]
    )
    assert rc == 0
    handle = json.loads(capsys.readouterr().out)
    status_path = Path(handle["status_path"])
    assert status_path.is_file()
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["status"] == "succeeded"
    assert status["mode"] == "dry"
    assert status["n_attempts"] == 1
    submission = status["attempts"][0]["submission_dir"]
    assert (Path(submission) / "reproduce.sh").is_file()
    assert (Path(submission) / "paperbench_manifest.json").is_file()
    assert status["attempts"][0]["submission_validation"]["ok"] is True


def test_paperbench_status_reads_back_persisted_run(
    mini_bundle: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs_root = tmp_path / "runs"
    cli_main(
        [
            "--runs-root",
            str(runs_root),
            "paperbench",
            "run",
            "--no-pipeline",
            "--paper-id",
            "mini",
            "--bundles-root",
            str(mini_bundle),
        ]
    )
    started = json.loads(capsys.readouterr().out)
    rc = cli_main(
        [
            "--runs-root",
            str(runs_root),
            "paperbench",
            "status",
            "--run-group-id",
            started["run_group_id"],
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["run_group_id"] == started["run_group_id"]
    assert payload["status"] == "succeeded"


def test_paperbench_summary_unknown_paper_id_exits_two(
    mini_bundle: Path,
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli_main(["paperbench", "summary", "--paper-id", "does-not-exist", "--bundles-root", str(mini_bundle)])
    assert excinfo.value.code == 2
