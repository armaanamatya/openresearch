from __future__ import annotations

import json
import os
from pathlib import Path

from backend.evals.paperbench import (
    PaperBenchJudgeCommand,
    code_development_ceiling,
    create_submission_manifest,
    load_paperbench_bundle,
    mean_standard_error,
    summarize_rubric,
    validate_submission_tree,
)
from backend.services.ingestion.paperbench import bundle_to_workspace_claim_map


def test_paperbench_bundle_adapter_and_weighted_code_ceiling(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "third_party" / "paperbench" / "mini"
    bundle_dir.mkdir(parents=True)
    (bundle_dir / "config.yaml").write_text(
        'id: mini\ntitle: "Mini Paper"\n',
        encoding="utf-8",
    )
    (bundle_dir / "paper.md").write_text("paper body", encoding="utf-8")
    (bundle_dir / "addendum.md").write_text("addendum body", encoding="utf-8")
    (bundle_dir / "task_instructions.md").write_text("instructions", encoding="utf-8")
    (bundle_dir / "blacklist.txt").write_text(
        "https://github.com/example/blocked\n",
        encoding="utf-8",
    )
    (bundle_dir / "rubric.json").write_text(
        json.dumps(
            {
                "id": "root",
                "requirements": "root",
                "weight": 1,
                "sub_tasks": [
                    {
                        "id": "code",
                        "requirements": "code",
                        "weight": 2,
                        "sub_tasks": [],
                        "task_category": "Code Development",
                        "finegrained_task_category": "Method Implementation",
                    },
                    {
                        "id": "exec",
                        "requirements": "exec",
                        "weight": 1,
                        "sub_tasks": [],
                        "task_category": "Code Execution",
                        "finegrained_task_category": "Evaluation",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    bundle = load_paperbench_bundle(tmp_path / "third_party" / "paperbench", "mini")
    summary = summarize_rubric(bundle.rubric())
    workspace = bundle_to_workspace_claim_map(bundle)

    assert bundle.paper_id == "mini"
    assert bundle.metadata()["title"] == "Mini Paper"
    assert bundle.blacklist_entries() == ("https://github.com/example/blocked",)
    assert summary.node_count == 3
    assert summary.leaf_count == 2
    assert code_development_ceiling(bundle.rubric()) == 2 / 3
    assert workspace["paperbench"]["blacklist_entries"] == [
        "https://github.com/example/blocked"
    ]
    assert any(entry["source_id"] == "rubric_summary" for entry in workspace["entries"])


def test_submission_validation_manifest_and_judge_command(tmp_path: Path) -> None:
    submission = tmp_path / "submission"
    submission.mkdir()
    reproduce = submission / "reproduce.sh"
    reproduce.write_text("#!/usr/bin/env bash\npython train.py\n", encoding="utf-8")
    os.chmod(reproduce, 0o755)
    (submission / "README.md").write_text("run with reproduce.sh", encoding="utf-8")

    validation = validate_submission_tree(submission)
    manifest = create_submission_manifest("mini", submission, write=True)
    command = PaperBenchJudgeCommand(
        frontier_evals_dir=tmp_path / "frontier-evals",
        submission_path=submission,
        paper_id="mini",
        out_dir=tmp_path / "judge-out",
        code_only=True,
        max_depth=3,
    )

    assert validation.ok is True
    assert manifest.validation.ok is True
    assert (submission / "paperbench_manifest.json").is_file()
    argv = command.argv()
    assert "paperbench.scripts.run_judge" in argv
    assert "paper_id=mini" in argv
    assert "code_only=True" in argv
    assert "max_depth=3" in argv
    assert "completer_config.model=o3-mini-2025-01-31" in argv
    assert "completer_config.reasoning_effort=high" in argv


def test_mean_standard_error_uses_sample_standard_error() -> None:
    mean, se, n = mean_standard_error([1.0, 2.0, 3.0])

    assert mean == 2.0
    assert round(se, 6) == 0.577350
    assert n == 3
