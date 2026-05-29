"""P1 (2026-05-29): commands.json manifest existence check.

`_harvest_baseline_artifacts` must reject envelopes whose ``commands.json``
references files the sub-agent didn't actually write. Catches the no-op
retry pattern observed adjacent to BUG-NEW-042 (sub-agent claimed success
but wrote nothing of substance in 2-3s follow-up retries).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.agents.rlm.primitives import (
    _commands_referenced_paths,
    _harvest_baseline_artifacts,
)


def _write_commands(code_dir: Path, commands: list) -> None:
    (code_dir / "commands.json").write_text(json.dumps(commands), encoding="utf-8")


class TestCommandsReferencedPaths:
    def test_extracts_python_script(self) -> None:
        refs = _commands_referenced_paths([["python", "train.py"]])
        assert refs == ["train.py"]

    def test_extracts_shell_script(self) -> None:
        refs = _commands_referenced_paths([["bash", "run.sh"]])
        assert refs == ["run.sh"]

    def test_extracts_config_files(self) -> None:
        refs = _commands_referenced_paths([["python", "train.py", "--config", "config.yaml"]])
        assert refs == ["train.py", "config.yaml"]

    def test_skips_flags_and_env(self) -> None:
        refs = _commands_referenced_paths([
            ["python", "train.py", "--epochs", "10", "--lr", "1e-4", "DEBUG=1"]
        ])
        assert refs == ["train.py"]

    def test_skips_urls(self) -> None:
        refs = _commands_referenced_paths([
            ["python", "train.py", "--dataset-url", "https://example.com/data.json"]
        ])
        assert refs == ["train.py"]

    def test_handles_string_command(self) -> None:
        refs = _commands_referenced_paths(["python train.py --config config.yaml"])
        assert refs == ["train.py", "config.yaml"]

    def test_handles_nested_paths(self) -> None:
        refs = _commands_referenced_paths([["python", "src/train.py", "configs/base.yaml"]])
        assert refs == ["src/train.py", "configs/base.yaml"]


class TestHarvestRejectsMissingFiles:
    def test_passes_when_all_referenced_files_exist(self, tmp_path: Path) -> None:
        code_dir = tmp_path / "code"
        code_dir.mkdir()
        (code_dir / "train.py").write_text("print('hello')\n", encoding="utf-8")
        _write_commands(code_dir, [["python", "train.py"]])

        result = _harvest_baseline_artifacts(code_dir)
        assert result["ok"] is True
        assert result["code_path"] == str(code_dir)

    def test_fails_when_referenced_file_missing(self, tmp_path: Path) -> None:
        code_dir = tmp_path / "code"
        code_dir.mkdir()
        # Write SOME runnable so the "no runnable" check passes — the missing
        # file is the one in commands.json, not the only file in the dir.
        (code_dir / "helper.py").write_text("# placeholder\n", encoding="utf-8")
        _write_commands(code_dir, [["python", "train.py"]])

        result = _harvest_baseline_artifacts(code_dir)
        assert result["ok"] is False
        assert result["error_code"] == "commands_missing_file"
        assert result["repairable"] is True
        assert "train.py" in result["missing_files"]
        assert "train.py" in result["error"]

    def test_fails_listing_all_missing_files(self, tmp_path: Path) -> None:
        code_dir = tmp_path / "code"
        code_dir.mkdir()
        (code_dir / "config.yaml").write_text("k: v\n", encoding="utf-8")
        (code_dir / "placeholder.py").write_text("# noop\n", encoding="utf-8")
        _write_commands(
            code_dir,
            [["python", "train.py", "--config", "config.yaml"], ["bash", "eval.sh"]],
        )

        result = _harvest_baseline_artifacts(code_dir)
        assert result["ok"] is False
        assert result["error_code"] == "commands_missing_file"
        missing = set(result["missing_files"])
        assert "train.py" in missing
        assert "eval.sh" in missing
        # config.yaml DOES exist — must not be flagged
        assert "config.yaml" not in missing

    def test_passes_when_no_file_args(self, tmp_path: Path) -> None:
        """Commands like `echo hello` reference no files — should pass."""
        code_dir = tmp_path / "code"
        code_dir.mkdir()
        (code_dir / "noop.py").write_text("pass\n", encoding="utf-8")
        _write_commands(code_dir, [["echo", "hello", "world"]])

        result = _harvest_baseline_artifacts(code_dir)
        assert result["ok"] is True
