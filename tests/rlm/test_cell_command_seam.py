"""Tests for the cell command/launcher seam (execute-mode Change #1).

A cell may declare ``cell["command"]`` — a raw shell command (e.g. ``conda run
-n sdar bash examples/sdar_trainer/run_search_3b.sh``) — to run the AUTHORS'
own launcher verbatim instead of the harness's default
``[sys.executable, train_cell.py, --cell-id=..., --output-dir=...]`` contract.

Default-OFF: a cell WITHOUT ``command`` must launch byte-identically to
before this change (same argv, no ``cwd`` override, no extra env keys).

Subprocess is mocked at the ``Popen`` level so no real process is launched;
mirrors the pattern in ``tests/agents/rlm/test_cell_checkpoint_env.py``.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import backend.agents.rlm.gpu_cell_runner as gcr
from backend.agents.rlm.gpu_cell_runner import _run_cell_subprocess, run_matrix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_popen_factory() -> tuple[Any, dict]:
    """Return (fake_popen, captured) — captured receives cmd/cwd/env kwargs.

    Accepts **kwargs so it tolerates an optional ``cwd`` kwarg being present
    or absent (unlike the fixed-signature fake in test_cell_checkpoint_env.py,
    which would TypeError on an unexpected ``cwd`` kwarg — this seam only adds
    ``cwd`` conditionally, so that other test stays valid untouched).
    """
    captured: dict[str, Any] = {}

    class _FakeProc:
        pid = 12345
        returncode = 0
        stdout = io.StringIO("")  # empty — reader thread exits immediately

        def wait(self, timeout=None):  # noqa: D401
            return 0

    fake_proc = _FakeProc()

    def _popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["cwd"] = kwargs.get("cwd")
        captured["env"] = dict(kwargs["env"])
        return fake_proc

    return _popen, captured


def _run(tmp_path: Path, cell: dict, cell_script: str) -> dict:
    output_dir = tmp_path / "c0"
    log_path = tmp_path / "c0.log"
    popen_stub, captured = _fake_popen_factory()

    with (
        patch("backend.agents.rlm.gpu_cell_runner.subprocess.Popen", popen_stub),
        patch("backend.agents.rlm.gpu_cell_runner._orphan_register"),
        patch("backend.agents.rlm.gpu_cell_runner._orphan_deregister"),
        patch("backend.agents.rlm.gpu_cell_runner._oom_enforce_enabled", return_value=False),
    ):
        _run_cell_subprocess(
            cell=cell,
            cell_script=cell_script,
            gpu_id="0",
            output_dir=output_dir,
            batch_scale=None,
            grad_checkpoint=False,
            timeout_s=None,
            log_path=log_path,
        )
    return captured


def _code_dir(tmp_path: Path) -> Path:
    code = tmp_path / "code"
    code.mkdir(parents=True, exist_ok=True)
    return code


# ---------------------------------------------------------------------------
# Tests — command branch
# ---------------------------------------------------------------------------

class TestCellCommandSeam:
    def test_command_cell_launches_via_bash_lc(self, tmp_path):
        code = _code_dir(tmp_path)
        cell_script = code / "train_cell.py"
        command = "conda run -n sdar bash examples/sdar_trainer/run_search_3b.sh"

        captured = _run(tmp_path, {"id": "c0", "command": command}, str(cell_script))

        assert captured["cmd"] == ["bash", "-lc", command]

    def test_command_cell_cwd_is_cell_script_parent(self, tmp_path):
        code = _code_dir(tmp_path)
        cell_script = code / "train_cell.py"

        captured = _run(tmp_path, {"id": "c0", "command": "bash run.sh"}, str(cell_script))

        assert captured["cwd"] == str(code)

    def test_command_cell_sets_output_dir_and_cell_id_env(self, tmp_path):
        code = _code_dir(tmp_path)
        cell_script = code / "train_cell.py"
        output_dir = tmp_path / "c0"

        captured = _run(tmp_path, {"id": "c0", "command": "bash run.sh"}, str(cell_script))

        assert captured["env"]["OUTPUT_DIR"] == str(output_dir)
        assert captured["env"]["OPENRESEARCH_CELL_ID"] == "c0"

    def test_command_cell_keeps_cuda_visible_devices_and_legacy_output_dir(self, tmp_path):
        code = _code_dir(tmp_path)
        cell_script = code / "train_cell.py"
        output_dir = tmp_path / "c0"

        captured = _run(tmp_path, {"id": "c0", "command": "bash run.sh"}, str(cell_script))

        assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "0"
        assert captured["env"]["OPENRESEARCH_CELL_OUTPUT_DIR"] == str(output_dir)

    def test_whitespace_only_command_falls_back_to_default_launcher(self, tmp_path):
        """A cell with command="   " (blank) must NOT take the command branch."""
        code = _code_dir(tmp_path)
        cell_script = code / "train_cell.py"

        captured = _run(tmp_path, {"id": "c0", "command": "   "}, str(cell_script))

        assert captured["cmd"][0] != "bash"
        assert captured["cmd"][0] == sys.executable


# ---------------------------------------------------------------------------
# Tests — no-command branch stays byte-identical
# ---------------------------------------------------------------------------

class TestNoCommandByteIdentical:
    def test_default_launcher_argv_unchanged(self, tmp_path):
        code = _code_dir(tmp_path)
        cell_script = code / "train_cell.py"
        output_dir = tmp_path / "c0"

        captured = _run(tmp_path, {"id": "c0"}, str(cell_script))

        assert captured["cmd"] == [
            sys.executable,
            str(cell_script),
            "--cell-id=c0",
            f"--output-dir={output_dir}",
        ]

    def test_default_launcher_has_no_cwd_override(self, tmp_path):
        code = _code_dir(tmp_path)
        cell_script = code / "train_cell.py"

        captured = _run(tmp_path, {"id": "c0"}, str(cell_script))

        assert captured["cwd"] is None

    def test_default_launcher_does_not_set_new_env_keys(self, tmp_path):
        code = _code_dir(tmp_path)
        cell_script = code / "train_cell.py"

        captured = _run(tmp_path, {"id": "c0"}, str(cell_script))

        assert "OUTPUT_DIR" not in captured["env"]
        assert "OPENRESEARCH_CELL_ID" not in captured["env"]

    def test_missing_command_key_entirely_is_also_byte_identical(self, tmp_path):
        code = _code_dir(tmp_path)
        cell_script = code / "train_cell.py"
        output_dir = tmp_path / "c0"

        captured = _run(tmp_path, {"id": "c0", "model_key": "m"}, str(cell_script))

        assert captured["cmd"] == [
            sys.executable,
            str(cell_script),
            "--cell-id=c0",
            f"--output-dir={output_dir}",
        ]
        assert captured["cwd"] is None


# ---------------------------------------------------------------------------
# run_matrix integration: the verl adapter (Change #3) fires only when a cell
# declares metrics_source.kind == "verl" AND no metrics.json landed on its own.
# ---------------------------------------------------------------------------

class TestRunMatrixVerlAdapterWiring:
    def _stub_writes_verl_log(self, *, cell, cell_script, gpu_id, output_dir, batch_scale,
                               grad_checkpoint, timeout_s, log_path):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "train.log").write_text("val/success_rate:0.456\n", encoding="utf-8")
        return 0, "authors' trainer finished"

    def test_verl_metrics_source_adapts_missing_metrics_json(self, tmp_path):
        cells = [{
            "id": "c0", "model_key": "m", "env": "e", "baseline": "b",
            "command": "bash run.sh",
            "metrics_source": {"kind": "verl"},
        }]
        with patch.object(gcr, "_run_cell_subprocess", self._stub_writes_verl_log):
            results = run_matrix(
                cells, str(tmp_path / "code" / "train_cell.py"),
                output_root=str(tmp_path / "out"), gpus=["0"],
            )
        assert results["c0"]["status"] == "ok"
        assert results["c0"]["metrics"]["success_rate"] == 0.456

    def test_no_metrics_source_leaves_metrics_none(self, tmp_path):
        """Default-OFF: a cell without metrics_source is unaffected — the log
        is never adapted even though it would parse cleanly."""
        cells = [{
            "id": "c0", "model_key": "m", "env": "e", "baseline": "b",
            "command": "bash run.sh",
        }]
        with patch.object(gcr, "_run_cell_subprocess", self._stub_writes_verl_log):
            results = run_matrix(
                cells, str(tmp_path / "code" / "train_cell.py"),
                output_root=str(tmp_path / "out"), gpus=["0"],
            )
        assert results["c0"]["metrics"] is None

    def test_existing_metrics_json_is_not_overwritten_by_adapter(self, tmp_path):
        """When train_cell.py-shaped metrics.json already exists, the verl
        adapter is never consulted — it only fires on a MISSING metrics.json."""
        def _stub(*, cell, cell_script, gpu_id, output_dir, batch_scale,
                  grad_checkpoint, timeout_s, log_path):
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "metrics.json").write_text('{"success_rate": 0.1}', encoding="utf-8")
            (output_dir / "train.log").write_text("val/success_rate:0.999\n", encoding="utf-8")
            return 0, "ok"

        cells = [{
            "id": "c0", "model_key": "m", "env": "e", "baseline": "b",
            "command": "bash run.sh",
            "metrics_source": {"kind": "verl"},
        }]
        with patch.object(gcr, "_run_cell_subprocess", _stub):
            results = run_matrix(
                cells, str(tmp_path / "code" / "train_cell.py"),
                output_root=str(tmp_path / "out"), gpus=["0"],
            )
        assert results["c0"]["metrics"]["success_rate"] == 0.1
