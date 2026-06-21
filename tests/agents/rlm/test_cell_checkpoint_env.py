"""Tests for T2 intra-cell checkpoint env injection (2026-06-18).

Verifies that ``_run_cell_subprocess`` injects
``OPENRESEARCH_CELL_CHECKPOINT_DIR`` and
``OPENRESEARCH_CELL_CHECKPOINT_INTERVAL_S`` into the child environment
WITHOUT requiring a real GPU or script.  Subprocess is patched at the
Popen level so no real process is launched.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any
from unittest.mock import patch

from backend.agents.rlm.gpu_cell_runner import _run_cell_subprocess


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_popen_factory() -> tuple[Any, dict]:
    """Return (fake_proc, captured) where captured['env'] receives the env kwarg."""
    captured: dict[str, Any] = {}

    class _FakeProc:
        pid = 99999
        returncode = 0
        stdout = io.StringIO("")  # empty — reader thread exits immediately

        def wait(self, timeout=None):  # noqa: D401
            return 0

    fake_proc = _FakeProc()

    def _popen(cmd, *, env, stdout, stderr, text, encoding, errors, start_new_session):
        captured["env"] = dict(env)
        return fake_proc

    return _popen, captured


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCellCheckpointEnvInjection:
    """_run_cell_subprocess must set both checkpoint env vars."""

    def _call(self, tmp_path: Path, monkeypatch, env_overrides: dict | None = None):
        """Run _run_cell_subprocess with a mocked Popen, return captured env."""
        if env_overrides:
            for k, v in env_overrides.items():
                monkeypatch.setenv(k, v)

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
                cell={"id": "c0"},
                cell_script="x.py",
                gpu_id="0",
                output_dir=output_dir,
                batch_scale=None,
                grad_checkpoint=False,
                timeout_s=None,
                log_path=log_path,
            )

        return captured["env"]

    def test_checkpoint_dir_is_output_dir_checkpoints(self, tmp_path, monkeypatch):
        """OPENRESEARCH_CELL_CHECKPOINT_DIR == output_dir/checkpoints."""
        monkeypatch.delenv("OPENRESEARCH_CELL_CHECKPOINT_INTERVAL_S", raising=False)
        env = self._call(tmp_path, monkeypatch)
        expected = str(tmp_path / "c0" / "checkpoints")
        assert env["OPENRESEARCH_CELL_CHECKPOINT_DIR"] == expected

    def test_checkpoint_interval_default_600(self, tmp_path, monkeypatch):
        """Default interval is 600 when parent env has no override."""
        monkeypatch.delenv("OPENRESEARCH_CELL_CHECKPOINT_INTERVAL_S", raising=False)
        env = self._call(tmp_path, monkeypatch)
        assert env["OPENRESEARCH_CELL_CHECKPOINT_INTERVAL_S"] == "600"

    def test_checkpoint_interval_parent_env_propagates(self, tmp_path, monkeypatch):
        """Setting OPENRESEARCH_CELL_CHECKPOINT_INTERVAL_S=120 in parent propagates."""
        env = self._call(
            tmp_path, monkeypatch,
            env_overrides={"OPENRESEARCH_CELL_CHECKPOINT_INTERVAL_S": "120"},
        )
        assert env["OPENRESEARCH_CELL_CHECKPOINT_INTERVAL_S"] == "120"

    def test_checkpoint_dir_is_stable_across_cells(self, tmp_path, monkeypatch):
        """Two different cell output dirs get distinct, stable checkpoint dirs."""
        monkeypatch.delenv("OPENRESEARCH_CELL_CHECKPOINT_INTERVAL_S", raising=False)

        def _call_for_cell(cell_id: str) -> str:
            output_dir = tmp_path / cell_id
            log_path = tmp_path / f"{cell_id}.log"
            popen_stub, captured = _fake_popen_factory()
            with (
                patch("backend.agents.rlm.gpu_cell_runner.subprocess.Popen", popen_stub),
                patch("backend.agents.rlm.gpu_cell_runner._orphan_register"),
                patch("backend.agents.rlm.gpu_cell_runner._orphan_deregister"),
                patch("backend.agents.rlm.gpu_cell_runner._oom_enforce_enabled", return_value=False),
            ):
                _run_cell_subprocess(
                    cell={"id": cell_id},
                    cell_script="x.py",
                    gpu_id="0",
                    output_dir=output_dir,
                    batch_scale=None,
                    grad_checkpoint=False,
                    timeout_s=None,
                    log_path=log_path,
                )
            return captured["env"]["OPENRESEARCH_CELL_CHECKPOINT_DIR"]

        dir_a = _call_for_cell("cell_a")
        dir_b = _call_for_cell("cell_b")
        assert dir_a != dir_b
        assert dir_a.endswith("cell_a/checkpoints")
        assert dir_b.endswith("cell_b/checkpoints")

    def test_checkpoint_dir_not_mkdir_by_harness(self, tmp_path, monkeypatch):
        """The harness must NOT create the checkpoint dir (trainer responsibility)."""
        monkeypatch.delenv("OPENRESEARCH_CELL_CHECKPOINT_INTERVAL_S", raising=False)
        self._call(tmp_path, monkeypatch)
        ckpt_dir = tmp_path / "c0" / "checkpoints"
        assert not ckpt_dir.exists(), (
            "Harness must not mkdir the checkpoint dir — the trainer creates it on first write"
        )
