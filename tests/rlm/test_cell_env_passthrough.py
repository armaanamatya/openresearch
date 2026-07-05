"""Tests for per-cell env overrides + operator staged-env passthrough (Task #2).

Final child_env precedence in ``_run_cell_subprocess`` (highest priority wins):

  1. inherited ``os.environ`` (base)
  2. harness advisory (PATH, ``PYTORCH_CUDA_ALLOC_CONF`` default-path-only, the
     ``OPENRESEARCH_CELL_*`` contract vars, batch-scale vars)
  3. ``cell["cell_env"]`` — per-cell overrides (NEW)
  4. ``OPENRESEARCH_CELL_ENV_PASSTHROUGH`` — operator staged-env allowlist (NEW)
  5. harness-protected, re-asserted LAST: ``CUDA_VISIBLE_DEVICES``, and (on the
     command branch only) ``OUTPUT_DIR`` / ``OPENRESEARCH_CELL_ID``

R1: a COMMAND cell (the authors' own launcher, execute mode) must NOT get the
harness's ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` default — vLLM's
CuMemAllocator asserts on it. The DEFAULT (non-command, train_cell.py) path
keeps it exactly as before this change.

Subprocess is mocked at the ``Popen`` level — mirrors the pattern in
``tests/rlm/test_cell_command_seam.py``.
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

from backend.agents.rlm.gpu_cell_runner import _run_cell_subprocess


# ---------------------------------------------------------------------------
# Helpers — mirrors tests/rlm/test_cell_command_seam.py
# ---------------------------------------------------------------------------

def _fake_popen_factory() -> tuple[Any, dict]:
    """Return (fake_popen, captured) — captured receives cmd/cwd/env kwargs."""
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


def _code_dir(tmp_path: Path) -> Path:
    code = tmp_path / "code"
    code.mkdir(parents=True, exist_ok=True)
    return code


def _run(tmp_path: Path, cell: dict, *, gpu_id: str = "0") -> dict:
    code = _code_dir(tmp_path)
    cell_script = code / "train_cell.py"
    cell_id = cell.get("id", "c0")
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
            cell=cell,
            cell_script=str(cell_script),
            gpu_id=gpu_id,
            output_dir=output_dir,
            batch_scale=None,
            grad_checkpoint=False,
            timeout_s=None,
            log_path=log_path,
        )
    return captured


# ---------------------------------------------------------------------------
# Operator staged-env passthrough (step 4)
# ---------------------------------------------------------------------------

class TestPassthroughAllowlist:
    def test_passthrough_forwards_listed_vars(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENRESEARCH_CELL_ENV_PASSTHROUGH", "HF_HOME,FOO")
        monkeypatch.setenv("HF_HOME", "/staged/hf")
        monkeypatch.setenv("FOO", "bar-value")

        captured = _run(tmp_path, {"id": "c0"})

        assert captured["env"]["HF_HOME"] == "/staged/hf"
        assert captured["env"]["FOO"] == "bar-value"

    def test_passthrough_ignores_names_absent_from_environ(self, tmp_path, monkeypatch):
        """A name on the allowlist with no actual os.environ value is a no-op
        (never sets e.g. child_env['UNSET_VAR'] = 'None')."""
        monkeypatch.setenv("OPENRESEARCH_CELL_ENV_PASSTHROUGH", "TOTALLY_UNSET_VAR_XYZ")
        monkeypatch.delenv("TOTALLY_UNSET_VAR_XYZ", raising=False)

        captured = _run(tmp_path, {"id": "c0"})

        assert "TOTALLY_UNSET_VAR_XYZ" not in captured["env"]


# ---------------------------------------------------------------------------
# cell_env per-cell overrides (step 3)
# ---------------------------------------------------------------------------

class TestCellEnvOverrides:
    def test_cell_env_sets_new_key(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OPENRESEARCH_CELL_ENV_PASSTHROUGH", raising=False)

        captured = _run(tmp_path, {"id": "c0", "cell_env": {"WANDB_MODE": "disabled"}})

        assert captured["env"]["WANDB_MODE"] == "disabled"

    def test_cell_env_overrides_harness_advisory_default(self, tmp_path, monkeypatch):
        """cell_env applies AFTER the harness advisory — it can override e.g.
        PYTORCH_CUDA_ALLOC_CONF on the default (non-command) path."""
        monkeypatch.delenv("OPENRESEARCH_CELL_ENV_PASSTHROUGH", raising=False)

        captured = _run(
            tmp_path,
            {"id": "c0", "cell_env": {"PYTORCH_CUDA_ALLOC_CONF": "backend:cudaMallocAsync"}},
        )

        assert captured["env"]["PYTORCH_CUDA_ALLOC_CONF"] == "backend:cudaMallocAsync"

    def test_cell_env_values_coerced_to_str(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OPENRESEARCH_CELL_ENV_PASSTHROUGH", raising=False)

        captured = _run(tmp_path, {"id": "c0", "cell_env": {"MY_INT_VAR": 5}})

        assert captured["env"]["MY_INT_VAR"] == "5"

    def test_non_dict_cell_env_is_ignored(self, tmp_path, monkeypatch):
        """A malformed cell_env (not a dict) must not crash the launch."""
        monkeypatch.delenv("OPENRESEARCH_CELL_ENV_PASSTHROUGH", raising=False)

        captured = _run(tmp_path, {"id": "c0", "cell_env": "not-a-dict"})

        assert captured["cmd"][0] == sys.executable

    def test_cell_env_coexists_with_env_axis_field(self, tmp_path, monkeypatch):
        """cell["env"] (the environment/dataset AXIS, e.g. "alfworld") and
        cell["cell_env"] (env-VAR overrides) are unrelated keys — the latter
        must be read/applied without disturbing the former's own consumer
        (_floor_cell_max_turns)."""
        monkeypatch.delenv("OPENRESEARCH_CELL_ENV_PASSTHROUGH", raising=False)

        captured = _run(
            tmp_path,
            {"id": "c0", "env": "alfworld", "cell_env": {"FOO": "real-cell-env-value"}},
        )

        assert captured["env"]["FOO"] == "real-cell-env-value"
        assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "0"


# ---------------------------------------------------------------------------
# Precedence ordering
# ---------------------------------------------------------------------------

class TestPrecedenceOrdering:
    def test_passthrough_overrides_conflicting_cell_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENRESEARCH_CELL_ENV_PASSTHROUGH", "HF_HOME")
        monkeypatch.setenv("HF_HOME", "/operator/staged")

        captured = _run(
            tmp_path,
            {"id": "c0", "cell_env": {"HF_HOME": "/cell/local"}},
        )

        assert captured["env"]["HF_HOME"] == "/operator/staged"

    def test_cuda_visible_devices_cannot_be_overridden_by_cell_env(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OPENRESEARCH_CELL_ENV_PASSTHROUGH", raising=False)

        captured = _run(
            tmp_path,
            {"id": "c0", "cell_env": {"CUDA_VISIBLE_DEVICES": "99"}},
            gpu_id="3",
        )

        assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "3"

    def test_cuda_visible_devices_cannot_be_overridden_by_passthrough(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENRESEARCH_CELL_ENV_PASSTHROUGH", "CUDA_VISIBLE_DEVICES")
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")

        captured = _run(tmp_path, {"id": "c0"}, gpu_id="2")

        assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "2"

    def test_output_dir_and_cell_id_cannot_be_overridden_on_command_branch(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENRESEARCH_CELL_ENV_PASSTHROUGH", "OUTPUT_DIR,OPENRESEARCH_CELL_ID")
        monkeypatch.setenv("OUTPUT_DIR", "/stale/passthrough/dir")
        monkeypatch.setenv("OPENRESEARCH_CELL_ID", "stale-id")

        captured = _run(tmp_path, {"id": "c0", "command": "bash run.sh", "cell_env": {
            "OUTPUT_DIR": "/stale/cell_env/dir", "OPENRESEARCH_CELL_ID": "also-stale",
        }})

        expected_output_dir = str(tmp_path / "c0")
        assert captured["env"]["OUTPUT_DIR"] == expected_output_dir
        assert captured["env"]["OPENRESEARCH_CELL_ID"] == "c0"


# ---------------------------------------------------------------------------
# R1: command cells must not carry the harness's expandable_segments default
# ---------------------------------------------------------------------------

class TestR1CommandCellAllocConf:
    def test_command_cell_without_cell_env_has_no_alloc_conf(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OPENRESEARCH_CELL_ENV_PASSTHROUGH", raising=False)

        captured = _run(tmp_path, {"id": "c0", "command": "bash run.sh"})

        assert "PYTORCH_CUDA_ALLOC_CONF" not in captured["env"]

    def test_command_cell_with_cell_env_alloc_conf_is_honored(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OPENRESEARCH_CELL_ENV_PASSTHROUGH", raising=False)

        captured = _run(
            tmp_path,
            {
                "id": "c0", "command": "bash run.sh",
                "cell_env": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
            },
        )

        assert captured["env"]["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"

    def test_command_cell_with_passthrough_alloc_conf_is_honored(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENRESEARCH_CELL_ENV_PASSTHROUGH", "PYTORCH_CUDA_ALLOC_CONF")
        monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8")

        captured = _run(tmp_path, {"id": "c0", "command": "bash run.sh"})

        assert captured["env"]["PYTORCH_CUDA_ALLOC_CONF"] == "garbage_collection_threshold:0.8"

    def test_non_command_cell_keeps_harness_default_byte_identical(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OPENRESEARCH_CELL_ENV_PASSTHROUGH", raising=False)

        captured = _run(tmp_path, {"id": "c0"})

        assert captured["env"]["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


# ---------------------------------------------------------------------------
# OFF-parity: unset allowlist + no cell_env + no command
# ---------------------------------------------------------------------------

class TestOffParity:
    def test_no_extra_keys_beyond_documented_harness_advisory(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OPENRESEARCH_CELL_ENV_PASSTHROUGH", raising=False)
        monkeypatch.delenv("OPENRESEARCH_CELL_BATCH_SCALE", raising=False)
        monkeypatch.delenv("OPENRESEARCH_CELL_GRAD_CHECKPOINT", raising=False)

        baseline_keys = set(os.environ.keys())
        captured = _run(tmp_path, {"id": "c0"})
        env = captured["env"]

        expected_new = {
            "PATH",
            "CUDA_VISIBLE_DEVICES",
            "PYTORCH_CUDA_ALLOC_CONF",
            "OPENRESEARCH_CELL_OUTPUT_DIR",
            "OPENRESEARCH_CELL_PARAMS",
            "OPENRESEARCH_CELL_CHECKPOINT_DIR",
            "OPENRESEARCH_CELL_CHECKPOINT_INTERVAL_S",
        }
        assert set(env.keys()) - baseline_keys - expected_new == set()

    def test_default_launcher_argv_and_cwd_unaffected(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OPENRESEARCH_CELL_ENV_PASSTHROUGH", raising=False)

        captured = _run(tmp_path, {"id": "c0"})

        assert captured["cwd"] is None
        assert "OUTPUT_DIR" not in captured["env"]
        assert "OPENRESEARCH_CELL_ID" not in captured["env"]
        assert captured["cmd"][0] == sys.executable
