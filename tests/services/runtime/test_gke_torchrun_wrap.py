"""Multi-GPU torchrun-wrap in the GKE cell entrypoint (SCOPE 2).

Hermetic: loads the standalone entrypoint by file path (no google-cloud, no GPU,
no subprocess). Tests the pure build_cell_launch_argv helper directly.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_ENTRYPOINT_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "docker" / "gke-cell-base" / "gke_cell_entrypoint.py"
)


def _load_entrypoint():
    spec = importlib.util.spec_from_file_location("gke_cell_entrypoint", _ENTRYPOINT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ep():
    return _load_entrypoint()


_DISTRIBUTED_SCRIPT = "from accelerate import Accelerator\nAccelerator()\n"
_PLAIN_SCRIPT = "import torch\nprint('single process trainer')\n"


def test_single_gpu_runs_plain(ep):
    argv = ep.build_cell_launch_argv(
        python_exe="/usr/bin/python", train_cell_path=Path("/code/train_cell.py"),
        cell_id="c0", output_dir=Path("/out"), gpu_count=1, script_text=_DISTRIBUTED_SCRIPT,
    )
    assert argv[0] == "/usr/bin/python"
    assert "torchrun" not in argv[0]
    assert str(Path("/code/train_cell.py")) in argv


def test_multi_gpu_with_markers_wraps_torchrun(ep):
    argv = ep.build_cell_launch_argv(
        python_exe="/usr/bin/python", train_cell_path=Path("/code/train_cell.py"),
        cell_id="c0", output_dir=Path("/out"), gpu_count=4, script_text=_DISTRIBUTED_SCRIPT,
    )
    joined = " ".join(argv)
    assert argv[0] == "torchrun" or argv[0].endswith("torchrun")
    assert "--nproc_per_node=4" in joined
    assert str(Path("/code/train_cell.py")) in argv
    assert any("--cell-id=c0" in a for a in argv)
    assert any("--output-dir=" in a for a in argv)


def test_multi_gpu_no_markers_runs_plain(ep):
    argv = ep.build_cell_launch_argv(
        python_exe="/usr/bin/python", train_cell_path=Path("/code/train_cell.py"),
        cell_id="c0", output_dir=Path("/out"), gpu_count=4, script_text=_PLAIN_SCRIPT,
    )
    assert "torchrun" not in " ".join(argv)
    assert argv[0] == "/usr/bin/python"


def test_disable_opt_out(ep, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_DISABLE_TORCHRUN_WRAP", "1")
    argv = ep.build_cell_launch_argv(
        python_exe="/usr/bin/python", train_cell_path=Path("/code/train_cell.py"),
        cell_id="c0", output_dir=Path("/out"), gpu_count=4, script_text=_DISTRIBUTED_SCRIPT,
    )
    assert "torchrun" not in " ".join(argv)
    assert argv[0] == "/usr/bin/python"


def test_gpu_count_read_from_env_when_unset(ep, monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_CELL_GPU_COUNT", raising=False)
    assert ep.resolve_cell_gpu_count() == 1
    monkeypatch.setenv("OPENRESEARCH_CELL_GPU_COUNT", "8")
    assert ep.resolve_cell_gpu_count() == 8
    monkeypatch.setenv("OPENRESEARCH_CELL_GPU_COUNT", "garbage")
    assert ep.resolve_cell_gpu_count() == 1  # fail-soft
