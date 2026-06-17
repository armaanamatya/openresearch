"""Reliability fixes from the 2026-06-07 All-Conv-Net post-mortem.

The run died TWICE at import-preflight and never produced a metric (honest 0.0):
  1. the agent's ``torch==2.2.0`` re-pin DOWNGRADED the harness's cu121 build →
     incoherent CUDA stack → ``libcupti.so.12: cannot open shared object file``;
  2. an in-repo ``from backend...`` import that doesn't resolve in the flat sandbox.

These tests cover the five fixes:
  A. env_pin wiring (``_local_core_bootstrap_commands``) — strip core re-pins, pin cu121.
  B. ``_venv_cuda_lib_dirs`` — make the venv's bundled CUDA libs loadable (LD_LIBRARY_PATH).
  C. ``cuda_shlib_load`` failure class — actionable repair instead of ``unknown``.
  D. ``cell_scheduler.py`` shipped + ``gpu_cell_runner`` import guarded (latent SDAR break).
  E. preflight smoke flags ``backend.*`` imports with an actionable hint.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


# --------------------------------------------------------------------------- A
def test_bootstrap_strips_torch_repin_and_pins_core(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OPENRESEARCH_DISABLE_ENV_PIN", raising=False)
    from backend.agents.rlm.primitives import _local_core_bootstrap_commands

    req = tmp_path / "requirements.txt"
    req.write_text("torch==2.2.0\ntorchvision==0.17.0\ntqdm\nnumpy>=1.24\n", encoding="utf-8")
    cmds = _local_core_bootstrap_commands(req, "https://download.pytorch.org/whl/cu121")
    joined = "\n".join(cmds)

    # harness-owned cu121 core installed FIRST (now guarded by a host-torch probe
    # so a coherent CUDA-≥12.1 venv keeps its build), from the cu121 index
    assert cmds[0].index("import torch") < cmds[0].index("pip install")  # probe precedes install
    assert "python -m pip install torch==2.5.1 torchvision==0.20.1" in cmds[0]
    assert "--index-url https://download.pytorch.org/whl/cu121" in cmds[0]
    # agent deps installed from the HARDENED file, not the raw requirements.txt
    assert any("requirements.hardened.txt" in c for c in cmds)
    hardened = (tmp_path / "requirements.hardened.txt").read_text(encoding="utf-8")
    assert "torch==2.2.0" not in hardened           # core re-pin stripped
    assert "torchvision==0.17.0" not in hardened
    assert "numpy" not in hardened                  # numpy is core-denied (ABI safety)
    assert "tqdm" in hardened                        # non-core dep kept


def test_bootstrap_disable_flag_uses_legacy(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OPENRESEARCH_DISABLE_ENV_PIN", "1")
    from backend.agents.rlm.primitives import _local_core_bootstrap_commands

    req = tmp_path / "requirements.txt"
    req.write_text("torch==2.2.0\ntqdm\n", encoding="utf-8")
    cmds = _local_core_bootstrap_commands(req, "https://download.pytorch.org/whl/cu121")
    joined = "\n".join(cmds)

    assert "torch==2.5.1" not in joined                       # no harness pin
    assert "pip install torch --index-url" in joined          # legacy bare-torch
    assert "requirements.hardened.txt" not in joined          # raw requirements
    assert not (tmp_path / "requirements.hardened.txt").exists()


def test_bootstrap_no_torch_index_is_raw(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OPENRESEARCH_DISABLE_ENV_PIN", raising=False)
    from backend.agents.rlm.primitives import _local_core_bootstrap_commands

    req = tmp_path / "requirements.txt"
    req.write_text("torch==2.2.0\n", encoding="utf-8")
    cmds = _local_core_bootstrap_commands(req, "")  # operator disabled the index
    assert cmds == ["python -m pip install -r requirements.txt || true"]
    assert not (tmp_path / "requirements.hardened.txt").exists()


# --------------------------------------------------------------------------- B
def test_venv_cuda_lib_dirs_finds_torch_and_nvidia(tmp_path: Path) -> None:
    from backend.services.runtime.local_process import _venv_cuda_lib_dirs

    venv = tmp_path / "venv"
    site = venv / "lib" / "python3.12" / "site-packages"
    (site / "torch" / "lib").mkdir(parents=True)
    (site / "nvidia" / "cuda_cupti" / "lib").mkdir(parents=True)
    (site / "nvidia" / "cudnn" / "lib").mkdir(parents=True)

    dirs = _venv_cuda_lib_dirs({"VIRTUAL_ENV": str(venv)})
    assert str(site / "torch" / "lib") in dirs
    assert str(site / "nvidia" / "cuda_cupti" / "lib") in dirs   # the libcupti home
    assert str(site / "nvidia" / "cudnn" / "lib") in dirs
    assert len(dirs) == len(set(dirs))                            # de-duped


def test_venv_cuda_lib_dirs_no_venv_is_empty() -> None:
    from backend.services.runtime.local_process import _venv_cuda_lib_dirs

    assert _venv_cuda_lib_dirs({}) == []
    assert _venv_cuda_lib_dirs({"VIRTUAL_ENV": "/does/not/exist"}) == []


def test_venv_cuda_lib_dirs_prefers_experiment_venv(tmp_path: Path) -> None:
    from backend.services.runtime.local_process import _venv_cuda_lib_dirs

    exp = tmp_path / "exp"
    (exp / "lib" / "python3.12" / "site-packages" / "torch" / "lib").mkdir(parents=True)
    dirs = _venv_cuda_lib_dirs({"OPENRESEARCH_EXPERIMENT_VENV": str(exp), "VIRTUAL_ENV": "/nope"})
    assert any(str(exp) in d for d in dirs)


def test_venv_cuda_lib_dirs_follows_base_inherit_pth(tmp_path: Path) -> None:
    """The Codex-review Q1 gap: the batch per-run venv ships a ``.pth`` pointing at the
    repo ``.venv``'s site-packages, where the shared cu121 torch PHYSICALLY lives. After
    env_pin strips the agent's re-pin the per-run venv has NO torch of its own — so the
    dirs glob MUST follow the ``.pth`` or the LD_LIBRARY_PATH prepend is a silent no-op."""
    from backend.services.runtime.local_process import _venv_cuda_lib_dirs

    base = tmp_path / "base.venv"
    base_site = base / "lib" / "python3.12" / "site-packages"
    (base_site / "torch" / "lib").mkdir(parents=True)
    (base_site / "nvidia" / "cuda_cupti" / "lib").mkdir(parents=True)

    run = tmp_path / "run.venv"
    run_site = run / "lib" / "python3.12" / "site-packages"
    run_site.mkdir(parents=True)  # per-run venv has NO torch of its own
    (run_site / "_reprolab_base_inherit.pth").write_text(str(base_site) + "\n", encoding="utf-8")

    dirs = _venv_cuda_lib_dirs({"OPENRESEARCH_EXPERIMENT_VENV": str(run)})
    assert str(base_site / "torch" / "lib") in dirs            # followed the .pth to the base
    assert str(base_site / "nvidia" / "cuda_cupti" / "lib") in dirs


def test_venv_cuda_lib_dirs_own_torch_ranks_before_base(tmp_path: Path) -> None:
    """A paper-installed torch in the per-run venv must win the prepend (it shadows the
    base on sys.path), so own-venv dirs rank BEFORE the .pth base dirs."""
    from backend.services.runtime.local_process import _venv_cuda_lib_dirs

    base = tmp_path / "base.venv"
    base_site = base / "lib" / "python3.12" / "site-packages"
    (base_site / "torch" / "lib").mkdir(parents=True)

    run = tmp_path / "run.venv"
    run_site = run / "lib" / "python3.12" / "site-packages"
    (run_site / "torch" / "lib").mkdir(parents=True)
    (run_site / "_reprolab_base_inherit.pth").write_text(str(base_site) + "\n", encoding="utf-8")

    dirs = _venv_cuda_lib_dirs({"OPENRESEARCH_EXPERIMENT_VENV": str(run)})
    assert dirs.index(str(run_site / "torch" / "lib")) < dirs.index(str(base_site / "torch" / "lib"))


def test_venv_cuda_lib_dirs_skips_executable_pth_lines(tmp_path: Path) -> None:
    """``.pth`` files routinely carry ``import …`` lines (distutils-precedence, editable
    installs). Those are executed by the import system, not paths — they must be skipped,
    never ``Path()``'d, and must not raise."""
    from backend.services.runtime.local_process import _venv_cuda_lib_dirs

    run = tmp_path / "venv"
    site = run / "lib" / "python3.12" / "site-packages"
    site.mkdir(parents=True)
    (site / "distutils-precedence.pth").write_text(
        "import os; os.environ.setdefault('OPENRESEARCH_X', '1')\n", encoding="utf-8"
    )
    (site / "comment.pth").write_text("# not a path\n", encoding="utf-8")
    assert _venv_cuda_lib_dirs({"OPENRESEARCH_EXPERIMENT_VENV": str(run)}) == []  # no torch, no crash


# --------------------------------------------------------------------------- C
@pytest.mark.parametrize("lib", ["libcupti.so.12", "libcudart.so.12", "libnvrtc.so.12"])
def test_cuda_shlib_load_classified(lib: str) -> None:
    from backend.agents.rlm.failure_classifier import classify_failure

    k, fix = classify_failure(
        {"success": False, "logs": f"ImportError: {lib}: cannot open shared object file: No such file or directory"}
    )
    assert k == "cuda_shlib_load"
    assert "torch" in fix.lower()  # the fix points at the torch re-pin


def test_real_nccl_timeout_not_misclassified_as_shlib() -> None:
    from backend.agents.rlm.failure_classifier import classify_failure

    k, _ = classify_failure(
        {"success": False, "logs": "NCCL watchdog: collective operation timed out after 600000 ms"}
    )
    assert k == "nccl_timeout"  # a real hang keeps its class (shlib is checked AFTER)


def test_cuda_shlib_load_is_repairable() -> None:
    from backend.agents.rlm.primitives import _RUN_EXPERIMENT_REPAIRABLE_FAILURES

    assert "cuda_shlib_load" in _RUN_EXPERIMENT_REPAIRABLE_FAILURES


def test_cuda_shlib_load_in_canonical_classes() -> None:
    from backend.agents.rlm.failure_classifier import FAILURE_CLASSES

    assert "cuda_shlib_load" in FAILURE_CLASSES


# --------------------------------------------------------------------------- D
def test_cell_scheduler_shipped_with_harness_helpers() -> None:
    from backend.agents.baseline_implementation import _HARNESS_CODE_HELPERS

    assert "cell_scheduler.py" in _HARNESS_CODE_HELPERS  # else flat `import` falls to backend.*
    assert "gpu_cell_runner.py" in _HARNESS_CODE_HELPERS


def test_gpu_cell_runner_import_is_guarded() -> None:
    import backend.agents.rlm.gpu_cell_runner as g  # in-repo import must still resolve

    src = Path(g.__file__).read_text(encoding="utf-8")
    assert "from cell_scheduler import" in src        # flat-first (sandbox)
    assert "except ImportError" in src                 # in-repo fallback


# --------------------------------------------------------------------------- E
def test_preflight_flags_backend_import_with_hint(tmp_path: Path) -> None:
    from backend.agents.rlm import preflight_smoke

    (tmp_path / "train.py").write_text(
        "from backend.agents.rlm.provenance import emit_provenance\nimport os\n",
        encoding="utf-8",
    )
    target = preflight_smoke.emit(tmp_path)
    r = subprocess.run(
        [sys.executable, target.name], cwd=tmp_path, capture_output=True, text=True
    )
    assert r.returncode == 3
    result = json.loads((tmp_path / "preflight_smoke_result.json").read_text(encoding="utf-8"))
    backend_fail = [f for f in result["failures"] if f["module"] == "backend"]
    assert backend_fail, result
    assert backend_fail[0]["error_type"] == "SandboxImportError"
    assert "flat" in backend_fail[0]["error"].lower()
    assert "provenance" in backend_fail[0]["error"].lower()  # names the correct flat import


def test_preflight_clean_imports_pass(tmp_path: Path) -> None:
    from backend.agents.rlm import preflight_smoke

    # only stdlib + a flat helper name (provenance is a local .py here → skipped)
    (tmp_path / "provenance.py").write_text("def emit_provenance(*a, **k):\n    pass\n", encoding="utf-8")
    (tmp_path / "train.py").write_text(
        "import os, json\nfrom provenance import emit_provenance\n", encoding="utf-8"
    )
    target = preflight_smoke.emit(tmp_path)
    r = subprocess.run(
        [sys.executable, target.name], cwd=tmp_path, capture_output=True, text=True
    )
    assert r.returncode == 0, r.stderr
