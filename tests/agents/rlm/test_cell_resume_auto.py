"""Tests for OPENRESEARCH_CELL_RESUME_AUTO — the checkpoint/resume auto-enable fix.

The cell-level resume machinery (``gpu_cell_runner.should_skip_cell`` /
``write_cell_manifest`` / ``cell_fingerprint.compute_fingerprint``) is already
correct and already unit-tested (``test_gpu_cell_runner_resume.py``). It is
defeated by two independent, default-off gaps:

* **Gap 2** (``primitives.py::run_experiment``) — a fresh uuid-suffixed
  ``run_id`` is minted on EVERY call, so ``code/outputs/<run_id>/`` changes
  call-to-call within the SAME attempt and ``should_skip_cell`` never finds
  the prior call's ``cell_manifest.json``.
* **Gap 1** (``run.py::_maybe_auto_arm_cell_resume``) — arms
  ``OPENRESEARCH_RESUME_CELLS`` on a detected warm/incomplete restart but never
  stabilizes ``run_id``, so even a genuine process-restart resume still falls
  into Gap 2.

``OPENRESEARCH_CELL_RESUME_AUTO`` (default OFF) closes both: it reuses
``ctx.project_id`` as a STABLE run_id (byte-identical to what
``OPENRESEARCH_STABLE_RUN_ID`` already produces) and arms
``OPENRESEARCH_RESUME_CELLS`` (site 1), and — under the same flag —
``_maybe_auto_arm_cell_resume`` ALSO arms ``OPENRESEARCH_STABLE_RUN_ID`` (site 2).
Both sites respect an explicit operator override (any value, including "0").

No GPU/CUDA is used anywhere in this file — ``train_cell.py`` is a trivial
sentinel script (mirroring ``test_gpu_cell_runner_resume.py``'s fixture
pattern) that proves a launch happened or did not.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from backend.agents.rlm import gpu_cell_runner, primitives
from backend.agents.rlm.run import _maybe_auto_arm_cell_resume

# ---------------------------------------------------------------------------
# Isolation — the code under test mutates os.environ DIRECTLY
# (os.environ.setdefault / os.environ[...] = "1"), which plain
# monkeypatch.setenv/delenv cannot auto-revert unless monkeypatch itself made
# the change. This suite runs FIRST in the specified 5-file invocation, ahead
# of test_run_experiment_cell_route.py (which asserts an un-suffixed run_id
# would break under a leaked OPENRESEARCH_STABLE_RUN_ID) — save + restore the
# real prior state around every test so nothing here can leak into a sibling
# file's test in the same pytest process.
# ---------------------------------------------------------------------------

_ENV_VARS = (
    "OPENRESEARCH_CELL_RESUME_AUTO",
    "OPENRESEARCH_STABLE_RUN_ID",
    "OPENRESEARCH_RESUME_CELLS",
    "OPENRESEARCH_CELLS_SEED_PATH",
)


@pytest.fixture(autouse=True)
def _isolate_resume_env():
    saved = {v: os.environ.get(v) for v in _ENV_VARS}
    for v in _ENV_VARS:
        os.environ.pop(v, None)
    try:
        yield
    finally:
        for v in _ENV_VARS:
            if saved[v] is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = saved[v]


@pytest.fixture(autouse=True)
def _no_events(monkeypatch):
    monkeypatch.setattr(primitives, "_emit_dashboard_event", lambda *a, **k: None)


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

def _caps(per_gpu=23.68, n=2, backend="local"):
    return SimpleNamespace(
        backend_kind=backend, num_gpus=n, per_gpu_vram_gb=per_gpu,
        free_gpu_ids=tuple(f"GPU-{i}" for i in range(n)), is_empty=(n <= 0),
    )


def _run_ctx(project_id: str, project_dir: Path) -> SimpleNamespace:
    """A minimal, EXPLICIT RunContext double.

    Every attribute ``run_experiment``'s cell-route path touches by DIRECT
    dot-access (not ``getattr(ctx, ..., default)``) is set explicitly;
    everything else is intentionally left unset so ``getattr``'s own default
    applies — exactly like the real ``RunContext``'s None/empty defaults.
    """
    return SimpleNamespace(
        project_id=project_id,
        project_dir=project_dir,
        sandbox_mode="local",
        run_budget=None,
        remaining_s=lambda: None,
        gpu_device_ids=(),
    )


_CELL_A = {"id": "cellA", "model_key": "qwen3_1_7b", "baseline": "sdar",
           "env": "search_qa", "seed": 42, "est_vram_gb": 4.0}
_CELL_B = {"id": "cellB", "model_key": "qwen3_1_7b", "baseline": "grpo",
           "env": "search_qa", "seed": 42, "est_vram_gb": 4.0}


def _write_cells(code_dir: Path, cells: list) -> None:
    code_dir.mkdir(parents=True, exist_ok=True)
    (code_dir / "cells.json").write_text(json.dumps({"cells": cells}), encoding="utf-8")


# A REAL single-cell trainer (mirror of test_gpu_cell_runner_resume.py's
# sentinel script): appends a launch sentinel EVERY time it runs (so
# re-launches are visible even across separate run_experiment calls), succeeds
# for "cellA", always OOMs for "cellB". Never touches CUDA/torch.
_SENTINEL_TRAIN_CELL_SRC = '''\
import json
import os
import sys

out = os.environ["OPENRESEARCH_CELL_OUTPUT_DIR"]
os.makedirs(out, exist_ok=True)
params = json.loads(os.environ.get("OPENRESEARCH_CELL_PARAMS", "{}"))
cell_id = params.get("id", "")

with open(os.path.join(out, "launched.sentinel"), "a", encoding="utf-8") as fh:
    fh.write("x")

if cell_id == "cellB":
    sys.stderr.write("CUDA out of memory. Tried to allocate 2.00 GiB\\n")
    sys.exit(1)

with open(os.path.join(out, "metrics.json"), "w", encoding="utf-8") as fh:
    json.dump({"status": "ok", "metric": 0.75, "reward_mean": 1.0}, fh)
sys.exit(0)
'''


def _write_sentinel_trainer(code_dir: Path) -> None:
    (code_dir / "train_cell.py").write_text(_SENTINEL_TRAIN_CELL_SRC, encoding="utf-8")


def _sentinel_launches(code_dir: Path, run_id: str, cell_id: str) -> int:
    """Launch count for one specific (run_id, cell_id) output dir."""
    p = code_dir / "outputs" / run_id / cell_id / "launched.sentinel"
    if not p.exists():
        return 0
    return len(p.read_text(encoding="utf-8"))


def _all_sentinel_launches(code_dir: Path, cell_id: str) -> int:
    """Sum launch counts across EVERY outputs/<run_id>/<cell_id>/ dir — used
    when run_id is NOT stable (flag off), so each call gets a fresh dir."""
    outputs = code_dir / "outputs"
    if not outputs.is_dir():
        return 0
    total = 0
    for run_dir in outputs.iterdir():
        p = run_dir / cell_id / "launched.sentinel"
        if p.exists():
            total += len(p.read_text(encoding="utf-8"))
    return total


def _distinct_output_dirs(code_dir: Path) -> set:
    outputs = code_dir / "outputs"
    if not outputs.is_dir():
        return set()
    return {p.name for p in outputs.iterdir() if p.is_dir()}


# ---------------------------------------------------------------------------
# 1. RED regression — pins Gap 2 (run_id churn) directly via a mocked
#    gpu_cell_runner.run_matrix (no subprocess needed for this one).
# ---------------------------------------------------------------------------

class TestRunIdStabilityAcrossRepeatedCalls:
    def test_default_mints_a_fresh_run_id_every_call(self, tmp_path, monkeypatch):
        """Pins Gap 2: two run_experiment calls on the SAME ctx/code_dir get a
        DIFFERENT output_root by default — should_skip_cell can never find the
        first call's manifest."""
        code = tmp_path / "code"
        _write_cells(code, [_CELL_A])
        _write_sentinel_trainer(code)

        seen_roots: list = []

        def fake_run_matrix(cells, script, *, output_root, **kw):
            seen_roots.append(str(output_root))
            return {c["id"]: {"status": "ok", "metrics": {"metric": 0.5},
                              "gpu": "GPU-0", "retries": 0, "error": None}
                    for c in cells}

        monkeypatch.setattr(gpu_cell_runner, "run_matrix", fake_run_matrix)
        ctx = _run_ctx("prj_gap2", tmp_path)

        with patch("backend.services.runtime.gpu_capacity.describe_capacity",
                   return_value=_caps()):
            primitives.run_experiment(str(code), env_id="", ctx=ctx)
            primitives.run_experiment(str(code), env_id="", ctx=ctx)

        assert len(seen_roots) == 2
        assert seen_roots[0] != seen_roots[1]
        assert os.environ.get("OPENRESEARCH_RESUME_CELLS") is None

    def test_cell_resume_auto_stabilizes_run_id_and_arms_resume(self, tmp_path, monkeypatch):
        """Fixed: with the new flag on, both calls get the SAME output_root
        (= ctx.project_id) and OPENRESEARCH_RESUME_CELLS is armed."""
        monkeypatch.setenv("OPENRESEARCH_CELL_RESUME_AUTO", "1")
        code = tmp_path / "code"
        _write_cells(code, [_CELL_A])
        _write_sentinel_trainer(code)

        seen_roots: list = []

        def fake_run_matrix(cells, script, *, output_root, **kw):
            seen_roots.append(str(output_root))
            return {c["id"]: {"status": "ok", "metrics": {"metric": 0.5},
                              "gpu": "GPU-0", "retries": 0, "error": None}
                    for c in cells}

        monkeypatch.setattr(gpu_cell_runner, "run_matrix", fake_run_matrix)
        ctx = _run_ctx("prj_gap2_fixed", tmp_path)

        with patch("backend.services.runtime.gpu_capacity.describe_capacity",
                   return_value=_caps()):
            primitives.run_experiment(str(code), env_id="", ctx=ctx)
            primitives.run_experiment(str(code), env_id="", ctx=ctx)

        assert len(seen_roots) == 2
        assert seen_roots[0] == seen_roots[1] == str(code / "outputs" / "prj_gap2_fixed")
        assert os.environ.get("OPENRESEARCH_RESUME_CELLS") == "1"

    def test_explicit_resume_cells_zero_still_wins(self, tmp_path, monkeypatch):
        """An operator's explicit OPENRESEARCH_RESUME_CELLS=0 must NOT be
        clobbered by the auto-arm (setdefault, not unconditional set)."""
        monkeypatch.setenv("OPENRESEARCH_CELL_RESUME_AUTO", "1")
        monkeypatch.setenv("OPENRESEARCH_RESUME_CELLS", "0")
        code = tmp_path / "code"
        _write_cells(code, [_CELL_A])
        _write_sentinel_trainer(code)

        monkeypatch.setattr(
            gpu_cell_runner, "run_matrix",
            lambda cells, script, **kw: {
                c["id"]: {"status": "ok", "metrics": {"metric": 0.5}, "gpu": "GPU-0",
                          "retries": 0, "error": None} for c in cells
            },
        )
        ctx = _run_ctx("prj_gap2_explicit0", tmp_path)
        with patch("backend.services.runtime.gpu_capacity.describe_capacity",
                   return_value=_caps()):
            primitives.run_experiment(str(code), env_id="", ctx=ctx)

        assert os.environ.get("OPENRESEARCH_RESUME_CELLS") == "0"


# ---------------------------------------------------------------------------
# 2. End-to-end — REAL gpu_cell_runner.run_matrix + REAL sentinel train_cell.
# ---------------------------------------------------------------------------

class TestEndToEndResumeSkipsOnlyCompletedCells:
    def test_flag_unset_control_relaunches_completed_cell(self, tmp_path):
        """Today's bug, pinned as an explicit control: with the flag OFF, the
        already-succeeded cell A is relaunched on the second call too (2
        launches total, in two DIFFERENT output dirs)."""
        code = tmp_path / "code"
        _write_cells(code, [_CELL_A, _CELL_B])
        _write_sentinel_trainer(code)
        ctx = _run_ctx("prj_e2e_control", tmp_path)

        with patch("backend.services.runtime.gpu_capacity.describe_capacity",
                   return_value=_caps()):
            primitives.run_experiment(str(code), env_id="", ctx=ctx)
            primitives.run_experiment(str(code), env_id="", ctx=ctx)

        assert _all_sentinel_launches(code, "cellA") == 2
        assert len(_distinct_output_dirs(code)) == 2

    def test_cell_resume_auto_skips_completed_cell_reruns_failed(self, tmp_path, monkeypatch):
        """The fix: with the flag on, cell A (already ok) launches EXACTLY
        ONCE across both calls; cell B (always OOMs) relaunches; and the
        second call's aggregated code/metrics.json still carries cell A's
        original metric."""
        monkeypatch.setenv("OPENRESEARCH_CELL_RESUME_AUTO", "1")
        code = tmp_path / "code"
        _write_cells(code, [_CELL_A, _CELL_B])
        _write_sentinel_trainer(code)
        ctx = _run_ctx("prj_e2e_fixed", tmp_path)

        with patch("backend.services.runtime.gpu_capacity.describe_capacity",
                   return_value=_caps()):
            primitives.run_experiment(str(code), env_id="", ctx=ctx)
            primitives.run_experiment(str(code), env_id="", ctx=ctx)

        assert _sentinel_launches(code, ctx.project_id, "cellA") == 1
        assert _sentinel_launches(code, ctx.project_id, "cellB") >= 2

        on_disk = json.loads((code / "metrics.json").read_text())
        leaf = on_disk["per_model"]["qwen3_1_7b"]["search_qa"]
        assert leaf["sdar"]["status"] == "ok"
        assert leaf["sdar"]["metric"] == 0.75


class TestFingerprintInvalidationControl:
    def test_edited_helper_file_forces_rerun_even_with_resume_auto(self, tmp_path, monkeypatch):
        """The 'unchanged cells only' guarantee holds: editing a fingerprinted
        helper file between calls invalidates cell A's fingerprint, so it
        re-runs even with OPENRESEARCH_CELL_RESUME_AUTO on."""
        monkeypatch.setenv("OPENRESEARCH_CELL_RESUME_AUTO", "1")
        code = tmp_path / "code"
        _write_cells(code, [_CELL_A])
        _write_sentinel_trainer(code)
        (code / "search_qa_env.py").write_text("# search_qa_env.py v1\n", encoding="utf-8")
        ctx = _run_ctx("prj_fp_invalidate", tmp_path)

        with patch("backend.services.runtime.gpu_capacity.describe_capacity",
                   return_value=_caps()):
            primitives.run_experiment(str(code), env_id="", ctx=ctx)
            # Mutate the fingerprinted env helper BETWEEN calls.
            (code / "search_qa_env.py").write_text(
                "# search_qa_env.py v2 — behavior changed\n", encoding="utf-8")
            primitives.run_experiment(str(code), env_id="", ctx=ctx)

        assert _sentinel_launches(code, ctx.project_id, "cellA") == 2


# ---------------------------------------------------------------------------
# 3. Site 2 — _maybe_auto_arm_cell_resume also stabilizes run_id under the
#    same new flag (run.py, Gap 1's genuine-process-restart sibling).
# ---------------------------------------------------------------------------

class TestAutoArmAlsoStabilizesRunId:
    def test_flag_off_does_not_set_stable_run_id(self, tmp_path):
        (tmp_path / "rlm_state").mkdir()
        result = _maybe_auto_arm_cell_resume(tmp_path)
        assert result is True
        assert os.environ.get("OPENRESEARCH_RESUME_CELLS") == "1"
        assert os.environ.get("OPENRESEARCH_STABLE_RUN_ID") is None

    def test_flag_on_also_arms_stable_run_id(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENRESEARCH_CELL_RESUME_AUTO", "1")
        (tmp_path / "rlm_state").mkdir()
        result = _maybe_auto_arm_cell_resume(tmp_path)
        assert result is True
        assert os.environ.get("OPENRESEARCH_RESUME_CELLS") == "1"
        assert os.environ.get("OPENRESEARCH_STABLE_RUN_ID") == "1"

    def test_flag_on_explicit_stable_run_id_zero_still_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENRESEARCH_CELL_RESUME_AUTO", "1")
        monkeypatch.setenv("OPENRESEARCH_STABLE_RUN_ID", "0")
        (tmp_path / "rlm_state").mkdir()
        result = _maybe_auto_arm_cell_resume(tmp_path)
        assert result is True
        assert os.environ.get("OPENRESEARCH_STABLE_RUN_ID") == "0"

    def test_flag_on_no_prior_attempt_does_not_arm_anything(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENRESEARCH_CELL_RESUME_AUTO", "1")
        result = _maybe_auto_arm_cell_resume(tmp_path)
        assert result is False
        assert os.environ.get("OPENRESEARCH_RESUME_CELLS") is None
        assert os.environ.get("OPENRESEARCH_STABLE_RUN_ID") is None
