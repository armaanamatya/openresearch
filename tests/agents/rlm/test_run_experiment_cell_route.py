"""Integration tests for the harness-owned cell-runner route (comp 4).

Exercises `_execute_cell_matrix` directly with a mocked `gpu_cell_runner.run_matrix`
so no GPU/subprocess is needed. Covers: leaf-shaped aggregation persisted to disk,
partial success, terminal oom_shrink_exhausted / capacity_exhausted stops, the
capacity gate dropping an over-budget model, and OOM-marker-safe logs.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from backend.agents.rlm import primitives
from backend.agents.rlm import gpu_cell_runner


def _caps(per_gpu=23.68, n=2, backend="local"):
    return SimpleNamespace(
        backend_kind=backend, num_gpus=n, per_gpu_vram_gb=per_gpu,
        free_gpu_ids=tuple(f"GPU-{i}" for i in range(n)), is_empty=(n <= 0),
    )


def _ctx(tmp_path):
    return SimpleNamespace(
        project_id="prj_test", project_dir=tmp_path, run_id="prj_test-abc",
        gpu_device_ids=(),
    )


def _write_cells(code_dir, cells):
    code_dir.mkdir(parents=True, exist_ok=True)
    (code_dir / "cells.json").write_text(json.dumps({"cells": cells}), encoding="utf-8")
    (code_dir / "train_cell.py").write_text("# single-cell trainer\n", encoding="utf-8")


_SMALL = {"id": "qwen3_1_7b__sdar__search_qa__s42", "model_key": "qwen3_1_7b",
          "baseline": "sdar", "env": "search_qa", "seed": 42, "est_vram_gb": 14.0}
_SMALL2 = {"id": "qwen3_1_7b__grpo__search_qa__s42", "model_key": "qwen3_1_7b",
           "baseline": "grpo", "env": "search_qa", "seed": 42, "est_vram_gb": 14.0}
_BIG = {"id": "qwen2_5_7b__sdar__search_qa__s42", "model_key": "qwen2_5_7b",
        "baseline": "sdar", "env": "search_qa", "seed": 42, "est_vram_gb": 28.0}


@pytest.fixture(autouse=True)
def _no_events(monkeypatch):
    monkeypatch.setattr(primitives, "_emit_dashboard_event", lambda *a, **k: None)


def test_partial_success_aggregates_and_persists_leaf_shape(tmp_path, monkeypatch):
    code = tmp_path / "code"
    _write_cells(code, [_SMALL, _SMALL2])

    def fake_run_matrix(cells, script, **kw):
        return {
            _SMALL["id"]: {"status": "ok", "metrics": {"status": "ok", "metric": 0.42, "steps_run": 50},
                           "gpu": "GPU-0", "retries": 0, "error": None},
            _SMALL2["id"]: {"status": "oom_failed", "metrics": None,
                            "gpu": "GPU-1", "retries": 2, "error": "CUDA out of memory. Tried to allocate..."},
        }
    monkeypatch.setattr(gpu_cell_runner, "run_matrix", fake_run_matrix)

    res = primitives._execute_cell_matrix(_ctx(tmp_path), str(code), _caps(), timeout_s=60, run_id="prj_test-rid")

    assert res["success"] is True  # one ok cell => real metrics to score
    leaf = res["metrics"]["per_model"]["qwen3_1_7b"]["search_qa"]
    assert leaf["sdar"]["status"] == "ok" and leaf["sdar"]["metric"] == 0.42
    assert leaf["grpo"]["status"] == "failed"
    # Aggregated metrics persisted where the scorer reads them.
    assert (code / "metrics.json").is_file()
    assert (code / "outputs" / "prj_test-rid" / "metrics.json").is_file()
    on_disk = json.loads((code / "metrics.json").read_text())
    assert on_disk["per_model"]["qwen3_1_7b"]["search_qa"]["sdar"]["metric"] == 0.42
    # The top-level logs must NOT carry raw OOM markers (would misfire silent_oom).
    low = res["logs"].lower()
    for marker in primitives._OOM_LOG_MARKERS:
        assert marker not in low


def test_all_cells_oom_is_terminal_stop(tmp_path, monkeypatch):
    code = tmp_path / "code"
    _write_cells(code, [_SMALL, _SMALL2])

    def fake_run_matrix(cells, script, **kw):
        return {c["id"]: {"status": "oom_failed", "metrics": None, "gpu": "GPU-0",
                          "retries": 2, "error": "CUDA out of memory"} for c in cells}
    monkeypatch.setattr(gpu_cell_runner, "run_matrix", fake_run_matrix)

    res = primitives._execute_cell_matrix(_ctx(tmp_path), str(code), _caps(), timeout_s=60, run_id="rid2")

    assert res["success"] is False
    assert res["failure_class"] == "oom_shrink_exhausted"
    assert res["stop_reason"]["kind"] == "oom_shrink_exhausted"
    assert (code / "metrics.json").is_file()  # report still written


def test_capacity_gate_drops_over_budget_model(tmp_path, monkeypatch):
    code = tmp_path / "code"
    _write_cells(code, [_SMALL, _BIG])  # 24GB card: 14GB fits, 28GB does not

    seen = {}

    def fake_run_matrix(cells, script, **kw):
        seen["ids"] = [c["id"] for c in cells]
        return {c["id"]: {"status": "ok", "metrics": {"metric": 0.5}, "gpu": "GPU-0",
                          "retries": 0, "error": None} for c in cells}
    monkeypatch.setattr(gpu_cell_runner, "run_matrix", fake_run_matrix)

    res = primitives._execute_cell_matrix(_ctx(tmp_path), str(code), _caps(per_gpu=23.68),
                                          timeout_s=60, run_id="rid3")

    # The 7B cell was dropped BEFORE run_matrix; only the 1.7B ran.
    assert seen["ids"] == [_SMALL["id"]]
    assert "qwen2_5_7b" in res["metrics"]["scope"]["models_skipped"]
    assert res["success"] is True


def test_all_dropped_is_capacity_exhausted(tmp_path, monkeypatch):
    code = tmp_path / "code"
    _write_cells(code, [_BIG])  # the only cell can't fit one card

    monkeypatch.setattr(gpu_cell_runner, "run_matrix",
                        lambda *a, **k: pytest.fail("run_matrix must not run when all cells are dropped"))

    res = primitives._execute_cell_matrix(_ctx(tmp_path), str(code), _caps(per_gpu=23.68),
                                          timeout_s=60, run_id="rid4")
    assert res["success"] is False
    assert res["failure_class"] == "capacity_exhausted"
    assert res["stop_reason"]["kind"] == "capacity_exhausted"


def test_missing_cells_json_returns_contract_guard(tmp_path, monkeypatch):
    code = tmp_path / "code"
    code.mkdir(parents=True)
    # no cells.json
    res = primitives._execute_cell_matrix(_ctx(tmp_path), str(code), _caps(), timeout_s=60, run_id="rid5")
    assert res["success"] is False
    assert res["failure_class"] == "contract_guard"
