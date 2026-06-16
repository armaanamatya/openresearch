"""Issue #3 (2026-06-15): HARNESS-owned provenance producer for the cell route.

emit_provenance is agent-facing; when the agent skips it, the recipe never reaches
the grader (All-CNN/Adam leaves stuck at 0.7: "weight_decay stated not verifiable",
"no artifacts confirm the lr search"). build_cell_provenance writes provenance.json
from cells.json + the aggregated metrics so the eval-protocol leaves can confirm the
recipe regardless. These tests pin the build, the lr_search capture, and the merge.
"""

from __future__ import annotations

import json
from pathlib import Path

from backend.agents.rlm import provenance


def _write(code: Path, cells: dict | list, metrics: dict | None = None) -> None:
    code.mkdir(parents=True, exist_ok=True)
    (code / "cells.json").write_text(json.dumps(cells), encoding="utf-8")
    if metrics is not None:
        (code / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")


def test_builds_per_cell_mechanical_facts(tmp_path):
    code = tmp_path / "code"
    cells = {"cells": [
        {"id": "a_base_cifar10_noaug", "model_key": "a_base", "env": "cifar10_noaug",
         "baseline": "base", "lr": 0.05, "epochs": 350, "seed": 42, "batch_size": 128,
         "augment": False, "use_zca": True},
    ]}
    metrics = {"per_model": {"a_base": {"cifar10_noaug": {"base": {
        "status": "ok", "best_lr": 0.05, "weight_decay": 0.001}}}}}
    _write(code, cells, metrics)
    provenance.build_cell_provenance(code, run_id="prj_test")
    prov = json.loads((code / "provenance.json").read_text())
    assert prov["source"] == "harness_cell_provenance"
    rec = prov["experiments"]["a_base_cifar10_noaug"]
    assert rec["model_key"] == "a_base" and rec["env"] == "cifar10_noaug"
    assert rec["lr"] == 0.05 and rec["epochs"] == 350 and rec["seed"] == 42
    # weight_decay not in the cell, but present in metrics → harness fills it.
    assert rec["weight_decay"] == 0.001


def test_captures_lr_search_grid(tmp_path):
    code = tmp_path / "code"
    cells = {
        "cells": [{"id": "c_allcnn", "model_key": "c_allcnn", "env": "cifar10", "baseline": "allcnn"}],
        "search": [{
            "group": "c_allcnn",
            "candidates": [
                {"id": "c_allcnn__lr_0.25", "params": {"lr": 0.25}},
                {"id": "c_allcnn__lr_0.1", "params": {"lr": 0.1}},
                {"id": "c_allcnn__lr_0.05", "params": {"lr": 0.05}},
                {"id": "c_allcnn__lr_0.01", "params": {"lr": 0.01}},
            ],
            "promote": {"id": "c_allcnn"},
        }],
    }
    _write(code, cells, {"per_model": {}})
    provenance.build_cell_provenance(code)
    prov = json.loads((code / "provenance.json").read_text())
    assert prov["lr_search"]["grid"] == [0.01, 0.05, 0.1, 0.25]
    assert prov["lr_search"]["groups"][0]["group"] == "c_allcnn"


def test_merges_agent_provenance_preserving_semantic_fields(tmp_path):
    code = tmp_path / "code"
    _write(code, {"cells": [
        {"id": "mnist_mlp", "model_key": "mnist_mlp", "env": "dropout", "baseline": "adam",
         "lr": 0.001, "epochs": 200}]},
        {"per_model": {}})
    # Agent already wrote a provenance.json with a SEMANTIC field the harness can't derive.
    (code / "provenance.json").write_text(json.dumps({
        "experiments": {"mnist_mlp": {"per_optimizer": {"adam": {"beta1": 0.9}}, "lr": 999}}
    }), encoding="utf-8")
    provenance.build_cell_provenance(code)
    prov = json.loads((code / "provenance.json").read_text())
    rec = prov["experiments"]["mnist_mlp"]
    assert rec["per_optimizer"] == {"adam": {"beta1": 0.9}}, "agent semantic field preserved"
    assert rec["lr"] == 999, "agent field wins on conflict"
    assert rec["epochs"] == 200, "harness fills the mechanical field the agent omitted"


def test_failsoft_on_missing_cells(tmp_path):
    code = tmp_path / "code"
    code.mkdir()
    # No cells.json at all → still writes a (near-empty) manifest, never raises.
    p = provenance.build_cell_provenance(code)
    assert p.exists()
    prov = json.loads(p.read_text())
    assert prov["experiments"] == {}


def test_real_allcnn_cells_smoke(tmp_path):
    """Real-data smoke: the actual ce9caf cells.json + metrics produce a sane manifest."""
    src = Path("runs/prj_0a3202fc187bb692/attempts/20260612T213230-651049-ce9caf/code")
    if not (src / "cells.json").is_file() or not (src / "metrics.json").is_file():
        import pytest
        pytest.skip("ce9caf run artifacts not present")
    code = tmp_path / "code"
    code.mkdir()
    (code / "cells.json").write_text((src / "cells.json").read_text(), encoding="utf-8")
    (code / "metrics.json").write_text((src / "metrics.json").read_text(), encoding="utf-8")
    provenance.build_cell_provenance(code, run_id="prj_real")
    prov = json.loads((code / "provenance.json").read_text())
    # The 14-cell All-CNN grid → 14 provenance records, each carrying its lr.
    assert len(prov["experiments"]) >= 10
    sample = next(iter(prov["experiments"].values()))
    assert "lr" in sample and "model_key" in sample
