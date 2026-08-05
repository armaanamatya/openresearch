"""EKS reproduction routing is cell-matrix only; never generic monolithic Jobs."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from backend.agents.rlm import primitives
from backend.agents.rlm import gke_cell_synth


def _ctx(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        project_id="project-a", project_dir=tmp_path, run_id="run-a",
        gpu_device_ids=(), sandbox_mode="aws",
    )


def _caps(*, empty: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        backend_kind="aws", num_gpus=0 if empty else 1,
        per_gpu_vram_gb=0.0 if empty else 80.0,
        free_gpu_ids=("0",) if not empty else (), is_empty=empty,
        detail={"configuration_error": "aws_gpu_usd_per_hour must be > 0"} if empty else {},
    )


def _common(monkeypatch):
    monkeypatch.setattr(primitives, "_emit_dashboard_event", lambda *a, **k: None)
    monkeypatch.setattr(
        "backend.agents.rlm.pre_flight_validator.validate_code_pre_flight", lambda *a, **k: []
    )


def test_aws_cells_dispatch_to_k8s_runner_not_generic_exec(monkeypatch, tmp_path: Path):
    _common(monkeypatch)
    code = tmp_path / "code"
    code.mkdir()
    (code / "cells.json").write_text(json.dumps({"cells": [{
        "id": "c", "model_key": "m", "baseline": "b", "env": "e", "est_vram_gb": 1,
    }]}))
    (code / "train_cell.py").write_text("# cell\n")
    seen = {"matrix": False}

    def matrix(*_args, **_kwargs):
        seen["matrix"] = True
        return {"success": False, "metrics": {}, "failure_class": "test"}

    monkeypatch.setattr(primitives, "_execute_cell_matrix", matrix)
    with patch("backend.services.runtime.gpu_capacity.describe_capacity", return_value=_caps()):
        primitives.run_experiment(str(code), env_id="prebaked", ctx=_ctx(tmp_path))

    assert seen["matrix"] is True


def test_aws_without_cells_returns_contract_error_before_generic_exec(monkeypatch, tmp_path: Path):
    _common(monkeypatch)
    monkeypatch.delenv("OPENRESEARCH_EKS_SYNTH_CELL", raising=False)
    code = tmp_path / "code"
    code.mkdir()
    (code / "commands.json").write_text(json.dumps(["python train.py"]))
    monkeypatch.setattr(
        primitives, "_execute_in_sandbox", lambda *a, **k: (_ for _ in ()).throw(AssertionError("generic EKS exec"))
    )

    with patch("backend.services.runtime.gpu_capacity.describe_capacity", return_value=_caps()):
        result = primitives.run_experiment(str(code), env_id="prebaked", ctx=_ctx(tmp_path))

    assert result["failure_class"] == "contract_guard"
    assert "generic monolithic EKS" in result["error"]


def test_eks_synth_flag_is_default_off_and_then_creates_cell_shim(monkeypatch, tmp_path: Path):
    code = tmp_path / "code"
    code.mkdir()
    (code / "commands.json").write_text(json.dumps(["python train.py"]))
    monkeypatch.delenv("OPENRESEARCH_EKS_SYNTH_CELL", raising=False)
    assert gke_cell_synth.maybe_synthesize_k8s_cell(code, "aws") is None
    assert not (code / "cells.json").exists()

    monkeypatch.setenv("OPENRESEARCH_EKS_SYNTH_CELL", "true")
    assert gke_cell_synth.maybe_synthesize_k8s_cell(code, "aws") is not None
    assert (code / "cells.json").is_file()
    assert (code / "train_cell.py").is_file()
