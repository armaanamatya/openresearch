"""Hybrid exec route — cells.json grid + commands.json families in ONE call.

Pins the Adam v6 lesson: the routes were mutually exclusive per
run_experiment call, so a multi-family paper burned a full iteration renaming
cells.json aside to reach its train.py families — v6 instead re-ran the whole
60-cell grid and died at the watchdog.
"""

from __future__ import annotations

import json

from backend.agents.rlm.primitives import _hybrid_route_enabled, _merge_hybrid_results

GRID = {
    "success": True,
    "logs": "cell-matrix: 60 cells",
    "contract_warnings": ["axes derived"],
    "metrics": {
        "status": "partial",
        "per_model": {"vae_bias_correction": {"default": {"adam": {"status": "ok", "metric": -101.2}}}},
        "scope": {"gaps": [{"item": "vae_b2_1", "reason": "diverged"}]},
    },
}


def test_flag_default_off(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_HYBRID_EXEC_ROUTE", raising=False)
    assert _hybrid_route_enabled() is False
    monkeypatch.setenv("OPENRESEARCH_HYBRID_EXEC_ROUTE", "1")
    assert _hybrid_route_enabled() is True
    monkeypatch.setenv("OPENRESEARCH_HYBRID_EXEC_ROUTE", "off")
    assert _hybrid_route_enabled() is False


def test_merge_grafts_grid_per_model_into_family_metrics(tmp_path):
    cmd = {
        "success": True,
        "logs": "families done",
        "metrics": {
            "status": "complete",
            "per_dataset": {"mnist": {"adam": {"nll": 0.31}}},
            "per_model": {"mnist_logreg": {"acc": 0.91}},
        },
    }
    out = _merge_hybrid_results(GRID, cmd, str(tmp_path))
    m = out["metrics"]
    assert m["per_dataset"]["mnist"]["adam"]["nll"] == 0.31        # agent base kept
    assert m["per_model"]["mnist_logreg"]["acc"] == 0.91           # agent families kept
    assert "vae_bias_correction" in m["per_model"]                  # grid grafted
    assert m["scope"]["gaps"] == [{"item": "vae_b2_1", "reason": "diverged"}]
    assert out["success"] is True
    assert "hybrid route" in out["logs"] and "cell-matrix" in out["logs"]
    assert out["contract_warnings"] == ["axes derived"]
    # merged blob persisted for the scorer
    on_disk = json.loads((tmp_path / "metrics.json").read_text())
    assert "vae_bias_correction" in on_disk["per_model"]


def test_failed_commands_keeps_grid_evidence_and_stays_repairable(tmp_path):
    cmd = {"success": False, "error": "train.py exit 1", "metrics": {}, "logs": "boom"}
    out = _merge_hybrid_results(GRID, cmd, str(tmp_path))
    assert out["success"] is False                                  # honest failure
    assert "grid SUCCEEDED" in out["error"]                         # repair scoped to families
    assert "vae_bias_correction" in out["metrics"]["per_model"]     # grid preserved


def test_agent_keys_never_overwritten(tmp_path):
    cmd = {
        "success": True,
        "metrics": {"per_model": {"vae_bias_correction": {"agent": "wrote this"}}},
        "logs": "",
    }
    out = _merge_hybrid_results(GRID, cmd, str(tmp_path))
    assert out["metrics"]["per_model"]["vae_bias_correction"] == {"agent": "wrote this"}
