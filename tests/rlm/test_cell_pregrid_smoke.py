"""U2/U3 — cell-aware pre-grid execution smoke decision logic.

Tests the smoke's status→block/proceed decision + the metrics-sanity check with a
mocked ``run_matrix`` (no GPU).  The smoke catches the All-CNN ``cell_execution_error``
on cell 1 and the ``degraded_no_metrics`` root cause, while never blocking on OOM /
timeout / its own infra flake (fail-soft).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from backend.agents.rlm import primitives
from backend.agents.rlm.primitives import (
    _cell_pregrid_smoke,
    _cell_smoke_repair,
    _smoke_metrics_violation,
)

_KEPT = [{"id": "small", "est_vram_gb": 1.0}, {"id": "big", "est_vram_gb": 8.0}]


def _mock_run_matrix(status: str, log: str = "", *, write_metrics: dict | None = None):
    def _fn(cells, cell_script, *, output_root, **kw):
        cid = cells[0]["id"]
        if write_metrics is not None:
            out = Path(output_root) / cid
            out.mkdir(parents=True, exist_ok=True)
            (out / "metrics.json").write_text(json.dumps(write_metrics), encoding="utf-8")
        return {cid: {"status": status, "log": log}}
    return _fn


def _run(monkeypatch, tmp_path, mock):
    from backend.agents.rlm import gpu_cell_runner as gcr
    monkeypatch.setattr(gcr, "run_matrix", mock)
    return _cell_pregrid_smoke(
        _KEPT, tmp_path, tmp_path / "outputs" / "run1", ["0"], 1, 300.0, ctx=None
    )


# --------------------------------------------------------------------------- #
# _smoke_metrics_violation (U3)
# --------------------------------------------------------------------------- #

def test_metrics_missing_is_flagged(tmp_path):
    assert _smoke_metrics_violation(tmp_path, "c") is not None  # no metrics.json anywhere


def test_metrics_nan_is_flagged(tmp_path):
    (tmp_path / "metrics.json").write_text('{"acc": NaN}'.replace("NaN", "Infinity"), encoding="utf-8")
    assert _smoke_metrics_violation(tmp_path, "c") is not None


def test_metrics_finite_is_ok(tmp_path):
    (tmp_path / "metrics.json").write_text('{"acc": 0.94, "loss": 0.1}', encoding="utf-8")
    assert _smoke_metrics_violation(tmp_path, "c") is None


def test_metrics_empty_dict_not_flagged(tmp_path):
    # a 1-step smoke may legitimately write partial metrics — don't false-positive
    (tmp_path / "metrics.json").write_text("{}", encoding="utf-8")
    assert _smoke_metrics_violation(tmp_path, "c") is None


# --------------------------------------------------------------------------- #
# _cell_smoke_repair shape
# --------------------------------------------------------------------------- #

def test_repair_is_repairable_not_terminal():
    r = _cell_smoke_repair("cell_smoke_failed", "small", "status=crash", "boom")
    assert r["success"] is False
    assert r["failure_class"] == "cell_smoke_failed"
    assert "stop_reason" not in r          # repairable, NOT a terminal stop
    assert "repair_context" in r


# --------------------------------------------------------------------------- #
# _cell_pregrid_smoke decision logic (mocked run_matrix)
# --------------------------------------------------------------------------- #

def test_crash_blocks_and_repairs(monkeypatch, tmp_path):
    """The All-CNN case: a code bug crashes the smallest cell → block + repair."""
    out = _run(monkeypatch, tmp_path, _mock_run_matrix("cell_execution_error", "AttributeError: ..."))
    assert out is not None
    assert out["failure_class"] == "cell_smoke_failed"
    assert "small" in out["error"]


def test_ok_with_good_metrics_proceeds(monkeypatch, tmp_path):
    out = _run(monkeypatch, tmp_path, _mock_run_matrix("ok", write_metrics={"acc": 0.9}))
    assert out is None  # proceed to the grid


def test_ok_without_metrics_blocks(monkeypatch, tmp_path):
    """degraded_no_metrics caught early: ran ok but wrote no metrics → repairable."""
    out = _run(monkeypatch, tmp_path, _mock_run_matrix("ok"))  # no metrics written
    assert out is not None
    assert out["failure_class"] == "incomplete_metrics"


def test_oom_does_not_block(monkeypatch, tmp_path):
    assert _run(monkeypatch, tmp_path, _mock_run_matrix("oom_failed")) is None


def test_timeout_does_not_block(monkeypatch, tmp_path):
    assert _run(monkeypatch, tmp_path, _mock_run_matrix("timeout")) is None


def test_infra_exception_is_failsoft(monkeypatch, tmp_path):
    def _boom(*a, **k):
        raise RuntimeError("nvidia-smi exploded")
    assert _run(monkeypatch, tmp_path, _boom) is None  # never blocks on its own flake


def test_smoke_steps_env_restored(monkeypatch, tmp_path):
    """The temporary REPROLAB_SMOKE_STEPS must not leak after the smoke."""
    monkeypatch.delenv("REPROLAB_SMOKE_STEPS", raising=False)
    _run(monkeypatch, tmp_path, _mock_run_matrix("ok", write_metrics={"x": 1.0}))
    assert "REPROLAB_SMOKE_STEPS" not in os.environ
