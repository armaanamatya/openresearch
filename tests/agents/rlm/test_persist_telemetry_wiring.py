"""Phase-2b wiring tests for the run_experiment persist telemetry hooks (E-3b / G-S1a).

Exercises the two new module-level helpers in ``primitives.py`` directly (they
only touch ``ctx.project_dir``), proving the OFF+ON pair for each flag and — the
load-bearing property — that with every flag OFF the experiment_runs.jsonl row is
byte-identical and no sidecar is written.
"""

from __future__ import annotations

from types import SimpleNamespace

from backend.agents.rlm.dag_nodes import read_dag_nodes
from backend.agents.rlm.ok_receipt import count_ok_receipts
from backend.agents.rlm.primitives import (
    _emit_experiment_telemetry,
    _stamp_efficiency_row_fields,
)

_TELEMETRY_FLAGS = ("OPENRESEARCH_OK_RECEIPT", "OPENRESEARCH_GPU_LEDGER", "OPENRESEARCH_DAG_BACKBONE")


def _ctx(tmp_path):
    return SimpleNamespace(project_dir=tmp_path, sandbox_mode="gcp")


# --------------------------------------------------------------------------- #
# Row-field stamping (OPENRESEARCH_GPU_LEDGER)
# --------------------------------------------------------------------------- #


def test_row_fields_byte_identical_when_off(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_GPU_LEDGER", raising=False)
    entry = {"timestamp": "t", "success": True, "model_id": "m"}
    before = dict(entry)
    _stamp_efficiency_row_fields(entry, _ctx(tmp_path), start_ts="s")
    assert entry == before  # no new keys, byte-identical row off-flag


def test_row_fields_added_when_on(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_GPU_LEDGER", "1")
    entry = {"timestamp": "end", "success": True}
    _stamp_efficiency_row_fields(entry, _ctx(tmp_path), start_ts="start")
    assert entry["start_ts"] == "start"
    assert entry["end_ts"] == "end"
    assert "retry_id" in entry  # stamped (value may be None — best-effort)


def test_row_fields_start_ts_falls_back_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_GPU_LEDGER", "1")
    entry = {"timestamp": "end", "success": True}
    _stamp_efficiency_row_fields(entry, _ctx(tmp_path), start_ts=None)
    assert entry["start_ts"] == "end"  # no captured start => start == end (hours 0)


# --------------------------------------------------------------------------- #
# Telemetry sidecars (each self-gated inside its own writer)
# --------------------------------------------------------------------------- #


def _emit(tmp_path, *, success=True):
    _emit_experiment_telemetry(
        _ctx(tmp_path),
        {"success": success, "experiment_run_id": "e1", "metrics_sha256": "sha1"},
        {"timestamp": "2026-07-11T01:00:00+00:00"},
        start_ts="2026-07-11T00:00:00+00:00",
    )


def test_all_off_writes_no_sidecars(tmp_path, monkeypatch):
    for f in _TELEMETRY_FLAGS:
        monkeypatch.delenv(f, raising=False)
    _emit(tmp_path)
    assert not (tmp_path / "gpu_ledger.jsonl").exists()
    assert not (tmp_path / "rlm_state" / "experiment_ok_receipts.jsonl").exists()
    assert not (tmp_path / "rlm_state" / "dag_nodes.jsonl").exists()


def test_ok_receipt_written_on_success(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_OK_RECEIPT", "1")
    _emit(tmp_path, success=True)
    assert count_ok_receipts(tmp_path) == 1


def test_ok_receipt_not_written_on_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_OK_RECEIPT", "1")
    _emit(tmp_path, success=False)
    assert count_ok_receipts(tmp_path) == 0  # forge-resistance: only genuine successes


def test_gpu_ledger_and_dag_written_when_on(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_GPU_LEDGER", "1")
    monkeypatch.setenv("OPENRESEARCH_DAG_BACKBONE", "1")
    _emit(tmp_path)
    assert (tmp_path / "gpu_ledger.jsonl").exists()
    nodes = read_dag_nodes(tmp_path)
    assert nodes and nodes[0]["node_id"] == "e1" and nodes[0]["kind"] == "run_experiment"
