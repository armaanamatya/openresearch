"""Track E Task 4: out-of-process re-grade ok-receipt (forge-resistant).

Hermetic OFF+ON pair (tests/CLAUDE.md): the writer is default-OFF + byte-identical
off (no file, no write), and a receipt only counts as a forge-resistant success
when it carries both ``ok is True`` and a non-empty ``metrics_sha256`` -- a
failed or sha-less row must never count. No network; tmp_path fixtures only.
"""

from __future__ import annotations

from backend.agents.rlm.ok_receipt import (
    count_ok_receipts,
    ok_receipt_enabled,
    write_ok_receipt,
)


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_OK_RECEIPT", raising=False)
    assert ok_receipt_enabled() is False


def test_write_and_count(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_OK_RECEIPT", "1")
    (tmp_path / "rlm_state").mkdir()
    assert write_ok_receipt(tmp_path, experiment_run_id="e1", ok=True,
                            metrics_sha256="abc", ts="t") is True
    assert count_ok_receipts(tmp_path) == 1


def test_failed_or_shaless_receipt_not_counted(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_OK_RECEIPT", "1")
    (tmp_path / "rlm_state").mkdir()
    write_ok_receipt(tmp_path, experiment_run_id="e1", ok=False, metrics_sha256="abc", ts="t")
    write_ok_receipt(tmp_path, experiment_run_id="e2", ok=True, metrics_sha256="", ts="t")
    assert count_ok_receipts(tmp_path) == 0  # neither is a forge-proof success


def test_distinct_run_ids_counted_once(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_OK_RECEIPT", "1")
    (tmp_path / "rlm_state").mkdir()
    write_ok_receipt(tmp_path, experiment_run_id="e1", ok=True, metrics_sha256="abc", ts="t1")
    write_ok_receipt(tmp_path, experiment_run_id="e1", ok=True, metrics_sha256="abc", ts="t2")
    assert count_ok_receipts(tmp_path) == 1
