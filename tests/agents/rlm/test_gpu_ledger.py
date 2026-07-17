"""Track E Task 3a: per-experiment GPU ledger (display-only efficiency).

Hermetic OFF+ON pair (tests/CLAUDE.md): the writer is default-OFF + byte-identical
off (no file), and the aggregate is display-only. No network; tmp_path only.
"""

from __future__ import annotations

import pytest

from backend.agents.rlm.gpu_ledger import (
    aggregate_gpu_cost,
    append_gpu_ledger,
    gpu_ledger_enabled,
)


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_GPU_LEDGER", raising=False)
    assert gpu_ledger_enabled() is False


def test_append_and_aggregate(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_GPU_LEDGER", "1")
    append_gpu_ledger(
        tmp_path,
        experiment_run_id="e1",
        start_ts="2026-07-10T00:00:00+00:00",
        end_ts="2026-07-10T01:00:00+00:00",
        gpu_plan={"sku": "A100"},
        provider="gcp",
        rate_usd_per_hr=3.0,
    )
    agg = aggregate_gpu_cost(tmp_path)
    assert agg["total_gpu_hours"] == pytest.approx(1.0)
    assert agg["total_est_cost_usd"] == pytest.approx(3.0)
    assert agg["by_experiment"]["e1"]["gpu_hours"] == pytest.approx(1.0)


def test_off_state_no_write(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_GPU_LEDGER", raising=False)
    assert (
        append_gpu_ledger(
            tmp_path,
            experiment_run_id="e1",
            start_ts="a",
            end_ts="b",
            gpu_plan={},
            provider="gcp",
            rate_usd_per_hr=3.0,
        )
        is False
    )
    assert not (tmp_path / "gpu_ledger.jsonl").exists()


def test_gpu_hours_zero_when_timestamps_unparseable(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_GPU_LEDGER", "1")
    append_gpu_ledger(
        tmp_path,
        experiment_run_id="e2",
        start_ts="bad",
        end_ts="also-bad",
        gpu_plan={},
        provider="local",
        rate_usd_per_hr=0.0,
    )
    assert aggregate_gpu_cost(tmp_path)["by_experiment"]["e2"]["gpu_hours"] == 0.0


def test_repeated_calls_sum_per_experiment(tmp_path, monkeypatch):
    # Two ledger rows for the same experiment_run_id (e.g. a repair-loop re-run)
    # sum in by_experiment and in the totals.
    monkeypatch.setenv("OPENRESEARCH_GPU_LEDGER", "1")
    for start, end in (
        ("2026-07-10T00:00:00+00:00", "2026-07-10T01:00:00+00:00"),
        ("2026-07-10T02:00:00+00:00", "2026-07-10T02:30:00+00:00"),
    ):
        append_gpu_ledger(
            tmp_path,
            experiment_run_id="e3",
            start_ts=start,
            end_ts=end,
            gpu_plan=None,
            provider="gcp",
            rate_usd_per_hr=2.0,
        )
    agg = aggregate_gpu_cost(tmp_path)
    assert agg["by_experiment"]["e3"]["gpu_hours"] == pytest.approx(1.5)
    assert agg["total_est_cost_usd"] == pytest.approx(3.0)
