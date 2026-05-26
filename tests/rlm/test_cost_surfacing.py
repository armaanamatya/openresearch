"""Tests for PR-ι.3 — rolling cost surfacing.

Covers:
1. _compute_cost_summary returns zeros when ledger absent.
2. _compute_cost_summary sums all entries correctly.
3. _compute_cost_summary computes usd_this_iter for the latest iteration.
4. _compute_cost_summary computes usd_per_iter_p50 as median.
5. _compute_cost_summary handles a single entry.
6. _compute_cost_summary handles malformed JSONL lines (skip-and-continue).
7. _compute_cost_summary sets iter_count from argument.
8. _update_cost_summary_loop writes cost_summary into demo_status.json.
9. cost_summary updated_at is a valid ISO timestamp.
10. _compute_cost_summary handles iteration_count=0 without division error.
"""

from __future__ import annotations

import json
import statistics
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from backend.agents.rlm.run import _compute_cost_summary, _update_cost_summary_loop


def _write_ledger(project_dir: Path, entries: list[dict]) -> None:
    ledger = project_dir / "cost_ledger.jsonl"
    with ledger.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def _entry(cost_usd: float) -> dict:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent_id": "baseline-implementation",
        "attempt_index": 0,
        "provider": "anthropic",
        "model": "claude-sonnet-4-5",
        "cost_usd": cost_usd,
        "tokens_in": 1000,
        "tokens_out": 200,
    }


# --- 1. Missing ledger → zeros ---

def test_compute_cost_summary_missing_ledger() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        result = _compute_cost_summary(project_dir, iteration_count=2)
    assert result["usd_total"] == 0.0
    assert result["iter_count"] == 2
    assert "updated_at" in result


# --- 2. Sum all entries ---

def test_compute_cost_summary_sum() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        _write_ledger(project_dir, [_entry(0.10), _entry(0.20), _entry(0.30)])
        result = _compute_cost_summary(project_dir, iteration_count=3)
    assert abs(result["usd_total"] - 0.60) < 1e-6


# --- 3. usd_this_iter for latest iteration ---

def test_compute_cost_summary_this_iter() -> None:
    """The latest iteration's cost slice is non-zero."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        # 3 iterations × 2 entries each
        entries = [_entry(0.1), _entry(0.1), _entry(0.2), _entry(0.2), _entry(0.3), _entry(0.3)]
        _write_ledger(project_dir, entries)
        result = _compute_cost_summary(project_dir, iteration_count=3)
    # Last iteration should have a non-zero cost
    assert result["usd_this_iter"] > 0.0
    assert result["usd_total"] > 0.0


# --- 4. usd_per_iter_p50 is median ---

def test_compute_cost_summary_p50() -> None:
    """p50 is the median of per-iteration costs when >= 2 iterations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        # 3 iterations with costs 0.1, 0.3, 0.5
        # Each iteration = 1 entry (slice_size=1 when len=3, iter=3)
        _write_ledger(project_dir, [_entry(0.1), _entry(0.3), _entry(0.5)])
        result = _compute_cost_summary(project_dir, iteration_count=3)
    # p50 ≈ median([0.1, 0.3, 0.5]) = 0.3
    assert abs(result["usd_per_iter_p50"] - 0.3) < 1e-5


# --- 5. Single entry ---

def test_compute_cost_summary_single_entry() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        _write_ledger(project_dir, [_entry(0.42)])
        result = _compute_cost_summary(project_dir, iteration_count=1)
    assert abs(result["usd_total"] - 0.42) < 1e-6
    assert result["iter_count"] == 1


# --- 6. Malformed JSONL lines are skipped ---

def test_compute_cost_summary_skips_malformed() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        ledger = project_dir / "cost_ledger.jsonl"
        with ledger.open("w") as f:
            f.write(json.dumps(_entry(0.1)) + "\n")
            f.write("NOT JSON!!!\n")
            f.write(json.dumps(_entry(0.2)) + "\n")
        result = _compute_cost_summary(project_dir, iteration_count=2)
    assert abs(result["usd_total"] - 0.3) < 1e-6


# --- 7. iter_count from argument ---

def test_compute_cost_summary_iter_count_matches_argument() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        _write_ledger(project_dir, [_entry(0.1)])
        result = _compute_cost_summary(project_dir, iteration_count=7)
    assert result["iter_count"] == 7


# --- 8. _update_cost_summary_loop writes to demo_status.json ---

def test_update_cost_summary_loop_writes_status() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        _write_ledger(project_dir, [_entry(0.05), _entry(0.15)])

        stop_event = threading.Event()
        t = threading.Thread(
            target=_update_cost_summary_loop,
            kwargs={
                "project_dir": project_dir,
                "stop_event": stop_event,
                "iteration_count": lambda: 2,
                "interval_s": 0.05,  # fast for testing
            },
            daemon=True,
        )
        t.start()
        # Wait for the first write.
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            status_path = project_dir / "demo_status.json"
            if status_path.exists():
                data = json.loads(status_path.read_text())
                if "cost_summary" in data:
                    break
            time.sleep(0.05)
        stop_event.set()
        t.join(timeout=2.0)

    assert "cost_summary" in data
    cs = data["cost_summary"]
    assert abs(cs["usd_total"] - 0.20) < 1e-5
    assert cs["iter_count"] == 2


# --- 9. updated_at is a valid ISO timestamp ---

def test_compute_cost_summary_updated_at_is_iso() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        result = _compute_cost_summary(project_dir, iteration_count=0)
    # Should not raise
    dt = datetime.fromisoformat(result["updated_at"].replace("Z", "+00:00"))
    assert dt is not None


# --- 10. iteration_count=0 does not divide by zero ---

def test_compute_cost_summary_zero_iterations_no_crash() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        _write_ledger(project_dir, [_entry(0.1)])
        result = _compute_cost_summary(project_dir, iteration_count=0)
    assert result["usd_total"] > 0.0
    assert result["iter_count"] == 0
    # No ZeroDivisionError
