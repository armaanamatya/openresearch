"""Tests for the batched-write behaviour of RunCostLedger.

TDD coverage for the append-batching feature introduced in the cost-ledger-batch
lane:

T1 – buffer threshold: 24 appends leave the file empty; the 25th triggers a flush
     and the file contains all 25 lines.
T2 – explicit flush: 5 appends + flush() writes all 5 lines.
T3 – concurrent appends from 2 threads (10 each); final flush has exactly 20 lines.
T4 – flush() is idempotent / safe to call when the buffer is empty.
T5 – custom batch_size is respected.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest

from backend.agents.resilience.cost import CostLedgerEntry, RunCostLedger


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _entry(agent_id: str = "test_agent") -> CostLedgerEntry:
    return CostLedgerEntry(
        timestamp=datetime.now(timezone.utc),
        agent_id=agent_id,
        attempt_index=0,
        provider="anthropic",
        model="claude-test",
    )


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


# ---------------------------------------------------------------------------
# T1 – threshold flush
# ---------------------------------------------------------------------------

def test_buffer_threshold_flush(tmp_path: Path) -> None:
    """24 appends must NOT write to disk; the 25th triggers the flush."""
    ledger_path = tmp_path / "cost_ledger.jsonl"
    ledger = RunCostLedger(project_id="proj-t1", path=ledger_path)

    for _ in range(24):
        ledger.append(_entry())

    # File should not exist (or be empty) after 24 appends
    assert _count_lines(ledger_path) == 0, (
        "expected no lines on disk after 24 appends (< batch_size=25)"
    )

    # 25th append triggers the flush
    ledger.append(_entry())

    assert _count_lines(ledger_path) == 25, (
        "expected 25 lines on disk after 25th append triggered flush"
    )


# ---------------------------------------------------------------------------
# T2 – explicit flush
# ---------------------------------------------------------------------------

def test_explicit_flush(tmp_path: Path) -> None:
    """5 appends followed by explicit flush() must write 5 lines."""
    ledger_path = tmp_path / "cost_ledger.jsonl"
    ledger = RunCostLedger(project_id="proj-t2", path=ledger_path)

    for i in range(5):
        ledger.append(_entry(agent_id=f"agent_{i}"))

    assert _count_lines(ledger_path) == 0, "no disk write before flush()"

    ledger.flush()

    assert _count_lines(ledger_path) == 5, "expected 5 lines after explicit flush()"


# ---------------------------------------------------------------------------
# T3 – concurrent appends
# ---------------------------------------------------------------------------

def test_concurrent_appends(tmp_path: Path) -> None:
    """Two threads each appending 10 entries; all 20 should be flushed correctly."""
    ledger_path = tmp_path / "cost_ledger.jsonl"
    # Use batch_size=100 so the threshold never fires mid-test; we flush at end
    ledger = RunCostLedger(project_id="proj-t3", path=ledger_path, batch_size=100)

    barrier = threading.Barrier(2)

    def _worker(n: int) -> None:
        barrier.wait()
        for _ in range(n):
            ledger.append(_entry())

    t1 = threading.Thread(target=_worker, args=(10,))
    t2 = threading.Thread(target=_worker, args=(10,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Flush whatever is buffered
    ledger.flush()

    lines_on_disk = _count_lines(ledger_path)
    assert lines_on_disk == 20, (
        f"expected 20 lines on disk after concurrent appends, got {lines_on_disk}"
    )

    # Verify the in-memory list is also intact
    assert len(ledger.entries) == 20


# ---------------------------------------------------------------------------
# T4 – idempotent empty flush
# ---------------------------------------------------------------------------

def test_flush_idempotent_when_empty(tmp_path: Path) -> None:
    """flush() on an empty buffer must be a no-op and must not raise."""
    ledger_path = tmp_path / "cost_ledger.jsonl"
    ledger = RunCostLedger(project_id="proj-t4", path=ledger_path)

    ledger.flush()  # buffer is empty — must not raise or create an empty file
    ledger.flush()  # second call also safe

    assert not ledger_path.exists() or _count_lines(ledger_path) == 0


# ---------------------------------------------------------------------------
# T5 – custom batch_size
# ---------------------------------------------------------------------------

def test_custom_batch_size(tmp_path: Path) -> None:
    """A ledger with batch_size=3 should flush after every 3rd append."""
    ledger_path = tmp_path / "cost_ledger.jsonl"
    ledger = RunCostLedger(project_id="proj-t5", path=ledger_path, batch_size=3)

    ledger.append(_entry())
    ledger.append(_entry())
    assert _count_lines(ledger_path) == 0

    ledger.append(_entry())  # 3rd → triggers flush
    assert _count_lines(ledger_path) == 3

    ledger.append(_entry())
    ledger.append(_entry())
    assert _count_lines(ledger_path) == 3  # still 3 on disk

    ledger.append(_entry())  # 6th → second flush
    assert _count_lines(ledger_path) == 6


# ---------------------------------------------------------------------------
# T6 – subsequent flushes are additive (append-mode check)
# ---------------------------------------------------------------------------

def test_subsequent_flushes_append(tmp_path: Path) -> None:
    """Two separate flush cycles must both append to the same file."""
    ledger_path = tmp_path / "cost_ledger.jsonl"
    ledger = RunCostLedger(project_id="proj-t6", path=ledger_path)

    for _ in range(3):
        ledger.append(_entry())
    ledger.flush()

    for _ in range(2):
        ledger.append(_entry())
    ledger.flush()

    assert _count_lines(ledger_path) == 5


# ---------------------------------------------------------------------------
# T7 – no path: flush() is safe
# ---------------------------------------------------------------------------

def test_no_path_flush_is_safe() -> None:
    """A ledger without a path should not raise on flush()."""
    ledger = RunCostLedger(project_id="proj-t7")  # no path

    for _ in range(5):
        ledger.append(_entry())

    ledger.flush()  # must not raise
    assert len(ledger.entries) == 5


# ---------------------------------------------------------------------------
# T8 – written JSON lines are valid
# ---------------------------------------------------------------------------

def test_flushed_lines_are_valid_json(tmp_path: Path) -> None:
    """Each line written during a flush must be valid JSON with expected keys."""
    ledger_path = tmp_path / "cost_ledger.jsonl"
    ledger = RunCostLedger(project_id="proj-t8", path=ledger_path)

    for i in range(3):
        ledger.append(_entry(agent_id=f"prim_{i}"))
    ledger.flush()

    lines = [l for l in ledger_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 3
    for line in lines:
        data = json.loads(line)
        assert "agent_id" in data
        assert "cost_usd" in data
