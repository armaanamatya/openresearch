"""Tests for the parallel batch execution path in score_reproduction.

Verifies:
1. Parallel execution produces the SAME leaf_scores as a serial reference.
2. Multiple batches' complete() calls overlap in wall-clock time
   (mock sleeps briefly; total elapsed < sum-of-sleeps proves concurrency).
3. A failing batch is handled gracefully — the other batches' scores are
   preserved (no batch kills all).
"""

from __future__ import annotations

import json
import tempfile
import threading
import time
from pathlib import Path
from typing import Any


from backend.evals.paperbench.leaf_scorer import score_reproduction

# ---------------------------------------------------------------------------
# Shared rubric fixture — 6 leaves spread across 3 branches so that with
# batch_size=2 we get exactly 3 batches, enough to observe concurrency.
# ---------------------------------------------------------------------------

SIX_LEAF_TREE: dict[str, Any] = {
    "id": "root",
    "requirements": "root",
    "weight": 1,
    "sub_tasks": [
        {
            "id": "branch-a",
            "requirements": "branch a",
            "weight": 1,
            "sub_tasks": [
                {"id": "leaf-a1", "requirements": "req a1", "weight": 1, "sub_tasks": []},
                {"id": "leaf-a2", "requirements": "req a2", "weight": 1, "sub_tasks": []},
            ],
        },
        {
            "id": "branch-b",
            "requirements": "branch b",
            "weight": 1,
            "sub_tasks": [
                {"id": "leaf-b1", "requirements": "req b1", "weight": 1, "sub_tasks": []},
                {"id": "leaf-b2", "requirements": "req b2", "weight": 1, "sub_tasks": []},
            ],
        },
        {
            "id": "branch-c",
            "requirements": "branch c",
            "weight": 1,
            "sub_tasks": [
                {"id": "leaf-c1", "requirements": "req c1", "weight": 1, "sub_tasks": []},
                {"id": "leaf-c2", "requirements": "req c2", "weight": 1, "sub_tasks": []},
            ],
        },
    ],
}

# Deterministic per-leaf scores that the mock will return.
_LEAF_SCORES: dict[str, float] = {
    "leaf-a1": 1.0,
    "leaf-a2": 0.5,
    "leaf-b1": 0.8,
    "leaf-b2": 0.2,
    "leaf-c1": 0.6,
    "leaf-c2": 0.4,
}


def _make_final_report(tmp_dir: Path) -> None:
    """Write a minimal honest final_report.json so degraded-run detection is False."""
    (tmp_dir / "final_report.json").write_text(
        json.dumps({
            "reproduction_summary": "parallel test run",
            "baseline_metrics": {"accuracy": 0.99},
        }),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Mock LLM client: deterministic per-leaf responses with a configurable sleep.
# ---------------------------------------------------------------------------

class _RecordingLlmClient:
    """Mock client that:
    - Returns deterministic scores keyed by leaf_id.
    - Records (start_time, end_time) for each call.
    - Optionally sleeps to make concurrency visible in wall-clock time.
    """

    def __init__(self, sleep_s: float = 0.0) -> None:
        self._sleep_s = sleep_s
        self.call_intervals: list[tuple[float, float]] = []
        self._lock = threading.Lock()

    def complete(self, *, system: str, user: str) -> str:
        t0 = time.monotonic()
        if self._sleep_s > 0:
            time.sleep(self._sleep_s)
        t1 = time.monotonic()
        with self._lock:
            self.call_intervals.append((t0, t1))

        # Parse the leaf_ids out of the tasks_json block in user message.
        # The template embeds them as JSON under "Rubric leaf tasks to grade".
        import re
        m = re.search(r"\[.*\]", user, re.DOTALL)
        if m:
            tasks = json.loads(m.group())
        else:
            tasks = []
        response = [
            {
                "leaf_id": t["leaf_id"],
                "score": _LEAF_SCORES.get(t["leaf_id"], 0.0),
                "justification": f"mock score for {t['leaf_id']}",
            }
            for t in tasks
        ]
        return json.dumps(response)


# ---------------------------------------------------------------------------
# Test 1: same leaf_scores as serial reference (correctness)
# ---------------------------------------------------------------------------

def test_parallel_produces_same_scores_as_serial():
    """Parallel path must return identical leaf_scores to the serial reference.

    We run score_reproduction with batch_size=2 (3 batches for 6 leaves).
    The parallel executor submits all 3 batches concurrently.  Keys are
    disjoint per batch so merge order does not matter — final leaf_scores
    dict must equal the deterministic _LEAF_SCORES mapping.
    """
    client = _RecordingLlmClient(sleep_s=0.0)
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        _make_final_report(run_dir)
        result = score_reproduction(
            SIX_LEAF_TREE, run_dir, client, batch_size=2, degraded=False
        )

    scored = {rec["id"]: rec["score"] for rec in result["leaf_scores"]}
    assert scored == _LEAF_SCORES, (
        f"parallel scores differ from expected: {scored!r} != {_LEAF_SCORES!r}"
    )
    assert result["graded"] == 6
    assert result["leaf_count"] == 6
    assert result["degraded"] is False


# ---------------------------------------------------------------------------
# Test 2: concurrency — total wall-clock < sum-of-sleeps
# ---------------------------------------------------------------------------

def test_parallel_batches_overlap_in_wall_clock():
    """Multiple batches must execute concurrently.

    Each mock complete() sleeps 0.15 s across 3 batches. Concurrency is proven
    by overlapping call intervals — deterministic under scheduler load, unlike
    the original `elapsed < serial_sum` wall-clock assertion, which flaked on
    loaded runners (same class as the fixed test_gpu_cell_runner_heterogeneous).
    """
    sleep_per_batch = 0.15
    n_batches = 3  # 6 leaves / batch_size=2

    client = _RecordingLlmClient(sleep_s=sleep_per_batch)
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        _make_final_report(run_dir)
        score_reproduction(SIX_LEAF_TREE, run_dir, client, batch_size=2, degraded=False)

    assert len(client.call_intervals) == n_batches, (
        f"expected {n_batches} LLM calls, got {len(client.call_intervals)}"
    )

    # At least two calls must have overlapping intervals,
    # i.e. one call started before another ended.
    intervals = client.call_intervals
    overlaps = 0
    for i in range(len(intervals)):
        for j in range(i + 1, len(intervals)):
            s1, e1 = intervals[i]
            s2, e2 = intervals[j]
            if s1 < e2 and s2 < e1:  # intervals overlap
                overlaps += 1
    assert overlaps >= 1, (
        f"no overlapping LLM call intervals detected — intervals={intervals!r}; "
        "concurrency is not being exploited"
    )


# ---------------------------------------------------------------------------
# Test 3: a failing batch does not kill the other batches
# ---------------------------------------------------------------------------

class _PartiallyFailingClient:
    """Fails for batch_num==2 (second call); succeeds for all others."""

    def __init__(self) -> None:
        self._call_count = 0
        self._lock = threading.Lock()

    def complete(self, *, system: str, user: str) -> str:
        with self._lock:
            self._call_count += 1
            n = self._call_count
        if n == 2:
            raise RuntimeError("simulated LLM failure for batch 2")
        import re
        m = re.search(r"\[.*\]", user, re.DOTALL)
        tasks = json.loads(m.group()) if m else []
        response = [
            {
                "leaf_id": t["leaf_id"],
                "score": _LEAF_SCORES.get(t["leaf_id"], 0.0),
                "justification": f"ok for {t['leaf_id']}",
            }
            for t in tasks
        ]
        return json.dumps(response)


def test_failing_batch_does_not_kill_other_batches(monkeypatch):
    """When one batch fails on EVERY transport (primary + the whole fallback
    chain), its leaves are marked ``ungraded`` (score None, EXCLUDED from the
    rollup) — never a phantom 0.0 — while the other batches' real scores are
    still present (no batch kills all). This is the T5 grader-hardening
    contract (spec §4.5): a leaf the grader could not score is excluded from
    the diagnostic, never silently zeroed.

    6 leaves, batch_size=2 → 3 batches.  Batch 2 fails on all transports.
    Batches 1 and 3 return real scores.  After merging:
    - 4 leaves must carry their deterministic scores.
    - 2 leaves (batch 2) must be ungraded: score None, state "ungraded".
    """
    # Isolate the "batch fully failed -> ungraded" contract from real fallback
    # transports: with an empty fallback chain the primary is the ONLY
    # candidate, so a batch whose primary call raises is genuinely ungradable.
    # (The cross-provider fallback ladder itself is covered by
    # test_leaf_scorer_fallback.py; a real transport here can return a
    # degenerate non-raising response that masks the ungraded path.)
    monkeypatch.setattr(
        "backend.agents.rlm.grader_transport.build_fallback_chain", lambda: []
    )
    client = _PartiallyFailingClient()
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        _make_final_report(run_dir)
        result = score_reproduction(
            SIX_LEAF_TREE, run_dir, client, batch_size=2, degraded=False
        )

    records = {rec["id"]: rec for rec in result["leaf_scores"]}
    assert len(records) == 6, f"expected 6 leaf records, got {len(records)}"

    # Exactly 2 leaves (the batch that failed every transport) must be
    # ungraded — score None, NEVER a phantom 0.0.
    ungraded = [lid for lid, rec in records.items() if rec.get("state") == "ungraded"]
    assert len(ungraded) == 2, (
        f"expected 2 ungraded leaves (one batch failed every transport), got {ungraded!r}"
    )
    for lid in ungraded:
        assert records[lid]["score"] is None, (
            f"ungraded leaf {lid} must be score=None, never a phantom 0.0"
        )
        assert records[lid]["justification"].startswith("grader_unavailable"), (
            f"ungraded leaf {lid} justification={records[lid]['justification']!r}"
        )

    # The other 4 leaves must carry real scores.
    good_leaves = [lid for lid in records if records[lid].get("state") != "ungraded"]
    assert len(good_leaves) == 4, f"expected 4 graded leaves, got {good_leaves!r}"
    for lid in good_leaves:
        assert records[lid]["score"] == _LEAF_SCORES[lid], (
            f"leaf {lid}: expected {_LEAF_SCORES[lid]}, got {records[lid]['score']}"
        )
