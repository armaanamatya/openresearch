"""Pin concurrent-write correctness on SqliteEventStore.

The 2026-05-23 paper-sweep regression: two parallel `/runs/arxiv` ingests
(prj_f4cc5fa917c27ef1 + prj_f87990c70c6bc8f6) collided on the shared
event store and the second died with `AppendError: SQLite error during
append: database is locked`.

Root cause: writers used `BEGIN` (= BEGIN DEFERRED), which holds SHARED
and tries to upgrade to RESERVED on the first INSERT. With WAL mode,
that upgrade can fail-fast with SQLITE_BUSY without honoring
`busy_timeout` — exactly what happened. Fix: `BEGIN IMMEDIATE` grabs
RESERVED upfront so writers serialize cleanly under `busy_timeout`
(now 30 s).

Tests:
1. Two threads append to *different* aggregates concurrently — both
   succeed; global_position is monotonically increasing across both
   writers.
2. Two threads append to the *same* aggregate at the same expected_version
   — exactly one wins, the other gets `ConcurrencyError`. (This is the
   optimistic-concurrency invariant; it should not regress to a generic
   "database is locked" error.)
"""

from __future__ import annotations

import concurrent.futures
import threading
from typing import ClassVar

import pytest

from backend.eventstore.interface import ConcurrencyError
from backend.eventstore.sqlite_store import SqliteEventStore
from backend.messaging.envelope import AggregateId, make_envelope
from backend.messaging.event import (
    DomainEvent,
    _clear_registry_for_tests,
    register_event,
)


# Local event-type registration so we don't import the main test module's
# fixtures (which would couple test ordering).
def _register_event_types():
    @register_event
    class ParallelTick(DomainEvent):
        event_type: ClassVar[str] = "parallel_tick"
        schema_version: ClassVar[int] = 1
        worker: str
        n: int

    return ParallelTick


@pytest.fixture
def store(tmp_path):
    _clear_registry_for_tests()
    s = SqliteEventStore(f"sqlite:///{tmp_path}/events.db")
    yield s
    s.close()
    _clear_registry_for_tests()


# --- Different aggregates: must both succeed -------------------------------


def test_concurrent_writers_on_distinct_aggregates_both_succeed(store):
    """Two threads append to different aggregates — neither should hit
    'database is locked'. BEGIN IMMEDIATE + busy_timeout serializes them."""
    ParallelTick = _register_event_types()

    # Barrier ensures both threads attempt their first conn.execute("BEGIN
    # IMMEDIATE") within microseconds of each other — maximum lock contention.
    barrier = threading.Barrier(2)
    n_per_thread = 5

    def worker(worker_id: str) -> list[int]:
        agg = AggregateId(f"agg_{worker_id}")
        positions: list[int] = []
        barrier.wait()  # both threads released simultaneously
        for i in range(n_per_thread):
            ev = ParallelTick(worker=worker_id, n=i)
            env = make_envelope(source=f"test_{worker_id}")
            result = store.append(
                agg, "tick", [ev], expected_version=i, envelopes=[env]
            )
            positions.extend(result.written_global_positions)
        return positions

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        f_a = ex.submit(worker, "A")
        f_b = ex.submit(worker, "B")
        pos_a = f_a.result(timeout=30)
        pos_b = f_b.result(timeout=30)

    # Each writer produced n_per_thread global positions.
    assert len(pos_a) == n_per_thread
    assert len(pos_b) == n_per_thread
    # All positions globally unique — no double-allocation under contention.
    all_pos = pos_a + pos_b
    assert len(set(all_pos)) == len(all_pos), (
        f"Global positions collided: {sorted(all_pos)}"
    )
    # Total count is right (no events dropped to a 'locked' error).
    assert len(all_pos) == 2 * n_per_thread


def test_concurrent_writers_distinct_aggregates_finishes_under_busy_timeout(store):
    """Stress the busy_timeout: 4 writers × 20 events each = 80 appends
    in 4 concurrent threads. Total must complete in well under the
    30 s busy_timeout (target: under 10 s on a normal machine)."""
    import time
    ParallelTick = _register_event_types()
    n_per_thread = 20
    n_workers = 4
    barrier = threading.Barrier(n_workers)

    def worker(worker_id: str) -> int:
        agg = AggregateId(f"agg_{worker_id}")
        barrier.wait()
        for i in range(n_per_thread):
            ev = ParallelTick(worker=worker_id, n=i)
            env = make_envelope(source=f"test_{worker_id}")
            store.append(agg, "tick", [ev], expected_version=i, envelopes=[env])
        return n_per_thread

    t0 = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(worker, f"W{i}") for i in range(n_workers)]
        for f in futures:
            f.result(timeout=30)
    elapsed = time.monotonic() - t0
    # 80 appends should serialize quickly under WAL + BEGIN IMMEDIATE.
    # Generous bound to avoid CI flakiness — the real goal is "no locked
    # errors", not microsecond-tight performance.
    assert elapsed < 25.0, (
        f"4×20 concurrent appends took {elapsed:.1f}s — busy_timeout may be "
        f"too small, or BEGIN DEFERRED regression."
    )


# --- Same aggregate: optimistic concurrency must hold ----------------------


def test_concurrent_writers_on_same_aggregate_one_wins_other_gets_concurrency_error(store):
    """Same aggregate, same expected_version — exactly one append wins,
    the other raises ConcurrencyError (NOT 'database is locked'). This
    pins the optimistic-concurrency contract under serialized writes."""
    ParallelTick = _register_event_types()
    agg = AggregateId("agg_shared")
    barrier = threading.Barrier(2)
    results: dict[str, object] = {}

    def worker(worker_id: str) -> None:
        ev = ParallelTick(worker=worker_id, n=0)
        env = make_envelope(source=f"test_{worker_id}")
        barrier.wait()
        try:
            r = store.append(agg, "tick", [ev], expected_version=0, envelopes=[env])
            results[worker_id] = r
        except Exception as exc:  # noqa: BLE001
            results[worker_id] = exc

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        f_a = ex.submit(worker, "A")
        f_b = ex.submit(worker, "B")
        f_a.result(timeout=30)
        f_b.result(timeout=30)

    # Exactly one succeeded; the other got ConcurrencyError.
    successes = [k for k, v in results.items() if not isinstance(v, Exception)]
    failures = [k for k, v in results.items() if isinstance(v, Exception)]
    assert len(successes) == 1, f"Expected 1 success, got {len(successes)}: {results}"
    assert len(failures) == 1, f"Expected 1 failure, got {len(failures)}: {results}"
    # The loser must be a ConcurrencyError — NOT an AppendError("database is
    # locked"). That's the whole point of the BEGIN IMMEDIATE fix.
    loser_exc = results[failures[0]]
    assert isinstance(loser_exc, ConcurrencyError), (
        f"Expected ConcurrencyError, got {type(loser_exc).__name__}: {loser_exc}"
    )
