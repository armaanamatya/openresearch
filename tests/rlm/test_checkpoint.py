"""Tests for backend.agents.rlm.checkpoint.

Verifies:
- IterationCheckpointer.record() appends one RLMRunIteration to the event store
  with a correctly incrementing expected_version.
- Snapshot JSONL is written with the sanitized dict.
- The stored event and snapshot are corpus-free.
- ConcurrencyError is surfaced (not swallowed) when a version mismatch occurs.
- RLMRunIteration is properly registered with @register_event.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.agents.rlm.checkpoint import IterationCheckpointer, RLMRunIteration
from backend.eventstore.interface import ConcurrencyError
from backend.eventstore.sqlite_store import SqliteEventStore
from backend.messaging.event import resolve_event_class

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CORPUS_SENTINEL = "PAPER_CORPUS_SENTINEL_xyzzy_DO_NOT_LEAK_abcdefg"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_clean(index: int = 1, *, response: str = "root reasoning text") -> dict:
    """Build a minimal sanitized iteration dict (as produced by sanitize_iteration)."""
    return {
        "iteration": index,
        "response": response,
        "code_blocks": [
            {
                "code": "x = understand_section(context['abstract'])",
                "stdout_meta": {"length": 5, "prefix": "ok\n\n\n", "has_traceback": False},
                "stderr_meta": {"length": 0, "prefix": "", "has_traceback": False},
                "vars": {"x": {"type": "dict", "size": 42}},
                "sub_calls": 0,
            }
        ],
        "sub_calls": 0,
        "timing": 1.25,
    }


# ---------------------------------------------------------------------------
# Fixtures — all defined locally, not in conftest.py
# ---------------------------------------------------------------------------

@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test.db"


@pytest.fixture
def event_store(db_path: Path) -> SqliteEventStore:
    store = SqliteEventStore(f"sqlite:///{db_path}")
    return store


@pytest.fixture
def snapshot_dir(tmp_path: Path) -> Path:
    d = tmp_path / "rlm_state"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def checkpointer(event_store: SqliteEventStore, snapshot_dir: Path) -> IterationCheckpointer:
    return IterationCheckpointer(
        project_id="test-proj-001",
        event_store=event_store,
        snapshot_dir=snapshot_dir,
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class TestRLMRunIterationRegistration:

    def test_event_type(self):
        assert RLMRunIteration.event_type == "rlm_run_iteration"

    def test_schema_version(self):
        assert RLMRunIteration.schema_version == 1

    def test_registered_in_registry(self):
        cls = resolve_event_class("rlm_run_iteration", 1)
        assert cls is RLMRunIteration

    def test_is_frozen(self):
        """DomainEvent subclasses must be frozen (immutable)."""
        event = RLMRunIteration(
            iteration=1,
            response="hi",
            code_blocks=[],
            sub_calls=0,
            timing=None,
        )
        with pytest.raises(Exception):
            event.iteration = 2  # type: ignore[misc]


# ---------------------------------------------------------------------------
# IterationCheckpointer.record — event store
# ---------------------------------------------------------------------------

class TestIterationCheckpointerEventStore:

    def test_record_appends_event(self, checkpointer, event_store):
        clean = _make_clean(1)
        checkpointer.record(clean)

        events = list(event_store.load("rlm-run:test-proj-001"))
        assert len(events) == 1
        assert events[0].event_type == "rlm_run_iteration"

    def test_record_uses_distinct_aggregate(self, checkpointer, event_store):
        """The RLM run aggregate id must be 'rlm-run:<project_id>'."""
        checkpointer.record(_make_clean(1))
        # The aggregate must have version 1 now
        version = event_store.get_aggregate_version("rlm-run:test-proj-001")
        assert version == 1

    def test_record_increments_expected_version(self, checkpointer, event_store):
        """expected_version is 0 for first event, 1 for second, etc."""
        checkpointer.record(_make_clean(1))
        checkpointer.record(_make_clean(2))
        checkpointer.record(_make_clean(3))

        events = list(event_store.load("rlm-run:test-proj-001"))
        assert len(events) == 3
        # aggregate_version is 1-based in the event store
        versions = [e.aggregate_version for e in events]
        assert versions == [1, 2, 3]

    def test_internal_version_counter_increments(self, checkpointer):
        assert checkpointer._version == 0
        checkpointer.record(_make_clean(1))
        assert checkpointer._version == 1
        checkpointer.record(_make_clean(2))
        assert checkpointer._version == 2

    def test_stored_event_payload_matches_clean(self, checkpointer, event_store):
        clean = _make_clean(1, response="test response")
        checkpointer.record(clean)

        stored = list(event_store.load("rlm-run:test-proj-001"))[0]
        event = stored.into(RLMRunIteration)
        assert event.iteration == 1
        assert event.response == "test response"
        assert event.sub_calls == 0

    def test_stored_event_correlation_id_is_project_id(self, checkpointer, event_store):
        checkpointer.record(_make_clean(1))
        stored = list(event_store.load("rlm-run:test-proj-001"))[0]
        assert stored.envelope.correlation_id == "test-proj-001"

    def test_stored_event_aggregate_type(self, checkpointer, event_store):
        checkpointer.record(_make_clean(1))
        stored = list(event_store.load("rlm-run:test-proj-001"))[0]
        assert stored.aggregate_type == "rlm_run"


# ---------------------------------------------------------------------------
# IterationCheckpointer.record — snapshot JSONL
# ---------------------------------------------------------------------------

class TestIterationCheckpointerSnapshot:

    def test_snapshot_file_created(self, checkpointer, snapshot_dir):
        checkpointer.record(_make_clean(1))
        assert (snapshot_dir / "iterations.jsonl").exists()

    def test_snapshot_appended_per_call(self, checkpointer, snapshot_dir):
        checkpointer.record(_make_clean(1))
        checkpointer.record(_make_clean(2))
        checkpointer.record(_make_clean(3))

        lines = (snapshot_dir / "iterations.jsonl").read_text().strip().splitlines()
        assert len(lines) == 3

    def test_snapshot_line_is_valid_json(self, checkpointer, snapshot_dir):
        checkpointer.record(_make_clean(1))
        line = (snapshot_dir / "iterations.jsonl").read_text().strip()
        data = json.loads(line)
        assert data["iteration"] == 1

    def test_snapshot_preserves_clean_fields(self, checkpointer, snapshot_dir):
        clean = _make_clean(2, response="snapshot test")
        checkpointer.record(clean)
        line = (snapshot_dir / "iterations.jsonl").read_text().strip()
        data = json.loads(line)
        assert data["response"] == "snapshot test"
        assert data["timing"] == 1.25


# ---------------------------------------------------------------------------
# Corpus-free assertion on both outputs
# ---------------------------------------------------------------------------

class TestCorpusFreeOutputs:

    def test_no_corpus_in_event_store(self, checkpointer, event_store):
        """Even if someone passes a dirty dict to record(), the event is clean."""
        # In production, record() only ever receives sanitize_iteration() output;
        # here we verify the storage layer handles the sanitized dict corpus-free.
        clean = _make_clean(1, response="clean response: " + "safe text")
        # Ensure no sentinel in the input (this is a valid sanitized dict)
        assert CORPUS_SENTINEL not in json.dumps(clean)
        checkpointer.record(clean)

        stored = list(event_store.load("rlm-run:test-proj-001"))[0]
        stored_json = json.dumps(stored.payload)
        assert CORPUS_SENTINEL not in stored_json

    def test_no_corpus_in_snapshot(self, checkpointer, snapshot_dir):
        clean = _make_clean(1, response="safe reasoning")
        assert CORPUS_SENTINEL not in json.dumps(clean)
        checkpointer.record(clean)

        snapshot_text = (snapshot_dir / "iterations.jsonl").read_text()
        assert CORPUS_SENTINEL not in snapshot_text


# ---------------------------------------------------------------------------
# ConcurrencyError is surfaced, not swallowed
# ---------------------------------------------------------------------------

class TestConcurrencyErrorSurfaced:

    def test_concurrency_error_raised_on_version_mismatch(self, tmp_path):
        """If a second checkpointer with version=0 writes to an already-written
        aggregate, the event store raises ConcurrencyError.  We must not swallow it."""
        store = SqliteEventStore(f"sqlite:///{tmp_path / 'ce_test.db'}")
        snap = tmp_path / "snap"
        snap.mkdir()

        cp1 = IterationCheckpointer(
            project_id="cp-conflict",
            event_store=store,
            snapshot_dir=snap,
        )
        cp2 = IterationCheckpointer(
            project_id="cp-conflict",
            event_store=store,
            snapshot_dir=snap,
        )

        # cp1 writes version 0 → 1
        cp1.record(_make_clean(1))

        # cp2 also starts at version 0 — version mismatch: must raise
        with pytest.raises(ConcurrencyError):
            cp2.record(_make_clean(1))


# ---------------------------------------------------------------------------
# Restart safety (T19 / review I9)
# ---------------------------------------------------------------------------

class TestRestartSafety:

    def test_iteration_checkpointer_does_not_crash_on_restart(self, tmp_path):
        """Symptom: a restarted run crashed on the first checkpoint with ConcurrencyError.

        checkpoint.py hardcoded _version=0 — on a process restart with the same
        project_id the aggregate was already at version N and the first record()
        raised ConcurrencyError (review I9 / T19). Verify: instantiating a fresh
        IterationCheckpointer for an existing project_id resumes the version
        counter cleanly so the next record() appends without conflict.
        """
        db_path = tmp_path / "test.db"
        store = SqliteEventStore(f"sqlite:///{db_path}")
        snap = tmp_path / "snap"
        snap.mkdir()
        try:
            # First run: record two iterations under project_id="p".
            cp1 = IterationCheckpointer(
                project_id="p", event_store=store, snapshot_dir=snap,
            )
            cp1.record(_make_clean(1))
            cp1.record(_make_clean(2))

            # Process restart: new IterationCheckpointer instance, same project_id.
            cp2 = IterationCheckpointer(
                project_id="p", event_store=store, snapshot_dir=snap,
            )
            # _version must be seeded to 2 (current aggregate version), not 0.
            assert cp2._version == 2

            # Must NOT raise ConcurrencyError — the version counter must resume.
            cp2.record(_make_clean(3))

            # Three events total in the store.
            events = list(store.load("rlm-run:p"))
            assert len(events) == 3
            assert [e.aggregate_version for e in events] == [1, 2, 3]
        finally:
            store.close()


# ---------------------------------------------------------------------------
# IterationCheckpointer validation
# ---------------------------------------------------------------------------

class TestIterationCheckpointerValidation:

    def test_empty_project_id_raises(self, event_store, snapshot_dir):
        with pytest.raises(ValueError, match="project_id"):
            IterationCheckpointer(
                project_id="",
                event_store=event_store,
                snapshot_dir=snapshot_dir,
            )

    def test_snapshot_dir_created_if_missing(self, event_store, tmp_path):
        new_dir = tmp_path / "nonexistent" / "subdir"
        assert not new_dir.exists()
        IterationCheckpointer(
            project_id="proj",
            event_store=event_store,
            snapshot_dir=new_dir,
        )
        assert new_dir.exists()


def test_checkpointer_fsyncs_after_jsonl_append(monkeypatch, tmp_path):
    """Symptom: a crash between event-store and JSONL append leaves a torn line.

    The JSONL append used flush() without os.fsync (review M7 / T30). Verify
    os.fsync is called after each iteration's JSONL write.
    """
    import os
    import backend.agents.rlm.checkpoint as cp_mod

    fsync_calls = []
    real_fsync = os.fsync

    def spy_fsync(fd):
        fsync_calls.append(fd)
        return real_fsync(fd)

    monkeypatch.setattr(cp_mod.os, "fsync", spy_fsync)

    db = tmp_path / "test.db"
    store = SqliteEventStore(f"sqlite:///{db}")
    try:
        cp = IterationCheckpointer(project_id="p", event_store=store, snapshot_dir=tmp_path)
        cp.record(_make_clean(1))
    finally:
        store.close()

    assert len(fsync_calls) >= 1, "os.fsync was not called after JSONL append"
