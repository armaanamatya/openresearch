"""Tests for backend.services.runs.purge.purge_project (handoff P1-I5 / T13)."""

from __future__ import annotations

from typing import ClassVar

import pytest

from backend.eventstore.sqlite_store import SqliteEventStore
from backend.messaging.envelope import AggregateId, make_envelope
from backend.messaging.event import (
    DomainEvent,
    _clear_registry_for_tests,
    register_event,
)


# ---------------------------------------------------------------------------
# Minimal DomainEvent for seeding the store
# ---------------------------------------------------------------------------


def _register_event():
    @register_event
    class PurgeTestEvent(DomainEvent):
        event_type: ClassVar[str] = "purge_test_event_t13"
        schema_version: ClassVar[int] = 1
        msg: str

    return PurgeTestEvent


def _append_one(store: SqliteEventStore, aggregate_id: str, PurgeTestEvent) -> None:
    ev = PurgeTestEvent(msg="seed")
    env = make_envelope(source="test")
    store.append(AggregateId(aggregate_id), "test", [ev], 0, [env])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_purge_project_clears_both_stores(tmp_path, monkeypatch):
    """Symptom: re-running a paper raises ConcurrencyError because aggregates persist.

    rm -rf runs/<id> does not clear event_store_events; first record() then
    crashes with expected_version=0 against version N (handoff P1-I5 / T13).
    Verify: purge_project deletes BOTH the run dir AND the project's aggregates,
    leaving the next run free to start at version 0.
    """
    from backend.config import Settings
    from backend.services.runs.purge import purge_project

    _clear_registry_for_tests()
    PurgeTestEvent = _register_event()

    db_url = f"sqlite:///{tmp_path}/test.db"
    settings_for_test = Settings(database_url=db_url)
    monkeypatch.setattr(
        "backend.services.runs.purge.get_settings", lambda: settings_for_test
    )

    runs_root = tmp_path / "runs"
    run_dir = runs_root / "prj_x"
    run_dir.mkdir(parents=True)
    (run_dir / "marker.txt").write_text("delete me")

    # Seed the event store with aggregates under the same prefix.
    store = SqliteEventStore(db_url)
    try:
        _append_one(store, "prj_x", PurgeTestEvent)
        _append_one(store, "prj_x:parsed", PurgeTestEvent)
        _append_one(store, "prj_x:index", PurgeTestEvent)
    finally:
        store.close()

    result = purge_project("prj_x", runs_root)

    # Both stores must be cleared.
    assert result["run_dir_removed"] is True
    assert not run_dir.exists()
    assert result["aggregates_removed"] >= 3

    # Re-appending at version 0 must succeed (symptom was ConcurrencyError here).
    _clear_registry_for_tests()
    PurgeTestEvent2 = _register_event()
    store2 = SqliteEventStore(db_url)
    try:
        _append_one(store2, "prj_x", PurgeTestEvent2)
    finally:
        store2.close()


def test_purge_project_idempotent_on_missing_dir(tmp_path, monkeypatch):
    """purge_project must not fail when the run dir doesn't exist.

    A caller invoking --fresh on a project_id whose run dir is already gone
    (or has never existed) must still succeed and clear any straggler aggregates.
    """
    from backend.config import Settings
    from backend.services.runs.purge import purge_project

    db_url = f"sqlite:///{tmp_path}/test.db"
    settings_for_test = Settings(database_url=db_url)
    monkeypatch.setattr(
        "backend.services.runs.purge.get_settings", lambda: settings_for_test
    )

    result = purge_project("nonexistent", tmp_path / "runs")
    assert result["run_dir_removed"] is False
    assert result["aggregates_removed"] == 0
