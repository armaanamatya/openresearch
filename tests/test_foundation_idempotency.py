"""Tests for backend.messaging.idempotency."""

from __future__ import annotations

import time
from datetime import timedelta

import pytest

from backend.messaging.command import CommandId
from backend.messaging.envelope import AggregateId, EventId
from backend.messaging.idempotency import DuplicateCommandError, IdempotencyTable
from backend.persistence.database import Database


@pytest.fixture
def db(tmp_path) -> Database:
    db = Database(f"sqlite:///{tmp_path}/test.db")
    db.connection  # noqa: B018  - eager init
    yield db
    db.close()


def test_lookup_returns_none_for_unknown(db):
    table = IdempotencyTable(db)
    assert table.lookup(AggregateId("agg_1"), CommandId("cmd_1")) is None


def test_record_then_lookup_returns_event_ids(db):
    table = IdempotencyTable(db)
    agg = AggregateId("agg_1")
    cmd = CommandId("cmd_1")
    eids = (EventId("evt_a"), EventId("evt_b"))

    table.record(agg, cmd, eids)
    db.connection.commit()

    found = table.lookup(agg, cmd)
    assert found == eids


def test_record_with_same_result_is_idempotent_no_op(db):
    """Re-recording the same result for the same (agg, cmd) is a no-op.
    The original row is preserved unchanged."""
    table = IdempotencyTable(db)
    agg = AggregateId("agg_1")
    cmd = CommandId("cmd_1")
    eids = (EventId("evt_a"), EventId("evt_b"))

    table.record(agg, cmd, eids)
    db.connection.commit()
    # Idempotent: same result, no exception, no overwrite.
    table.record(agg, cmd, eids)
    db.connection.commit()

    assert table.lookup(agg, cmd) == eids


def test_record_with_divergent_result_raises_duplicate_command_error(db):
    """A real bug: caller skipped lookup() and re-executed IO with
    a different outcome. The table refuses to corrupt the original."""
    table = IdempotencyTable(db)
    agg = AggregateId("agg_1")
    cmd = CommandId("cmd_1")

    table.record(agg, cmd, (EventId("evt_first"),))
    db.connection.commit()
    with pytest.raises(DuplicateCommandError) as exc_info:
        table.record(agg, cmd, (EventId("evt_second"),))
    err = exc_info.value
    assert err.aggregate_id == agg
    assert err.command_id == cmd
    assert err.existing == (EventId("evt_first"),)
    assert err.incoming == (EventId("evt_second"),)
    # Original is preserved.
    assert table.lookup(agg, cmd) == (EventId("evt_first"),)


def test_lookup_treats_expired_rows_as_missing(db):
    table = IdempotencyTable(db, default_retention=timedelta(milliseconds=10))
    agg = AggregateId("agg_1")
    cmd = CommandId("cmd_1")
    table.record(agg, cmd, (EventId("evt_a"),))
    db.connection.commit()

    time.sleep(0.05)
    assert table.lookup(agg, cmd) is None


def test_record_replaces_expired_row(db):
    """An expired row is treated as missing by lookup(). A new record
    on the same (agg, cmd) replaces it cleanly."""
    table = IdempotencyTable(db, default_retention=timedelta(milliseconds=10))
    agg = AggregateId("agg_1")
    cmd = CommandId("cmd_1")
    table.record(agg, cmd, (EventId("evt_old"),))
    db.connection.commit()

    time.sleep(0.05)
    # Expired — record() replaces.
    table.record(agg, cmd, (EventId("evt_new"),))
    db.connection.commit()
    assert table.lookup(agg, cmd) == (EventId("evt_new"),)


def test_purge_expired_removes_rows(db):
    table = IdempotencyTable(db, default_retention=timedelta(milliseconds=10))
    table.record(AggregateId("a1"), CommandId("c1"), (EventId("e1"),))
    table.record(AggregateId("a2"), CommandId("c2"), (EventId("e2"),))
    db.connection.commit()

    time.sleep(0.05)
    purged = table.purge_expired()
    assert purged == 2
    assert table.lookup(AggregateId("a1"), CommandId("c1")) is None
