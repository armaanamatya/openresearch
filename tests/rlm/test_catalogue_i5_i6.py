"""Regression tests for catalogue issues I5 and I6.

I5 — re-running a paper conflicts (no --fresh purge):
    Symptom: a second `reproduce` run on the same paper hits ConcurrencyError
    because the SQLite event store still holds the previous run's aggregates
    even after `rm -rf runs/<id>/`.  `purge_project_aggregates` must clear
    all five aggregate patterns atomically.

I6 — concurrent runs 429 with no backoff:
    Symptom: a second concurrent run against Featherless gets HTTP 429 and
    fails hard because (a) the OpenAI SDK client was constructed without
    max_retries and (b) generate_rubric_tree had no inter-attempt sleep.
    Both fixes must be verifiable without a live network call.
"""

from __future__ import annotations

from typing import ClassVar
from unittest.mock import MagicMock, patch

import pytest

from backend.eventstore.sqlite_store import SqliteEventStore
from backend.messaging.envelope import AggregateId, make_envelope
from backend.messaging.event import (
    DomainEvent,
    _clear_registry_for_tests,
    register_event,
)


# ---------------------------------------------------------------------------
# Helpers shared by I5 tests
# ---------------------------------------------------------------------------


def _register_minimal_event():
    """Register (or re-use) a minimal DomainEvent class for appending."""
    @register_event
    class PingEvent(DomainEvent):
        event_type: ClassVar[str] = "ping_event_i5"
        schema_version: ClassVar[int] = 1
        msg: str

    return PingEvent


def _append_one(store: SqliteEventStore, aggregate_id: str, PingEvent) -> None:
    """Write a single event to ``aggregate_id`` at version 1."""
    ev = PingEvent(msg="hi")
    env = make_envelope(source="test")
    store.append(
        AggregateId(aggregate_id),
        "test",
        [ev],
        0,
        [env],
    )


# ---------------------------------------------------------------------------
# I5 — purge_project_aggregates
# ---------------------------------------------------------------------------


def test_purge_removes_all_project_aggregates(tmp_path):
    """purge_project_aggregates deletes root, sub-aggregates, and rlm-run aggregate.

    Symptom: without purge, a re-run raises ConcurrencyError because the
    event store already holds events for the project's aggregates.
    """
    _clear_registry_for_tests()
    PingEvent = _register_minimal_event()

    store = SqliteEventStore(f"sqlite:///{tmp_path}/events.db")
    try:
        pid = "prj_abc123"

        # Write one event to each of the three aggregate id patterns.
        _append_one(store, pid, PingEvent)
        _append_one(store, f"{pid}:parsed", PingEvent)
        _append_one(store, f"rlm-run:{pid}", PingEvent)

        # Write an unrelated aggregate that must survive.
        _append_one(store, "prj_other_unrelated", PingEvent)

        # Confirm all four aggregates have events.
        assert store.get_aggregate_version(AggregateId(pid)) == 1
        assert store.get_aggregate_version(AggregateId(f"{pid}:parsed")) == 1
        assert store.get_aggregate_version(AggregateId(f"rlm-run:{pid}")) == 1
        assert store.get_aggregate_version(AggregateId("prj_other_unrelated")) == 1

        deleted = store.purge_project_aggregates(pid)

        # All three project aggregates are gone.
        assert store.get_aggregate_version(AggregateId(pid)) == 0
        assert store.get_aggregate_version(AggregateId(f"{pid}:parsed")) == 0
        assert store.get_aggregate_version(AggregateId(f"rlm-run:{pid}")) == 0

        # Unrelated aggregate must survive.
        assert store.get_aggregate_version(AggregateId("prj_other_unrelated")) == 1

        # Must report the correct deletion count (3 rows).
        assert deleted == 3
    finally:
        store.close()


def test_purge_of_empty_project_returns_zero(tmp_path):
    """purge_project_aggregates on a project with no events returns 0."""
    _clear_registry_for_tests()
    store = SqliteEventStore(f"sqlite:///{tmp_path}/events.db")
    try:
        result = store.purge_project_aggregates("prj_nonexistent")
        assert result == 0
    finally:
        store.close()


def test_purge_sub_aggregates_with_colon_suffix(tmp_path):
    """Sub-aggregates like <pid>:index and <pid>:discovery are all purged."""
    _clear_registry_for_tests()
    PingEvent = _register_minimal_event()

    store = SqliteEventStore(f"sqlite:///{tmp_path}/events.db")
    try:
        pid = "prj_multi"
        suffixes = ["", ":parsed", ":index", ":discovery"]
        for suffix in suffixes:
            _append_one(store, f"{pid}{suffix}", PingEvent)
        _append_one(store, f"rlm-run:{pid}", PingEvent)

        deleted = store.purge_project_aggregates(pid)

        for suffix in suffixes:
            assert store.get_aggregate_version(AggregateId(f"{pid}{suffix}")) == 0
        assert store.get_aggregate_version(AggregateId(f"rlm-run:{pid}")) == 0
        assert deleted == 5
    finally:
        store.close()


# ---------------------------------------------------------------------------
# I6 — OpenAILlmClient max_retries
# ---------------------------------------------------------------------------


def test_openai_llm_client_sets_max_retries():
    """OpenAILlmClient must construct the SDK client with max_retries set.

    Symptom: without max_retries, a 429 from Featherless fails hard instead
    of being retried with exponential backoff.
    """
    try:
        from backend.services.context.workspace.tools.openai_client import (
            OpenAILlmClient,
        )
    except ImportError:
        pytest.skip("openai package not installed")

    captured_kwargs: dict = {}

    class _CapturingOpenAI:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)
            # Minimal stub so the client attribute exists.
            self.chat = MagicMock()

    # OpenAI is imported lazily inside __init__ via `from openai import OpenAI`,
    # so we patch the canonical location in the openai package itself.
    with patch("openai.OpenAI", _CapturingOpenAI):
        OpenAILlmClient(
            model="gpt-4o-mini",
            api_key="test-key",
            base_url="https://api.featherless.ai/v1",
        )

    assert "max_retries" in captured_kwargs, (
        "OpenAILlmClient must pass max_retries to the OpenAI SDK constructor"
    )
    assert captured_kwargs["max_retries"] >= 1, (
        "max_retries must be a positive integer"
    )


# ---------------------------------------------------------------------------
# I6 — generate_rubric_tree exponential backoff
# ---------------------------------------------------------------------------


def test_rubric_gen_sleeps_between_retries():
    """generate_rubric_tree must sleep between retries after an LLM exception.

    Symptom: without a sleep, a transient 429 is immediately retried and
    fails again before the server's rate-limit window resets.
    """
    from backend.agents.rlm.rubric_gen import generate_rubric_tree

    sleep_calls: list[float] = []

    class _FailOnceThenSucceedClient:
        """Raises on the first call, returns valid JSON on the second."""

        _VALID = (
            '{"categories": [{"name": "Method fidelity", "weight": 0.6, '
            '"leaves": [{"requirements": "The model implements X as described", '
            '"weight": 1.0}]}]}'
        )

        def __init__(self):
            self.call_count = 0

        def complete(self, *, system: str, user: str) -> str:
            self.call_count += 1
            if self.call_count == 1:
                raise RuntimeError("simulated 429")
            return self._VALID

    client = _FailOnceThenSucceedClient()
    long_paper = "x " * 300

    with patch("backend.agents.rlm.rubric_gen.time.sleep", side_effect=lambda s: sleep_calls.append(s)):
        result = generate_rubric_tree(long_paper, client, max_attempts=3)

    # Should succeed on attempt 2.
    assert result is not None, "should succeed after one failure"
    assert client.call_count == 2

    # At least one sleep must have been called between the failed attempt and
    # the successful retry.
    assert len(sleep_calls) >= 1, "must sleep at least once between retry attempts"
    assert all(s > 0 for s in sleep_calls), "sleep duration must be positive"


def test_rubric_gen_no_sleep_on_first_success():
    """generate_rubric_tree must not sleep when the first attempt succeeds."""
    from backend.agents.rlm.rubric_gen import generate_rubric_tree

    sleep_calls: list[float] = []

    _VALID = (
        '{"categories": [{"name": "Cat A", "weight": 1.0, '
        '"leaves": [{"requirements": "criterion 1", "weight": 1.0}]}]}'
    )

    class _ImmediateSuccessClient:
        def complete(self, *, system: str, user: str) -> str:
            return _VALID

    with patch("backend.agents.rlm.rubric_gen.time.sleep", side_effect=lambda s: sleep_calls.append(s)):
        result = generate_rubric_tree("x " * 300, _ImmediateSuccessClient(), max_attempts=3)

    assert result is not None
    assert sleep_calls == [], "no sleep on first-attempt success"
