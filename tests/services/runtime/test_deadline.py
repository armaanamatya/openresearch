"""Unit tests for ``backend.services.runtime.deadline`` -- pure absolute-epoch
deadline helpers.

WS3 (durable cloud-native orchestration) design: when a controller pod is
killed and a successor adopts an in-flight GPU cell Job, the successor must
inherit the run's REMAINING wall-clock budget, not a fresh full one. This
module is PURE -- no I/O, no clock reads -- every test below drives
``now_epoch`` explicitly as an injected float; the module never calls
``time.time()`` itself so determinism here is structural, not incidental.
"""

from __future__ import annotations

from backend.services.runtime import deadline as dl

# ---------------------------------------------------------------------------
# make_deadline / remaining_s
# ---------------------------------------------------------------------------


class TestMakeDeadlineAndRemaining:
    def test_deadline_epoch_is_now_plus_budget(self) -> None:
        rec = dl.make_deadline(1000.0, 600.0)
        assert rec["deadline_epoch"] == 1600.0

    def test_remaining_s_at_creation_equals_full_budget(self) -> None:
        rec = dl.make_deadline(1000.0, 600.0)
        assert dl.remaining_s(rec, 1000.0) == 600.0

    def test_remaining_s_partway_through(self) -> None:
        rec = dl.make_deadline(1000.0, 600.0)
        assert dl.remaining_s(rec, 1300.0) == 300.0

    def test_remaining_s_at_exact_deadline_is_zero(self) -> None:
        rec = dl.make_deadline(1000.0, 600.0)
        assert dl.remaining_s(rec, 1600.0) == 0.0

    def test_remaining_s_past_deadline_clamps_to_zero_never_negative(self) -> None:
        rec = dl.make_deadline(1000.0, 600.0)
        assert dl.remaining_s(rec, 5000.0) == 0.0

    def test_record_shape_carries_version_created_and_budget(self) -> None:
        rec = dl.make_deadline(1000.0, 600.0)
        assert rec["version"] == 1
        assert rec["created_epoch"] == 1000.0
        assert rec["budget_s"] == 600.0


# ---------------------------------------------------------------------------
# is_expired
# ---------------------------------------------------------------------------


class TestIsExpired:
    def test_false_just_before_the_deadline(self) -> None:
        rec = dl.make_deadline(1000.0, 600.0)
        assert dl.is_expired(rec, 1599.9) is False

    def test_true_exactly_at_the_deadline_boundary(self) -> None:
        rec = dl.make_deadline(1000.0, 600.0)
        assert dl.is_expired(rec, 1600.0) is True

    def test_true_well_past_the_deadline(self) -> None:
        rec = dl.make_deadline(1000.0, 600.0)
        assert dl.is_expired(rec, 2000.0) is True


# ---------------------------------------------------------------------------
# Negative budget clamps to zero (already-expired record).
# ---------------------------------------------------------------------------


class TestNegativeBudgetClamps:
    def test_negative_budget_deadline_equals_now(self) -> None:
        rec = dl.make_deadline(1000.0, -5.0)
        assert rec["deadline_epoch"] == 1000.0

    def test_negative_budget_stored_budget_s_clamped_to_zero(self) -> None:
        rec = dl.make_deadline(1000.0, -5.0)
        assert rec["budget_s"] == 0.0

    def test_negative_budget_record_is_already_expired_at_creation(self) -> None:
        rec = dl.make_deadline(1000.0, -5.0)
        assert dl.is_expired(rec, 1000.0) is True


# ---------------------------------------------------------------------------
# serialize / parse round-trip + byte-determinism.
# ---------------------------------------------------------------------------


class TestSerializeParseRoundTrip:
    def test_parse_of_serialize_reproduces_the_record(self) -> None:
        rec = dl.make_deadline(1000.0, 600.0)
        assert dl.parse(dl.serialize(rec)) == rec

    def test_serialize_is_byte_deterministic_across_calls(self) -> None:
        rec_a = dl.make_deadline(1000.0, 600.0)
        rec_b = dl.make_deadline(1000.0, 600.0)
        assert dl.serialize(rec_a) == dl.serialize(rec_b)

    def test_serialize_returns_bytes(self) -> None:
        rec = dl.make_deadline(1000.0, 600.0)
        assert isinstance(dl.serialize(rec), bytes)


# ---------------------------------------------------------------------------
# A parsed-from-bytes record behaves identically to a fresh record.
# ---------------------------------------------------------------------------


class TestParsedRecordBehavesLikeFresh:
    def test_remaining_s_matches_between_fresh_and_parsed(self) -> None:
        fresh = dl.make_deadline(1000.0, 600.0)
        parsed = dl.parse(dl.serialize(fresh))
        assert dl.remaining_s(parsed, 1300.0) == dl.remaining_s(fresh, 1300.0)

    def test_is_expired_matches_between_fresh_and_parsed_before_deadline(self) -> None:
        fresh = dl.make_deadline(1000.0, 600.0)
        parsed = dl.parse(dl.serialize(fresh))
        assert dl.is_expired(parsed, 1599.9) == dl.is_expired(fresh, 1599.9)

    def test_is_expired_matches_between_fresh_and_parsed_at_boundary(self) -> None:
        fresh = dl.make_deadline(1000.0, 600.0)
        parsed = dl.parse(dl.serialize(fresh))
        assert dl.is_expired(parsed, 1600.0) == dl.is_expired(fresh, 1600.0)
        assert dl.is_expired(parsed, 1600.0) is True
