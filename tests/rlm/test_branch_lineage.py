"""Branch-tree lineage domain events: validated construction, round-trip
serialization (the event-store persistence contract), frozen immutability, and
the aggregate-id helper."""
import pytest

from backend.agents.rlm.branch_lineage import (
    BranchPromoted,
    BranchRevived,
    BranchSpawned,
    BranchTrueKilled,
    DedupHit,
    FrozenPoolEviction,
    RungClimbed,
    branch_tree_aggregate_id,
)

_ALL = [
    BranchSpawned(branch_id="b1", hypothesis_fingerprint="fp1"),
    BranchSpawned(
        branch_id="b2", parent_branch_id="b1", branch_type="ambiguity",
        rung=1, hypothesis_fingerprint="fp2",
    ),
    RungClimbed(branch_id="b1", rung=1, step_budget=5000, measured_score=0.42, gpu_usd=1.3),
    BranchPromoted(branch_id="b1", from_rung=1, to_rung=2, cohort_ids=("b1", "b2")),
    FrozenPoolEviction(branch_id="b2", rung=1, ckpt_uri="gs://ck/b2", reason="halved_below_topk"),
    BranchRevived(branch_id="b2", from_ckpt="gs://ck/b2"),
    BranchTrueKilled(branch_id="b3", termination_cause="frozen_params"),
    DedupHit(hypothesis_fingerprint="fp2", existing_branch_id="b2"),
]


@pytest.mark.parametrize("event", _ALL, ids=lambda e: e.event_type)
def test_round_trip_serialization(event):
    """model_dump → model_validate must reproduce the event exactly — this is the
    event-store persist/load contract."""
    restored = type(event).model_validate(event.model_dump())
    assert restored == event


@pytest.mark.parametrize("event", _ALL, ids=lambda e: e.event_type)
def test_events_are_frozen(event):
    with pytest.raises((TypeError, ValueError)):  # pydantic frozen → ValidationError
        event.branch_id = "mutated"  # type: ignore[misc]


def test_event_types_are_unique_and_stamped():
    types = [type(e).event_type for e in _ALL]
    # BranchSpawned appears twice in _ALL; dedupe by class.
    by_class = {type(e).__name__: type(e).event_type for e in _ALL}
    assert len(set(by_class.values())) == len(by_class)  # one event_type per class
    assert all(t for t in types)  # every event_type non-empty
    assert all(type(e).schema_version == 1 for e in _ALL)


def test_cohort_ids_coerced_to_tuple():
    e = BranchPromoted(branch_id="b1", from_rung=0, to_rung=1, cohort_ids=["a", "b"])
    assert e.cohort_ids == ("a", "b")
    assert isinstance(e.cohort_ids, tuple)


def test_aggregate_id_helper():
    assert branch_tree_aggregate_id("prj_abc") == "branch-tree:prj_abc"
