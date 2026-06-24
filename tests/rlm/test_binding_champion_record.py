"""Tests for _champion_rubric_block — the pure helper that builds the rubric
block snapshotted alongside each champion code snapshot (Task 3 of the
BES conversion-correctness plan).
"""

from backend.agents.rlm.binding import _champion_rubric_block


def test_champion_block_built_from_result():
    result = {
        "leaf_scores": [{"id": "a", "score": 0.6}],
        "weak_leaves": [],
        "leaf_count": 1,
        "meets_target": False,
        "compute_adjusted_score": 0.61,
    }
    block = _champion_rubric_block(result, score=0.6, target=0.6)
    assert block["overall_score"] == 0.6
    assert block["leaf_scores"] == [{"id": "a", "score": 0.6}]
    assert block["meets_target"] is False
    assert block["target_score"] == 0.6


def test_champion_block_none_target():
    """When target is None, target_score should be None (not raise)."""
    result = {"leaf_scores": [], "weak_leaves": [], "leaf_count": 0,
              "meets_target": None, "compute_adjusted_score": None}
    block = _champion_rubric_block(result, score=0.0, target=None)
    assert block["target_score"] is None
    assert block["overall_score"] == 0.0


def test_champion_block_missing_keys():
    """Missing optional keys in result should not raise — return None for those."""
    result = {}
    block = _champion_rubric_block(result, score=0.5, target=0.7)
    assert block["leaf_scores"] == []
    assert block["weak_leaves"] == []
    assert block["leaf_count"] is None
    assert block["meets_target"] is None
    assert block["compute_adjusted_score"] is None
    assert block["overall_score"] == 0.5
    assert block["target_score"] == 0.7
