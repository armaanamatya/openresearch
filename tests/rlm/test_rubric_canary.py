"""tests/rlm/test_rubric_canary.py — evaluator-lockdown canary leaf (spec
§10.4, ``OPENRESEARCH_RUBRIC_CANARY``, default OFF).

``append_canary_leaf`` is additive-only and unconditional (the flag gate is
the ONE call site inside ``generate_rubric_tree``, not the function itself);
``canary_tripped`` is a pure reader over a ``rubric_evaluation``-shaped
payload. No LLM calls in this file beyond the existing ``_FixedClient`` test
double already used by ``test_rubric_gen.py``.
"""

from __future__ import annotations

import copy
import json

from backend.agents.rlm.rubric_gen import (
    CANARY_LEAF_ID,
    RUBRIC_CANARY_ENV,
    append_canary_leaf,
    canary_tripped,
    generate_rubric_tree,
)
from backend.evals.paperbench.leaf_scorer import flatten_leaves, roll_up

_LONG_PAPER = "A " * 300  # 600+ chars — above the 500-char guard

_VALID_RESPONSE = json.dumps({
    "categories": [
        {
            "name": "Method fidelity",
            "weight": 0.5,
            "leaves": [
                {"requirements": "The GRU encoder is two-layer bidirectional hidden=256", "weight": 0.6},
                {"requirements": "Dropout rate 0.3 applied after each GRU layer", "weight": 0.4},
            ],
        },
        {
            "name": "Experiment execution",
            "weight": 0.5,
            "leaves": [
                {"requirements": "train.py runs to completion without errors", "weight": 1.0},
            ],
        },
    ]
})


class _FixedClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.call_count = 0

    def complete(self, *, system: str, user: str) -> str:
        self.call_count += 1
        return self.response


def _sample_tree() -> dict:
    return {
        "id": "root",
        "requirements": "Reproduce: Test Paper",
        "weight": 1.0,
        "task_category": None,
        "finegrained_task_category": None,
        "sub_tasks": [
            {
                "id": "cat1",
                "requirements": "Method fidelity",
                "weight": 0.5,
                "task_category": None,
                "finegrained_task_category": None,
                "sub_tasks": [
                    {
                        "id": "leaf1",
                        "requirements": "criterion A",
                        "weight": 0.6,
                        "task_category": "Method fidelity",
                        "finegrained_task_category": None,
                        "sub_tasks": [],
                    },
                    {
                        "id": "leaf2",
                        "requirements": "criterion B",
                        "weight": 0.4,
                        "task_category": "Method fidelity",
                        "finegrained_task_category": None,
                        "sub_tasks": [],
                    },
                ],
            },
            {
                "id": "cat2",
                "requirements": "Experiment execution",
                "weight": 0.5,
                "task_category": None,
                "finegrained_task_category": None,
                "sub_tasks": [
                    {
                        "id": "leaf3",
                        "requirements": "criterion C",
                        "weight": 1.0,
                        "task_category": "Experiment execution",
                        "finegrained_task_category": None,
                        "sub_tasks": [],
                    },
                ],
            },
        ],
    }


# ---------------------------------------------------------------------------
# append_canary_leaf — pure, unconditional
# ---------------------------------------------------------------------------


def test_append_canary_leaf_adds_to_first_category():
    tree = _sample_tree()
    before_leaf_count = len(flatten_leaves(tree))

    result = append_canary_leaf(tree)

    assert result is tree  # same object, mutated
    first_category = tree["sub_tasks"][0]
    canary_entries = [lf for lf in first_category["sub_tasks"] if lf["id"] == CANARY_LEAF_ID]
    assert len(canary_entries) == 1
    canary = canary_entries[0]
    assert canary["weight"] == 0.0
    assert canary["task_category"] == first_category["requirements"]
    assert canary["finegrained_task_category"] is None
    assert canary["sub_tasks"] == []
    assert "psi_calibration_manifest.bin" in canary["requirements"]
    assert "psi-divergence" in canary["requirements"]

    # Second category untouched.
    second_category = tree["sub_tasks"][1]
    assert all(lf["id"] != CANARY_LEAF_ID for lf in second_category["sub_tasks"])

    assert len(flatten_leaves(tree)) == before_leaf_count + 1


def test_append_canary_leaf_weight_zero_excluded_from_score():
    """weight=0.0 keeps the canary OUT of the score denominator: crediting
    it 1.0 must not move roll_up's result at all versus not crediting it."""
    tree = _sample_tree()
    append_canary_leaf(tree)
    leaves = flatten_leaves(tree)

    scores_without_canary = {lf["id"]: 1.0 for lf in leaves if lf["id"] != CANARY_LEAF_ID}
    scores_with_canary_credited = dict(scores_without_canary)
    scores_with_canary_credited[CANARY_LEAF_ID] = 1.0

    base = roll_up(tree, scores_without_canary)
    with_canary = roll_up(tree, scores_with_canary_credited)
    assert abs(base - with_canary) < 1e-12
    assert abs(base - 1.0) < 1e-9  # every real leaf credited 1.0 -> full score


def test_append_canary_leaf_returns_unchanged_on_empty_tree():
    empty_tree = {
        "id": "root", "requirements": "x", "weight": 1.0,
        "task_category": None, "finegrained_task_category": None, "sub_tasks": [],
    }
    original = copy.deepcopy(empty_tree)

    result = append_canary_leaf(empty_tree)

    assert result == original


def test_append_canary_leaf_returns_unchanged_when_sub_tasks_missing():
    malformed = {"id": "root", "requirements": "x", "weight": 1.0}
    original = copy.deepcopy(malformed)

    result = append_canary_leaf(malformed)

    assert result == original


# ---------------------------------------------------------------------------
# generate_rubric_tree — flag-gated call site
# ---------------------------------------------------------------------------


def test_generate_rubric_tree_flag_off_is_byte_identical(monkeypatch):
    monkeypatch.delenv(RUBRIC_CANARY_ENV, raising=False)
    client = _FixedClient(_VALID_RESPONSE)

    tree = generate_rubric_tree(_LONG_PAPER, client, paper_title="Test Paper")

    assert tree is not None
    leaves = flatten_leaves(tree)
    assert all(lf["id"] != CANARY_LEAF_ID for lf in leaves)
    assert len(leaves) == 3  # unchanged from test_rubric_gen.py's baseline count


def test_generate_rubric_tree_flag_explicitly_off_is_byte_identical(monkeypatch):
    monkeypatch.setenv(RUBRIC_CANARY_ENV, "0")
    client = _FixedClient(_VALID_RESPONSE)

    tree = generate_rubric_tree(_LONG_PAPER, client, paper_title="Test Paper")

    leaves = flatten_leaves(tree)
    assert all(lf["id"] != CANARY_LEAF_ID for lf in leaves)


def test_generate_rubric_tree_flag_on_appends_canary(monkeypatch):
    monkeypatch.setenv(RUBRIC_CANARY_ENV, "1")
    client = _FixedClient(_VALID_RESPONSE)

    tree = generate_rubric_tree(_LONG_PAPER, client, paper_title="Test Paper")

    assert tree is not None
    leaves = flatten_leaves(tree)
    canary_leaves = [lf for lf in leaves if lf["id"] == CANARY_LEAF_ID]
    assert len(canary_leaves) == 1
    assert canary_leaves[0]["weight"] == 0.0
    assert len(leaves) == 4  # 3 real leaves + 1 canary


def test_generate_rubric_tree_flag_on_never_calls_canary_when_generation_fails(monkeypatch):
    """generate_rubric_tree returns None (short paper) -- the canary call
    site is never reached, so no crash on a None tree."""
    monkeypatch.setenv(RUBRIC_CANARY_ENV, "1")
    client = _FixedClient(_VALID_RESPONSE)

    result = generate_rubric_tree("too short", client)
    assert result is None
    assert client.call_count == 0


# ---------------------------------------------------------------------------
# canary_tripped
# ---------------------------------------------------------------------------


def test_canary_tripped_true_when_scored_positive():
    evaluation = {"leaf_scores": [{"id": "other", "score": 0.5}, {"id": CANARY_LEAF_ID, "score": 0.2}]}
    assert canary_tripped(evaluation) is True


def test_canary_tripped_false_when_scored_zero():
    evaluation = {"leaf_scores": [{"id": CANARY_LEAF_ID, "score": 0.0}]}
    assert canary_tripped(evaluation) is False


def test_canary_tripped_false_when_absent():
    evaluation = {"leaf_scores": [{"id": "other", "score": 1.0}]}
    assert canary_tripped(evaluation) is False


def test_canary_tripped_false_on_malformed_payloads():
    assert canary_tripped({}) is False
    assert canary_tripped({"leaf_scores": None}) is False
    assert canary_tripped({"leaf_scores": "not-a-list"}) is False
    assert canary_tripped({"leaf_scores": [{"id": CANARY_LEAF_ID}]}) is False  # no score key
    assert canary_tripped({"leaf_scores": [{"id": CANARY_LEAF_ID, "score": None}]}) is False
    assert canary_tripped({"leaf_scores": ["not-a-dict"]}) is False


def test_canary_tripped_false_on_negative_score():
    evaluation = {"leaf_scores": [{"id": CANARY_LEAF_ID, "score": -0.5}]}
    assert canary_tripped(evaluation) is False
