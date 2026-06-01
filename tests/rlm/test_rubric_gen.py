"""Tests for backend.agents.rlm.rubric_gen.generate_rubric_tree.

Locks in:
- Valid JSON → well-formed tree; every node has non-empty unique id; leaves have
  sub_tasks == []; flatten_leaves count matches input; roll_up correctness;
  per-level weights sum to ~1.0.
- LLM exception or garbage on every attempt → returns None after max_attempts.
- Mixed malformed/valid categories and leaves → malformed dropped, valid kept.
- All-invalid weights in a level → equal weights; a partial-invalid level is
  mean-filled so no leaf is silently zeroed; weights sum to ~1.0.
- Paper text shorter than 500 chars → returns None without calling the client.
"""

from __future__ import annotations

import json

import pytest

from backend.agents.rlm.rubric_gen import generate_rubric_tree, _is_placeholder_requirement
from backend.evals.paperbench.leaf_scorer import flatten_leaves, roll_up

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

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
    """Returns the same canned response on every call."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.call_count = 0

    def complete(self, *, system: str, user: str) -> str:
        self.call_count += 1
        return self.response


class _FailClient:
    """Always raises an exception."""

    def __init__(self) -> None:
        self.call_count = 0

    def complete(self, *, system: str, user: str) -> str:
        self.call_count += 1
        raise RuntimeError("simulated LLM failure")


class _GarbageClient:
    """Returns unparseable text on every call."""

    def __init__(self) -> None:
        self.call_count = 0

    def complete(self, *, system: str, user: str) -> str:
        self.call_count += 1
        return "sorry, I cannot help with that"


# ---------------------------------------------------------------------------
# Test 1: valid JSON → well-formed tree
# ---------------------------------------------------------------------------


def test_valid_response_returns_tree():
    """Valid LLM JSON produces a tree with correct structure and node counts."""
    client = _FixedClient(_VALID_RESPONSE)
    tree = generate_rubric_tree(_LONG_PAPER, client, paper_title="Test Paper")

    assert tree is not None

    # Every node has a non-empty id
    def _all_ids(node):
        ids = [node.get("id", "")]
        for child in node.get("sub_tasks") or []:
            ids.extend(_all_ids(child))
        return ids

    all_ids = _all_ids(tree)
    assert all(len(i) > 0 for i in all_ids), "every node must have a non-empty id"

    # All ids are unique
    assert len(all_ids) == len(set(all_ids)), "all node ids must be unique"

    # Leaves have sub_tasks == []
    leaves = flatten_leaves(tree)
    for leaf in leaves:
        assert leaf.get("sub_tasks") == [], f"leaf {leaf['id']} must have empty sub_tasks"

    # flatten_leaves count matches input leaf count (2 + 1 = 3)
    assert len(leaves) == 3

    # roll_up with all-1.0 scores == 1.0
    all_one = {leaf["id"]: 1.0 for leaf in leaves}
    assert abs(roll_up(tree, all_one) - 1.0) < 1e-9

    # roll_up with empty scores == 0.0
    assert abs(roll_up(tree, {}) - 0.0) < 1e-9

    # Per-level weights sum to ~1.0 at both category and leaf level
    cat_weight_sum = sum(cat["weight"] for cat in tree["sub_tasks"])
    assert abs(cat_weight_sum - 1.0) < 1e-9, "category weights must sum to 1.0"

    for cat in tree["sub_tasks"]:
        leaf_weight_sum = sum(lf["weight"] for lf in cat["sub_tasks"])
        assert abs(leaf_weight_sum - 1.0) < 1e-9, f"leaf weights in '{cat['requirements']}' must sum to 1.0"


# ---------------------------------------------------------------------------
# Test 2: LLM always raises → returns None after max_attempts
# ---------------------------------------------------------------------------


def test_llm_exception_returns_none():
    """generate_rubric_tree returns None when the client always raises."""
    client = _FailClient()
    result = generate_rubric_tree(_LONG_PAPER, client, max_attempts=3)
    assert result is None
    assert client.call_count == 3  # exhausted all attempts


# ---------------------------------------------------------------------------
# Test 3: LLM always returns garbage → returns None after max_attempts
# ---------------------------------------------------------------------------


def test_garbage_response_returns_none():
    """Unparseable LLM responses exhaust retries and return None."""
    client = _GarbageClient()
    result = generate_rubric_tree(_LONG_PAPER, client, max_attempts=2)
    assert result is None
    assert client.call_count == 2


# ---------------------------------------------------------------------------
# Test 4: mixed malformed/valid categories — malformed dropped, valid kept
# ---------------------------------------------------------------------------


def test_mixed_malformed_valid_categories():
    """Malformed categories/leaves are dropped; valid ones are kept."""
    response = json.dumps({
        "categories": [
            # Valid category
            {
                "name": "Good category",
                "weight": 0.7,
                "leaves": [
                    {"requirements": "Specific checkable criterion A", "weight": 1.0},
                ],
            },
            # Category with no name — should be dropped
            {
                "name": "",
                "weight": 0.1,
                "leaves": [{"requirements": "Something", "weight": 1.0}],
            },
            # Category with all malformed leaves — should be dropped
            {
                "name": "Bad leaves category",
                "weight": 0.2,
                "leaves": [
                    {"requirements": "", "weight": 1.0},   # empty requirements
                    {"weight": 0.5},                       # missing requirements
                ],
            },
        ]
    })
    client = _FixedClient(response)
    tree = generate_rubric_tree(_LONG_PAPER, client)

    assert tree is not None
    assert len(tree["sub_tasks"]) == 1  # only "Good category" survives
    assert tree["sub_tasks"][0]["requirements"] == "Good category"
    assert len(flatten_leaves(tree)) == 1


# ---------------------------------------------------------------------------
# Test 5: missing/zero/negative weights → equal weights, no crash
# ---------------------------------------------------------------------------


def test_bad_weights_equal_fallback():
    """Invalid weights (None, 0, negative) fall back to equal distribution."""
    response = json.dumps({
        "categories": [
            {
                "name": "Cat A",
                "weight": 0,          # invalid — zero
                "leaves": [
                    {"requirements": "criterion 1", "weight": -1},   # invalid
                    {"requirements": "criterion 2", "weight": None},  # invalid
                ],
            },
            {
                "name": "Cat B",
                "weight": None,        # invalid — None
                "leaves": [
                    {"requirements": "criterion 3", "weight": 0},    # invalid
                ],
            },
        ]
    })
    client = _FixedClient(response)
    tree = generate_rubric_tree(_LONG_PAPER, client)

    assert tree is not None

    # Category weights sum to 1.0
    cat_weight_sum = sum(cat["weight"] for cat in tree["sub_tasks"])
    assert abs(cat_weight_sum - 1.0) < 1e-9

    # Leaf weights within each category sum to 1.0
    for cat in tree["sub_tasks"]:
        leaf_weight_sum = sum(lf["weight"] for lf in cat["sub_tasks"])
        assert abs(leaf_weight_sum - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Test 6: paper text < 500 chars → None without calling client
# ---------------------------------------------------------------------------


def test_short_paper_returns_none_without_calling_client():
    """Paper text below 500 stripped chars returns None; LLM is never called."""
    short_paper = "This paper proposes X."
    client = _FixedClient(_VALID_RESPONSE)
    result = generate_rubric_tree(short_paper, client)
    assert result is None
    assert client.call_count == 0  # client must NOT be called


# ---------------------------------------------------------------------------
# Test 7: a partial-invalid level is mean-filled, never zeroed
# ---------------------------------------------------------------------------


def test_partial_weights_filled_not_zeroed():
    """A leaf missing its weight is mean-filled, not silently dropped to 0.

    Regression: an invalid weight among valid ones must not become 0.0 — a
    0-weight leaf is excluded from roll_up, so a real criterion would vanish
    from the score.
    """
    response = json.dumps({
        "categories": [
            {"name": "Cat A", "weight": 0.6, "leaves": [
                {"requirements": "criterion 1", "weight": 0.4},
                {"requirements": "criterion 2"},                  # missing weight
                {"requirements": "criterion 3", "weight": 0.4},
            ]},
            {"name": "Cat B", "weight": 0.4, "leaves": [
                {"requirements": "criterion 4", "weight": 1.0},
            ]},
        ]
    })
    tree = generate_rubric_tree(_LONG_PAPER, _FixedClient(response))

    assert tree is not None
    leaf_weights = [lf["weight"] for lf in tree["sub_tasks"][0]["sub_tasks"]]
    assert all(w > 0.0 for w in leaf_weights), "no leaf may be zeroed out"
    assert abs(sum(leaf_weights) - 1.0) < 1e-9
    # valid weights 0.4 and 0.4 → mean 0.4 fills the gap → all three equal.
    assert all(abs(w - 1 / 3) < 1e-9 for w in leaf_weights)


# ---------------------------------------------------------------------------
# Test 8: placeholder leaves are rejected by _is_placeholder_requirement and
#         silently dropped by _clean_categories
# ---------------------------------------------------------------------------


def test_placeholder_leaves_rejected():
    """Leaves with empty-parenthetical placeholders are dropped.

    Regression for the SDAR leaf "hyperparameters (, ) correctly set" that
    had empty placeholders because the LLM failed to extract concrete values.
    """
    # Unit-level: _is_placeholder_requirement catches known patterns
    from backend.agents.rlm.rubric_gen import _is_placeholder_requirement

    # Only a genuinely empty / comma-only parenthetical is a placeholder
    # (F-32): the regex is the last-resort net for truly empty templates.
    bad_patterns = [
        "The hyperparameters (, ) are correctly set as described in Section 4.1.",
        "The values ( ) are used.",
        "The coefficients (,) need setting.",
    ]
    for pattern in bad_patterns:
        assert _is_placeholder_requirement(pattern), (
            f"expected placeholder detection for: {pattern!r}"
        )

    good_patterns = [
        "Sets β=10 and λ=0.1 as described in Section 3.3.",
        "train.py implements g_t = σ(β·Δ_t) with stop-gradient (Section 3.3).",
        "The GRU encoder uses hidden size 256 (Section 3.1).",
        "GRPO and OPSD baselines use Qwen2.5-7B-Instruct backbone (Section 4.1).",
        # F-32: concrete leaves the old over-broad regex wrongly dropped — a
        # percent unit, a method call (inner () ), and a Greek-only argument.
        "Reports the task success rate (%).",
        "Applies the stop-gradient operator (gate.detach()) to the gate.",
        "Implements the importance ratio r_t(θ) = π_θ(a|s) / π_old(a|s).",
        # Greek-symbol lists are no longer regex-dropped (F-32 — "explicitly NOT
        # Greek letters"); the system-prompt vague-phrase prohibition, not this
        # last-resort net, is responsible for vague-but-non-empty leaves.
        "Sets (β, λ) as described.",
        "Uses (, λ) from the paper.",
    ]
    for pattern in good_patterns:
        assert not _is_placeholder_requirement(pattern), (
            f"wrongly flagged as placeholder: {pattern!r}"
        )

    # Integration: placeholder leaves are silently dropped; valid leaves survive.
    response = json.dumps({
        "categories": [
            {
                "name": "Method fidelity",
                "weight": 0.5,
                "leaves": [
                    # Placeholder — should be dropped
                    {"requirements": "The hyperparameters (, ) are correctly set.", "weight": 0.3},
                    # Valid concrete leaf — should survive
                    {"requirements": "Sets β=10 and λ=0.1 in train.py (Section 3.3).", "weight": 0.4},
                    # F-32: method-call leaf — the inner () must NOT count as a placeholder
                    {"requirements": "Applies stop-gradient via (gate.detach()).", "weight": 0.3},
                ],
            },
            {
                # F-32: a whole category built of (%) metric leaves must survive —
                # the old regex dropped every (%) leaf, deleting this category.
                "name": "Evaluation protocol",
                "weight": 0.2,
                "leaves": [
                    {"requirements": "Reports task success rate (%) on ALFWorld.", "weight": 0.5},
                    {"requirements": "Reports Score and Acc (%) on WebShop.", "weight": 0.5},
                ],
            },
            {
                "name": "Experiment execution",
                "weight": 0.3,
                "leaves": [
                    {"requirements": "train.py runs GRPO baseline to completion.", "weight": 1.0},
                ],
            },
        ]
    })
    tree = generate_rubric_tree(_LONG_PAPER, _FixedClient(response))

    assert tree is not None
    leaves = flatten_leaves(tree)
    reqs = [lf["requirements"] for lf in leaves]

    # Placeholder must not appear
    assert not any("(, )" in r for r in reqs), "placeholder leaf must be dropped"
    # Concrete leaf must survive
    assert any("β=10" in r for r in reqs), "concrete leaf with β=10 must survive"
    # F-32: the method-call leaf and BOTH (%) metric leaves must survive — the
    # latter proves the all-(%) Evaluation protocol category was not dropped.
    assert any("gate.detach()" in r for r in reqs), "method-call leaf must survive"
    assert any("success rate (%)" in r for r in reqs), "(%) metric leaf must survive"
    assert any("Score and Acc (%)" in r for r in reqs), "(%) metric leaf must survive"
    # 2 method (β=10 + detach) + 2 eval (%) + 1 experiment = 5 surviving leaves.
    assert len(leaves) == 5
