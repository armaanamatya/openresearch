"""Unit tests for the shared Exclusion contract (backend/agents/rlm/exclusion.py).

The verified-only rubric-exclusion contract both halves of the full-scope work
build against. Anti-gaming invariant under test: only ``verified=True`` records
reduce the denominator (enter a skip list); an agent-declared, un-corroborated
exclusion is recorded for transparency but never excluded.
"""
from __future__ import annotations

import pytest

from backend.agents.rlm import exclusion as X


def test_exclusion_validates_and_normalises():
    e = X.Exclusion(item="ALFWorld", axis="Environment", kind="Operator_Scope",
                    reason="x", verified=True, evidence="e")
    assert e.axis == "environment" and e.kind == "operator_scope"
    assert e.to_gap() == {
        "item": "ALFWorld", "axis": "environment", "kind": "operator_scope",
        "reason": "x", "verified": True, "evidence": "e",
    }
    assert e.is_hard_limit is False  # operator_scope is a decision, not a measured limit
    assert X.Exclusion(item="x", axis="model", kind="capacity_vram",
                       reason="", verified=True).is_hard_limit is True


@pytest.mark.parametrize("bad", [
    dict(item="", axis="environment", kind="operator_scope", reason="x", verified=True),
    dict(item="x", axis="bogus", kind="operator_scope", reason="x", verified=True),
    dict(item="x", axis="environment", kind="nope", reason="x", verified=True),
])
def test_exclusion_rejects_malformed(bad):
    with pytest.raises(ValueError):
        X.Exclusion(**bad)


def test_evidence_is_truncated():
    e = X.Exclusion(item="x", axis="model", kind="oom_shrink_exhausted",
                    reason="r", verified=True, evidence="z" * 5000)
    assert len(e.evidence) <= 500


def test_operator_scope_exclusions_diffs_case_insensitively():
    ex = X.operator_scope_exclusions(
        ["ALFWorld", "WebShop", "Search-QA"], ["search-qa"],
        X.AXIS_ENVIRONMENT, evidence="scope.json")
    assert [e.item for e in ex] == ["ALFWorld", "WebShop"]
    assert all(e.verified and e.kind == X.KIND_OPERATOR_SCOPE for e in ex)
    # invalid axis → empty (splice-safe)
    assert X.operator_scope_exclusions(["a"], [], "bogus") == []


def test_build_scope_block_derives_legacy_lists_from_verified_only():
    ver = X.Exclusion(item="ALFWorld", axis="environment", kind="operator_scope",
                      reason="r", verified=True)
    unver = X.Exclusion(item="WebShop", axis="environment", kind="env_setup_failed",
                        reason="agent said", verified=False)
    blk = X.build_scope_block([ver, unver], models_run=["Qwen3-1.7B"])
    assert blk["environments_skipped"] == ["ALFWorld"]   # unverified NOT in skip list
    assert {e["item"] for e in blk["exclusions"]} == {"ALFWorld", "WebShop"}  # both recorded
    assert blk["models_run"] == ["Qwen3-1.7B"]
    assert [g["item"] for g in blk["gaps"]] == ["ALFWorld"]  # only verified → gaps


def test_build_scope_block_merges_existing_and_dedupes():
    ex = X.Exclusion(item="WebShop", axis="environment", kind="operator_scope",
                     reason="r", verified=True)
    existing = {"models_run": ["A"], "gaps": [{"item": "foo", "kind": "capacity"}],
                "environments_skipped": ["ALFWorld"]}
    blk = X.build_scope_block([ex], models_run=["B"], existing=existing)
    assert set(blk["models_run"]) == {"A", "B"}
    assert set(blk["environments_skipped"]) == {"ALFWorld", "WebShop"}
    assert {g["item"] for g in blk["gaps"]} == {"foo", "WebShop"}  # existing + minted


def test_gap_roundtrip_legacy_kinds():
    e = X.Exclusion.from_gap({"item": "WebShop", "kind": "dataset_unavailable", "reason": "404"})
    assert e and e.kind == X.KIND_DATASET_DEAD and e.verified is True and e.axis == X.AXIS_ENVIRONMENT
    e2 = X.Exclusion.from_gap({"item": "qwen7b", "kind": "capacity"})
    assert e2 and e2.kind == X.KIND_CAPACITY_VRAM and e2.axis == X.AXIS_MODEL
    assert X.Exclusion.from_gap({"reason": "no item"}) is None
    assert X.exclusions_from_gaps([{"item": "x", "kind": "capacity"}, "junk", 5]) and \
        len(X.exclusions_from_gaps([{"item": "x", "kind": "capacity"}, "junk", 5])) == 1


def test_verified_items_by_axis():
    exs = [
        X.Exclusion(item="ALFWorld", axis="environment", kind="operator_scope", reason="", verified=True),
        X.Exclusion(item="WebShop", axis="environment", kind="env_setup_failed", reason="", verified=False),
        X.Exclusion(item="Qwen7B", axis="model", kind="operator_scope", reason="", verified=True),
    ]
    assert X.verified_items_by_axis(exs) == {"environment": {"ALFWorld"}, "model": {"Qwen7B"}}
    assert [e.item for e in X.verified_only(exs)] == ["ALFWorld", "Qwen7B"]
