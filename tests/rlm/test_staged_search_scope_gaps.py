"""Task 11 — staged-search budget-dropped cells must appear in scope.gaps.

The test drives the new ``budget_dropped`` kwarg added to
``aggregate_cell_metrics``; the implementation folds each dropped-cell dict
into ``scope.gaps`` with ``reason="staged_search_budget_dropped"``.
"""
from backend.agents.rlm import cell_matrix as cm


def test_dropped_full_cells_become_scope_gaps():
    result = cm.aggregate_cell_metrics(
        {"cells": []}, [],                      # matrix_result, cells (positional)
        capacity_gaps=[], dataset_gaps=[],
        budget_dropped=[{"model_key": "qwen3-1.7b", "env": "alfworld", "baseline": "grpo"}],
    )
    gaps = result["scope"]["gaps"]
    assert any(g.get("reason") == "staged_search_budget_dropped" for g in gaps)


def test_dropped_full_cells_gap_fields():
    """Each dropped cell gets a gap dict with the three axis fields + reason."""
    dropped = [
        {"model_key": "qwen3-1.7b", "env": "alfworld", "baseline": "grpo"},
        {"model_key": "qwen3-3b",   "env": "webshop",  "baseline": "sft"},
    ]
    result = cm.aggregate_cell_metrics(
        {}, [], capacity_gaps=[], dataset_gaps=[], budget_dropped=dropped,
    )
    gaps = result["scope"]["gaps"]
    bd_gaps = [g for g in gaps if g.get("reason") == "staged_search_budget_dropped"]
    assert len(bd_gaps) == 2
    assert bd_gaps[0]["model_key"] == "qwen3-1.7b"
    assert bd_gaps[0]["env"] == "alfworld"
    assert bd_gaps[0]["baseline"] == "grpo"
    assert bd_gaps[1]["model_key"] == "qwen3-3b"


def test_budget_dropped_none_does_not_change_gaps():
    """Default None leaves gaps identical to no argument passed."""
    result_default = cm.aggregate_cell_metrics(
        {}, [], capacity_gaps=[], dataset_gaps=[],
    )
    result_none = cm.aggregate_cell_metrics(
        {}, [], capacity_gaps=[], dataset_gaps=[], budget_dropped=None,
    )
    assert result_default["scope"]["gaps"] == result_none["scope"]["gaps"] == []


def test_budget_dropped_merges_with_capacity_gaps():
    """budget_dropped gaps appear alongside capacity_gaps, not instead of them."""
    cap_gap = {"item": "qwen3-7b", "reason": "vram_exceeded"}
    dropped = [{"model_key": "qwen3-1.7b", "env": "alfworld", "baseline": "grpo"}]
    result = cm.aggregate_cell_metrics(
        {}, [], capacity_gaps=[cap_gap], dataset_gaps=[], budget_dropped=dropped,
    )
    gaps = result["scope"]["gaps"]
    assert any(g.get("reason") == "vram_exceeded" for g in gaps)
    assert any(g.get("reason") == "staged_search_budget_dropped" for g in gaps)
    assert len(gaps) == 2
