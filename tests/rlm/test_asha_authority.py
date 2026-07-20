"""Authority seam tests: default-off identity and fail-closed adoption.

The current campaign lacks a provenance-gated deterministic defining metric and
paper-step checkpoint lineage, so authority may audit its evaluation but must not
adopt the grade-derived shadow advisory.
"""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from backend.agents.rlm.campaign_composition import _maybe_apply_asha_authority


def _continue_decision():
    return {
        "kind": "CONTINUE",
        "rule": "continue",
        "stop_reason": None,
        "next_plan": {
            "lineage": "fresh",
            "seed_attempt_n": None,
            "seed_pointer": None,
            "scope_rung": 2,
            "width": 1,
        },
        "champion_attempt_n": None,
    }


def test_authority_off_is_byte_identical_even_when_tree_is_on(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_SCHEDULER_TREE", "1")
    monkeypatch.delenv("OPENRESEARCH_SCHEDULER_AUTHORITATIVE", raising=False)
    result = _continue_decision()
    before = deepcopy(result)
    _maybe_apply_asha_authority(result)
    assert result == before


def test_authority_without_tree_is_byte_identical(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_SCHEDULER_TREE", raising=False)
    monkeypatch.setenv("OPENRESEARCH_SCHEDULER_AUTHORITATIVE", "true")
    result = _continue_decision()
    before = deepcopy(result)
    _maybe_apply_asha_authority(result)
    assert result == before


def test_noncanonical_tree_token_keeps_authority_off(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_SCHEDULER_TREE", "on")
    monkeypatch.setenv("OPENRESEARCH_SCHEDULER_AUTHORITATIVE", "1")
    result = _continue_decision()
    before = deepcopy(result)
    _maybe_apply_asha_authority(result)
    assert result == before


@pytest.mark.parametrize("kind", ("REPRODUCED", "CONTRADICTED", "INFEASIBLE", "EXHAUSTED"))
def test_authority_preserves_every_terminal_decision(monkeypatch, kind):
    monkeypatch.setenv("OPENRESEARCH_SCHEDULER_TREE", "yes")
    monkeypatch.setenv("OPENRESEARCH_SCHEDULER_AUTHORITATIVE", "1")
    result = _continue_decision()
    result.update(kind=kind, rule=f"terminal_{kind.lower()}", next_plan=None, champion_attempt_n=7)
    before = {key: deepcopy(result[key]) for key in _continue_decision()}
    _maybe_apply_asha_authority(result)
    assert {key: result[key] for key in before} == before
    assert result["asha_authority_audit"] == {
        "enabled": True,
        "applied": False,
        "action": "continue",
        "deterministic_evidence_basis": "base_terminal_precedence",
    }


def test_authority_refuses_grade_derived_shadow_adoption(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_SCHEDULER_TREE", "1")
    monkeypatch.setenv("OPENRESEARCH_SCHEDULER_AUTHORITATIVE", "true")
    result = _continue_decision()
    before = deepcopy(result)
    # A tempting high-vs-low report grade is deliberately irrelevant: it is
    # not a deterministic defining metric and must never choose a live branch.
    reports = [SimpleNamespace(score=0.99), SimpleNamespace(score=0.01)]
    _maybe_apply_asha_authority(result)
    assert reports[0].score > reports[1].score  # fixture sanity only
    assert {key: result[key] for key in before} == before
    assert result["next_plan"]["scope_rung"] == 2
    assert result["asha_authority_audit"] == {
        "enabled": True,
        "applied": False,
        "action": "continue",
        "deterministic_evidence_basis": (
            "unavailable: provenance-gated deterministic defining metric and "
            "optimizer-step lineage are required"
        ),
    }
