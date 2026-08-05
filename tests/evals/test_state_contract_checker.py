"""W2 state-contracts — GroundEval-style trace-coherence predicates as a new
``deterministic:state_contract`` check_kind (OPENRESEARCH_STATE_CONTRACTS).

Grounded in the EXISTING ``eval_provenance.json`` artifact (produced by
``eval_provenance.record_eval``): eval-coverage (``n_eval >= min``) and held-out
(``held_out is True``). These close a gap the eval-provenance *guard* does not —
it checks the reported mean is correct, not that enough held-out examples were
evaluated. Default-OFF ⇒ the new kind is unrecognized ⇒ routed to the LLM
exactly as an un-annotated leaf (byte-identical baseline).
"""
from __future__ import annotations

import json
from pathlib import Path

from backend.evals.paperbench.deterministic_leaf_checker import check_leaf


def _write_provenance(run_dir: Path, cell: str, payload: dict) -> None:
    d = run_dir / "code" / "outputs" / cell
    d.mkdir(parents=True, exist_ok=True)
    (d / "eval_provenance.json").write_text(json.dumps(payload), encoding="utf-8")


def _leaf(assertion: dict) -> dict:
    return {"id": "L1", "check_kind": "deterministic:state_contract", "assertion": assertion}


def test_off_routes_state_contract_to_llm(tmp_path: Path, monkeypatch):
    """Flag OFF: the new kind is unrecognized → check_leaf returns None (LLM),
    byte-identical to an un-annotated leaf even when a sidecar exists."""
    monkeypatch.delenv("OPENRESEARCH_STATE_CONTRACTS", raising=False)
    _write_provenance(tmp_path, "c0", {"n_eval": 3, "held_out": True})
    assert check_leaf(_leaf({"min_eval_n": 100}), tmp_path) is None


def test_on_eval_coverage_satisfied(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_STATE_CONTRACTS", "1")
    _write_provenance(tmp_path, "c0", {"n_eval": 500, "held_out": True})
    result = check_leaf(_leaf({"min_eval_n": 100}), tmp_path)
    assert result is not None
    assert result["score"] == 1.0
    assert result["_graded"] is True


def test_on_eval_coverage_violated(tmp_path: Path, monkeypatch):
    """The attack: claim a benchmark result but evaluate only a handful."""
    monkeypatch.setenv("OPENRESEARCH_STATE_CONTRACTS", "1")
    _write_provenance(tmp_path, "c0", {"n_eval": 3, "held_out": True})
    result = check_leaf(_leaf({"min_eval_n": 100}), tmp_path)
    assert result is not None
    assert result["score"] == 0.0


def test_on_require_held_out_violated(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_STATE_CONTRACTS", "1")
    _write_provenance(tmp_path, "c0", {"n_eval": 500, "held_out": False})
    result = check_leaf(_leaf({"require_held_out": True}), tmp_path)
    assert result is not None
    assert result["score"] == 0.0


def test_on_require_held_out_satisfied(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_STATE_CONTRACTS", "1")
    _write_provenance(tmp_path, "c0", {"n_eval": 500, "held_out": True})
    result = check_leaf(_leaf({"require_held_out": True}), tmp_path)
    assert result is not None
    assert result["score"] == 1.0


def test_on_missing_sidecar_is_failed_grade(tmp_path: Path, monkeypatch):
    """Well-formed assertion + no evidence → graded 0.0 (missing evidence), the
    module's existing contract — NOT a route-to-LLM fall-through."""
    monkeypatch.setenv("OPENRESEARCH_STATE_CONTRACTS", "1")
    result = check_leaf(_leaf({"min_eval_n": 100}), tmp_path)
    assert result is not None
    assert result["score"] == 0.0
    assert "eval_provenance" in result["justification"]


def test_on_malformed_assertion_routes_to_llm(tmp_path: Path, monkeypatch):
    """No recognized predicate key → cannot interpret → route to LLM (None)."""
    monkeypatch.setenv("OPENRESEARCH_STATE_CONTRACTS", "1")
    _write_provenance(tmp_path, "c0", {"n_eval": 500, "held_out": True})
    assert check_leaf(_leaf({"unknown_predicate": 1}), tmp_path) is None


def test_on_degenerate_require_held_out_false_routes_to_llm(tmp_path: Path, monkeypatch):
    """A no-op ``require_held_out: false`` as the ONLY predicate must not auto-pass
    at 1.0 — it carries no active predicate, so route to the LLM (None)."""
    monkeypatch.setenv("OPENRESEARCH_STATE_CONTRACTS", "1")
    _write_provenance(tmp_path, "c0", {"n_eval": 500, "held_out": True})
    assert check_leaf(_leaf({"require_held_out": False}), tmp_path) is None
