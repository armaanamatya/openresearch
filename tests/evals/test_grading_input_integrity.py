"""W1-M1 grading-input integrity — rubric pinning (OPENRESEARCH_GRADER_INTEGRITY).

The root writes Python in a sandboxed REPL; it cannot mutate the host grader, but
it CAN rewrite the grading-input artifacts it legitimately produces
(``rubric_tree.json``, ``provenance.json``, ``metrics.json``). W1 pins the rubric:
at rubric-gen the tree is fingerprinted; at grade time the fingerprint is
recomputed and a mismatch is a fail-closed ``evidence_tampered`` signal.

This file starts with the pure fingerprint function (no disk, no flag, no network).
Default-OFF behavior + the gate wiring get their own OFF/ON pair once the pure core
is proven (the flag-pair convention in tests/CLAUDE.md).
"""
from __future__ import annotations

import json
from pathlib import Path

from backend.evals.paperbench.grading_input_integrity import (
    check_grading_input_integrity,
    rubric_fingerprint,
    verify_rubric_integrity,
    write_rubric_pin,
)


def _write_rubric(run_dir: Path, rubric: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "rubric_tree.json").write_text(json.dumps(rubric), encoding="utf-8")


RUBRIC = {"id": "root", "leaves": [{"id": "L1", "criteria": "test_error <= 8.75"}]}


def test_rubric_fingerprint_is_stable_across_key_order():
    """Semantically identical rubrics fingerprint identically regardless of the
    order keys were inserted (canonical serialization)."""
    a = {"id": "root", "weight": 1.0, "leaves": [{"id": "L1", "criteria": "acc >= 0.9"}]}
    b = {"leaves": [{"criteria": "acc >= 0.9", "id": "L1"}], "weight": 1.0, "id": "root"}
    assert rubric_fingerprint(a) == rubric_fingerprint(b)


def test_rubric_fingerprint_changes_when_criteria_change():
    """Editing a leaf's criteria (the weakening attack) changes the fingerprint."""
    original = {"id": "root", "leaves": [{"id": "L1", "criteria": "test_error <= 8.75"}]}
    weakened = {"id": "root", "leaves": [{"id": "L1", "criteria": "test_error <= 99.0"}]}
    assert rubric_fingerprint(original) != rubric_fingerprint(weakened)


def test_rubric_fingerprint_is_hex_sha256():
    """Fingerprint is a 64-char lowercase hex sha256 digest (stable, comparable)."""
    fp = rubric_fingerprint({"id": "root"})
    assert isinstance(fp, str)
    assert len(fp) == 64
    assert all(c in "0123456789abcdef" for c in fp)


def test_rubric_fingerprint_matches_canonical_campaign_hash():
    """W1-M1 must agree byte-for-byte with the campaign layer's canonical
    ``attempt_assessment.rubric_sha256`` — including for the non-ASCII rubric
    text (β, σ, λ) that is ubiquitous in ML papers. A second, subtly-different
    hash would make single-run pinning and cross-attempt pinning disagree."""
    from backend.agents.rlm.attempt_assessment import rubric_sha256

    for tree in (
        {"id": "root"},
        {"id": "root", "leaves": [{"id": "L1", "criteria": "gate g_t=σ(β·Δ_t), β=10, λ=0.1"}]},
    ):
        assert rubric_fingerprint(tree) == rubric_sha256(tree)


# --------------------------------------------------------------------------- #
# disk-backed pin + fail-closed verify
# --------------------------------------------------------------------------- #
def test_pinned_then_unchanged_rubric_verifies_ok(tmp_path: Path):
    """Round trip: pin at gen time, rubric unchanged at grade time → match."""
    _write_rubric(tmp_path, RUBRIC)
    write_rubric_pin(tmp_path, RUBRIC)
    result = verify_rubric_integrity(tmp_path)
    assert result["ok"] is True
    assert result["reason"] == "match"


def test_pinned_then_weakened_rubric_is_tampered(tmp_path: Path):
    """The attack: rubric rewritten weaker after pinning → fail-closed mismatch."""
    _write_rubric(tmp_path, RUBRIC)
    write_rubric_pin(tmp_path, RUBRIC)
    _write_rubric(tmp_path, {"id": "root", "leaves": [{"id": "L1", "criteria": "test_error <= 99.0"}]})
    result = verify_rubric_integrity(tmp_path)
    assert result["ok"] is False
    assert result["reason"] == "rubric_mismatch"


def test_no_pin_never_vetoes(tmp_path: Path):
    """Absence of a pin is not evidence of tampering — verify stays ok (we only
    veto on positive evidence, never on a missing baseline)."""
    _write_rubric(tmp_path, RUBRIC)
    result = verify_rubric_integrity(tmp_path)
    assert result["ok"] is True
    assert result["reason"] == "no_pin"


def test_pinned_then_deleted_rubric_is_tampered(tmp_path: Path):
    """Pinned but the rubric file vanished at grade time → fail-closed."""
    _write_rubric(tmp_path, RUBRIC)
    write_rubric_pin(tmp_path, RUBRIC)
    (tmp_path / "rubric_tree.json").unlink()
    result = verify_rubric_integrity(tmp_path)
    assert result["ok"] is False
    assert result["reason"] == "rubric_missing"


# --------------------------------------------------------------------------- #
# flag-gated entrypoint (the OFF/ON pair — flag-pair convention)
# --------------------------------------------------------------------------- #
def test_entrypoint_off_returns_none(tmp_path: Path, monkeypatch):
    """Default-OFF: the entrypoint is a no-op (returns None) even on a tampered
    run — byte-identical to the prior baseline."""
    monkeypatch.delenv("OPENRESEARCH_GRADER_INTEGRITY", raising=False)
    _write_rubric(tmp_path, RUBRIC)
    write_rubric_pin(tmp_path, RUBRIC)
    _write_rubric(tmp_path, {"id": "root", "leaves": [{"id": "L1", "criteria": "test_error <= 99.0"}]})
    assert check_grading_input_integrity(tmp_path) is None


def test_entrypoint_on_flags_tamper(tmp_path: Path, monkeypatch):
    """Default-ON: the entrypoint returns the fail-closed verdict on a tampered run."""
    monkeypatch.setenv("OPENRESEARCH_GRADER_INTEGRITY", "1")
    _write_rubric(tmp_path, RUBRIC)
    write_rubric_pin(tmp_path, RUBRIC)
    _write_rubric(tmp_path, {"id": "root", "leaves": [{"id": "L1", "criteria": "test_error <= 99.0"}]})
    verdict = check_grading_input_integrity(tmp_path)
    assert verdict is not None
    assert verdict["ok"] is False
    assert verdict["reason"] == "rubric_mismatch"
