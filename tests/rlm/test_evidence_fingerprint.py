"""A3: evidence-fingerprint aggregation — median-within-state, no global MAX.

OPENRESEARCH_EVIDENCE_FINGERPRINT default OFF → legacy global-max best-of-run floor
(byte-for-byte today). On → the floor is the MEDIAN of the rubric_score events
sharing the LATEST evidence_key; keyless events degrade to 'latest score'. Either
way the upward-biased global max is gone.
"""

from __future__ import annotations

import json

import pytest

from backend.agents.rlm.report import (
    _apply_best_of_run_floor,
    _evidence_aware_best_score,
)


def _write_events(tmp_path, rows):
    lines = [json.dumps({"type": "rubric_score", "payload": r}) for r in rows]
    (tmp_path / "dashboard_events.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_floor_off_uses_legacy_global_max(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENRESEARCH_EVIDENCE_FINGERPRINT", raising=False)
    _write_events(tmp_path, [
        {"overall_score": 0.7, "evidence_key": "k1"},
        {"overall_score": 0.9, "evidence_key": "k1"},  # lucky draw
        {"overall_score": 0.6, "evidence_key": "k2"},  # latest state, lower
    ])
    r = _apply_best_of_run_floor({"overall_score": 0.6}, tmp_path)
    assert r["overall_score"] == pytest.approx(0.9)  # legacy: global max banks the lucky draw


def test_floor_on_uses_median_at_latest_key(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENRESEARCH_EVIDENCE_FINGERPRINT", "1")
    _write_events(tmp_path, [
        {"overall_score": 0.7, "evidence_key": "k1"},
        {"overall_score": 0.9, "evidence_key": "k1"},  # earlier state, lucky
        {"overall_score": 0.60, "evidence_key": "k2"},  # latest state
        {"overall_score": 0.64, "evidence_key": "k2"},
    ])
    # latest key k2 → median(0.60, 0.64) = 0.62, NOT the global max 0.9
    assert _evidence_aware_best_score(tmp_path) == pytest.approx(0.62)
    r = _apply_best_of_run_floor({"overall_score": 0.60}, tmp_path)
    assert r["overall_score"] == pytest.approx(0.62)  # within-state anti-regression, not 0.9


def test_floor_on_keyless_degrades_to_latest_not_max(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENRESEARCH_EVIDENCE_FINGERPRINT", "1")
    _write_events(tmp_path, [
        {"overall_score": 0.9},  # no evidence_key (older run / flag was off)
        {"overall_score": 0.6},  # latest
    ])
    # keyless → singleton groups → latest score 0.6, never the global max 0.9
    assert _evidence_aware_best_score(tmp_path) == pytest.approx(0.6)


def test_floor_on_salvages_when_current_is_none(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENRESEARCH_EVIDENCE_FINGERPRINT", "1")
    _write_events(tmp_path, [
        {"overall_score": 0.55, "evidence_key": "k1"},
        {"overall_score": 0.59, "evidence_key": "k1"},
    ])
    # current None (run killed) → salvage the latest-state median (0.57)
    r = _apply_best_of_run_floor({"overall_score": None}, tmp_path)
    assert r["overall_score"] == pytest.approx(0.57)


def test_no_events_returns_input_unchanged(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENRESEARCH_EVIDENCE_FINGERPRINT", "1")
    assert _evidence_aware_best_score(tmp_path) is None
    r = _apply_best_of_run_floor({"overall_score": 0.5}, tmp_path)
    assert r["overall_score"] == pytest.approx(0.5)
