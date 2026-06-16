"""A6: count-based per-cell digest + measured-value metrics-path rank (2026-06-16).

OPENRESEARCH_GRADER_DIGEST default OFF → byte-slice + truthiness rank (today's
behavior). On → no cell silently vanishes on a wide grid; a placeholder
per_model:{m:{}} cannot outrank genuinely-measured older data.
"""

from __future__ import annotations

import json
import os

import pytest

from backend.evals.paperbench import leaf_scorer
from backend.evals.paperbench.leaf_scorer import _gather_evidence, _latest_metrics_path


def _write(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj), encoding="utf-8")


def _measured():
    return {"per_model": {"m": {"env": {"base": {"acc": 0.9}}}}}


def _placeholder():
    return {"per_model": {"m": {}}}  # truthy but measures nothing


def test_rank_prefers_measured_over_newer_placeholder_when_flag_on(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENRESEARCH_GRADER_DIGEST", "1")
    out = tmp_path / "code" / "outputs"
    older = out / "r1" / "metrics.json"
    newer = out / "r2" / "metrics.json"
    _write(older, _measured())
    _write(newer, _placeholder())
    os.utime(older, (1000, 1000))
    os.utime(newer, (2000, 2000))
    # measured (older) beats newer-but-placeholder
    assert _latest_metrics_path(tmp_path) == older


def test_rank_truthiness_when_flag_off(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENRESEARCH_GRADER_DIGEST", raising=False)
    out = tmp_path / "code" / "outputs"
    older = out / "r1" / "metrics.json"
    newer = out / "r2" / "metrics.json"
    _write(older, _measured())
    _write(newer, _placeholder())
    os.utime(older, (1000, 1000))
    os.utime(newer, (2000, 2000))
    # flag off: both per_model truthy → newer wins (today's behavior, byte-for-byte)
    assert _latest_metrics_path(tmp_path) == newer


def test_gather_uses_digest_on_overflow_when_flag_on(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENRESEARCH_GRADER_DIGEST", "1")
    monkeypatch.setattr(leaf_scorer, "_MAX_METRICS_BYTES", 50)  # force overflow
    per_model = {f"m{i}": {"e": {"b": {"acc": 0.5 + i / 100}}} for i in range(20)}
    _write(tmp_path / "code" / "metrics.json", {"per_model": per_model})
    ev = _gather_evidence(tmp_path)
    assert "per-cell DIGEST" in ev
    # the trailing cell a raw byte slice would have dropped must survive
    assert '"model_key": "m19"' in ev
    # every one of the 20 cells is present — nothing silently vanished
    assert ev.count('"model_key"') == 20


def test_gather_byteslice_on_overflow_when_flag_off(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENRESEARCH_GRADER_DIGEST", raising=False)
    monkeypatch.setattr(leaf_scorer, "_MAX_METRICS_BYTES", 50)
    per_model = {f"m{i}": {"e": {"b": {"acc": 0.5}}} for i in range(20)}
    _write(tmp_path / "code" / "metrics.json", {"per_model": per_model})
    ev = _gather_evidence(tmp_path)
    assert "truncated" in ev
    assert "per-cell DIGEST" not in ev
