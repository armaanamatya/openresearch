"""Tests for the Phase 0 LMCostTracker shared across optimization lanes."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.agents.optimization.cost_tracker import (
    CostSnapshot,
    LMCostTracker,
    wrap_callable,
)


def test_record_accumulates_per_model(tmp_path: Path) -> None:
    t = LMCostTracker(run_dir=tmp_path / "r")
    t.record(model="openai/gpt-4.1", tokens_in=100, tokens_out=200, cost_usd=0.015)
    t.record(model="openai/gpt-4.1", tokens_in=50, tokens_out=100, cost_usd=0.008)
    t.record(model="openai/gpt-5", tokens_in=10, tokens_out=20, cost_usd=0.001)

    snap = t.snapshot()
    assert snap.total_cost_usd == pytest.approx(0.024)
    assert snap.total_tokens_in == 160
    assert snap.total_tokens_out == 320
    assert snap.call_count == 3
    assert snap.per_model["openai/gpt-4.1"]["calls"] == 2
    assert snap.per_model["openai/gpt-5"]["cost_usd"] == pytest.approx(0.001)


def test_persists_to_disk(tmp_path: Path) -> None:
    t = LMCostTracker(run_dir=tmp_path / "r")
    t.record(model="x", tokens_in=1, tokens_out=2, cost_usd=0.5)
    saved = json.loads((tmp_path / "r" / "cost.json").read_text())
    assert saved["total_cost_usd"] == 0.5
    assert saved["per_model"]["x"]["cost_usd"] == 0.5


def test_round_trip_across_instances(tmp_path: Path) -> None:
    t1 = LMCostTracker(run_dir=tmp_path / "r")
    t1.record(model="x", tokens_in=1, tokens_out=2, cost_usd=0.10)
    # Simulate a crash: new instance reads existing cost.json.
    t2 = LMCostTracker(run_dir=tmp_path / "r")
    snap = t2.snapshot()
    assert snap.total_cost_usd == pytest.approx(0.10)
    assert snap.call_count == 1
    # Subsequent records add to the recovered total.
    t2.record(model="x", tokens_in=3, tokens_out=4, cost_usd=0.05)
    assert t2.snapshot().total_cost_usd == pytest.approx(0.15)
    assert t2.snapshot().call_count == 2


def test_absorb_gepa_lm(tmp_path: Path) -> None:
    """LMCostTracker.absorb_gepa_lm pulls totals from a finished LM instance."""
    t = LMCostTracker(run_dir=tmp_path / "r")
    fake_lm = SimpleNamespace(
        model="openai/gpt-4.1",
        total_cost=1.234,
        total_tokens_in=12345,
        total_tokens_out=6789,
    )
    t.absorb_gepa_lm(fake_lm)
    snap = t.snapshot()
    assert snap.total_cost_usd == pytest.approx(1.234)
    assert snap.total_tokens_in == 12345
    assert snap.total_tokens_out == 6789
    assert snap.per_model["openai/gpt-4.1"]["cost_usd"] == pytest.approx(1.234)


def test_absorb_gepa_lm_handles_missing_attrs(tmp_path: Path) -> None:
    """Absorb degrades gracefully if the LM hasn't exposed totals yet."""
    t = LMCostTracker(run_dir=tmp_path / "r")
    fake_lm = SimpleNamespace(model="openai/gpt-4.1")  # no totals
    t.absorb_gepa_lm(fake_lm)
    snap = t.snapshot()
    assert snap.total_cost_usd == 0.0
    assert snap.call_count == 1  # the absorb call is still counted


def test_wrap_callable_records_each_call(tmp_path: Path) -> None:
    t = LMCostTracker(run_dir=tmp_path / "r")

    def inner(messages, **kwargs):
        return "hello world"

    wrapped = wrap_callable(inner, tracker=t, model="claude-sonnet-4-6")
    out = wrapped([{"role": "user", "content": "hi"}])
    assert out == "hello world"
    snap = t.snapshot()
    assert snap.call_count == 1
    assert snap.per_model["claude-sonnet-4-6"]["calls"] == 1


def test_wrap_callable_with_cost_estimator(tmp_path: Path) -> None:
    t = LMCostTracker(run_dir=tmp_path / "r")

    def inner(messages, **kwargs):
        return "x" * 400

    def estimate(tin: int, tout: int) -> float:
        return tin * 0.001 + tout * 0.002

    wrapped = wrap_callable(
        inner, tracker=t, model="m", estimate_cost=estimate,
    )
    wrapped([{"role": "user", "content": "a" * 400}])
    snap = t.snapshot()
    # 400 chars / 4 = 100 tokens in; 400 / 4 = 100 tokens out
    # cost = 100 * 0.001 + 100 * 0.002 = 0.3
    assert snap.total_cost_usd == pytest.approx(0.3)
    assert snap.total_tokens_in == 100
    assert snap.total_tokens_out == 100
