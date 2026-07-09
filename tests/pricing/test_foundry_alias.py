"""Foundry role aliases (opus-foundry / sonnet-foundry) must price at their
priced Claude siblings, not $0. Regression: the ledger records the bare role
ids (backend.agents.rlm.role_models), which did not resolve against PRICING."""

from __future__ import annotations

import pytest

from backend.agents.resilience.pricing import (
    PRICING,
    _resolve_pricing,
    estimate_cost_usd,
)


def test_opus_foundry_resolves_to_opus_pricing():
    assert _resolve_pricing("opus-foundry") is PRICING["claude-opus-4-8"]


def test_sonnet_foundry_resolves_to_sonnet_pricing():
    assert _resolve_pricing("sonnet-foundry") is PRICING["claude-sonnet-5"]


def test_opus_foundry_estimate_is_nonzero():
    usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}
    # 15/1M input + 75/1M output = 90.0
    assert estimate_cost_usd("opus-foundry", usage) == pytest.approx(90.0)


def test_unknown_model_still_returns_none():
    assert estimate_cost_usd("mystery-model", {"input_tokens": 100}) is None
