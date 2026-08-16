"""Foundry role-token aliases (opus-foundry/sonnet-foundry) must price via their
bare Claude PRICING entries — else a Foundry-routed ledger row prices at $0.

Rate-agnostic on purpose: assert the alias prices *like its Claude sibling*, not a
specific rate (the $15/$75-vs-$5/$25 CCU-vs-API rate question is unresolved and
untouched here)."""
from backend.agents.resilience.pricing import estimate_cost_usd

_USAGE = {"input_tokens": 1_000_000, "output_tokens": 500_000}


def test_opus_foundry_prices_like_claude_opus_4_8():
    got = estimate_cost_usd("opus-foundry", _USAGE)
    assert got is not None  # previously None → $0
    assert got == estimate_cost_usd("claude-opus-4-8", _USAGE)


def test_sonnet_foundry_prices_like_claude_sonnet_5():
    got = estimate_cost_usd("sonnet-foundry", _USAGE)
    assert got is not None
    assert got == estimate_cost_usd("claude-sonnet-5", _USAGE)


def test_bare_claude_ids_still_price():
    assert estimate_cost_usd("claude-opus-4-8", _USAGE) is not None
    assert estimate_cost_usd("claude-sonnet-4-6", _USAGE) is not None


def test_unknown_model_still_none():
    # A genuinely unpriced model (e.g. grok) stays None — the alias map is narrow.
    assert estimate_cost_usd("grok-4.3", _USAGE) is None
