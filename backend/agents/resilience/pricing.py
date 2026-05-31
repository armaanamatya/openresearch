"""Best-effort model pricing for run cost estimation.

The token ledger is authoritative even when pricing is unknown. Pricing is
only used for budget enforcement and operator visibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


PRICING_UPDATED_AT = "2026-05-10"


@dataclass(frozen=True)
class ModelPricing:
    input_per_1m: float
    output_per_1m: float
    cache_read_input_per_1m: float = 0.0
    cache_creation_input_per_1m: float | None = None
    reasoning_per_1m: float | None = None


PRICING: dict[str, ModelPricing] = {
    "claude-sonnet-4-5": ModelPricing(
        input_per_1m=3.00,
        output_per_1m=15.00,
        cache_read_input_per_1m=0.30,
        cache_creation_input_per_1m=3.75,
    ),
    "claude-sonnet-4-6": ModelPricing(
        input_per_1m=3.00,
        output_per_1m=15.00,
        cache_read_input_per_1m=0.30,
        cache_creation_input_per_1m=3.75,
    ),
    "claude-opus-4-7": ModelPricing(
        input_per_1m=15.00,
        output_per_1m=75.00,
        cache_read_input_per_1m=1.50,
        cache_creation_input_per_1m=18.75,
    ),
    "gpt-4o": ModelPricing(input_per_1m=2.50, output_per_1m=10.00),
    "o4-mini": ModelPricing(
        input_per_1m=1.10,
        output_per_1m=4.40,
        reasoning_per_1m=4.40,
    ),
    # Claude subscription (OAuth) — actual billed cost is $0.
    # Use equivalent_cost_usd() to compute the hypothetical API cost.
    "claude-oauth": ModelPricing(
        input_per_1m=0.00,
        output_per_1m=0.00,
        cache_read_input_per_1m=0.00,
        cache_creation_input_per_1m=0.00,
    ),
}


def _resolve_pricing(model: str) -> ModelPricing | None:
    """Resolve a ModelPricing entry for *model*.

    Lookup order:
    1. Exact match in PRICING (bare names like ``claude-sonnet-4-6``).
    2. Strip a ``provider.`` prefix from any PRICING key and match the suffix
       (e.g. key ``anthropic.claude-sonnet-4-6`` → suffix ``claude-sonnet-4-6``
       matches ``model == "claude-sonnet-4-6"``).
    3. Strip a ``provider.`` prefix from *model* and match the remainder against
       PRICING keys directly (e.g. ``model == "anthropic.claude-sonnet-4-6"``
       resolves to the bare key ``claude-sonnet-4-6`` if that is in PRICING).

    This makes the bare ledger model names (``claude-oauth``,
    ``claude-sonnet-4-6``) resolve correctly even when PRICING keys are stored
    with provider prefixes, and vice-versa — without requiring callers to
    normalise the key.
    """
    # 1. Exact match.
    entry = PRICING.get(model)
    if entry is not None:
        return entry
    # 2. Match model against suffixes of PRICING keys (strip ``provider.`` prefix).
    for key, entry in PRICING.items():
        dot = key.find(".")
        if dot != -1 and key[dot + 1:] == model:
            return entry
    # 3. Strip provider prefix from model, look up the remainder in PRICING.
    dot = model.find(".")
    if dot != -1:
        bare = model[dot + 1:]
        entry = PRICING.get(bare)
        if entry is not None:
            return entry
    return None


def estimate_cost_usd(model: str, usage: dict[str, Any]) -> float | None:
    pricing = _resolve_pricing(model)
    if pricing is None:
        return None
    input_tokens = _int(usage.get("input_tokens"))
    output_tokens = _int(usage.get("output_tokens"))
    cache_read = _int(usage.get("cache_read_input_tokens"))
    cache_creation = _int(usage.get("cache_creation_input_tokens"))
    reasoning = _int(usage.get("reasoning_tokens"))
    cache_creation_price = (
        pricing.cache_creation_input_per_1m
        if pricing.cache_creation_input_per_1m is not None
        else pricing.input_per_1m
    )
    reasoning_price = (
        pricing.reasoning_per_1m
        if pricing.reasoning_per_1m is not None
        else pricing.output_per_1m
    )
    total = (
        input_tokens * pricing.input_per_1m
        + output_tokens * pricing.output_per_1m
        + cache_read * pricing.cache_read_input_per_1m
        + cache_creation * cache_creation_price
        + reasoning * reasoning_price
    ) / 1_000_000
    return round(total, 8)


# ---------------------------------------------------------------------------
# OAuth equivalent-cost mapping (C2)
# ---------------------------------------------------------------------------

# Maps zero-cost subscription model keys to the paid API model whose price
# represents the "equivalent API cost" — useful for budget estimation and
# operator visibility even though the actual charge is $0.
# Keys are bare model names (matching PRICING keys or the bare suffix of a
# provider-prefixed key).  Values are also bare names resolved via _resolve_pricing.
OAUTH_EQUIVALENT_MODEL: dict[str, str] = {
    "claude-oauth": "claude-sonnet-4-6",
}


def equivalent_cost_usd(model: str, usage: dict[str, Any]) -> float | None:
    """Return the hypothetical API cost if *model* were billed at its equivalent rate.

    For subscription models (e.g. ``claude-oauth``) this returns what the same
    token counts would cost under the equivalent paid-API model (``claude-sonnet-4-6``).
    For non-subscription models the return value is identical to
    ``estimate_cost_usd(model, usage)`` — the real cost is already the API cost.

    Returns ``None`` when no pricing data is available for the resolved model.
    The actual billed cost (always $0 for OAuth) is unchanged — this helper
    is ONLY for visibility/estimation; never use it as the cost field in a
    ledger entry.
    """
    bare = model
    dot = model.find(".")
    if dot != -1:
        bare = model[dot + 1:]
    equivalent_model = OAUTH_EQUIVALENT_MODEL.get(bare)
    if equivalent_model is not None:
        return estimate_cost_usd(equivalent_model, usage)
    # Not an OAuth model — real cost is already the API cost.
    return estimate_cost_usd(model, usage)


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


__all__ = [
    "ModelPricing",
    "PRICING",
    "PRICING_UPDATED_AT",
    "_resolve_pricing",
    "estimate_cost_usd",
    "equivalent_cost_usd",
]
