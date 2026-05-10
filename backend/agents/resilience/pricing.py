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
}


def estimate_cost_usd(model: str, usage: dict[str, Any]) -> float | None:
    pricing = PRICING.get(model)
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


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


__all__ = ["ModelPricing", "PRICING", "PRICING_UPDATED_AT", "estimate_cost_usd"]
