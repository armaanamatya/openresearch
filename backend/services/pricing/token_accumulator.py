"""TokenAccumulator — lightweight per-call usage aggregator for claude-agent-sdk.

The claude-agent-sdk's ``query()`` stream emits:
  - ``ResultMessage.usage`` — dict with input_tokens, output_tokens,
    cache_creation_input_tokens, cache_read_input_tokens, reasoning_tokens.

``TokenAccumulator`` consumes that dict defensively and exposes the totals
in the shape expected by ``CostLedgerEntry.from_usage``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TokenAccumulator:
    """Accumulate token counts from claude-agent-sdk stream events.

    Usage:
        acc = TokenAccumulator()
        acc.absorb_usage(result_message_usage)
        entry = CostLedgerEntry.from_usage(..., usage=acc.as_dict())
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    reasoning_tokens: int = 0

    def absorb_usage(self, usage: object) -> None:
        """Consume a usage dict (from ResultMessage.usage).

        Handles both the flat dict shape returned by the SDK and nested
        shapes defensively. Missing or non-integer fields default to 0.
        """
        if not isinstance(usage, dict):
            return
        self.input_tokens += _int(usage.get("input_tokens"))
        self.output_tokens += _int(usage.get("output_tokens"))
        self.cache_creation_input_tokens += _int(
            usage.get("cache_creation_input_tokens")
        )
        self.cache_read_input_tokens += _int(usage.get("cache_read_input_tokens"))
        # reasoning_tokens is present on extended-thinking responses only
        self.reasoning_tokens += _int(usage.get("reasoning_tokens"))

    def as_dict(self) -> dict[str, int]:
        """Return a dict compatible with ``CostLedgerEntry.from_usage``."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }

    def has_any(self) -> bool:
        """Return True if any token count is non-zero."""
        return bool(
            self.input_tokens
            or self.output_tokens
            or self.cache_creation_input_tokens
            or self.cache_read_input_tokens
            or self.reasoning_tokens
        )


def _int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


__all__ = ["TokenAccumulator"]
