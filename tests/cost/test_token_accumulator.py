"""Tests for TokenAccumulator — PR-α.5.

Covers:
1. Normal usage dict accumulates correctly.
2. Missing fields default to 0.
3. Only message_start (no output) gives input but 0 output.
4. Non-integer values are coerced safely.
5. Multiple absorb_usage calls are additive.
6. as_dict() returns the correct shape.
"""

from __future__ import annotations


from backend.services.pricing.token_accumulator import TokenAccumulator


def test_accumulate_full_usage():
    acc = TokenAccumulator()
    acc.absorb_usage({
        "input_tokens": 1000,
        "output_tokens": 200,
        "cache_creation_input_tokens": 50,
        "cache_read_input_tokens": 10,
        "reasoning_tokens": 5,
    })
    assert acc.input_tokens == 1000
    assert acc.output_tokens == 200
    assert acc.cache_creation_input_tokens == 50
    assert acc.cache_read_input_tokens == 10
    assert acc.reasoning_tokens == 5
    assert acc.has_any()


def test_missing_fields_default_to_zero():
    acc = TokenAccumulator()
    acc.absorb_usage({})  # empty dict — all fields missing
    assert acc.input_tokens == 0
    assert acc.output_tokens == 0
    assert acc.cache_creation_input_tokens == 0
    assert acc.cache_read_input_tokens == 0
    assert acc.reasoning_tokens == 0
    assert not acc.has_any()


def test_only_input_tokens():
    """Simulates a message_start event with no output yet."""
    acc = TokenAccumulator()
    acc.absorb_usage({"input_tokens": 800})
    assert acc.input_tokens == 800
    assert acc.output_tokens == 0
    assert acc.has_any()


def test_only_output_tokens():
    """Simulates a message_delta event with only output tokens."""
    acc = TokenAccumulator()
    acc.absorb_usage({"output_tokens": 150})
    assert acc.input_tokens == 0
    assert acc.output_tokens == 150
    assert acc.has_any()


def test_multiple_absorb_calls_additive():
    """Two absorb_usage calls add up (covers multi-event scenario)."""
    acc = TokenAccumulator()
    acc.absorb_usage({"input_tokens": 500, "cache_read_input_tokens": 20})
    acc.absorb_usage({"output_tokens": 100})
    assert acc.input_tokens == 500
    assert acc.output_tokens == 100
    assert acc.cache_read_input_tokens == 20


def test_non_integer_values_coerced():
    """Float / string values from the SDK are coerced to int."""
    acc = TokenAccumulator()
    acc.absorb_usage({"input_tokens": "1000", "output_tokens": 50.9})
    assert acc.input_tokens == 1000
    assert acc.output_tokens == 50


def test_none_value_defaults_to_zero():
    acc = TokenAccumulator()
    acc.absorb_usage({"input_tokens": None, "output_tokens": None})
    assert acc.input_tokens == 0
    assert acc.output_tokens == 0


def test_non_dict_usage_is_ignored():
    """Non-dict usage (e.g. None from a missing field) does not crash."""
    acc = TokenAccumulator()
    acc.absorb_usage(None)   # type: ignore[arg-type]
    acc.absorb_usage("bad")  # type: ignore[arg-type]
    assert not acc.has_any()


def test_as_dict_shape():
    acc = TokenAccumulator()
    acc.absorb_usage({"input_tokens": 400, "output_tokens": 80})
    d = acc.as_dict()
    assert set(d.keys()) == {
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
        "reasoning_tokens",
    }
    assert d["input_tokens"] == 400
    assert d["output_tokens"] == 80
    assert d["cache_creation_input_tokens"] == 0
