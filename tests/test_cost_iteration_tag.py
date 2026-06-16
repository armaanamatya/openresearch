"""F1 — per-iteration cost tagging (2026-06-16).

Each cost-ledger entry carries the root-loop ``iteration`` it occurred in, for
per-iteration cost attribution. Additive metadata: default 0, never affects cost
math, and old ledgers without the field read back as 0.
"""

from __future__ import annotations

from backend.agents.resilience.cost import CostLedgerEntry


def test_from_usage_records_iteration():
    e = CostLedgerEntry.from_usage(
        agent_id="run_experiment", attempt_index=0, provider="anthropic",
        model="m", usage={"input_tokens": 10}, iteration=3,
    )
    assert e.iteration == 3
    assert e.to_json()["iteration"] == 3


def test_default_iteration_zero_and_backward_compat():
    e = CostLedgerEntry.from_usage(
        agent_id="x", attempt_index=0, provider="anthropic", model="m", usage={},
    )
    assert e.iteration == 0
    # an old ledger row (no iteration field) reads back as 0 (no crash)
    old = CostLedgerEntry.from_json({
        "timestamp": "2026-06-16T00:00:00+00:00", "agent_id": "x",
        "attempt_index": 0, "provider": "anthropic", "model": "m",
    })
    assert old.iteration == 0


def test_roundtrip_preserves_iteration():
    e = CostLedgerEntry.from_usage(
        agent_id="x", attempt_index=0, provider="anthropic", model="m",
        usage={"input_tokens": 1}, iteration=7,
    )
    assert CostLedgerEntry.from_json(e.to_json()).iteration == 7
