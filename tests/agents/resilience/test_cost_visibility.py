"""Cost visibility — surface the ledger's $0 blind spot instead of hiding it.

The cost summary sums ``estimated_usd`` and treats an unpriceable row (Foundry /
unknown model → ``estimated_usd is None``) as $0, so real spend is under-reported
with no signal. ``audit_cost_ledger`` instead COUNTS the unpriced rows + their
token volume and flags confidence, so an operator sees "priced $X, plus N rows /
M tokens of UNKNOWN cost" rather than a misleading total.
"""
from __future__ import annotations

from backend.agents.resilience.cost_visibility import audit_cost_ledger


def _row(model, in_tok, out_tok, usd):
    return {"model": model, "input_tokens": in_tok, "output_tokens": out_tok, "estimated_usd": usd}


def test_all_priced_is_complete():
    rows = [_row("gpt-4o", 1000, 500, 0.02), _row("gpt-4o", 2000, 1000, 0.04)]
    a = audit_cost_ledger(rows)
    assert a["priced_usd"] == 0.06
    assert a["unpriced_rows"] == 0
    assert a["confidence"] == "complete"


def test_unpriced_foundry_rows_are_surfaced():
    rows = [
        _row("gpt-4o", 1000, 500, 0.02),
        _row("opus-foundry", 3000, 1500, None),   # unpriceable → invisible today
        _row("grok-4.3", 4000, 2000, None),
    ]
    a = audit_cost_ledger(rows)
    assert a["priced_usd"] == 0.02
    assert a["unpriced_rows"] == 2
    assert a["unpriced_tokens"] == 3000 + 1500 + 4000 + 2000
    assert set(a["unpriced_models"]) == {"opus-foundry", "grok-4.3"}
    assert a["confidence"] == "partial"


def test_zero_token_unpriced_row_is_not_flagged():
    """A $0/0-token row (e.g. a no-op) is not a hidden-cost signal."""
    rows = [_row("opus-foundry", 0, 0, None)]
    a = audit_cost_ledger(rows)
    assert a["unpriced_rows"] == 0
    assert a["confidence"] == "complete"


def test_empty_ledger():
    a = audit_cost_ledger([])
    assert a["priced_usd"] == 0.0
    assert a["unpriced_rows"] == 0
    assert a["confidence"] == "complete"


def test_malformed_rows_fail_soft():
    rows = ["not a dict", {}, _row("gpt-4o", 100, 50, 0.01), {"estimated_usd": "bad"}]
    a = audit_cost_ledger(rows)
    assert a["priced_usd"] == 0.01  # only the well-formed priced row counts
