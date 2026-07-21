"""Cost visibility — make the ledger's $0 blind spot auditable.

``cost_ledger.jsonl`` records a per-primitive ``estimated_usd`` that is ``None``
when the model has no pricing entry (Foundry-routed / grok / any unknown model).
The cost summary sums those as $0, so unpriceable spend is silently invisible —
"a $0 there is not proof of $0" (learn.md). This module counts the unpriced rows
and their token volume so an operator SEES the gap instead of a misleading total.

Pure / fail-soft: malformed rows are skipped, never raise.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["audit_cost_ledger", "audit_cost_ledger_file"]


def _num(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _int(v: Any) -> int:
    n = _num(v)
    return int(n) if n is not None and n >= 0 else 0


def audit_cost_ledger(rows: list[Any]) -> dict[str, Any]:
    """Audit ledger rows, surfacing priced vs unpriced (unknown-cost) spend.

    Returns ``{priced_usd, unpriced_rows, unpriced_tokens, unpriced_models,
    confidence}`` where ``confidence`` is ``"complete"`` when every token-bearing
    row is priced, else ``"partial"`` (real cost exceeds ``priced_usd``).
    """
    priced_usd = 0.0
    unpriced_rows = 0
    unpriced_tokens = 0
    unpriced_models: set[str] = set()

    for row in rows:
        if not isinstance(row, dict):
            continue
        usd = _num(row.get("estimated_usd"))
        tokens = _int(row.get("input_tokens")) + _int(row.get("output_tokens"))
        if usd is not None:
            priced_usd += usd
            continue
        # Unpriced: only a signal when the row actually consumed tokens.
        if tokens > 0:
            unpriced_rows += 1
            unpriced_tokens += tokens
            model = row.get("model")
            if isinstance(model, str) and model:
                unpriced_models.add(model)

    return {
        "priced_usd": round(priced_usd, 8),
        "unpriced_rows": unpriced_rows,
        "unpriced_tokens": unpriced_tokens,
        "unpriced_models": sorted(unpriced_models),
        "confidence": "complete" if unpriced_rows == 0 else "partial",
    }


def audit_cost_ledger_file(run_dir: Path | str) -> dict[str, Any]:
    """Audit ``<run_dir>/cost_ledger.jsonl`` (fail-soft; missing → empty audit)."""
    path = Path(run_dir) / "cost_ledger.jsonl"
    rows: list[Any] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:  # noqa: BLE001 — skip a torn line.
                continue
    except Exception:  # noqa: BLE001 — missing / unreadable → empty audit.
        pass
    return audit_cost_ledger(rows)
