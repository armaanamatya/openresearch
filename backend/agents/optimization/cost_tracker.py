"""LM cost accumulator shared across optimization lanes.

Wraps any LM client (gepa.lm.LM, custom callable, OpenAI SDK, etc.) and
accumulates token + dollar spend across an optimization run. Persists
to ``<opt_run_dir>/cost.json`` after each call so a crash leaves a
recoverable record.

Used by GEPA Lane B (via gepa.lm.LM passthrough) and any future lane.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class CostSnapshot:
    total_cost_usd: float = 0.0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    call_count: int = 0
    per_model: dict[str, dict[str, float]] = field(default_factory=dict)


class LMCostTracker:
    """Thread-safe accumulator that persists after every recorded call.

    Usage:
        tracker = LMCostTracker(run_dir=Path("optimization_runs/foo"))
        tracker.record(model="openai/gpt-4.1", tokens_in=120, tokens_out=400, cost_usd=0.018)
        tracker.snapshot()  # → CostSnapshot
    """

    def __init__(self, run_dir: Path) -> None:
        self._run_dir = Path(run_dir)
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._state = CostSnapshot()
        self._lock = threading.Lock()
        self._cost_path = self._run_dir / "cost.json"
        if self._cost_path.exists():
            self._state = self._load_existing()

    def _load_existing(self) -> CostSnapshot:
        data = json.loads(self._cost_path.read_text())
        return CostSnapshot(
            total_cost_usd=data.get("total_cost_usd", 0.0),
            total_tokens_in=data.get("total_tokens_in", 0),
            total_tokens_out=data.get("total_tokens_out", 0),
            call_count=data.get("call_count", 0),
            per_model=data.get("per_model", {}),
        )

    def record(
        self,
        *,
        model: str,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
    ) -> None:
        with self._lock:
            self._state.total_cost_usd += cost_usd
            self._state.total_tokens_in += tokens_in
            self._state.total_tokens_out += tokens_out
            self._state.call_count += 1
            slot = self._state.per_model.setdefault(
                model,
                {"cost_usd": 0.0, "tokens_in": 0, "tokens_out": 0, "calls": 0},
            )
            slot["cost_usd"] += cost_usd
            slot["tokens_in"] += tokens_in
            slot["tokens_out"] += tokens_out
            slot["calls"] += 1
            self._persist_unlocked()

    def _persist_unlocked(self) -> None:
        tmp = self._cost_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(self._state), indent=2, sort_keys=True))
        tmp.replace(self._cost_path)

    def snapshot(self) -> CostSnapshot:
        with self._lock:
            return CostSnapshot(
                total_cost_usd=self._state.total_cost_usd,
                total_tokens_in=self._state.total_tokens_in,
                total_tokens_out=self._state.total_tokens_out,
                call_count=self._state.call_count,
                per_model=dict(self._state.per_model),
            )

    def absorb_gepa_lm(self, lm: Any) -> None:
        """Pull totals from a finished gepa.lm.LM instance into this tracker.

        gepa.lm.LM exposes ``total_cost``, ``total_tokens_in``,
        ``total_tokens_out``. We record one synthetic call carrying the
        full delta — finer-grained per-call accounting is not exposed
        by the LM class.
        """
        model = getattr(lm, "model", "unknown")
        cost = float(getattr(lm, "total_cost", 0.0) or 0.0)
        tin = int(getattr(lm, "total_tokens_in", 0) or 0)
        tout = int(getattr(lm, "total_tokens_out", 0) or 0)
        self.record(model=model, tokens_in=tin, tokens_out=tout, cost_usd=cost)


def wrap_callable(
    fn: Callable[..., str],
    *,
    tracker: LMCostTracker,
    model: str,
    estimate_cost: Callable[[int, int], float] | None = None,
) -> Callable[..., str]:
    """Wrap a ``(messages, **kwargs) -> str`` callable so calls are recorded.

    Token + cost estimation is best-effort; pass an ``estimate_cost`` if
    the caller has provider-specific knowledge. Default is zero-cost
    accounting (still counts the call).
    """

    def wrapped(messages: Any, **kwargs: Any) -> str:
        out = fn(messages, **kwargs)
        # crude token estimate: 4 chars per token, both directions.
        # ``messages`` may be a raw string prompt OR a list[{role,content}].
        if isinstance(messages, str):
            tin = len(messages) // 4
        else:
            try:
                tin = sum(len(str(m.get("content", ""))) for m in messages) // 4
            except (AttributeError, TypeError):
                tin = len(str(messages)) // 4
        tout = len(out) // 4
        cost = estimate_cost(tin, tout) if estimate_cost else 0.0
        tracker.record(model=model, tokens_in=tin, tokens_out=tout, cost_usd=cost)
        return out

    return wrapped
