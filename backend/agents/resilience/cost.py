"""Append-only cost ledger for provider attempts."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.agents.resilience.pricing import estimate_cost_usd
from backend.agents.runtime.base import ProviderName


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CostLedgerEntry:
    timestamp: datetime
    agent_id: str
    attempt_index: int
    provider: ProviderName
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    reasoning_tokens: int = 0
    estimated_usd: float | None = None

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "CostLedgerEntry":
        payload = dict(data)
        ts = payload.get("timestamp")
        if isinstance(ts, str):
            payload["timestamp"] = datetime.fromisoformat(ts)
        elif ts is None:
            payload["timestamp"] = datetime.now(timezone.utc)
        return cls(**payload)

    @classmethod
    def from_usage(
        cls,
        *,
        agent_id: str,
        attempt_index: int,
        provider: ProviderName,
        model: str,
        usage: dict[str, Any],
        timestamp: datetime | None = None,
    ) -> "CostLedgerEntry":
        normalized = {
            "input_tokens": _int(usage.get("input_tokens")),
            "output_tokens": _int(usage.get("output_tokens")),
            "cache_read_input_tokens": _int(usage.get("cache_read_input_tokens")),
            "cache_creation_input_tokens": _int(
                usage.get("cache_creation_input_tokens")
            ),
            "reasoning_tokens": _int(usage.get("reasoning_tokens")),
        }
        return cls(
            timestamp=timestamp or datetime.now(timezone.utc),
            agent_id=agent_id,
            attempt_index=attempt_index,
            provider=provider,
            model=model,
            estimated_usd=estimate_cost_usd(model, normalized),
            **normalized,
        )


@dataclass(frozen=True)
class ProviderTotals:
    provider: ProviderName
    attempts: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    estimated_usd: float = 0.0


@dataclass
class RunCostLedger:
    project_id: str
    entries: list[CostLedgerEntry] = field(default_factory=list)
    path: Path | None = None

    def append(self, entry: CostLedgerEntry) -> None:
        self.entries.append(entry)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry.to_json(), sort_keys=True) + "\n")

    def total_usd(self) -> float:
        return round(sum(entry.estimated_usd or 0.0 for entry in self.entries), 8)

    def total_by_provider(self) -> dict[ProviderName, ProviderTotals]:
        raw: dict[ProviderName, dict[str, Any]] = {}
        for entry in self.entries:
            totals = raw.setdefault(
                entry.provider,
                {
                    "provider": entry.provider,
                    "attempts": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "reasoning_tokens": 0,
                    "estimated_usd": 0.0,
                },
            )
            totals["attempts"] += 1
            totals["input_tokens"] += entry.input_tokens
            totals["output_tokens"] += entry.output_tokens
            totals["reasoning_tokens"] += entry.reasoning_tokens
            totals["estimated_usd"] += entry.estimated_usd or 0.0
        return {
            provider: ProviderTotals(
                provider=provider,
                attempts=values["attempts"],
                input_tokens=values["input_tokens"],
                output_tokens=values["output_tokens"],
                reasoning_tokens=values["reasoning_tokens"],
                estimated_usd=round(values["estimated_usd"], 8),
            )
            for provider, values in raw.items()
        }

    def write_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for entry in self.entries:
                handle.write(json.dumps(entry.to_json(), sort_keys=True) + "\n")

    @classmethod
    def load_jsonl(
        cls,
        path: Path,
        *,
        project_id: str | None = None,
        attach_path: bool = True,
    ) -> "RunCostLedger":
        entries: list[CostLedgerEntry] = []
        if path.exists():
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    entries.append(CostLedgerEntry.from_json(json.loads(line)))
                except Exception:
                    logger.warning(
                        "Skipping malformed cost ledger line %s in %s",
                        line_number,
                        path,
                    )
        inferred_project_id = project_id or path.parent.name
        return cls(
            project_id=inferred_project_id,
            entries=entries,
            path=path if attach_path else None,
        )


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


__all__ = ["CostLedgerEntry", "ProviderTotals", "RunCostLedger"]
