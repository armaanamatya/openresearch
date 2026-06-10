"""Append-only cost ledger for provider attempts."""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.agents.resilience.pricing import estimate_cost_usd
from backend.agents.runtime.base import ProviderName


logger = logging.getLogger(__name__)

_DEFAULT_BATCH_SIZE = 25


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
    # Per-row provenance (audit 2026-06-10): how the primitive call ENDED, as
    # observed by binding.wrap_primitive — "ok" (returned non-failure),
    # "failed" (returned a failure-shaped dict), "raised" (exception path).
    # "" = unknown (legacy rows, non-primitive appenders); treated as
    # success-compatible by the evidence gate so old artifacts never
    # over-downgrade.
    outcome: str = ""

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        data["primitive"] = self.agent_id
        data["cost_usd"] = self.estimated_usd or 0.0
        data["tokens_in"] = self.input_tokens
        data["tokens_out"] = self.output_tokens
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "CostLedgerEntry":
        payload = dict(data)
        for alias in ("primitive", "cost_usd", "tokens_in", "tokens_out"):
            payload.pop(alias, None)
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
        outcome: str = "",
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
            outcome=outcome,
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
    batch_size: int = _DEFAULT_BATCH_SIZE

    def __post_init__(self) -> None:
        # Buffer for batched writes; entries are accumulated here until a flush.
        # The single _lock guards BOTH buffer mutations AND the file write inside
        # _flush_locked — so this design subsumes the parallel-RDR concurrency
        # concern that an earlier sibling branch addressed with a per-append lock:
        # concurrent appends from parallel cluster tasks queue on this lock; the
        # actual disk write is serialized through the same lock at flush time, so
        # line-tearing is structurally impossible.
        self._buffer: list[dict] = []
        self._lock: threading.Lock = threading.Lock()
        # Entries present at construction time were seeded from disk (load_jsonl
        # of the root-writable cost_ledger.jsonl on a warm retry) and are NOT
        # trustworthy as a record of in-process primitive calls. Appends always
        # go to the end of the list under _lock, so entries[_seeded_len:] is
        # exactly the rows recorded by THIS process (binding.wrap_primitive).
        self._seeded_len: int = len(self.entries)

    def append(self, entry: CostLedgerEntry) -> None:
        """Append an entry to the in-memory list and the write buffer.

        Flushes to disk when the buffer reaches ``self.batch_size``.
        Safe under concurrent appends from parallel RDR cluster tasks: the lock
        serializes both buffer mutations and the eventual disk write.
        """
        with self._lock:
            self.entries.append(entry)
            if self.path is not None:
                self._buffer.append(entry.to_json())
                if len(self._buffer) >= self.batch_size:
                    self._flush_locked()

    def flush(self) -> None:
        """Flush any buffered entries to disk.

        Idempotent — safe to call when the buffer is empty.
        If no path is set, this is a no-op.
        """
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        """Write the buffer to disk (must be called with _lock held)."""
        if not self._buffer or self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            for row in self._buffer:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._buffer.clear()

    def total_usd(self) -> float:
        return round(sum(entry.estimated_usd or 0.0 for entry in self.entries), 8)

    def session_call_count(self, agent_id: str) -> int:
        """Count entries appended IN THIS PROCESS for ``agent_id``.

        Excludes rows seeded from disk at ``load_jsonl`` time: those live in the
        root-writable ``cost_ledger.jsonl`` and can be forged through the REPL's
        live ``open()``, then re-ingested on a warm retry of the same project
        dir. Only post-construction appends (made by ``binding.wrap_primitive``
        inside the orchestrator) are trusted. Budget math (``total_usd``)
        intentionally stays cumulative across retries — this counter is the
        trust signal, not the cost accumulator.
        """
        return sum(
            1 for entry in self.entries[self._seeded_len :] if entry.agent_id == agent_id
        )

    def session_success_compatible_count(self, agent_id: str) -> int:
        """In-process entries for ``agent_id`` whose outcome can back a SUCCESS
        verdict: ``"ok"`` or ``""`` (unknown/legacy — conservative, never
        over-downgrades). Rows explicitly stamped ``"failed"``/``"raised"`` by
        ``binding.wrap_primitive`` do NOT count: a root that makes one real but
        FAILED ``run_experiment`` call and then forges a success row into
        ``experiment_runs.jsonl`` used to pass the >=1 call cross-check (the
        documented KNOWN RESIDUAL); with per-row provenance it no longer does.
        """
        return sum(
            1
            for entry in self.entries[self._seeded_len :]
            if entry.agent_id == agent_id
            and (entry.outcome or "") in ("", "ok")
        )

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


def record_subagent_usage_to_path(
    project_dir: "Path",
    agent_id: str,
    model: str,
    provider: "ProviderName",
    usage: "dict[str, int]",
    *,
    attempt_index: int = 0,
) -> None:
    """Append a sub-agent usage row to cost_ledger.jsonl and emit an SSE event.

    Path-based (no RunCostLedger instance needed) — callable from invoke.py
    which has no RunContext. Fail-soft: IO errors are logged, never raised.
    """
    import json as _json
    from datetime import datetime as _dt, timezone as _tz

    entry = CostLedgerEntry.from_usage(
        agent_id=agent_id,
        attempt_index=attempt_index,
        provider=provider,
        model=model,
        usage=usage,
    )
    ledger_path = Path(project_dir) / "cost_ledger.jsonl"
    try:
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with ledger_path.open("a", encoding="utf-8") as fh:
            fh.write(_json.dumps(entry.to_json(), sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    except Exception:
        logger.warning("record_subagent_usage_to_path: ledger write failed for %s", agent_id)

    # SSE event — fail-soft
    events_file = Path(project_dir) / "dashboard_events.jsonl"
    payload: dict = {
        "agent_id": agent_id,
        "model": model,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
        "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
        "estimated_usd": entry.estimated_usd or 0.0,
    }
    line = _json.dumps({
        "ts": _dt.now(_tz.utc).isoformat(),
        "event": "subagent_usage",
        "data": payload,
    }, default=str)
    try:
        with events_file.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        logger.warning("record_subagent_usage_to_path: SSE emit failed for %s", agent_id)


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


__all__ = ["CostLedgerEntry", "ProviderTotals", "RunCostLedger"]
