"""Per-iteration event log: event store + sanitized REPL-state snapshot.

Each time ``IterationCheckpointer.record(clean)`` is called:

1. One ``RLMRunIteration`` domain event is appended to the SQLite event store
   under a dedicated aggregate ``"rlm-run:<project_id>"``.  ``expected_version``
   is tracked and incremented by this single-writer instance — a
   ``ConcurrencyError`` is a hard bug and is never swallowed.

2. The sanitized dict is appended as a JSONL line to
   ``<snapshot_dir>/iterations.jsonl`` — the forensic trajectory + the
   variable-shape manifest per iteration.

Both outputs are corpus-free by construction: this module only ever receives the
output of :func:`~backend.agents.rlm.sse_bridge.sanitize_iteration`.

Note: this module is one-way (emit-only). Reading ``iterations.jsonl`` back for
replay is not implemented (T19 — deferred to a follow-up).

Design spec §10.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, ClassVar

from backend.eventstore.interface import EventStore
from backend.messaging.envelope import (
    CorrelationId,
    EventEnvelope,
    new_event_id,
)
from backend.messaging.event import DomainEvent, register_event


# ---------------------------------------------------------------------------
# RLMRunIteration domain event
# ---------------------------------------------------------------------------


@register_event
class RLMRunIteration(DomainEvent):
    """Domain event recording one sanitized RLM iteration.

    Payload is the corpus-free projection produced by
    :func:`~backend.agents.rlm.sse_bridge.sanitize_iteration`:

    - ``iteration``: 1-based iteration index.
    - ``response``: Root model reasoning text, bounded to ≤4 000 chars.
    - ``code_blocks``: Per-block metadata (code, stdout/stderr metadata, var
      shapes, sub-call count).  No locals values; no ``context`` key or value.
    - ``sub_calls``: Total sub-calls across all blocks in this iteration.
    - ``timing``: Wall-clock duration in seconds, or ``None``.

    This event is stored on the ``"rlm-run:<project_id>"`` aggregate stream,
    distinct from the ingestion / workspace aggregate streams.
    """

    event_type: ClassVar[str] = "rlm_run_iteration"
    schema_version: ClassVar[int] = 1

    # Core fields from the sanitized iteration dict (§9.1).
    iteration: int
    response: str
    code_blocks: list[Any]
    sub_calls: int
    timing: float | None = None


# ---------------------------------------------------------------------------
# IterationCheckpointer
# ---------------------------------------------------------------------------


class IterationCheckpointer:
    """Appends one sanitized iteration to both the event store and a JSONL file.

    This is a single-writer object: only one thread calls ``record()`` during a
    run (the ``rlms`` worker thread via :class:`~backend.agents.rlm.sse_bridge.
    OpenResearchRLMLogger`).  The ``expected_version`` counter is therefore
    race-free without additional locking.

    On instantiation the version counter is seeded from the event store so that
    a process restart with the same ``project_id`` appends at version N+1 rather
    than conflicting at version 0 (T19 / review I9).

    Note: this class is emit-only.  Reading the iteration log back for replay is
    not implemented (deferred — T19).

    Args:
        project_id:   The run's project identifier.  Used to form the aggregate
                      id ``"rlm-run:<project_id>"`` and as the ``correlation_id``
                      for every event envelope.
        event_store:  Any implementation of the
                      :class:`~backend.eventstore.interface.EventStore` protocol
                      (typically ``SqliteEventStore``).
        snapshot_dir: Directory where ``iterations.jsonl`` is written.  Created
                      if it does not exist.

    Raises:
        ConcurrencyError: If the event store's current aggregate version does not
                          match ``self._version`` — this is a hard bug (two writers
                          racing on the same aggregate), surfaced immediately.
    """

    def __init__(
        self,
        *,
        project_id: str,
        event_store: EventStore,
        snapshot_dir: Path,
    ) -> None:
        if not project_id:
            raise ValueError("project_id must be a non-empty string")
        self._project_id = project_id
        self._event_store = event_store
        self._snapshot_dir = Path(snapshot_dir)
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._snapshot_path = self._snapshot_dir / "iterations.jsonl"
        self._aggregate_id: str = f"rlm-run:{project_id}"
        # Seed from the store so a process restart appends at version N+1
        # instead of conflicting at version 0 (T19 / review I9).
        self._version: int = event_store.get_aggregate_version(self._aggregate_id)

    def record(self, clean: dict) -> None:
        """Persist one sanitized iteration to the event store and JSONL snapshot.

        Args:
            clean: The corpus-free dict returned by
                   :func:`~backend.agents.rlm.sse_bridge.sanitize_iteration`.
                   Must contain at least ``iteration``, ``response``,
                   ``code_blocks``, ``sub_calls``, and ``timing``.

        Raises:
            ConcurrencyError: If the event store detects a version mismatch.
                              This is a hard bug — never swallow it.
            KeyError: If ``clean`` is missing a required field.
        """
        event = RLMRunIteration(
            iteration=clean["iteration"],
            response=clean["response"],
            code_blocks=clean["code_blocks"],
            sub_calls=clean["sub_calls"],
            timing=clean.get("timing"),
        )
        envelope = EventEnvelope(
            event_id=new_event_id(),
            correlation_id=CorrelationId(self._project_id),
            source="agents.rlm.checkpoint",
        )
        # ConcurrencyError is intentionally not caught — it is a hard bug.
        self._event_store.append(
            aggregate_id=self._aggregate_id,
            aggregate_type="rlm_run",
            events=[event],
            expected_version=self._version,
            envelopes=[envelope],
        )
        self._version += 1

        # Snapshot: append the sanitized dict as a JSONL line.
        line = json.dumps(clean, default=str) + "\n"
        with self._snapshot_path.open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())  # T30 / review M7 — avoid torn lines on crash.


__all__ = [
    "IterationCheckpointer",
    "RLMRunIteration",
]
