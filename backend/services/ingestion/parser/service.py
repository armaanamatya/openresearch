"""ParserAppService — drives the parser, appends events.

Single command: `StartParsing(project_id)`.
1. Load ProjectAggregate; require state == FETCHED.
2. Load ParsedPaperAggregate; require not yet PARSING/PARSED.
3. Append ParsingStarted.
4. Invoke parser (the IO).
5. For each section / reference, append the corresponding *Extracted event.
6. Append ParsingCompleted (or ParsingFailed on parser error).
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import List

from pydantic import ConfigDict

from backend.eventstore.interface import EventStore
from backend.messaging.command import Command
from backend.messaging.envelope import (
    AggregateId,
    CorrelationId,
    EventEnvelope,
    make_envelope,
    new_correlation_id,
)
from backend.messaging.event import DomainEvent, resolve_event_class
from backend.services.ingestion.intake.aggregate import (
    ProjectAggregate,
    ProjectState,
)
from backend.services.ingestion.parser.aggregate import (
    InvalidParseTransition,
    ParsedPaperAggregate,
    ParsedPaperState,
)
from backend.services.ingestion.parser.events import (
    FigureExtracted,
    ParsingCompleted,
    ParsingFailed,
    ParsingStarted,
    ReferenceExtracted,
    SectionExtracted,
)
from backend.services.ingestion.parser.extractor import NullExtractor, PaperExtractor
from backend.services.ingestion.parser.interface import Parser, ParseError


class StartParsing(Command):
    model_config = ConfigDict(frozen=True)
    project_id: str


class ParserError(Exception):
    """Errors that should NOT be modeled as ParsingFailed events
    (e.g., the project doesn't exist or isn't in FETCHED state)."""


def write_parsed_full_text(project_dir: Path, text: str | None) -> None:
    """Write parsed_full_text.txt atomically, or delete it on parse failure.

    On a failed parse a stale blob from a prior paper would silently feed the
    RLM the wrong corpus (review I6 / T18). Idempotent.
    """
    path = project_dir / "parsed_full_text.txt"
    if not text:
        # Parse failed (or text empty): invalidate any stale blob.
        if path.exists():
            path.unlink()
        return
    tmp = path.with_suffix(".txt.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _aggregate_id(project_id: str, suffix: str) -> AggregateId:
    """Compose the parsed-paper aggregate id deterministically from project_id."""
    return AggregateId(f"{project_id}:{suffix}")


class ParserAppService:
    """The IO orchestrator for parsing."""

    def __init__(
        self,
        store: EventStore,
        parser: Parser,
        runs_root: Path = Path("runs"),
        extractor: PaperExtractor = NullExtractor(),
    ) -> None:
        self._store = store
        self._parser = parser
        self._runs_root = runs_root
        self._extractor = extractor

    # --- Public ------------------------------------------------------------

    def start_parsing(
        self,
        cmd: StartParsing,
        *,
        correlation_id: CorrelationId | None = None,
    ) -> bool:
        """Returns True if parsing completed successfully, False on
        recorded parser failure. Raises ParserError for state-protocol
        violations (project missing, not yet fetched)."""
        cid = correlation_id or new_correlation_id()
        project_id = cmd.project_id

        project = self._load_project_aggregate(project_id)
        if project.state is not ProjectState.FETCHED:
            raise ParserError(
                f"Cannot start parsing: project {project_id!r} is in state "
                f"{project.state.value!r}; must be {ProjectState.FETCHED.value!r}."
            )
        # Find the fetched paper path from the project's events.
        paper_path = self._find_paper_path(project_id)

        parsed_agg_id = _aggregate_id(project_id, "parsed")
        parsed = self._load_parsed_aggregate(project_id)
        if parsed.state is ParsedPaperState.PARSED:
            return True  # idempotent — already parsed
        if parsed.state is ParsedPaperState.PARSING:
            # Mid-parse re-issue. Fail fast — concurrent parsers risk
            # interleaving events.
            raise ParserError(
                f"Project {project_id!r} is already PARSING; cannot start again."
            )

        # Step 1: ParsingStarted.
        try:
            start_events = list(
                parsed.handle_start(self._parser.name, self._parser.version)
            )
        except InvalidParseTransition as exc:
            raise ParserError(str(exc)) from exc
        self._append(parsed, parsed_agg_id, start_events, cid)

        # Step 2: invoke parser (the IO).
        t0 = time.monotonic()
        try:
            result = self._parser.parse(project_id=project_id, paper_path=paper_path)
        except ParseError as exc:
            failure = ParsingFailed(
                project_id=project_id,
                parser_name=self._parser.name,
                cause_kind=exc.cause_kind,
                cause_message=str(exc),
                retryable=exc.retryable,
            )
            self._append(parsed, parsed_agg_id, [failure], cid)
            # Invalidate any stale blob — a re-run into a dir that holds a
            # prior paper's blob would silently feed the RLM the wrong corpus
            # (review I6 / T18).
            blob_dir = self._runs_root / project_id
            blob_dir.mkdir(parents=True, exist_ok=True)
            write_parsed_full_text(blob_dir, None)
            return False

        # Step 2b: augmentation pass (fail-soft by contract).
        try:
            result = self._extractor.extract(
                project_id=project_id, paper_path=paper_path, base=result
            )
        except Exception:
            import logging as _logging
            _logging.getLogger(__name__).exception(
                "Extractor %r raised unexpectedly for project %s; continuing with base result",
                self._extractor.name,
                project_id,
            )

        # Step 3: emit one event per section / reference / figure.
        events: List[DomainEvent] = []
        for section in result.sections:
            events.append(
                SectionExtracted(project_id=project_id, section=section)
            )
        for reference in result.references:
            events.append(
                ReferenceExtracted(project_id=project_id, reference=reference)
            )
        for figure in result.figures:
            events.append(
                FigureExtracted(project_id=project_id, figure=figure)
            )
        if events:
            self._append(parsed, parsed_agg_id, events, cid)

        # Step 4: store full text as a blob (atomic write), append ParsingCompleted.
        blob_dir = self._runs_root / project_id
        blob_dir.mkdir(parents=True, exist_ok=True)
        write_parsed_full_text(blob_dir, result.full_text)
        blob_path = blob_dir / "parsed_full_text.txt"
        full_text_sha = hashlib.sha256(result.full_text.encode()).hexdigest()

        completed = ParsingCompleted(
            project_id=project_id,
            section_count=len(result.sections),
            reference_count=len(result.references),
            figure_count=len(result.figures),
            parser_name=self._parser.name,
            parser_version=self._parser.version,
            full_text_blob_path=str(blob_path.resolve()),
            full_text_sha256=full_text_sha,
        )
        # parse_duration_ms intentionally NOT in the event payload — the
        # spec's byte-identical replay (§8.5) is downgraded to ID
        # stability for this slice (per Codex 2026-05-09).
        _ = time.monotonic() - t0  # for future structured-log emission

        self._append(parsed, parsed_agg_id, [completed], cid)
        return True

    def get_state(self, project_id: str) -> ParsedPaperState:
        return self._load_parsed_aggregate(project_id).state

    # --- Internal ----------------------------------------------------------

    def _load_project_aggregate(self, project_id: str) -> ProjectAggregate:
        agg = ProjectAggregate.empty(project_id)
        for stored in self._store.load(AggregateId(project_id)):
            cls = resolve_event_class(stored.event_type, stored.schema_version)
            agg.apply(stored.into(cls))
        return agg

    def _load_parsed_aggregate(self, project_id: str) -> ParsedPaperAggregate:
        agg = ParsedPaperAggregate.empty(project_id)
        for stored in self._store.load(_aggregate_id(project_id, "parsed")):
            cls = resolve_event_class(stored.event_type, stored.schema_version)
            agg.apply(stored.into(cls))
        return agg

    def _find_paper_path(self, project_id: str) -> Path:
        for stored in self._store.load(AggregateId(project_id)):
            if stored.event_type == "paper_fetched":
                return Path(stored.payload["raw_paper_path"])
        raise ParserError(
            f"No PaperFetched event found for project {project_id!r}; cannot parse."
        )

    def _append(
        self,
        agg: ParsedPaperAggregate,
        agg_id: AggregateId,
        events: list,
        correlation_id: CorrelationId,
    ) -> None:
        envelopes: list[EventEnvelope] = [
            make_envelope(
                source="ingestion.parser.service",
                correlation_id=correlation_id,
            )
            for _ in events
        ]
        self._store.append(
            aggregate_id=agg_id,
            aggregate_type="parsed_paper",
            events=events,
            expected_version=agg.version,
            envelopes=envelopes,
        )
        # _append also applies events to the aggregate so its `version`
        # tracks the store's. Callers must NOT call apply_all separately
        # or the version drifts.
        agg.apply_all(events)


__all__ = ["ParserAppService", "ParserError", "StartParsing"]
