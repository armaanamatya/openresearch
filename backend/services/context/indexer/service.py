"""IndexerAppService — drives the chunker, appends index events.

Single command: `StartIndexing(project_id)`.
1. Load IndexAggregate; require state PENDING or FAILED.
2. Read parsed sections + references from the parsed-paper event stream.
3. Build SourceRefs (one per parsed item) and Chunks (one per section
   via SectionChunker) using deterministic content-addressed IDs.
4. Append IndexingStarted, then SourceRegistered + ChunkCreated events,
   then IndexingCompleted.
"""

from __future__ import annotations

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
from backend.services.context.indexer.aggregate import (
    IndexAggregate,
    IndexState,
    InvalidIndexTransition,
)
from backend.services.context.indexer.chunkers.section import SectionChunker
from backend.services.context.indexer.events import (
    ChunkCreated,
    IndexingCompleted,
    IndexingFailed,
    IndexingStarted,
    SourceRegistered,
)
from backend.services.context.indexer.model import (
    Chunk,
    SourceKind,
    SourceRef,
    source_id_for,
)
from backend.services.context.indexer.projections import SourcesProjection
from backend.services.ingestion.parser.aggregate import (
    ParsedPaperAggregate,
    ParsedPaperState,
)
from backend.services.ingestion.parser.model import Reference, Section


class StartIndexing(Command):
    model_config = ConfigDict(frozen=True)
    project_id: str


class IndexerError(Exception):
    """Protocol violations (e.g., parsed paper isn't PARSED yet)."""


def _index_aggregate_id(project_id: str) -> AggregateId:
    return AggregateId(f"{project_id}:index")


def _parsed_aggregate_id(project_id: str) -> AggregateId:
    return AggregateId(f"{project_id}:parsed")


class IndexerAppService:
    def __init__(self, store: EventStore) -> None:
        self._store = store
        self._chunker = SectionChunker()

    # --- Public ------------------------------------------------------------

    def start_indexing(
        self,
        cmd: StartIndexing,
        *,
        correlation_id: CorrelationId | None = None,
    ) -> bool:
        cid = correlation_id or new_correlation_id()
        project_id = cmd.project_id

        parsed = self._load_parsed_aggregate(project_id)
        if parsed.state is not ParsedPaperState.PARSED:
            raise IndexerError(
                f"Cannot index: parsed paper for {project_id!r} is in state "
                f"{parsed.state.value!r}; must be {ParsedPaperState.PARSED.value!r}."
            )

        index = self._load_index_aggregate(project_id)
        if index.state is IndexState.INDEXED:
            return True  # idempotent
        if index.state is IndexState.INDEXING:
            raise IndexerError(
                f"Project {project_id!r} is already INDEXING; cannot start again."
            )

        # Step 1: IndexingStarted.
        try:
            start_events = list(
                index.handle_start(self._chunker.name, self._chunker.version)
            )
        except InvalidIndexTransition as exc:
            raise IndexerError(str(exc)) from exc
        self._append(index, project_id, start_events, cid)

        # Step 2: build sources + chunks from parsed events.
        sections, references = self._read_parsed(project_id)

        try:
            source_events, chunk_events = self._build_events(
                project_id, sections, references
            )
        except Exception as exc:  # defensive: any chunker error -> failure event
            failure = IndexingFailed(
                project_id=project_id,
                cause_kind="chunker_failure",
                cause_message=str(exc),
                retryable=False,
            )
            self._append(index, project_id, [failure], cid)
            return False

        # Append sources first (so the projection sees them before chunks).
        if source_events:
            self._append(index, project_id, source_events, cid)
        if chunk_events:
            self._append(index, project_id, chunk_events, cid)

        completed = IndexingCompleted(
            project_id=project_id,
            source_count=len(source_events),
            chunk_count=len(chunk_events),
            chunker_name=self._chunker.name,
            chunker_version=self._chunker.version,
        )
        self._append(index, project_id, [completed], cid)
        return True

    def get_state(self, project_id: str) -> IndexState:
        return self._load_index_aggregate(project_id).state

    def project_into_projection(
        self, project_id: str, projection: SourcesProjection
    ) -> SourcesProjection:
        """Replay this project's index events into `projection`. Used by
        downstream services (workspace) and tests."""
        for stored in self._store.load(_index_aggregate_id(project_id)):
            if stored.event_type == "source_registered":
                src = SourceRef.model_validate(stored.payload["source"])
                projection.apply_source(src)
            elif stored.event_type == "chunk_created":
                chunk = Chunk.model_validate(stored.payload["chunk"])
                projection.apply_chunk(chunk)
        return projection

    # --- Internal ----------------------------------------------------------

    def _read_parsed(
        self, project_id: str
    ) -> tuple[list[Section], list[Reference]]:
        sections: list[Section] = []
        references: list[Reference] = []
        for stored in self._store.load(_parsed_aggregate_id(project_id)):
            if stored.event_type == "section_extracted":
                sections.append(Section.model_validate(stored.payload["section"]))
            elif stored.event_type == "reference_extracted":
                references.append(
                    Reference.model_validate(stored.payload["reference"])
                )
        return sections, references

    def _build_events(
        self,
        project_id: str,
        sections: list[Section],
        references: list[Reference],
    ) -> tuple[list[DomainEvent], list[DomainEvent]]:
        # Build a SourceRef per section + per reference, deterministically.
        sources_by_upstream: dict[str, SourceRef] = {}
        source_events: list[DomainEvent] = []

        # Sort sections deterministically before generating sources.
        for section in sorted(sections, key=lambda s: (s.depth, s.char_offset, s.id)):
            src = SourceRef(
                id=source_id_for(
                    project_id=project_id,
                    kind=SourceKind.paper_section,
                    upstream_id=section.id,
                ),
                project_id=project_id,
                kind=SourceKind.paper_section,
                locator=section.title,
                upstream_id=section.id,
            )
            sources_by_upstream[section.id] = src
            source_events.append(
                SourceRegistered(project_id=project_id, source=src)
            )

        for ref in sorted(references, key=lambda r: r.id):
            label = ref.title or (ref.arxiv_id or ref.doi or ref.id)
            src = SourceRef(
                id=source_id_for(
                    project_id=project_id,
                    kind=SourceKind.paper_reference,
                    upstream_id=ref.id,
                ),
                project_id=project_id,
                kind=SourceKind.paper_reference,
                locator=f"ref: {label}",
                upstream_id=ref.id,
            )
            source_events.append(
                SourceRegistered(project_id=project_id, source=src)
            )

        # Chunks: one per section via SectionChunker.
        chunks = self._chunker.chunk(
            sources_by_upstream=sources_by_upstream,
            sections=sections,
        )
        chunk_events: list[DomainEvent] = [
            ChunkCreated(project_id=project_id, chunk=c) for c in chunks
        ]
        return source_events, chunk_events

    def _load_parsed_aggregate(self, project_id: str) -> ParsedPaperAggregate:
        agg = ParsedPaperAggregate.empty(project_id)
        for stored in self._store.load(_parsed_aggregate_id(project_id)):
            cls = resolve_event_class(stored.event_type, stored.schema_version)
            agg.apply(stored.into(cls))
        return agg

    def _load_index_aggregate(self, project_id: str) -> IndexAggregate:
        agg = IndexAggregate.empty(project_id)
        for stored in self._store.load(_index_aggregate_id(project_id)):
            cls = resolve_event_class(stored.event_type, stored.schema_version)
            agg.apply(stored.into(cls))
        return agg

    def _append(
        self,
        agg: IndexAggregate,
        project_id: str,
        events: list[DomainEvent],
        correlation_id: CorrelationId,
    ) -> None:
        envelopes: list[EventEnvelope] = [
            make_envelope(
                source="context.indexer.service",
                correlation_id=correlation_id,
            )
            for _ in events
        ]
        self._store.append(
            aggregate_id=_index_aggregate_id(project_id),
            aggregate_type="index",
            events=events,
            expected_version=agg.version,
            envelopes=envelopes,
        )
        agg.apply_all(events)


__all__ = ["IndexerAppService", "IndexerError", "StartIndexing"]
