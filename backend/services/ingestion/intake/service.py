"""IntakeAppService — the IO boundary that orchestrates intake.

Handles two commands:
  - RegisterProject(source) -> project_id
      Creates (or returns existing) Project aggregate with deterministic
      project_id derived from the source. Idempotent on retry.
  - FetchPaper(project_id)
      Calls the matching fetcher (PdfPathFetcher for PdfPath sources)
      and appends PaperFetched (success) or PaperFetchFailed (failure).

This is where IO happens. The Project aggregate validates state
transitions; the service performs IO and appends the resulting
events to the event store with optimistic concurrency. The aggregate
NEVER calls IO.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

from pydantic import BaseModel, ConfigDict

from backend.eventstore.interface import (
    AppendError,
    ConcurrencyError,
    EventStore,
)
from backend.messaging.command import Command
from backend.messaging.envelope import (
    AggregateId,
    CorrelationId,
    EventEnvelope,
    make_envelope,
    new_correlation_id,
)
from backend.services.ingestion.intake.aggregate import (
    InvalidStateTransition,
    ProjectAggregate,
    ProjectState,
)
from backend.services.ingestion.intake.events import (
    PaperFetched,
    PaperFetchFailed,
    ProjectCreated,
)
from backend.services.ingestion.intake.fetchers.interface import (
    IntakeFetcher,
    IntakeFetchError,
)
from backend.services.ingestion.intake.sources import ArxivId, DoiRef, PaperSource, PdfPath


class RegisterProject(Command):
    """Submit a paper source. Returns the deterministic project_id.

    Re-registering the same source is a no-op that returns the
    same project_id without writing duplicate events.
    """

    model_config = ConfigDict(frozen=True)

    source: PaperSource


class FetchPaper(Command):
    """Materialize the paper to disk for an already-registered project."""

    model_config = ConfigDict(frozen=True)

    project_id: str


# --- Public errors --------------------------------------------------------


class IntakeError(Exception):
    """Base class for intake-level errors callers may want to catch."""


class UnknownProject(IntakeError):
    """Raised when FetchPaper references a project that was never registered."""


class IntakeAppService:
    """Application service layered over EventStore + a registry of fetchers
    keyed by source kind."""

    def __init__(
        self,
        store: EventStore,
        fetchers: Mapping[str, IntakeFetcher],
    ) -> None:
        self._store = store
        self._fetchers = dict(fetchers)

    # --- RegisterProject ---------------------------------------------------

    def register_project(
        self,
        cmd: RegisterProject,
        *,
        correlation_id: CorrelationId | None = None,
    ) -> str:
        project_id = project_id_for(cmd.source)
        cid = correlation_id or new_correlation_id()

        agg = self._load_aggregate(project_id)
        if agg.state is not ProjectState.NEW:
            # Already registered. Re-running is idempotent — same project_id,
            # no new events.
            return project_id

        events = list(agg.handle_register(cmd.source))
        envelopes = [
            make_envelope(source="ingestion.intake.service", correlation_id=cid)
            for _ in events
        ]
        try:
            self._store.append(
                aggregate_id=AggregateId(project_id),
                aggregate_type="project",
                events=events,
                expected_version=agg.version,
                envelopes=envelopes,
            )
        except ConcurrencyError:
            # Another writer raced us into the same project_id. Reload — the
            # winner will be in REGISTERED already; we just return.
            agg = self._load_aggregate(project_id)
            if agg.state is ProjectState.NEW:
                raise  # actual race we lost without recovery
        return project_id

    # --- FetchPaper --------------------------------------------------------

    def fetch_paper(
        self,
        cmd: FetchPaper,
        *,
        correlation_id: CorrelationId | None = None,
    ) -> bool:
        """Returns True on success (PaperFetched appended), False on
        recorded failure (PaperFetchFailed appended). Raises only on
        protocol errors (UnknownProject, AppendError)."""
        cid = correlation_id or new_correlation_id()
        agg = self._load_aggregate(cmd.project_id)
        if agg.state is ProjectState.NEW:
            raise UnknownProject(
                f"Project {cmd.project_id!r} has never been registered"
            )
        if agg.state is ProjectState.FETCHED:
            # Idempotent — already fetched.
            return True
        if agg.source is None:
            # Should be impossible if state is REGISTERED — defensive.
            raise IntakeError(
                f"Project {cmd.project_id!r} in state REGISTERED has no source on aggregate"
            )

        fetcher = self._fetchers.get(agg.source.kind)
        if fetcher is None:
            failure_events = list(
                agg.handle_fetch_failure(
                    cause_kind="unsupported_source_kind",
                    cause_message=f"No fetcher registered for source kind {agg.source.kind!r}",
                    retryable=False,
                )
            )
            self._append(agg, failure_events, cid)
            return False

        try:
            result = fetcher.fetch(agg.source, project_id=cmd.project_id)
        except IntakeFetchError as exc:
            failure_events = list(
                agg.handle_fetch_failure(
                    cause_kind=exc.cause_kind,
                    cause_message=str(exc),
                    retryable=exc.retryable,
                )
            )
            self._append(agg, failure_events, cid)
            return False

        success_events = list(
            agg.handle_fetch(
                raw_paper_path=result.raw_paper_path,
                pdf_sha256=result.pdf_sha256,
                pdf_size_bytes=result.pdf_size_bytes,
                fetched_via=fetcher.kind,
            )
        )
        self._append(agg, success_events, cid)
        return True

    # --- Read helpers ------------------------------------------------------

    def get_state(self, project_id: str) -> ProjectState:
        return self._load_aggregate(project_id).state

    # --- Internal ----------------------------------------------------------

    def _load_aggregate(self, project_id: str) -> ProjectAggregate:
        agg = ProjectAggregate.empty(project_id)
        for stored in self._store.load(AggregateId(project_id)):
            try:
                from backend.messaging.event import resolve_event_class

                cls = resolve_event_class(stored.event_type, stored.schema_version)
            except KeyError:
                # An unregistered event type means our app and the stored
                # data have diverged — this is an upcaster gap, not a
                # silent skip.
                raise IntakeError(
                    f"Cannot replay project {project_id!r}: no class registered for "
                    f"event_type={stored.event_type!r} v{stored.schema_version}"
                )
            agg.apply(stored.into(cls))
        return agg

    def _append(
        self,
        agg: ProjectAggregate,
        events: list,  # list[DomainEvent], typed loosely to avoid import cycle
        correlation_id: CorrelationId,
    ) -> None:
        envelopes: list[EventEnvelope] = [
            make_envelope(
                source="ingestion.intake.service",
                correlation_id=correlation_id,
            )
            for _ in events
        ]
        self._store.append(
            aggregate_id=AggregateId(agg.project_id),
            aggregate_type="project",
            events=events,
            expected_version=agg.version,
            envelopes=envelopes,
        )


# --- Project id derivation -----------------------------------------------


def project_id_for(source: PaperSource) -> str:
    """Compute the deterministic project_id for a source.

    For PdfPath we use the **resolved absolute path** (not file content)
    as the keyed material. This means re-running with the same path is
    idempotent; the same PDF under different absolute paths produces
    different project_ids, which is the right behavior — relocating a
    file is a new logical project until the user explicitly aliases it.
    Content-addressed project_ids are a future enhancement (would
    require reading the file at command time).
    """
    if isinstance(source, PdfPath):
        abs_path = str(Path(source.path).expanduser().resolve())
        digest = hashlib.sha256(f"pdf_path:{abs_path}".encode()).hexdigest()
        return f"prj_{digest[:16]}"
    if isinstance(source, ArxivId):
        digest = hashlib.sha256(f"arxiv:{source.arxiv_id.lower()}".encode()).hexdigest()
        return f"prj_{digest[:16]}"
    if isinstance(source, DoiRef):
        digest = hashlib.sha256(f"doi:{source.doi.lower()}".encode()).hexdigest()
        return f"prj_{digest[:16]}"
    raise NotImplementedError(
        f"project_id_for: unsupported source kind {source.kind!r}"
    )


__all__ = [
    "FetchPaper",
    "IntakeAppService",
    "IntakeError",
    "RegisterProject",
    "UnknownProject",
    "project_id_for",
]
