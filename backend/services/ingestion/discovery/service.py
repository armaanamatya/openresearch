"""ArtifactDiscoveryAppService — extracts external artifacts from parsed papers."""

from __future__ import annotations

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
from backend.services.ingestion.discovery.adapters.interface import (
    DiscoveryAdapter,
    DiscoveryAdapterError,
)
from backend.services.ingestion.discovery.aggregate import (
    DiscoveryAggregate,
    DiscoveryState,
    InvalidDiscoveryTransition,
)
from backend.services.ingestion.discovery.events import (
    ArtifactDiscovered,
    DiscoveryCompleted,
    DiscoveryFailed,
)
from backend.services.ingestion.discovery.model import DiscoveredArtifact
from backend.services.ingestion.parser.aggregate import (
    ParsedPaperAggregate,
    ParsedPaperState,
)


class DiscoverArtifacts(Command):
    model_config = ConfigDict(frozen=True)
    project_id: str


class DiscoveryError(Exception):
    pass


def _discovery_aggregate_id(project_id: str) -> AggregateId:
    return AggregateId(f"{project_id}:discovery")


def _parsed_aggregate_id(project_id: str) -> AggregateId:
    return AggregateId(f"{project_id}:parsed")


class ArtifactDiscoveryAppService:
    def __init__(self, store: EventStore, adapters: list[DiscoveryAdapter]) -> None:
        self._store = store
        self._adapters = list(adapters)

    def discover(
        self,
        cmd: DiscoverArtifacts,
        *,
        correlation_id: CorrelationId | None = None,
    ) -> bool:
        cid = correlation_id or new_correlation_id()
        project_id = cmd.project_id

        parsed = self._load_parsed_aggregate(project_id)
        if parsed.state is not ParsedPaperState.PARSED:
            raise DiscoveryError(
                f"Cannot discover artifacts: parsed paper for {project_id!r} is in "
                f"state {parsed.state.value!r}; must be {ParsedPaperState.PARSED.value!r}."
            )

        agg = self._load_discovery_aggregate(project_id)
        if agg.state is DiscoveryState.COMPLETED:
            return True
        if agg.state is DiscoveryState.DISCOVERING:
            raise DiscoveryError(f"Project {project_id!r} is already DISCOVERING.")

        try:
            start_events = list(
                agg.handle_start(tuple(adapter.name for adapter in self._adapters))
            )
        except InvalidDiscoveryTransition as exc:
            raise DiscoveryError(str(exc)) from exc
        self._append(agg, project_id, start_events, cid)

        text = self._read_full_text(project_id)
        try:
            artifacts = self._run_adapters(project_id=project_id, text=text)
        except DiscoveryAdapterError as exc:
            failure = DiscoveryFailed(
                project_id=project_id,
                cause_kind=exc.cause_kind,
                cause_message=str(exc),
                retryable=exc.retryable,
            )
            self._append(agg, project_id, [failure], cid)
            return False

        events: list[DomainEvent] = [
            ArtifactDiscovered(project_id=project_id, artifact=artifact)
            for artifact in artifacts
        ]
        if events:
            self._append(agg, project_id, events, cid)

        completed = DiscoveryCompleted(
            project_id=project_id,
            artifact_count=len(artifacts),
        )
        self._append(agg, project_id, [completed], cid)
        return True

    def list_artifacts(self, project_id: str) -> list[DiscoveredArtifact]:
        artifacts: list[DiscoveredArtifact] = []
        for stored in self._store.load(_discovery_aggregate_id(project_id)):
            if stored.event_type == "artifact_discovered":
                artifacts.append(
                    DiscoveredArtifact.model_validate(stored.payload["artifact"])
                )
        return artifacts

    def get_state(self, project_id: str) -> DiscoveryState:
        return self._load_discovery_aggregate(project_id).state

    def _run_adapters(self, *, project_id: str, text: str) -> list[DiscoveredArtifact]:
        by_key: dict[tuple[str, str], DiscoveredArtifact] = {}
        for adapter in self._adapters:
            for artifact in adapter.discover(project_id=project_id, text=text):
                by_key.setdefault((artifact.kind.value, artifact.locator), artifact)
        return sorted(by_key.values(), key=lambda item: (item.kind.value, item.locator))

    def _read_full_text(self, project_id: str) -> str:
        for stored in self._store.load(_parsed_aggregate_id(project_id)):
            if stored.event_type == "parsing_completed":
                path = stored.payload["full_text_blob_path"]
                try:
                    with open(path, encoding="utf-8") as f:
                        return f.read()
                except OSError as exc:
                    raise DiscoveryError(
                        f"Cannot read parsed full-text blob for {project_id!r}: {exc}"
                    ) from exc
        raise DiscoveryError(
            f"No ParsingCompleted event found for project {project_id!r}."
        )

    def _load_parsed_aggregate(self, project_id: str) -> ParsedPaperAggregate:
        agg = ParsedPaperAggregate.empty(project_id)
        for stored in self._store.load(_parsed_aggregate_id(project_id)):
            cls = resolve_event_class(stored.event_type, stored.schema_version)
            agg.apply(stored.into(cls))
        return agg

    def _load_discovery_aggregate(self, project_id: str) -> DiscoveryAggregate:
        agg = DiscoveryAggregate.empty(project_id)
        for stored in self._store.load(_discovery_aggregate_id(project_id)):
            cls = resolve_event_class(stored.event_type, stored.schema_version)
            agg.apply(stored.into(cls))
        return agg

    def _append(
        self,
        agg: DiscoveryAggregate,
        project_id: str,
        events: list[DomainEvent],
        correlation_id: CorrelationId,
    ) -> None:
        envelopes: list[EventEnvelope] = [
            make_envelope(
                source="ingestion.discovery.service",
                correlation_id=correlation_id,
            )
            for _ in events
        ]
        self._store.append(
            aggregate_id=_discovery_aggregate_id(project_id),
            aggregate_type="discovery",
            events=events,
            expected_version=agg.version,
            envelopes=envelopes,
        )
        agg.apply_all(events)


__all__ = [
    "ArtifactDiscoveryAppService",
    "DiscoverArtifacts",
    "DiscoveryError",
]
