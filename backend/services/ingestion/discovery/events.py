"""Artifact discovery domain events."""

from __future__ import annotations

from typing import ClassVar

from backend.messaging.event import DomainEvent, register_event
from backend.services.ingestion.discovery.model import DiscoveredArtifact


@register_event
class DiscoveryStarted(DomainEvent):
    event_type: ClassVar[str] = "discovery_started"
    schema_version: ClassVar[int] = 1

    project_id: str
    adapter_names: tuple[str, ...]


@register_event
class ArtifactDiscovered(DomainEvent):
    event_type: ClassVar[str] = "artifact_discovered"
    schema_version: ClassVar[int] = 1

    project_id: str
    artifact: DiscoveredArtifact


@register_event
class DiscoveryCompleted(DomainEvent):
    event_type: ClassVar[str] = "discovery_completed"
    schema_version: ClassVar[int] = 1

    project_id: str
    artifact_count: int


@register_event
class DiscoveryFailed(DomainEvent):
    event_type: ClassVar[str] = "discovery_failed"
    schema_version: ClassVar[int] = 1

    project_id: str
    cause_kind: str
    cause_message: str
    retryable: bool


__all__ = [
    "ArtifactDiscovered",
    "DiscoveryCompleted",
    "DiscoveryFailed",
    "DiscoveryStarted",
]
