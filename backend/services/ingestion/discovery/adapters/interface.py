"""Discovery adapter protocol."""

from __future__ import annotations

from typing import Protocol, Sequence

from backend.services.ingestion.discovery.model import DiscoveredArtifact


class DiscoveryAdapterError(Exception):
    def __init__(self, message: str, *, cause_kind: str, retryable: bool) -> None:
        super().__init__(message)
        self.cause_kind = cause_kind
        self.retryable = retryable


class DiscoveryAdapter(Protocol):
    @property
    def name(self) -> str: ...

    def discover(self, *, project_id: str, text: str) -> Sequence[DiscoveredArtifact]:
        """Extract normalized external artifacts from parsed paper text."""


__all__ = ["DiscoveryAdapter", "DiscoveryAdapterError"]
