"""External artifact discovery model."""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any, NewType

from pydantic import BaseModel, ConfigDict, HttpUrl


ArtifactRefId = NewType("ArtifactRefId", str)


class DiscoveredArtifactKind(str, Enum):
    repository = "repository"
    dataset = "dataset"
    issue = "issue"
    discussion = "discussion"


def artifact_ref_id_for(
    *, project_id: str, kind: DiscoveredArtifactKind, locator: str
) -> ArtifactRefId:
    h = hashlib.sha256()
    h.update(f"artifact:{project_id}:{kind.value}:".encode())
    h.update(locator.encode())
    return ArtifactRefId(f"art_{h.hexdigest()[:16]}")


class DiscoveredArtifact(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    project_id: str
    kind: DiscoveredArtifactKind
    locator: str
    url: HttpUrl
    title: str | None = None
    metadata: dict[str, Any] = {}
    evidence_quote: str
    confidence: float = 0.8


__all__ = [
    "ArtifactRefId",
    "DiscoveredArtifact",
    "DiscoveredArtifactKind",
    "artifact_ref_id_for",
]
