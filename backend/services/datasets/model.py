"""Dataset cache models."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


DatasetCacheStatus = Literal["planned", "downloading", "available", "failed", "blocked"]


class DatasetCacheEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset_id: str
    name: str
    source_url: str = ""
    version: str = ""
    checksum: str = ""
    size_bytes: int | None = None
    local_path: str = ""
    status: DatasetCacheStatus = "planned"
    source_project_id: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    failure_reason: str = ""

    @property
    def size_gb(self) -> float:
        return (self.size_bytes or 0) / (1024 ** 3)


__all__ = ["DatasetCacheEntry", "DatasetCacheStatus"]
