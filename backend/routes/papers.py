"""GET /papers — the bundled reproduction targets (selectable presets).

Surfaces ``backend/services/ingestion/paper_registry`` so the lab UI can offer the
in-repo papers a fresh clone can reproduce offline (no network fetch). Read-only,
not demo-gated — mirrors the leaderboard route's lightweight, request-time shape.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter
from pydantic import BaseModel

from backend.services.ingestion import paper_registry

logger = logging.getLogger(__name__)

router = APIRouter()


class PaperPreset(BaseModel):
    id: str
    title: str
    arxiv_id: str | None = None
    datasets: list[str] = []
    default: bool = False


class PapersResponse(BaseModel):
    papers: list[PaperPreset]


@router.get("/papers", response_model=PapersResponse)
def list_papers() -> PapersResponse:
    """List the bundled, in-repo reproduction targets (id + title + datasets)."""
    try:
        presets = paper_registry.list_presets()
    except Exception:  # noqa: BLE001 — a read-only preset list must never 500.
        logger.exception("list_papers: registry read failed")
        presets = []
    return PapersResponse(papers=[PaperPreset(**p) for p in presets])
