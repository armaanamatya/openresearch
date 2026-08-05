"""Seed loading — assemble a ``TargetPaper`` from an ingested project.

Reads the parsed-paper event stream (``<project_id>:parsed`` aggregate — the
same stream ``IndexerAppService._read_parsed`` consumes) for the target's
reference list, plus best-effort title/arxiv-id recovery from the intake
aggregate and the run dir. Every lookup is fail-soft: a missing stream or odd
payload degrades to an emptier ``TargetPaper``, never an exception.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from backend.services.knowledge.corpus.builder import TargetPaper

logger = logging.getLogger(__name__)

_MAX_RAW_TEXT = 300


def load_target(
    project_id: str,
    *,
    store: Any,
    runs_root: Path,
    arxiv_id: str | None = None,
    title: str = "",
) -> TargetPaper:
    """Best-effort ``TargetPaper`` for an already-ingested project."""
    refs: list[dict] = []
    try:
        for stored in store.load(f"{project_id}:parsed"):
            if stored.event_type != "reference_extracted":
                continue
            payload = stored.payload.get("reference") or {}
            if not isinstance(payload, dict):
                continue
            refs.append(
                {
                    "arxiv_id": payload.get("arxiv_id"),
                    "doi": payload.get("doi"),
                    "title": payload.get("title"),
                    "raw_text": str(payload.get("raw_text") or "")[:_MAX_RAW_TEXT],
                }
            )
    except Exception:  # noqa: BLE001 — an unreadable stream is an emptier seed
        logger.debug("corpus inputs: parsed-stream read failed for %s", project_id, exc_info=True)

    if not arxiv_id:
        arxiv_id = _arxiv_id_from_intake(project_id, store)
    if not title:
        title = _title_from_run_dir(Path(runs_root) / project_id)

    return TargetPaper(
        project_id=project_id,
        arxiv_id=arxiv_id,
        title=title,
        references=tuple(refs),
    )


def _arxiv_id_from_intake(project_id: str, store: Any) -> str | None:
    """Recover the arXiv id from the intake aggregate's registration payload."""
    try:
        for stored in store.load(project_id):
            payload = getattr(stored, "payload", None)
            if not isinstance(payload, dict):
                continue
            source = payload.get("source")
            if isinstance(source, dict) and source.get("arxiv_id"):
                return str(source["arxiv_id"])
            if payload.get("arxiv_id"):
                return str(payload["arxiv_id"])
    except Exception:  # noqa: BLE001
        logger.debug("corpus inputs: intake read failed for %s", project_id, exc_info=True)
    return None


def _title_from_run_dir(project_dir: Path) -> str:
    """Best-effort paper title from run-dir status artifacts."""
    for name, keys in (
        ("demo_status.json", ("paper_title", "title")),
        ("artifact_index.json", ("paper_title", "title")),
    ):
        try:
            path = project_dir / name
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                continue
            for key in keys:
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        except Exception:  # noqa: BLE001
            continue
    return ""


__all__ = ["load_target"]
