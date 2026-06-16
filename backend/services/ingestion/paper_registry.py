"""Bundled-paper registry — the top reproduction targets shipped in-repo.

Motivation (2026-06-15): a coworker who clones the repo should be able to select +
reproduce the main papers immediately, with no network fetch. The SDAR paper's
arXiv id (``2605.15155``) is future-dated and does NOT resolve on arxiv.org, so a
plain ``reproduce 2605.15155`` produced a degraded 469-char run. Bundling the
papers (``papers/*.pdf`` + ``papers/registry.json``) and resolving a registered
id/alias/arXiv-id to the in-repo PDF fixes that and makes the trio offline-first.

Pure stdlib, fail-soft: a missing/maformed registry returns ``[]`` / ``None`` so
the normal arXiv/PDF/DOI path is unaffected. The PDFs are consumed by the existing
``pdf_path`` fetcher — no new ingestion surface.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RegistryEntry:
    """One bundled paper."""

    id: str
    title: str
    arxiv_id: str | None
    aliases: tuple[str, ...]
    pdf: str  # repo-relative path, e.g. "papers/sdar.pdf"
    hint: str | None
    datasets: tuple[str, ...]
    default: bool
    note: str = ""


def _repo_root() -> Path:
    # this file: <repo>/backend/services/ingestion/paper_registry.py → parents[3] = <repo>
    return Path(__file__).resolve().parents[3]


def _registry_path(registry_path: Path | None = None) -> Path:
    return registry_path or (_repo_root() / "papers" / "registry.json")


def load_registry(registry_path: Path | None = None) -> list[RegistryEntry]:
    """Parse ``papers/registry.json`` → list of entries. ``[]`` on any error."""
    try:
        data = json.loads(_registry_path(registry_path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — fail-soft: no registry ⇒ normal fetch path.
        return []
    entries: list[RegistryEntry] = []
    for e in data.get("papers", []) if isinstance(data, dict) else []:
        if not isinstance(e, dict) or not e.get("id") or not e.get("pdf"):
            continue
        entries.append(
            RegistryEntry(
                id=str(e["id"]),
                title=str(e.get("title", "")),
                arxiv_id=(str(e["arxiv_id"]) if e.get("arxiv_id") else None),
                aliases=tuple(str(a).strip().lower() for a in (e.get("aliases") or []) if str(a).strip()),
                pdf=str(e["pdf"]),
                hint=(str(e["hint"]) if e.get("hint") else None),
                datasets=tuple(str(d) for d in (e.get("datasets") or [])),
                default=bool(e.get("default", False)),
                note=str(e.get("note", "")),
            )
        )
    return entries


def _norm(s: str) -> str:
    return s.strip().lower()


def resolve(source: str, registry_path: Path | None = None) -> RegistryEntry | None:
    """Return the entry matching ``source`` by id, alias, or arXiv id — else None."""
    if not source or not str(source).strip():
        return None
    key = _norm(str(source))
    for e in load_registry(registry_path):
        if key == _norm(e.id) or (e.arxiv_id and key == _norm(e.arxiv_id)) or key in e.aliases:
            return e
    return None


def resolve_pdf_path(source: str, registry_path: Path | None = None) -> Path | None:
    """The absolute bundled-PDF path for ``source`` if registered AND on disk."""
    e = resolve(source, registry_path)
    if e is None:
        return None
    p = _repo_root() / e.pdf
    return p if p.is_file() else None


def list_presets(registry_path: Path | None = None) -> list[dict]:
    """Serializable preset list for the UI / ``GET /papers`` (existing PDFs only)."""
    out: list[dict] = []
    for e in load_registry(registry_path):
        if not (_repo_root() / e.pdf).is_file():
            continue
        out.append({
            "id": e.id,
            "title": e.title,
            "arxiv_id": e.arxiv_id,
            "datasets": list(e.datasets),
            "default": e.default,
        })
    return out
