"""Per-run ``prior_work_refs`` manifest — constant-size, trim-on-overflow.

Reads the per-target ``rlm_state/literature_spec.json`` written by the
builder and produces the bounded list mounted at
``context["prior_work_refs"]``. Same posture as
``ingestion/repo/manifest.py::as_context``: a HARD serialized-size ceiling
(``MAX_CONTEXT_BYTES``) enforced by deterministic trimming — drop
lowest-ranked entries first, then shorten abstracts — so the root model's
context stays constant-size no matter how big the corpus grows (RLM
Algorithm-1 invariant).

Fail-soft: any error (missing/corrupt spec, odd shapes) returns ``[]`` —
byte-identical to the reserved empty ``prior_work_refs`` the context has
carried since the pivot brief.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_CONTEXT_BYTES = 8192
MAX_ENTRIES = 25
_TITLE_MAX = 200
_ABSTRACT_MAX = 300

_SPEC_FILENAME = "literature_spec.json"

# Budget-gated campaign width attempts (spec §8.3 / F16) launch under
# ``<project_id>_w<k>`` child run dirs, but UNDERSTAND writes the literature
# spec under the campaign's own project id — a width child falls back to the
# sibling parent dir's spec.
_WIDTH_CHILD_RE = re.compile(r"^(?P<parent>.+)_w\d+$")


def resolve_spec_path(project_dir: Path) -> Path | None:
    """This run's ``literature_spec.json`` path, or ``None`` when absent.

    Checks the run's own ``rlm_state/`` first; a campaign width child
    (``<project_id>_w<k>``) that has no spec of its own resolves to the
    campaign parent's spec. Fail-soft: never raises.
    """
    try:
        project_dir = Path(project_dir)
        own = project_dir / "rlm_state" / _SPEC_FILENAME
        if own.exists():
            return own
        match = _WIDTH_CHILD_RE.match(project_dir.name)
        if match:
            parent = (
                project_dir.parent / match.group("parent") / "rlm_state" / _SPEC_FILENAME
            )
            if parent.exists():
                return parent
    except Exception:  # noqa: BLE001 — advisory lookup must never break a run
        logger.debug("corpus manifest: resolve_spec_path failed", exc_info=True)
    return None


def _entry(paper: dict) -> dict | None:
    if not isinstance(paper, dict):
        return None
    title = str(paper.get("title") or "")[:_TITLE_MAX]
    pid = str(paper.get("id") or "")
    if not (pid or title):
        return None
    out: dict = {"id": pid, "title": title, "relation": str(paper.get("relation") or "")}
    year = paper.get("year")
    if isinstance(year, int):
        out["year"] = year
    abstract = paper.get("abstract")
    if isinstance(abstract, str) and abstract.strip():
        out["abstract"] = abstract.strip()[:_ABSTRACT_MAX]
    return out


def _serialized_size(entries: list[dict]) -> int:
    return len(json.dumps(entries, ensure_ascii=False).encode("utf-8"))


def build_prior_work_refs(project_dir: Path) -> list[dict]:
    """The bounded ``prior_work_refs`` list for a run. ``[]`` on any trouble."""
    try:
        spec_path = resolve_spec_path(project_dir)
        if spec_path is None:
            return []
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        papers = spec.get("papers")
        if not isinstance(papers, list):
            return []
        entries = [e for e in (_entry(p) for p in papers[:MAX_ENTRIES]) if e is not None]

        # Ceiling enforcement, in order: (1) drop abstracts from the tail up,
        # (2) drop whole tail entries — spec order is rank order, so the least
        # relevant paper is always the first casualty.
        while entries and _serialized_size(entries) > MAX_CONTEXT_BYTES:
            stripped = False
            for e in reversed(entries):
                if "abstract" in e:
                    del e["abstract"]
                    stripped = True
                    break
            if not stripped:
                entries.pop()
        return entries
    except Exception:  # noqa: BLE001 — a broken manifest must never break a run
        logger.debug("corpus manifest: build_prior_work_refs failed", exc_info=True)
        return []


__all__ = [
    "MAX_CONTEXT_BYTES",
    "MAX_ENTRIES",
    "build_prior_work_refs",
    "resolve_spec_path",
]
