"""GET /papers/{arxiv_id}/repo — best-effort pre-run repo-resolve suggestion.

Cheap, resolve-WITHOUT-clone suggestion feeding the `_4` repo-confirm UI
screen shown before a run launches. The in-run pipeline discovers artifacts
from the already-parsed, event-stored paper and clones the winning repo
inside the run; neither of those is available pre-ingestion, so this route
does its own bounded, best-effort text fetch + pure discovery + pure resolve:

1. Fetch the paper's arXiv HTML rendering (bounded, ~5s, fail-soft —
   ``_fetch_paper_text``; NOT the persisting/project-scoped ingestion fetcher
   at ``backend/services/ingestion/intake/fetchers/arxiv.py``).
2. Run the pure regex artifact-discovery adapter over that text
   (``RegexArtifactDiscoveryAdapter`` — no network, no store).
3. Feed the discovered artifacts through the same pure ``RepoResolver`` the
   in-run pipeline uses, WITHOUT provisioning/cloning anything.

Fail-soft end to end: this is a read-only, best-effort GET and must NEVER
500. Any failure at any stage (bad id, network error, no repo mentioned,
resolver exception) collapses to the empty suggestion
``{repo_url: None, provider: None, confidence: None}`` — the UI field just
renders blank/editable and the run's own in-loop resolver still finds the
repo.
"""

from __future__ import annotations

import logging
from urllib.error import URLError
from urllib.request import Request, urlopen

from fastapi import APIRouter
from pydantic import BaseModel

from backend.services.ingestion.discovery.adapters.regex import (
    RegexArtifactDiscoveryAdapter,
)
from backend.services.ingestion.repo.resolver import RepoResolver, normalize_repo_url

logger = logging.getLogger(__name__)

router = APIRouter()

_HTML_BASE_URL = "https://arxiv.org/html"
_USER_AGENT = "Mozilla/5.0 (compatible; openresearch-repo-resolve/0.1)"
_FETCH_TIMEOUT_S = 5.0
# Bounded read — this is a cheap suggestion source, not the real ingestion
# fetch; a paper's rendered HTML is comfortably under this cap.
_MAX_BYTES = 5 * 1024 * 1024


class RepoSuggestion(BaseModel):
    repo_url: str | None = None
    provider: str | None = None
    confidence: float | None = None


def _fetch_paper_text(arxiv_id: str, *, timeout_s: float = _FETCH_TIMEOUT_S) -> str | None:
    """Fetch the paper's arXiv HTML rendering, bounded and fail-soft.

    Returns ``None`` on ANY failure (bad id, network/timeout error, non-2xx,
    empty body) — never raises. Monkeypatched in tests so the suite stays
    socket-hermetic.
    """
    url = f"{_HTML_BASE_URL}/{arxiv_id}"
    try:
        request = Request(url, headers={"User-Agent": _USER_AGENT})
        with urlopen(request, timeout=timeout_s) as response:  # noqa: S310 — fixed https arxiv.org host
            body = response.read(_MAX_BYTES)
    except (URLError, OSError, ValueError):
        return None
    except Exception:  # noqa: BLE001 — this helper's contract is "never raises", full stop.
        logger.exception("_fetch_paper_text: unexpected failure fetching %s", arxiv_id)
        return None
    if not body:
        return None
    return body.decode("utf-8", errors="ignore")


def _artifact_repo_url(artifact) -> str | None:
    """Mirror ``RepoResolver.resolve``'s own normalization order (locator, then url)."""
    return normalize_repo_url(getattr(artifact, "locator", None)) or normalize_repo_url(
        str(getattr(artifact, "url", "")) or None
    )


@router.get("/papers/{arxiv_id}/repo", response_model=RepoSuggestion)
def get_repo_suggestion(arxiv_id: str) -> RepoSuggestion:
    """Best-effort repo suggestion for the pre-run repo-confirm screen."""
    try:
        text = _fetch_paper_text(arxiv_id)
        if not text:
            return RepoSuggestion()

        discovered = RegexArtifactDiscoveryAdapter().discover(project_id=arxiv_id, text=text)
        spec = RepoResolver.resolve(
            user_url=None, discovered=discovered, blacklist=set(), mode_override=None
        )
        if spec.url is None:
            return RepoSuggestion()

        repo_artifacts = [
            a for a in discovered
            if getattr(getattr(a, "kind", None), "value", None) == "repository"
        ]
        matched = [a for a in repo_artifacts if _artifact_repo_url(a) == spec.url]
        if matched:
            confidence = float(matched[0].confidence)
        elif repo_artifacts:
            confidence = float(max(a.confidence for a in repo_artifacts))
        else:
            confidence = None

        return RepoSuggestion(repo_url=spec.url, provider="github", confidence=confidence)
    except Exception:  # noqa: BLE001 — a read-only best-effort GET must never 500.
        logger.exception("get_repo_suggestion: repo resolve failed for %s", arxiv_id)
        return RepoSuggestion()
