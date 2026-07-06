"""HTTP-level tests for GET /papers/{arxiv_id}/repo.

Best-effort, fail-soft pre-run repo-resolve suggestion for the `_4`
repo-confirm UI screen. The only network dependency
(``backend.routes.papers_resolve._fetch_paper_text``) is monkeypatched so the
suite stays socket-hermetic (pytest-socket blocks real network — see
tests/conftest.py). This endpoint must NEVER 500: any fetch/discover/resolve
failure collapses to the empty ``{repo_url: None, provider: None,
confidence: None}`` suggestion.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app import create_app
import backend.routes.papers_resolve as papers_resolve


def _client() -> TestClient:
    return TestClient(create_app())


def test_repo_suggestion_found_returns_repo_url_provider_and_confidence(monkeypatch):
    monkeypatch.setattr(
        papers_resolve,
        "_fetch_paper_text",
        lambda arxiv_id, **kwargs: "See our code at https://github.com/ZJU-REAL/SDAR for details.",
    )
    r = _client().get("/papers/2605.15155/repo")
    assert r.status_code == 200
    body = r.json()
    assert body["repo_url"] == "https://github.com/ZJU-REAL/SDAR"
    assert body["provider"] == "github"
    assert isinstance(body["confidence"], float)
    assert 0.0 < body["confidence"] <= 1.0


def test_repo_suggestion_fetch_failure_returns_empty_suggestion(monkeypatch):
    monkeypatch.setattr(
        papers_resolve, "_fetch_paper_text", lambda arxiv_id, **kwargs: None
    )
    r = _client().get("/papers/2605.15155/repo")
    assert r.status_code == 200
    assert r.json() == {"repo_url": None, "provider": None, "confidence": None}


def test_repo_suggestion_no_repo_link_returns_empty_suggestion(monkeypatch):
    monkeypatch.setattr(
        papers_resolve,
        "_fetch_paper_text",
        lambda arxiv_id, **kwargs: "This paper has no public code release at this time.",
    )
    r = _client().get("/papers/2605.15155/repo")
    assert r.status_code == 200
    assert r.json() == {"repo_url": None, "provider": None, "confidence": None}


def test_repo_suggestion_handler_exception_is_fail_soft_never_500(monkeypatch):
    def _raise(arxiv_id, **kwargs):
        raise RuntimeError("simulated fetch blowup")

    monkeypatch.setattr(papers_resolve, "_fetch_paper_text", _raise)
    r = _client().get("/papers/2605.15155/repo")
    assert r.status_code == 200
    assert r.json() == {"repo_url": None, "provider": None, "confidence": None}
