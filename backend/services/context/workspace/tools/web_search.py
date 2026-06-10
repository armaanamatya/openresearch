"""Web search tool — Tavily AI-native search with DuckDuckGo fallback.

When tavily-python is installed and TAVILY_API_KEY is set, uses Tavily's
AI-native search (citation-ready, relevance-scored results). Falls back
to DuckDuckGo HTML search via httpx (no API key, no extra deps) when
Tavily is unavailable.

Both paths return evidence-grade citations via Cited[dict].
"""

from __future__ import annotations

import logging
import os
import re
from html import unescape
from typing import Any

import httpx

from backend.schemas.citations import Citation
from backend.services.context.workspace.model import Cited
from backend.services.context.workspace.tools.interface import WorkspaceToolError

logger = logging.getLogger(__name__)

_QUOTE_TRUNCATE = 240
_DDG_URL = "https://html.duckduckgo.com/html/"
_DDG_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ReproLab/0.1; research-agent)",
}

# Regex to extract DuckDuckGo HTML search results.
_DDG_RESULT_RE = re.compile(
    r'<a\s+rel="nofollow"\s+class="result__a"\s+href="([^"]+)"[^>]*>'
    r"(.*?)</a>",
    re.DOTALL,
)
_DDG_SNIPPET_RE = re.compile(
    r'<a\s+class="result__snippet"[^>]*>(.*?)</a>',
    re.DOTALL,
)
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    return unescape(_HTML_TAG_RE.sub("", text)).strip()


def _tavily_available() -> bool:
    try:
        import tavily  # noqa: F401
        return bool(os.environ.get("TAVILY_API_KEY"))
    except ImportError:
        return False


class WebSearchTool:
    """Search the web with Tavily or DuckDuckGo fallback.

    When constructed with a ``tavily_api_key`` (or TAVILY_API_KEY env var
    is set and tavily-python is installed), uses Tavily's AI-native search.
    Otherwise falls back to DuckDuckGo HTML search via httpx.
    """

    name = "web_search"

    def __init__(self, tavily_api_key: str | None = None) -> None:
        self._tavily_key = tavily_api_key or os.environ.get("TAVILY_API_KEY")
        self._tavily_client: Any | None = None

        if self._tavily_key:
            try:
                from tavily import TavilyClient
                self._tavily_client = TavilyClient(api_key=self._tavily_key)
                logger.info("WebSearchTool initialized with Tavily backend")
            except ImportError:
                logger.info("tavily-python not installed; using DuckDuckGo fallback")
                self._tavily_client = None

    @property
    def backend(self) -> str:
        """Return which search backend is active."""
        return "tavily" if self._tavily_client is not None else "ddg"

    def call(
        self,
        *,
        workspace_id: str,
        query: str,
        limit: int = 5,
        **kwargs: Any,
    ) -> Cited[dict[str, Any]]:
        del workspace_id
        if limit < 1:
            raise WorkspaceToolError("web_search limit must be >= 1")
        if not query.strip():
            raise WorkspaceToolError("web_search query must contain searchable text")

        if self._tavily_client is not None:
            return self._tavily_search(query, limit)
        return self._ddg_search(query, limit)

    # --- Tavily search -------------------------------------------------------

    def _tavily_search(
        self, query: str, limit: int
    ) -> Cited[dict[str, Any]]:
        try:
            response = self._tavily_client.search(
                query=query,
                max_results=limit,
                include_answer=True,
            )
        except Exception as exc:
            logger.warning("Tavily search failed, falling back to DuckDuckGo: %s", exc)
            return self._ddg_search(query, limit)

        results: list[dict[str, Any]] = []
        citations: list[Citation] = []

        for hit in response.get("results", []):
            url = hit.get("url", "")
            title = hit.get("title", "")
            snippet = hit.get("content", "")[:_QUOTE_TRUNCATE]
            score = hit.get("score", 0.5)

            results.append({
                "url": url,
                "title": title,
                "snippet": snippet,
                "score": score,
                "backend": "tavily",
            })
            citations.append(Citation(
                source_id=f"web:{url}",
                chunk_id=None,
                quote=snippet,
                locator=url,
                confidence=min(1.0, max(0.2, score)),
            ))

        if not results:
            raise WorkspaceToolError("web_search found no results")

        answer = response.get("answer", "")

        return Cited(
            value={
                "query": query,
                "answer": answer,
                "results": results,
                "result_count": len(results),
                "backend": "tavily",
            },
            citations=tuple(citations),
        )

    # --- DuckDuckGo fallback -------------------------------------------------

    def _ddg_search(
        self, query: str, limit: int
    ) -> Cited[dict[str, Any]]:
        try:
            resp = httpx.post(
                _DDG_URL,
                data={"q": query},
                headers=_DDG_HEADERS,
                timeout=15.0,
                follow_redirects=True,
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise WorkspaceToolError(
                f"web_search DuckDuckGo request failed: {exc}"
            ) from exc

        html = resp.text
        titles_urls = _DDG_RESULT_RE.findall(html)
        snippets = _DDG_SNIPPET_RE.findall(html)

        if not titles_urls:
            raise WorkspaceToolError("web_search found no results")

        results: list[dict[str, Any]] = []
        citations: list[Citation] = []

        for i, (url, raw_title) in enumerate(titles_urls[:limit]):
            title = _strip_html(raw_title)
            snippet = _strip_html(snippets[i]) if i < len(snippets) else ""
            quote = snippet[:_QUOTE_TRUNCATE] if snippet else title[:_QUOTE_TRUNCATE]

            results.append({
                "url": url,
                "title": title,
                "snippet": snippet,
                "score": round(1.0 - (i * 0.1), 2),
                "backend": "ddg",
            })
            citations.append(Citation(
                source_id=f"web:{url}",
                chunk_id=None,
                quote=quote,
                locator=url,
                confidence=max(0.3, round(0.8 - (i * 0.05), 2)),
            ))

        return Cited(
            value={
                "query": query,
                "answer": "",
                "results": results,
                "result_count": len(results),
                "backend": "ddg",
            },
            citations=tuple(citations),
        )


__all__ = ["WebSearchTool"]
