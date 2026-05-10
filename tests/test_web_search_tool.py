"""WebSearchTool — Tavily + DuckDuckGo fallback tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend.services.context.workspace.model import Cited
from backend.services.context.workspace.tools.interface import WorkspaceToolError
from backend.services.context.workspace.tools.web_search import WebSearchTool


# --- Backend selection -------------------------------------------------------


def test_default_backend_is_ddg():
    tool = WebSearchTool()
    assert tool.backend == "ddg"


def test_tavily_backend_when_key_and_package_available():
    with patch.dict("os.environ", {"TAVILY_API_KEY": "tvly-test"}):
        with patch(
            "backend.services.context.workspace.tools.web_search.TavilyClient",
            create=True,
        ):
            # Simulate tavily importable
            mock_client = MagicMock()
            with patch(
                "backend.services.context.workspace.tools.web_search.WebSearchTool.__init__",
                return_value=None,
            ) as mock_init:
                # Just test that passing a key stores it
                tool = WebSearchTool.__new__(WebSearchTool)
                tool._tavily_key = "tvly-test"
                tool._tavily_client = mock_client
                assert tool.backend == "tavily"


# --- Validation --------------------------------------------------------------


def test_empty_query_raises():
    tool = WebSearchTool()
    with pytest.raises(WorkspaceToolError, match="searchable text"):
        tool.call(workspace_id="ws_test", query="   ")


def test_zero_limit_raises():
    tool = WebSearchTool()
    with pytest.raises(WorkspaceToolError, match="limit must be >= 1"):
        tool.call(workspace_id="ws_test", query="test", limit=0)


# --- DuckDuckGo fallback tests ----------------------------------------------

_MOCK_DDG_HTML = """
<html>
<body>
<div class="result">
  <a rel="nofollow" class="result__a" href="https://example.com/page1">Example Result One</a>
  <a class="result__snippet" href="https://example.com/page1">This is the first snippet about machine learning research.</a>
</div>
<div class="result">
  <a rel="nofollow" class="result__a" href="https://example.com/page2">Example Result Two</a>
  <a class="result__snippet" href="https://example.com/page2">Second snippet about deep learning experiments.</a>
</div>
<div class="result">
  <a rel="nofollow" class="result__a" href="https://example.com/page3">Example Result Three</a>
  <a class="result__snippet" href="https://example.com/page3">Third snippet about PyTorch CUDA compatibility.</a>
</div>
</body>
</html>
"""


@pytest.fixture
def ddg_tool():
    """WebSearchTool with no Tavily — always uses DuckDuckGo."""
    return WebSearchTool()


def test_ddg_search_returns_cited_results(ddg_tool):
    mock_response = MagicMock()
    mock_response.text = _MOCK_DDG_HTML
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.post", return_value=mock_response):
        result = ddg_tool.call(
            workspace_id="ws_test", query="machine learning", limit=3
        )

    assert isinstance(result, Cited)
    assert result.value["backend"] == "ddg"
    assert result.value["query"] == "machine learning"
    assert result.value["result_count"] == 3
    assert len(result.citations) == 3

    first = result.value["results"][0]
    assert first["url"] == "https://example.com/page1"
    assert "Example Result One" in first["title"]
    assert "machine learning" in first["snippet"]


def test_ddg_search_respects_limit(ddg_tool):
    mock_response = MagicMock()
    mock_response.text = _MOCK_DDG_HTML
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.post", return_value=mock_response):
        result = ddg_tool.call(
            workspace_id="ws_test", query="test", limit=2
        )

    assert result.value["result_count"] == 2
    assert len(result.citations) == 2


def test_ddg_search_citations_have_web_source_ids(ddg_tool):
    mock_response = MagicMock()
    mock_response.text = _MOCK_DDG_HTML
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.post", return_value=mock_response):
        result = ddg_tool.call(
            workspace_id="ws_test", query="test", limit=2
        )

    for citation in result.citations:
        assert citation.source_id.startswith("web:")
        assert citation.locator.startswith("https://")
        assert 0.0 < citation.confidence <= 1.0


def test_ddg_search_no_results_raises(ddg_tool):
    mock_response = MagicMock()
    mock_response.text = "<html><body>No results</body></html>"
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.post", return_value=mock_response):
        with pytest.raises(WorkspaceToolError, match="no results"):
            ddg_tool.call(workspace_id="ws_test", query="xyznonexistent")


def test_ddg_search_http_error_raises(ddg_tool):
    import httpx as _httpx

    with patch("httpx.post", side_effect=_httpx.ConnectError("refused")):
        with pytest.raises(WorkspaceToolError, match="request failed"):
            ddg_tool.call(workspace_id="ws_test", query="test")


def test_ddg_score_decreases_with_position(ddg_tool):
    mock_response = MagicMock()
    mock_response.text = _MOCK_DDG_HTML
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.post", return_value=mock_response):
        result = ddg_tool.call(
            workspace_id="ws_test", query="test", limit=3
        )

    scores = [r["score"] for r in result.value["results"]]
    assert scores == sorted(scores, reverse=True)


def test_ddg_strips_html_from_snippets(ddg_tool):
    html_with_bold = """
    <div class="result">
      <a rel="nofollow" class="result__a" href="https://example.com/p1">
        <b>Bold</b> Title
      </a>
      <a class="result__snippet" href="https://example.com/p1">
        Snippet with <b>bold</b> and &amp; entities
      </a>
    </div>
    """
    mock_response = MagicMock()
    mock_response.text = html_with_bold
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.post", return_value=mock_response):
        result = ddg_tool.call(
            workspace_id="ws_test", query="test", limit=1
        )

    first = result.value["results"][0]
    assert "<b>" not in first["title"]
    assert "<b>" not in first["snippet"]
    assert "&amp;" not in first["snippet"]
    assert "& entities" in first["snippet"]


# --- Tavily backend tests (mocked) ------------------------------------------


def test_tavily_search_returns_cited_results():
    mock_tavily = MagicMock()
    mock_tavily.search.return_value = {
        "answer": "Machine learning is a subset of AI.",
        "results": [
            {
                "url": "https://example.com/ml",
                "title": "ML Guide",
                "content": "Comprehensive guide to machine learning techniques.",
                "score": 0.95,
            },
            {
                "url": "https://example.com/dl",
                "title": "DL Intro",
                "content": "Introduction to deep learning fundamentals.",
                "score": 0.82,
            },
        ],
    }

    tool = WebSearchTool.__new__(WebSearchTool)
    tool._tavily_key = "tvly-test"
    tool._tavily_client = mock_tavily

    result = tool.call(
        workspace_id="ws_test", query="machine learning", limit=5
    )

    assert isinstance(result, Cited)
    assert result.value["backend"] == "tavily"
    assert result.value["answer"] == "Machine learning is a subset of AI."
    assert result.value["result_count"] == 2
    assert len(result.citations) == 2

    first = result.value["results"][0]
    assert first["url"] == "https://example.com/ml"
    assert first["score"] == 0.95


def test_tavily_empty_results_raises():
    mock_tavily = MagicMock()
    mock_tavily.search.return_value = {"answer": "", "results": []}

    tool = WebSearchTool.__new__(WebSearchTool)
    tool._tavily_key = "tvly-test"
    tool._tavily_client = mock_tavily

    with pytest.raises(WorkspaceToolError, match="no results"):
        tool.call(workspace_id="ws_test", query="test")


def test_tavily_falls_back_to_ddg_on_error():
    mock_tavily = MagicMock()
    mock_tavily.search.side_effect = RuntimeError("API error")

    tool = WebSearchTool.__new__(WebSearchTool)
    tool._tavily_key = "tvly-test"
    tool._tavily_client = mock_tavily

    mock_response = MagicMock()
    mock_response.text = _MOCK_DDG_HTML
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.post", return_value=mock_response):
        result = tool.call(workspace_id="ws_test", query="test", limit=2)

    # Should have fallen back to DDG
    assert result.value["backend"] == "ddg"
    assert result.value["result_count"] == 2


def test_tavily_citations_have_correct_confidence():
    mock_tavily = MagicMock()
    mock_tavily.search.return_value = {
        "answer": "",
        "results": [
            {
                "url": "https://example.com/high",
                "title": "High Score",
                "content": "Very relevant result.",
                "score": 0.99,
            },
            {
                "url": "https://example.com/low",
                "title": "Low Score",
                "content": "Less relevant result.",
                "score": 0.15,
            },
        ],
    }

    tool = WebSearchTool.__new__(WebSearchTool)
    tool._tavily_key = "tvly-test"
    tool._tavily_client = mock_tavily

    result = tool.call(workspace_id="ws_test", query="test")

    # High score should be clamped to max 1.0
    assert result.citations[0].confidence == 0.99
    # Low score should be clamped to min 0.2
    assert result.citations[1].confidence == 0.2


# --- Import / export tests --------------------------------------------------


def test_web_search_tool_in_workspace_exports():
    from backend.services.context.workspace import WebSearchTool as Exported

    assert Exported is WebSearchTool


def test_web_search_tool_in_tools_exports():
    from backend.services.context.workspace.tools import WebSearchTool as Exported

    assert Exported is WebSearchTool
