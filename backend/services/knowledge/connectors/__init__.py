"""Key-free literature connectors — arXiv / OpenAlex / Semantic Scholar.

Adapted from OpenScience (Apache-2.0, Synthetic Sciences /
github.com/synthetic-sciences/openscience)
``science/connectors/literature/{shared,http,arxiv,openalex,semantic-scholar}.ts``
— a clean-room Python reimplementation of the same small ``Connector``
protocol, credited the same (see ``base.py`` module docstring).

Gated behind ``OPENRESEARCH_LITERATURE_GROUNDING`` at the call site
(``backend/agents/rlm/literature_claim_gate.py``) — this package always
imports; connectors are simply never invoked when the flag is unset.
"""

from __future__ import annotations

from backend.services.knowledge.connectors.arxiv import ArxivConnector
from backend.services.knowledge.connectors.base import (
    Connector,
    FetchJson,
    FetchText,
    LiteratureRecord,
    cache_get,
    cache_put,
    fetch_json,
    fetch_text,
    literature_grounding_enabled,
    snippet,
)
from backend.services.knowledge.connectors.openalex import OpenAlexConnector
from backend.services.knowledge.connectors.semantic_scholar import SemanticScholarConnector


def default_connectors() -> list[Connector]:
    """The three key-free connectors, wired to the real httpx transport
    (no injection) — the production default for ``literature_claim_gate.py``."""
    return [ArxivConnector(), OpenAlexConnector(), SemanticScholarConnector()]


__all__ = [
    "ArxivConnector",
    "Connector",
    "FetchJson",
    "FetchText",
    "LiteratureRecord",
    "OpenAlexConnector",
    "SemanticScholarConnector",
    "cache_get",
    "cache_put",
    "default_connectors",
    "fetch_json",
    "fetch_text",
    "literature_grounding_enabled",
    "snippet",
]
