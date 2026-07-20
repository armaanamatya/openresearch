"""Shared base for key-free literature connectors (arXiv / OpenAlex / Semantic Scholar).

Adapted from OpenScience's ``science/connectors/literature/{shared,http}.ts``
(Apache-2.0, Synthetic Sciences / github.com/synthetic-sciences/openscience) —
a clean-room Python reimplementation of the same small ``Connector`` protocol
and text-normalization helpers, credited the same. These are read-only,
key-free HTTP clients over public scholarly APIs (Semantic Scholar accepts an
OPTIONAL API key for a higher rate limit; key-free still works), used to
advisory-ground the paper's OWN claimed numbers/baselines against real
literature records — the input-side counterpart to
``backend/agents/rlm/claim_grounding.py``.

Design contract (house rules — see
``docs/history/specs/2026-07-05-openscience-skill-library-and-harness-enhancements-design.md``
§6②):
  - Gated at the CALL SITE by ``OPENRESEARCH_LITERATURE_GROUNDING`` — this
    module is always importable; ``.search()``/``.fetch()`` are simply never
    invoked by ``literature_claim_gate.py`` when the flag is unset.
  - Fail-soft throughout: a blocked network, timeout, non-2xx status, or
    malformed response degrades to ``None``/``[]``, never raises (mirrors the
    ``OPENRESEARCH_USE_AUTHOR_REPO`` blocked-clone posture).
  - Every connector accepts an INJECTABLE transport (``fetch_json_fn`` /
    ``fetch_text_fn``) so tests exercise real parsing code against a canned
    in-memory response — the test suite is socket-hermetic
    (``pytest-socket``), so no connector test may reach a real network host.
"""

from __future__ import annotations

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

_TRUTHY = ("1", "true", "yes")

_DEFAULT_TIMEOUT_S = 10.0
_USER_AGENT = "openresearch-literature-grounding/0.1 (+https://github.com/lolout1/openresearch)"


def literature_grounding_enabled() -> bool:
    """True iff ``OPENRESEARCH_LITERATURE_GROUNDING`` is truthy.

    Gates every REAL connector network call. The module always imports; a
    caller (``literature_claim_gate.py``) checks this before ever invoking
    ``.search()``/``.fetch()`` on a connector.
    """
    return os.environ.get("OPENRESEARCH_LITERATURE_GROUNDING", "").strip().lower() in _TRUTHY


# ---------------------------------------------------------------------------
# LiteratureRecord — the normalized shape every connector returns.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LiteratureRecord:
    """A single normalized literature hit, uniform across connectors."""

    source: str
    id: str
    title: str
    authors: str | None = None
    year: int | None = None
    abstract_snippet: str | None = None
    url: str | None = None
    venue: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Text helpers — ported from shared.ts (decodeEntities / stripTags / snippet /
# fromInverted / xmlText / xmlBlocks / xmlAttr). Pure and defensive: odd or
# partial input returns None/[] rather than raising.
# ---------------------------------------------------------------------------

_ENTITIES: dict[str, str] = {
    "&amp;": "&",
    "&lt;": "<",
    "&gt;": ">",
    "&quot;": '"',
    "&apos;": "'",
    "&#39;": "'",
    "&nbsp;": " ",
}
_HEX_ENTITY_RE = re.compile(r"&#x([0-9a-fA-F]+);")
_DEC_ENTITY_RE = re.compile(r"&#(\d+);")
_NAMED_ENTITY_RE = re.compile(r"&[a-zA-Z]+;")
_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_TRAILING_PARTIAL_WORD_RE = re.compile(r"\s+\S*$")


def _safe_codepoint(code: int) -> str:
    if not (0 <= code <= 0x10FFFF):
        return ""
    try:
        return chr(code)
    except (ValueError, OverflowError):
        return ""


def decode_entities(text: str) -> str:
    """Decode the handful of XML/HTML entities that show up in scholarly metadata."""
    text = _HEX_ENTITY_RE.sub(lambda m: _safe_codepoint(int(m.group(1), 16)), text)
    text = _DEC_ENTITY_RE.sub(lambda m: _safe_codepoint(int(m.group(1))), text)
    text = _NAMED_ENTITY_RE.sub(lambda m: _ENTITIES.get(m.group(0), m.group(0)), text)
    return text


def strip_tags(text: str | None) -> str | None:
    """Strip XML/HTML tags (e.g. JATS ``<jats:p>`` abstracts) and collapse whitespace."""
    if not text:
        return None
    cleaned = _WHITESPACE_RE.sub(" ", decode_entities(_TAG_RE.sub(" ", text))).strip()
    return cleaned or None


def snippet(text: str | None, max_chars: int = 600) -> str | None:
    """Clamp a snippet to a readable length without cutting mid-word too hard."""
    cleaned = strip_tags(text)
    if not cleaned:
        return None
    if len(cleaned) <= max_chars:
        return cleaned
    truncated = _TRAILING_PARTIAL_WORD_RE.sub("", cleaned[:max_chars])
    return truncated + "…"


def from_inverted(index: dict[str, list[int]] | None) -> str | None:
    """Reconstruct a plain abstract from OpenAlex's ``abstract_inverted_index``."""
    if not index:
        return None
    words: dict[int, str] = {}
    for word, positions in index.items():
        for pos in positions or ():
            if isinstance(pos, int) and pos >= 0:
                words[pos] = word
    if not words:
        return None
    text = " ".join(words[i] for i in sorted(words)).strip()
    return text or None


def xml_text(xml: str, tag: str) -> str | None:
    """Inner text of the first ``<tag>…</tag>``, entity-decoded."""
    m = re.search(rf"<{tag}(?:\s[^>]*)?>([\s\S]*?)</{tag}>", xml)
    if not m:
        return None
    text = _WHITESPACE_RE.sub(" ", decode_entities(m.group(1))).strip()
    return text or None


def xml_blocks(xml: str, tag: str) -> list[str]:
    """Inner text of every ``<tag>…</tag>`` block."""
    return re.findall(rf"<{tag}(?:\s[^>]*)?>([\s\S]*?)</{tag}>", xml)


def xml_attr(xml: str, tag: str, attr: str) -> str | None:
    """Value of ``attr`` on the first ``<tag …>`` occurrence."""
    m = re.search(rf'<{tag}\b[^>]*?\b{attr}="([^"]*)"', xml)
    return decode_entities(m.group(1)) if m else None


def join_authors(names: list[str], *, max_named: int = 4) -> str | None:
    """Join author names the way every upstream connector formats them:
    the first ``max_named`` names plus "et al." beyond that, else a plain join.
    """
    clean = [n for n in names if n]
    if not clean:
        return None
    if len(clean) > max_named:
        return f"{', '.join(clean[:max_named])} et al."
    return ", ".join(clean)


# ---------------------------------------------------------------------------
# Transport seam — httpx-based default, injectable, fail-soft.
# ---------------------------------------------------------------------------

FetchText = Callable[..., "str | None"]
FetchJson = Callable[..., "dict[str, Any] | None"]


def fetch_text(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> str | None:
    """GET ``url`` and return the raw text body, or ``None`` on any failure.

    httpx-based default transport. Every connector accepts this (or
    :func:`fetch_json`) as an injectable constructor argument so tests
    exercise real parsing code against a canned in-memory response instead of
    the network (the suite is socket-hermetic). Fail-soft: timeout / DNS /
    connection-refused / non-2xx all degrade to ``None`` — a literature
    connector must never raise into a run.
    """
    try:
        import httpx
    except Exception:  # noqa: BLE001 — httpx is a hard dependency, but stay fail-soft
        return None
    merged_headers = {"User-Agent": _USER_AGENT, **(headers or {})}
    try:
        response = httpx.get(
            url,
            params=params,
            headers=merged_headers,
            timeout=timeout_s,
            follow_redirects=True,
        )
    except Exception:  # noqa: BLE001 — network/timeout/DNS/blocked-socket/etc.
        logger.debug("literature connector: fetch_text failed for %s", url, exc_info=True)
        return None
    if response.status_code != 200:
        logger.debug("literature connector: HTTP %s for %s", response.status_code, url)
        return None
    return response.text


def fetch_json(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> dict[str, Any] | None:
    """GET ``url`` and JSON-decode the body, or ``None`` on any failure
    (network, non-2xx, or non-object JSON)."""
    text = fetch_text(url, params=params, headers=headers, timeout_s=timeout_s)
    if text is None:
        return None
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


# ---------------------------------------------------------------------------
# Connector ABC
# ---------------------------------------------------------------------------


class Connector(ABC):
    """Uniform contract every literature connector implements.

    Constructed with an injectable transport (``fetch_json_fn`` /
    ``fetch_text_fn``) defaulting to the real httpx-based transport above —
    tests inject a fake returning a canned in-memory response.
    """

    #: Stable, lowercase identifier stamped onto every LiteratureRecord.
    source: str = "connector"

    def __init__(
        self,
        *,
        fetch_json_fn: FetchJson | None = None,
        fetch_text_fn: FetchText | None = None,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
    ) -> None:
        self._fetch_json = fetch_json_fn or fetch_json
        self._fetch_text = fetch_text_fn or fetch_text
        self._timeout_s = timeout_s

    @abstractmethod
    def search(self, query: str, *, limit: int = 10) -> list[LiteratureRecord]:
        """Search the source; returns normalized, possibly-empty hits. Fail-soft."""

    @abstractmethod
    def fetch(self, record_id: str) -> LiteratureRecord | None:
        """Fetch a single record by id. Returns ``None`` on miss/failure. Fail-soft."""


# ---------------------------------------------------------------------------
# Small content-addressed disk cache for connector responses.
#
# ``primitive_cache.CACHEABLE_PRIMITIVES`` is a closed allow-list scoped to
# the registered RLM primitives (primitives.py) — this gate is a deterministic
# helper, not a primitive, so extending that allow-list is out of scope here
# (see the delivery report for why). We instead REUSE its versioned
# content-addressed hashing (``make_key``) and mirror its JSONL-append design
# in a small, separately-namespaced file, so a literature response survives
# across attempts of the same paper (``project_dir`` is keyed by
# ``arxiv_id``) without touching ``primitive_cache.py`` at all.
# ---------------------------------------------------------------------------

_CACHE_FILENAME = "literature_cache.jsonl"


def cache_get(project_dir: Path | None, source: str, *, payload: Any) -> list[dict] | None:
    """Look up a cached connector response. ``None`` on miss/disable/any I/O error."""
    if project_dir is None:
        return None
    try:
        from backend.agents.rlm.primitive_cache import make_key
    except Exception:  # noqa: BLE001
        return None
    cache_path = Path(project_dir) / "rlm_state" / _CACHE_FILENAME
    if not cache_path.exists():
        return None
    key = make_key(f"literature:{source}", payload=payload)
    try:
        with cache_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(entry, dict) and entry.get("key") == key:
                    result = entry.get("result")
                    return result if isinstance(result, list) else None
    except OSError:
        return None
    return None


def cache_put(project_dir: Path | None, source: str, *, payload: Any, result: list[dict]) -> None:
    """Append a connector response to the cache. Fail-soft; no-op without ``project_dir``."""
    if project_dir is None or not isinstance(result, list):
        return
    try:
        from backend.agents.rlm.primitive_cache import make_key

        cache_dir = Path(project_dir) / "rlm_state"
        cache_dir.mkdir(parents=True, exist_ok=True)
        entry = {
            "key": make_key(f"literature:{source}", payload=payload),
            "source": source,
            "result": result,
        }
        with (cache_dir / _CACHE_FILENAME).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
    except Exception:  # noqa: BLE001 — observability must never block the caller
        logger.debug("literature connector cache: put failed for %s", source, exc_info=True)


__all__ = [
    "Connector",
    "FetchJson",
    "FetchText",
    "LiteratureRecord",
    "cache_get",
    "cache_put",
    "decode_entities",
    "fetch_json",
    "fetch_text",
    "from_inverted",
    "join_authors",
    "literature_grounding_enabled",
    "snippet",
    "strip_tags",
    "xml_attr",
    "xml_blocks",
    "xml_text",
]
