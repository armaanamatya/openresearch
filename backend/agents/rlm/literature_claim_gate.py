"""Claim-grounding gate on the RUBRIC INPUT — WS② of the 2026-07-05
openscience-skill-library-and-harness-enhancements design.

Input-side counterpart to :mod:`backend.agents.rlm.report_claim_gate` (the
OUTPUT-side gate that caps a run's verdict when the final report's own
narrative claims a number ``claim_grounding.py`` can't find on disk). This
gate runs much earlier: it inspects the PAPER's own claimed numbers — the
same values ``rubric_gen.py::generate_rubric_tree`` turns into hard rubric
targets — and checks whether real literature (arXiv / OpenAlex / Semantic
Scholar) corroborates the textual neighborhood of each claim, via a fuzzy
token-overlap match against the connectors' returned titles.

This is advisory ONLY:
  - it NEVER changes the rubric tree, nor any target/weight in it;
  - a claim with no corroborating record is not proof of fabrication (the
    paper may simply be novel, or the extracted context too generic/short) —
    it is surfaced as a ``run_warning`` for a human/downstream signal to
    weigh, exactly like ``report_claim_gate`` never forces a verdict past
    "partial". "Fuzzy title match" here means "we could not find literature
    resembling this claim's context" — never a hard veto.

Flags:
  ``OPENRESEARCH_LITERATURE_CLAIM_GATE`` — master flag for this module.
  ``OPENRESEARCH_LITERATURE_GROUNDING``  — (``connectors/base.py``) gates
    whether the connectors are actually invoked; re-checked here too since
    this module is the "call site" that would otherwise touch the network.

Fail-soft everywhere: any exception (network, parsing, whatever) collapses to
an empty findings list — this must never break rubric generation.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.services.knowledge.connectors import (
    Connector,
    LiteratureRecord,
    cache_get,
    cache_put,
    default_connectors,
    literature_grounding_enabled,
)

logger = logging.getLogger(__name__)

_TRUTHY = ("1", "true", "yes")

#: Bound the number of distinct claims we spend network calls corroborating —
#: a paper can report dozens of numbers; this is an advisory sample, not an
#: exhaustive audit.
_DEFAULT_MAX_CLAIMS = 5
_DEFAULT_LIMIT_PER_CONNECTOR = 3
_DEFAULT_OVERLAP_THRESHOLD = 0.3

_STOPWORDS = frozenset({
    "the", "a", "an", "of", "on", "in", "for", "and", "to", "with", "using",
    "via", "is", "are", "by", "at", "as", "our", "we", "this", "that", "from",
})
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-_]*")


def literature_claim_gate_enabled() -> bool:
    """True iff ``OPENRESEARCH_LITERATURE_CLAIM_GATE`` is truthy."""
    return os.environ.get("OPENRESEARCH_LITERATURE_CLAIM_GATE", "").strip().lower() in _TRUTHY


@dataclass(frozen=True)
class LiteratureFinding:
    """One advisory "could not corroborate" result for a single claimed number."""

    term: str
    value: float
    context: str
    query: str
    checked_sources: tuple[str, ...]

    @property
    def message(self) -> str:
        sources = "/".join(self.checked_sources) or "no connectors"
        return (
            f"paper claims {self.term}={self.value!r} (context: {self.context.strip()!r}); "
            f"no corroborating literature record found via {sources}"
        )


def _tokenize(text: str) -> set[str]:
    words = _TOKEN_RE.findall((text or "").lower())
    return {w for w in words if len(w) > 2 and w not in _STOPWORDS}


def _title_overlap(query: str, title: str) -> float:
    """Containment-style token overlap in [0, 1]; 0.0 when either side is empty."""
    q, t = _tokenize(query), _tokenize(title or "")
    if not q or not t:
        return 0.0
    return len(q & t) / min(len(q), len(t))


def _dedupe_claims(claims: list) -> list:
    """Keep the first occurrence per (term, context) — a repeated identical
    window (the same number quoted twice) would otherwise waste a network call."""
    seen: set[tuple[str, str]] = set()
    out = []
    for c in claims:
        key = (c.term, c.context.strip())
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _search_cached(
    connector: Connector,
    query: str,
    *,
    limit: int,
    project_dir: Path | None,
) -> list[LiteratureRecord]:
    """``connector.search(query, limit=limit)``, cached (project_dir-scoped).

    A single connector's failure (network, malformed response, whatever) never
    aborts the gate — it just contributes no records for that source.
    """
    payload = {"source": connector.source, "query": query, "limit": limit}
    cached = cache_get(project_dir, connector.source, payload=payload)
    if cached is not None:
        try:
            return [LiteratureRecord(**r) for r in cached]
        except Exception:  # noqa: BLE001 — poisoned/stale cache entry, recompute live
            logger.debug("literature_claim_gate: cache hit failed validation for %s", connector.source)
    try:
        results = connector.search(query, limit=limit)
    except Exception:  # noqa: BLE001 — a single connector's failure never aborts the gate
        logger.debug("literature_claim_gate: %s.search failed", connector.source, exc_info=True)
        return []
    if not isinstance(results, list):
        return []
    cache_put(project_dir, connector.source, payload=payload, result=[r.to_dict() for r in results])
    return results


def gather_literature_findings(
    paper_text: str,
    *,
    connectors: list[Connector] | None = None,
    project_dir: Path | None = None,
    max_claims: int = _DEFAULT_MAX_CLAIMS,
    limit_per_connector: int = _DEFAULT_LIMIT_PER_CONNECTOR,
    overlap_threshold: float = _DEFAULT_OVERLAP_THRESHOLD,
) -> list[LiteratureFinding]:
    """Extract result claims from ``paper_text`` and flag the ones no
    configured connector's search results can corroborate via fuzzy title
    match.

    Pure/deterministic assembly around network calls that are themselves
    fail-soft (``connectors/base.py``). Never raises — any failure degrades
    to an empty list. Returns ``[]`` immediately when
    ``OPENRESEARCH_LITERATURE_GROUNDING`` is off (the connectors' own
    call-site gate) or when no claims / no connectors are available.
    """
    try:
        if not literature_grounding_enabled():
            return []
        from backend.agents.rlm.claim_grounding import extract_result_claims

        claims = _dedupe_claims(extract_result_claims(paper_text))[: max(0, int(max_claims))]
        if not claims:
            return []

        active_connectors = connectors if connectors is not None else default_connectors()
        if not active_connectors:
            return []

        findings: list[LiteratureFinding] = []
        for claim in claims:
            query = claim.context.strip()
            if not query:
                continue
            records: list[LiteratureRecord] = []
            checked: list[str] = []
            for connector in active_connectors:
                checked.append(connector.source)
                records.extend(
                    _search_cached(connector, query, limit=limit_per_connector, project_dir=project_dir)
                )
            corroborated = any(_title_overlap(query, rec.title) >= overlap_threshold for rec in records)
            if not corroborated:
                findings.append(
                    LiteratureFinding(
                        term=claim.term,
                        value=claim.value,
                        context=claim.context,
                        query=query,
                        checked_sources=tuple(checked),
                    )
                )
        return findings
    except Exception:  # noqa: BLE001 — fail-soft, never break rubric generation
        logger.debug("literature_claim_gate: gather_literature_findings failed", exc_info=True)
        return []


def run_literature_claim_gate(
    paper_text: str,
    *,
    project_dir: Path | None = None,
    emit_warning: Any | None = None,
    connectors: list[Connector] | None = None,
) -> list[LiteratureFinding]:
    """Top-level advisory hook — called from ``rubric_gen.generate_rubric_tree``.

    Returns ``[]`` immediately (no network calls at all) when
    ``OPENRESEARCH_LITERATURE_CLAIM_GATE`` is unset — safe to call
    unconditionally. NEVER mutates anything; ``emit_warning``, if given, is
    called once per finding as ``emit_warning(code, message)`` — mirrors
    ``report_claim_gate``'s optional-callable convention. Fail-soft throughout.
    """
    try:
        if not literature_claim_gate_enabled():
            return []
        findings = gather_literature_findings(paper_text, connectors=connectors, project_dir=project_dir)
        if findings and emit_warning is not None:
            for finding in findings:
                try:
                    emit_warning("literature_claim_ungrounded", finding.message)
                except Exception:  # noqa: BLE001
                    pass
        return findings
    except Exception:  # noqa: BLE001 — fail-soft, never break rubric generation
        logger.debug("literature_claim_gate: run_literature_claim_gate failed", exc_info=True)
        return []


__all__ = [
    "LiteratureFinding",
    "gather_literature_findings",
    "literature_claim_gate_enabled",
    "run_literature_claim_gate",
]
