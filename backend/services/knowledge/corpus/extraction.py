"""Structured extraction into ``lit_entities`` / ``lit_results`` — two lanes.

**Lane A — deterministic (free, runs inside every corpus build):** a curated
benchmark alias dictionary (the survey's recommendation — canonical name +
aliases, e.g. ``imagenet`` ≡ ``ILSVRC-2012``) matched literally against paper
text → ``lit_entities`` rows with ``source='deterministic'``. These rows are
the closed vocabulary the retrieval entity-linker uses. No LLM, no network,
byte-deterministic — safe inside the campaign UNDERSTAND $0 contract.

**Lane B — LLM (operator-invoked, ``corpus extract`` CLI / explicit caller):**
ONE bounded call per full-text paper asking for reported results as
``(method, dataset, metric, value, quote)`` rows. The **grounding rule**
(borrowed from rubric annotation — "no annotation beats a guessed one",
one-directional gates that can only REFUSE) is enforced in code, not prompt:

  - the ``quote`` must appear LITERALLY in the paper text (whitespace-
    normalized, case-insensitive),
  - the ``value`` must appear inside that quote,
  - anything else is silently dropped.

Accepted rows are stored ``source='llm'`` (lane rule: they may route/rank —
e.g. feed the advisory plausibility band — but are never citable evidence).
Every failure degrades per-paper; extraction can never break a build.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from backend.services.knowledge.corpus.store import CorpusStore

logger = logging.getLogger(__name__)

MAX_RESULTS_PER_PAPER = 20
MAX_ENTITIES_PER_PAPER = 15
MAX_PROMPT_CHARS = 24_000
MAX_EXTRACT_PAPERS = 8   # per invocation — bounded LLM spend

# Curated benchmark dictionary: canonical name -> aliases. Deliberately small
# and high-precision; extend by editing, not by LLM output (the whole point is
# a closed deterministic vocabulary). Canonical names are lowercase.
DATASET_ALIASES: dict[str, tuple[str, ...]] = {
    "imagenet": ("imagenet-1k", "ilsvrc-2012", "ilsvrc2012"),
    "cifar-10": ("cifar10",),
    "cifar-100": ("cifar100",),
    "mnist": (),
    "coco": ("ms-coco", "mscoco"),
    "squad": ("squad v1.1", "squad 1.1", "squad 2.0", "squadv2"),
    "glue": (),
    "superglue": (),
    "mmlu": (),
    "gsm8k": ("gsm-8k",),
    "math": ("math benchmark",),
    "humaneval": ("human-eval",),
    "mbpp": (),
    "hotpotqa": ("hotpot-qa", "hotpot qa"),
    "triviaqa": ("trivia-qa", "trivia qa"),
    "natural questions": ("naturalquestions", "nq"),
    "alfworld": ("alf-world", "alf world"),
    "webshop": ("web-shop",),
    "wikitext-103": ("wikitext103",),
    "c4": (),
    "the pile": ("pile",),
    "librispeech": (),
    "wmt14": ("wmt-14",),
    "wmt16": ("wmt-16",),
    "openwebtext": ("open-webtext",),
    "alpacaeval": ("alpaca-eval",),
    "mt-bench": ("mtbench", "mt bench"),
}

_WS_RE = re.compile(r"\s+")


@dataclass
class ExtractionStats:
    papers_seen: int = 0
    entities_added: int = 0
    results_added: int = 0
    results_rejected: int = 0   # failed the grounding gates
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "papers_seen": self.papers_seen,
            "entities_added": self.entities_added,
            "results_added": self.results_added,
            "results_rejected": self.results_rejected,
            "errors": list(self.errors),
        }


# ---------------------------------------------------------------------------
# Lane A — deterministic dictionary entities
# ---------------------------------------------------------------------------


def match_datasets(text: str) -> list[tuple[str, tuple[str, ...]]]:
    """Canonical datasets whose name or any alias appears literally in ``text``.

    Word-boundary, case-insensitive literal match — high precision, and pure
    (no state). Returns ``[(canonical_name, aliases)]`` sorted by name.
    """
    lowered = text.lower()
    hits: list[tuple[str, tuple[str, ...]]] = []
    for canonical, aliases in sorted(DATASET_ALIASES.items()):
        for candidate in (canonical, *aliases):
            if re.search(rf"(?<![a-z0-9]){re.escape(candidate)}(?![a-z0-9])", lowered):
                hits.append((canonical, aliases))
                break
    return hits


def run_deterministic_extraction(corpus: CorpusStore) -> ExtractionStats:
    """Populate deterministic dataset entities for every paper with text."""
    stats = ExtractionStats()
    for row in corpus.connection.execute(
        "SELECT id, abstract, fetched_level FROM lit_papers ORDER BY id"
    ).fetchall():
        try:
            text = _paper_text(corpus, row) or ""
            if not text:
                continue
            stats.papers_seen += 1
            for canonical, aliases in match_datasets(text):
                corpus.put_entity(
                    row["id"],
                    kind="dataset",
                    name=canonical,
                    aliases=list(aliases),
                    source="deterministic",
                )
                stats.entities_added += 1
        except Exception as exc:  # noqa: BLE001 — per-paper fail-soft
            if len(stats.errors) < 10:
                stats.errors.append(f"{row['id']}: {exc}"[:200])
    return stats


# ---------------------------------------------------------------------------
# Lane B — grounded LLM result extraction
# ---------------------------------------------------------------------------

_EXTRACT_SYSTEM = (
    "You extract REPORTED NUMERIC RESULTS from one research paper. Return ONLY "
    "a JSON object {\"results\": [...], \"entities\": [...]} — no prose.\n"
    "Each result: {\"method\": str, \"dataset\": str, \"metric\": str, "
    "\"value\": number, \"quote\": str}. The quote MUST be copied VERBATIM "
    "from the paper (<=250 chars) and MUST contain the numeric value; results "
    "you cannot quote verbatim must be omitted. Each entity: {\"kind\": "
    "\"dataset\"|\"method\"|\"metric\", \"name\": str} for names appearing "
    "literally in the text. Lowercase dataset/metric names. At most "
    f"{MAX_RESULTS_PER_PAPER} results and {MAX_ENTITIES_PER_PAPER} entities."
)


def _norm(text: str) -> str:
    return _WS_RE.sub(" ", text.lower()).strip()


def grounded(quote: str, value: Any, paper_text: str) -> bool:
    """The one-directional grounding gate: quote ⊂ text AND value ⊂ quote."""
    q = _norm(quote or "")
    if not q or len(q) < 8:
        return False
    if q not in _norm(paper_text):
        return False
    value_str = f"{value}"
    # Accept 58.1 matching "58.1" and 58.0 matching "58" — strip trailing
    # zeros ONLY after a decimal point ("100" must never reduce to "1").
    candidates = {value_str}
    if "." in value_str:
        candidates.add(value_str.rstrip("0").rstrip("."))
    return any(c and c in q for c in candidates)


def extract_paper_results(
    client: Any, paper_id: str, paper_text: str
) -> tuple[list[dict], list[dict], int]:
    """One bounded LLM call → (grounded_results, grounded_entities, rejected)."""
    raw = client.complete(
        system=_EXTRACT_SYSTEM,
        user=f"Paper text:\n\n{paper_text[:MAX_PROMPT_CHARS]}",
    )
    try:
        start, end = raw.find("{"), raw.rfind("}")
        payload = json.loads(raw[start : end + 1]) if start >= 0 else {}
    except (ValueError, TypeError):
        return [], [], 0

    results, rejected = [], 0
    for row in (payload.get("results") or [])[:MAX_RESULTS_PER_PAPER]:
        if not isinstance(row, dict):
            continue
        try:
            value = float(row.get("value"))
        except (TypeError, ValueError):
            rejected += 1
            continue
        quote = str(row.get("quote") or "")
        if not grounded(quote, row.get("value"), paper_text):
            rejected += 1
            continue
        results.append(
            {
                "method": _norm(str(row.get("method") or "unknown"))[:80],
                "dataset": _norm(str(row.get("dataset") or ""))[:80],
                "metric": _norm(str(row.get("metric") or ""))[:80],
                "value": value,
                "quote": quote[:300],
            }
        )

    entities = []
    lowered = _norm(paper_text)
    for ent in (payload.get("entities") or [])[:MAX_ENTITIES_PER_PAPER]:
        if not isinstance(ent, dict):
            continue
        kind = str(ent.get("kind") or "")
        name = _norm(str(ent.get("name") or ""))
        if kind in ("dataset", "method", "metric") and name and name in lowered:
            entities.append({"kind": kind, "name": name[:80]})
    return results, entities, rejected


def run_llm_extraction(
    corpus: CorpusStore,
    client: Any,
    *,
    max_papers: int = MAX_EXTRACT_PAPERS,
    force: bool = False,
) -> ExtractionStats:
    """Extract results for full-text papers that don't have any yet.

    ``force=True`` re-extracts papers that already carry rows. Per-paper
    fail-soft; at most ``max_papers`` LLM calls per invocation.
    """
    stats = ExtractionStats()
    rows = corpus.connection.execute(
        "SELECT p.id, p.abstract, p.fetched_level,"
        " (SELECT COUNT(*) FROM lit_results r WHERE r.paper_id = p.id) AS have"
        " FROM lit_papers p WHERE p.fetched_level = 'fulltext' ORDER BY p.id"
    ).fetchall()
    for row in rows:
        if stats.papers_seen >= max(1, int(max_papers)):
            break
        if row["have"] and not force:
            continue
        try:
            text = _paper_text(corpus, row) or ""
            if len(text) < 500:
                continue
            stats.papers_seen += 1
            results, entities, rejected = extract_paper_results(client, row["id"], text)
            stats.results_rejected += rejected
            for r in results:
                if not (r["dataset"] and r["metric"]):
                    continue
                corpus.put_result(
                    row["id"],
                    method=r["method"],
                    dataset=r["dataset"],
                    metric=r["metric"],
                    value=r["value"],
                    span_quote=r["quote"],
                    source="llm",
                )
                stats.results_added += 1
            for e in entities:
                corpus.put_entity(
                    row["id"], kind=e["kind"], name=e["name"], source="llm"
                )
                stats.entities_added += 1
        except Exception as exc:  # noqa: BLE001 — one paper must not sink the pass
            if len(stats.errors) < 10:
                stats.errors.append(f"{row['id']}: {exc}"[:200])
            logger.debug("extraction: paper %s failed", row["id"], exc_info=True)
    return stats


def _paper_text(corpus: CorpusStore, row: Any) -> str:
    if row["fetched_level"] == "fulltext":
        path = corpus.paper_dir(row["id"]) / "parsed_full_text.txt"
        try:
            if path.exists():
                return path.read_text(encoding="utf-8")
        except OSError:
            return row["abstract"] or ""
    return row["abstract"] or ""


__all__ = [
    "DATASET_ALIASES",
    "ExtractionStats",
    "extract_paper_results",
    "grounded",
    "match_datasets",
    "run_deterministic_extraction",
    "run_llm_extraction",
]
