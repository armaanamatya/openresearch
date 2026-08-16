"""Hybrid retrieval over the corpus: seed → expand → rerank, fully traced.

The survey-validated standard, deterministic end to end:

  1. **Seed** — weighted Reciprocal Rank Fusion (``k=60``, BM25-favored) over
     the FTS5 lexical channel (fallback: pure-Python Okapi BM25, same k1/b as
     ``context/workspace/tools/semantic_search.py::_rank_chunks``) and the
     optional sqlite-vec dense channel.
  2. **Expand** — 1-hop citation traversal from the seed papers + the target
     (PaperQA2's ablation-validated move, p=0.022) and exact-match entity
     linking against ``lit_entities`` names/aliases (``source='deterministic'``
     rows only — the closed-vocabulary dictionary lane).
  3. **Rerank** — fused score + a small bounded graph-metric boost
     (``graph_boost`` × ``_GRAPH_WEIGHT``); graph evidence reorders, never
     dominates.

Every hit carries ``lane`` (``"A"`` — everything this module produces is
deterministic/citable; LLM-derived channels, when they exist, must label
``"B"`` and may never be sole provenance) and a full per-channel rank trace.
The trace is the audit artifact: same corpus snapshot + same query ⇒
byte-identical trace.

PaperQA2 context-budget finding honored via ``top_n``: ~5 maximizes answer
accuracy, ~15 maximizes precision — callers pick per question type.
"""

from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
from pathlib import Path
from typing import Any, Callable, Sequence

from backend.services.knowledge.corpus.graph_metrics import graph_boost
from backend.services.knowledge.corpus.store import CorpusStore

logger = logging.getLogger(__name__)

RRF_K = 60
SEED_K = 50
_W_BM25 = 1.0
_W_VEC = 0.8
_W_EXPAND = 0.5
_GRAPH_WEIGHT = 0.005  # graph_boost ≤1.0 × this ≈ ≤30% of a top RRF rank share
_QUOTE_MAX = 300
_EXPAND_SEED_PAPERS = 10
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]{1,}")


def retrieve(
    corpus: CorpusStore,
    query: str,
    *,
    target_id: str | None = None,
    top_n: int = 5,
    embed_fn: Callable[[Sequence[str]], Sequence[Sequence[float]]] | None = None,
    trace_path: Path | None = None,
) -> dict[str, Any]:
    """Hybrid corpus retrieval. Returns ``{"hits": [...], "trace": {...}}``.

    Fail-soft: any channel error drops that channel (recorded in the trace),
    an empty index returns empty hits — never raises on environment trouble.
    """
    conn = corpus.connection
    trace: dict[str, Any] = {"query": query, "channels": {}}

    bm25_ranks = _lexical_ranks(conn, query, trace)
    vec_ranks = _vector_ranks(conn, query, embed_fn, trace)

    fused: dict[str, float] = {}
    for rank_map, weight in ((bm25_ranks, _W_BM25), (vec_ranks, _W_VEC)):
        for chunk_id, rank in rank_map.items():
            fused[chunk_id] = fused.get(chunk_id, 0.0) + weight / (RRF_K + rank)

    expand_ranks = _expansion_ranks(
        corpus, conn, query, fused, bm25_ranks, target_id, trace
    )
    for chunk_id, rank in expand_ranks.items():
        fused[chunk_id] = fused.get(chunk_id, 0.0) + _W_EXPAND / (RRF_K + rank)

    metrics = corpus.get_graph_metrics(target_id) if target_id else {}
    chunk_meta = _chunk_meta(conn, list(fused))
    scored: list[tuple[float, str]] = []
    for chunk_id, base in fused.items():
        meta = chunk_meta.get(chunk_id)
        if meta is None:
            continue
        boost = graph_boost(metrics.get(meta["paper_id"], {})) if metrics else 0.0
        scored.append((base + _GRAPH_WEIGHT * boost, chunk_id))
    scored.sort(key=lambda t: (-t[0], t[1]))

    hits = []
    for score, chunk_id in scored[: max(1, int(top_n))]:
        meta = chunk_meta[chunk_id]
        hits.append(
            {
                "chunk_id": chunk_id,
                "paper_id": meta["paper_id"],
                "title": meta["title"],
                "section": meta["section"],
                "quote": meta["text"][:_QUOTE_MAX],
                "lane": "A",
                "score": round(score, 6),
                "channels": {
                    name: ranks[chunk_id]
                    for name, ranks in (
                        ("bm25", bm25_ranks),
                        ("vec", vec_ranks),
                        ("graph_expand", expand_ranks),
                    )
                    if chunk_id in ranks
                },
            }
        )
    trace["hits"] = hits
    if trace_path is not None:
        _write_trace(trace_path, trace)
    return {"hits": hits, "trace": trace}


# --- channels ----------------------------------------------------------------


def _fts_query(query: str) -> str:
    """Quote tokens so user text can't break FTS5 MATCH syntax."""
    tokens = _TOKEN_RE.findall(query)[:16]
    return " OR ".join(f'"{t}"' for t in tokens)


def _lexical_ranks(
    conn: sqlite3.Connection, query: str, trace: dict
) -> dict[str, int]:
    """FTS5 bm25() when available; else pure-Python Okapi BM25 over lit_chunks."""
    match = _fts_query(query)
    if not match:
        trace["channels"]["bm25"] = {"used": False, "reason": "empty query"}
        return {}
    try:
        rows = conn.execute(
            "SELECT chunk_id FROM lit_chunks_fts WHERE lit_chunks_fts MATCH ?"
            " ORDER BY bm25(lit_chunks_fts), chunk_id LIMIT ?",
            (match, SEED_K),
        ).fetchall()
        trace["channels"]["bm25"] = {"used": True, "backend": "fts5", "hits": len(rows)}
        return {r["chunk_id"]: i for i, r in enumerate(rows)}
    except sqlite3.OperationalError:
        pass  # FTS5 table absent — fall back to the pure-Python ranker
    try:
        chunks = conn.execute(
            "SELECT chunk_id, text FROM lit_chunks ORDER BY chunk_id"
        ).fetchall()
    except sqlite3.OperationalError:
        trace["channels"]["bm25"] = {"used": False, "reason": "no chunk table"}
        return {}
    ranked = _fallback_bm25(query, [(r["chunk_id"], r["text"]) for r in chunks])
    trace["channels"]["bm25"] = {
        "used": True, "backend": "python", "hits": len(ranked)
    }
    return {chunk_id: i for i, chunk_id in enumerate(ranked[:SEED_K])}


def _fallback_bm25(query: str, chunks: list[tuple[str, str]]) -> list[str]:
    """Okapi BM25 (k1=1.2, b=0.75 — same constants as workspace `_rank_chunks`)."""
    k1, b = 1.2, 0.75
    q_terms = [t.lower() for t in _TOKEN_RE.findall(query)]
    if not q_terms or not chunks:
        return []
    docs = [(cid, [t.lower() for t in _TOKEN_RE.findall(text)]) for cid, text in chunks]
    avg_len = sum(len(toks) for _, toks in docs) / len(docs) or 1.0
    n = len(docs)
    df: dict[str, int] = {}
    for term in set(q_terms):
        df[term] = sum(1 for _, toks in docs if term in toks)
    scored = []
    for cid, toks in docs:
        score = 0.0
        for term in q_terms:
            tf = toks.count(term)
            if tf == 0 or df[term] == 0:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(toks) / avg_len))
        if score > 0:
            scored.append((score, cid))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [cid for _, cid in scored]


def _vector_ranks(
    conn: sqlite3.Connection,
    query: str,
    embed_fn: Callable[[Sequence[str]], Sequence[Sequence[float]]] | None,
    trace: dict,
) -> dict[str, int]:
    if embed_fn is None:
        trace["channels"]["vec"] = {"used": False, "reason": "no embed_fn"}
        return {}
    try:
        from backend.services.knowledge.corpus.index import try_load_sqlite_vec

        if not try_load_sqlite_vec(conn):
            trace["channels"]["vec"] = {"used": False, "reason": "sqlite-vec unavailable"}
            return {}
        vec = json.dumps([float(x) for x in embed_fn([query])[0]])
        rows = conn.execute(
            "SELECT chunk_id FROM lit_chunks_vec WHERE embedding MATCH ? AND k = ?"
            " ORDER BY distance",
            (vec, SEED_K),
        ).fetchall()
        trace["channels"]["vec"] = {"used": True, "hits": len(rows)}
        return {r["chunk_id"]: i for i, r in enumerate(rows)}
    except Exception:  # noqa: BLE001 — dense lane is optional
        logger.debug("retrieval: vector channel failed", exc_info=True)
        trace["channels"]["vec"] = {"used": False, "reason": "error"}
        return {}


def _expansion_ranks(
    corpus: CorpusStore,
    conn: sqlite3.Connection,
    query: str,
    fused: dict[str, float],
    bm25_ranks: dict[str, int],
    target_id: str | None,
    trace: dict,
) -> dict[str, int]:
    """1-hop citation neighbours of seed papers + entity-linked papers.

    For each expanded paper not already represented in the fused seed set,
    contribute its first chunk (seq 0 — abstract/opening) as a
    ``graph_expand`` channel hit. Deterministic ordering.
    """
    try:
        seed_papers: list[str] = []
        for chunk_id, _score in sorted(fused.items(), key=lambda t: (-t[1], t[0])):
            paper_id = chunk_id.rsplit("#c", 1)[0]
            if paper_id not in seed_papers:
                seed_papers.append(paper_id)
            if len(seed_papers) >= _EXPAND_SEED_PAPERS:
                break
        if target_id and target_id not in seed_papers:
            seed_papers.append(target_id)

        expanded: list[str] = []
        for paper_id in seed_papers:
            for n in corpus.neighbors(paper_id, depth=1):
                if n["id"] not in seed_papers and n["id"] not in expanded:
                    expanded.append(n["id"])
        expanded.extend(
            p for p in _entity_linked_papers(conn, query) if p not in expanded
        )

        seeded_papers = {c.rsplit("#c", 1)[0] for c in fused}
        ranks: dict[str, int] = {}
        rank = 0
        for paper_id in expanded:
            if paper_id in seeded_papers:
                continue
            row = conn.execute(
                "SELECT chunk_id FROM lit_chunks WHERE paper_id = ?"
                " ORDER BY seq LIMIT 1",
                (paper_id,),
            ).fetchone()
            if row is None:
                continue
            ranks[row["chunk_id"]] = rank
            rank += 1
        trace["channels"]["graph_expand"] = {
            "used": True, "seed_papers": len(seed_papers), "added": len(ranks)
        }
        return ranks
    except Exception:  # noqa: BLE001 — expansion is additive; seeds must survive
        logger.debug("retrieval: expansion failed", exc_info=True)
        trace["channels"]["graph_expand"] = {"used": False, "reason": "error"}
        return {}


def _entity_linked_papers(conn: sqlite3.Connection, query: str) -> list[str]:
    """Deterministic dictionary entity linking: exact token/alias match.

    Single-word names match against the query's token set; multi-word names
    ("natural questions", "the pile") match as a token-boundary phrase in the
    whitespace-normalized query. Lane-A only: restricted to
    ``source='deterministic'`` entity rows.
    """
    query_tokens = [t.lower() for t in _TOKEN_RE.findall(query)]
    if not query_tokens:
        return []
    tokens = set(query_tokens)
    padded_query = f" {' '.join(query_tokens)} "

    def _name_matches(name: str) -> bool:
        name_tokens = _TOKEN_RE.findall(name.lower())
        if not name_tokens:
            return False
        if len(name_tokens) == 1:
            return name_tokens[0] in tokens
        return f" {' '.join(name_tokens)} " in padded_query

    papers: list[str] = []
    try:
        rows = conn.execute(
            "SELECT paper_id, name, aliases FROM lit_entities"
            " WHERE source = 'deterministic' ORDER BY paper_id, kind, name"
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    for row in rows:
        names = {str(row["name"]).lower()}
        try:
            names.update(str(a).lower() for a in json.loads(row["aliases"] or "[]"))
        except (ValueError, TypeError):
            pass
        if row["paper_id"] not in papers and any(_name_matches(n) for n in names):
            papers.append(row["paper_id"])
    return papers


def _chunk_meta(conn: sqlite3.Connection, chunk_ids: list[str]) -> dict[str, dict]:
    if not chunk_ids:
        return {}
    meta: dict[str, dict] = {}
    for i in range(0, len(chunk_ids), 500):
        batch = chunk_ids[i : i + 500]
        marks = ",".join("?" * len(batch))
        for row in conn.execute(
            f"SELECT c.chunk_id, c.paper_id, c.section, c.text, p.title"
            f" FROM lit_chunks c LEFT JOIN lit_papers p ON p.id = c.paper_id"
            f" WHERE c.chunk_id IN ({marks})",
            batch,
        ).fetchall():
            meta[row["chunk_id"]] = {
                "paper_id": row["paper_id"],
                "section": row["section"],
                "text": row["text"],
                "title": row["title"] or "",
            }
    return meta


def _write_trace(trace_path: Path, trace: dict) -> None:
    try:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = trace_path.with_suffix(trace_path.suffix + ".tmp")
        tmp.write_text(json.dumps(trace, indent=2), encoding="utf-8")
        tmp.replace(trace_path)
    except Exception:  # noqa: BLE001 — the trace must never break retrieval
        logger.debug("retrieval: trace write failed", exc_info=True)


__all__ = ["RRF_K", "SEED_K", "retrieve"]
