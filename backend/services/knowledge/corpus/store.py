"""SQLite-backed corpus store — the ``lit_*`` tables under ``runs/_corpus/``.

Layout (papers are immutable, so the store is GLOBAL and shared across runs):

    runs/_corpus/
      corpus.db                # this module's SQLite file (lit_* tables)
      papers/<dir_id>/         # per-paper artifacts: raw_paper.pdf,
                               # parsed_full_text.txt, meta.json

Deliberately a SEPARATE file from the CQRS event store and ``openresearch.db``
(mirrors the ``event_store_`` table-prefix convention: ``lit_`` namespacing +
its own file means zero collision surface). Raw ``sqlite3`` + idempotent
``executescript`` schema, same idiom as ``backend/persistence/database.py``.

The store holds FACTS only (papers + citation edges). Per-target ranking and
relevance live in each run's ``rlm_state/literature_spec.json`` — a corpus
shared by many targets must not bake one target's relevance into global rows.

``lit_entities`` / ``lit_results`` are created now (schema stability for the
plausibility-band phase) but stay unpopulated until the grounded-extraction
slice lands — nothing reads them yet.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

_CORPUS_DIRNAME = "_corpus"
_DB_FILENAME = "corpus.db"
_PAPERS_DIRNAME = "papers"

# Relation kinds on lit_relations rows (src --kind--> dst).
RELATION_CITES = "cites"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS lit_papers (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    year INTEGER,
    venue TEXT,
    arxiv_id TEXT,
    doi TEXT,
    s2_id TEXT,
    openalex_id TEXT,
    corpus_id TEXT,
    abstract TEXT,
    fetched_level TEXT NOT NULL DEFAULT 'metadata'
);

CREATE TABLE IF NOT EXISTS lit_relations (
    src_id TEXT NOT NULL,
    dst_id TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'cites',
    context TEXT,
    intent TEXT,
    PRIMARY KEY (src_id, dst_id, kind)
);

CREATE TABLE IF NOT EXISTS lit_entities (
    paper_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    aliases TEXT NOT NULL DEFAULT '[]',
    source TEXT NOT NULL DEFAULT 'deterministic',
    PRIMARY KEY (paper_id, kind, name)
);

CREATE TABLE IF NOT EXISTS lit_results (
    paper_id TEXT NOT NULL,
    method TEXT NOT NULL,
    dataset TEXT NOT NULL,
    metric TEXT NOT NULL,
    value REAL NOT NULL,
    span_quote TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT 'llm',
    PRIMARY KEY (paper_id, method, dataset, metric)
);

CREATE TABLE IF NOT EXISTS lit_graph_metrics (
    target_id TEXT NOT NULL,
    paper_id TEXT NOT NULL,
    metric TEXT NOT NULL,
    value REAL NOT NULL,
    PRIMARY KEY (target_id, paper_id, metric)
);

CREATE INDEX IF NOT EXISTS idx_lit_papers_arxiv ON lit_papers(arxiv_id);
CREATE INDEX IF NOT EXISTS idx_lit_relations_src ON lit_relations(src_id);
CREATE INDEX IF NOT EXISTS idx_lit_relations_dst ON lit_relations(dst_id, kind);
CREATE INDEX IF NOT EXISTS idx_lit_relations_kind_src ON lit_relations(kind, src_id);
CREATE INDEX IF NOT EXISTS idx_lit_results_dataset_metric
    ON lit_results(dataset, metric);
"""

# Guarded column migrations for corpus.db files created by the v1 schema
# (executescript's IF NOT EXISTS never adds columns to existing tables).
_COLUMN_MIGRATIONS: tuple[tuple[str, str, str], ...] = (
    ("lit_papers", "openalex_id", "ALTER TABLE lit_papers ADD COLUMN openalex_id TEXT"),
    ("lit_papers", "corpus_id", "ALTER TABLE lit_papers ADD COLUMN corpus_id TEXT"),
    ("lit_relations", "context", "ALTER TABLE lit_relations ADD COLUMN context TEXT"),
    ("lit_relations", "intent", "ALTER TABLE lit_relations ADD COLUMN intent TEXT"),
    (
        "lit_entities",
        "source",
        "ALTER TABLE lit_entities ADD COLUMN source TEXT NOT NULL DEFAULT 'deterministic'",
    ),
    (
        "lit_results",
        "source",
        "ALTER TABLE lit_results ADD COLUMN source TEXT NOT NULL DEFAULT 'llm'",
    ),
)

_ARXIV_VERSION_RE = re.compile(r"v\d+$")
_UNSAFE_DIR_CHARS_RE = re.compile(r"[^A-Za-z0-9._-]")


def corpus_root(runs_root: Path) -> Path:
    """The global corpus directory for a given ``runs/`` root."""
    return Path(runs_root) / _CORPUS_DIRNAME


def normalize_paper_id(
    *,
    arxiv_id: str | None = None,
    doi: str | None = None,
    s2_id: str | None = None,
) -> str | None:
    """Canonical corpus id for a paper: ``arxiv:<id>`` > ``doi:<lower>`` > ``s2:<id>``.

    The arXiv version suffix is stripped (``2101.00001v3`` and ``2101.00001``
    are the same paper for corpus purposes). Returns ``None`` when no usable
    identifier is present — callers skip such entries rather than minting
    title-keyed ids that would collide across preprint revisions.
    """
    aid = (arxiv_id or "").strip()
    if aid:
        return f"arxiv:{_ARXIV_VERSION_RE.sub('', aid)}"
    d = (doi or "").strip().lower()
    if d:
        return f"doi:{d}"
    sid = (s2_id or "").strip()
    if sid:
        return f"s2:{sid}"
    return None


class CorpusStore:
    """Thin raw-sqlite wrapper over ``runs/_corpus/corpus.db``.

    Not thread-hardened — Phase 1 callers (CLI, campaign UNDERSTAND) are
    single-threaded; per-run readers only touch the JSON spec/manifest, never
    this connection.
    """

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._conn: sqlite3.Connection | None = None

    @property
    def root(self) -> Path:
        return self._root

    @property
    def connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._root.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self._root / _DB_FILENAME)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def initialize(self) -> None:
        """Create all ``lit_*`` tables + apply guarded column migrations. Idempotent."""
        conn = self.connection
        conn.executescript(_SCHEMA)
        for table, column, ddl in _COLUMN_MIGRATIONS:
            existing = {
                row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            if column not in existing:
                conn.execute(ddl)
        conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # --- papers -------------------------------------------------------------

    def upsert_paper(
        self,
        paper_id: str,
        *,
        title: str = "",
        year: int | None = None,
        venue: str | None = None,
        arxiv_id: str | None = None,
        doi: str | None = None,
        s2_id: str | None = None,
        abstract: str | None = None,
        fetched_level: str | None = None,
    ) -> None:
        """Insert or enrich a paper row.

        Enrich-only semantics: an existing non-empty column is never
        overwritten with an empty/None incoming value, and ``fetched_level``
        can only ratchet ``metadata`` -> ``fulltext`` (a re-run with less
        information must not degrade the row).
        """
        conn = self.connection
        existing = conn.execute(
            "SELECT * FROM lit_papers WHERE id = ?", (paper_id,)
        ).fetchone()
        if existing is None:
            conn.execute(
                "INSERT INTO lit_papers"
                " (id, title, year, venue, arxiv_id, doi, s2_id, abstract, fetched_level)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    paper_id,
                    title or "",
                    year,
                    venue,
                    arxiv_id,
                    doi,
                    s2_id,
                    abstract,
                    fetched_level or "metadata",
                ),
            )
        else:
            merged = {
                "title": title or existing["title"],
                "year": year if year is not None else existing["year"],
                "venue": venue or existing["venue"],
                "arxiv_id": arxiv_id or existing["arxiv_id"],
                "doi": doi or existing["doi"],
                "s2_id": s2_id or existing["s2_id"],
                "abstract": abstract or existing["abstract"],
                "fetched_level": (
                    "fulltext"
                    if "fulltext" in (fetched_level, existing["fetched_level"])
                    else "metadata"
                ),
            }
            conn.execute(
                "UPDATE lit_papers SET title=?, year=?, venue=?, arxiv_id=?, doi=?,"
                " s2_id=?, abstract=?, fetched_level=? WHERE id=?",
                (*merged.values(), paper_id),
            )
        conn.commit()

    def get_paper(self, paper_id: str) -> dict | None:
        row = self.connection.execute(
            "SELECT * FROM lit_papers WHERE id = ?", (paper_id,)
        ).fetchone()
        return dict(row) if row is not None else None

    def paper_count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM lit_papers").fetchone()[0])

    # --- relations ----------------------------------------------------------

    def add_relation(self, src_id: str, dst_id: str, *, kind: str = RELATION_CITES) -> None:
        """Record ``src --kind--> dst``. Idempotent (PK dedupe)."""
        conn = self.connection
        conn.execute(
            "INSERT OR IGNORE INTO lit_relations (src_id, dst_id, kind) VALUES (?, ?, ?)",
            (src_id, dst_id, kind),
        )
        conn.commit()

    def relation_count(self) -> int:
        return int(
            self.connection.execute("SELECT COUNT(*) FROM lit_relations").fetchone()[0]
        )

    def edges(self, *, kind: str = RELATION_CITES) -> list[tuple[str, str]]:
        """All ``(src, dst)`` pairs of one kind, deterministically ordered."""
        return [
            (r["src_id"], r["dst_id"])
            for r in self.connection.execute(
                "SELECT src_id, dst_id FROM lit_relations WHERE kind = ?"
                " ORDER BY src_id, dst_id",
                (kind,),
            ).fetchall()
        ]

    def neighbors(
        self,
        paper_id: str,
        *,
        depth: int = 1,
        kind: str = RELATION_CITES,
        limit: int = 200,
    ) -> list[dict]:
        """K-hop neighbourhood of ``paper_id`` over ``kind`` edges, both directions.

        Recursive-CTE breadth-first walk with a hop cap (``depth``, clamped to
        2) and a row ``limit`` — at corpus scale (10^2 nodes) this is
        microseconds. Returns ``[{"id", "hop", "direction"}]``: hop is the
        minimum hop the node was reached at; direction is ``out`` (reached via
        an outgoing/cites edge) or ``in`` (via an incoming/cited-by edge),
        with ``in`` reported when both apply. Deterministic ordering (hop, id).
        """
        d = max(1, min(int(depth), 2))
        rows = self.connection.execute(
            """
            WITH RECURSIVE walk(id, hop, direction) AS (
                SELECT ?, 0, ''
                UNION
                SELECT CASE WHEN r.src_id = w.id THEN r.dst_id ELSE r.src_id END,
                       w.hop + 1,
                       CASE WHEN r.src_id = w.id THEN 'out' ELSE 'in' END
                FROM lit_relations r
                JOIN walk w ON w.hop < ? AND r.kind = ?
                    AND (r.src_id = w.id OR r.dst_id = w.id)
            )
            SELECT id, MIN(hop) AS hop,
                   MIN(direction) AS direction
            FROM walk
            WHERE id != ?
            GROUP BY id
            ORDER BY hop, id
            LIMIT ?
            """,
            (paper_id, d, kind, paper_id, max(1, int(limit))),
        ).fetchall()
        return [dict(r) for r in rows]

    # --- graph metrics -------------------------------------------------------

    def put_graph_metrics(
        self, target_id: str, metrics: dict[str, dict[str, float]]
    ) -> None:
        """Replace the per-target metric rows: ``{paper_id: {metric: value}}``."""
        conn = self.connection
        conn.execute("DELETE FROM lit_graph_metrics WHERE target_id = ?", (target_id,))
        conn.executemany(
            "INSERT OR REPLACE INTO lit_graph_metrics"
            " (target_id, paper_id, metric, value) VALUES (?, ?, ?, ?)",
            [
                (target_id, paper_id, metric, float(value))
                for paper_id, per_paper in sorted(metrics.items())
                for metric, value in sorted(per_paper.items())
            ],
        )
        conn.commit()

    def get_graph_metrics(self, target_id: str) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for row in self.connection.execute(
            "SELECT paper_id, metric, value FROM lit_graph_metrics WHERE target_id = ?",
            (target_id,),
        ).fetchall():
            out.setdefault(row["paper_id"], {})[row["metric"]] = float(row["value"])
        return out

    # --- entities + results (the structured index) ----------------------------

    def put_entity(
        self,
        paper_id: str,
        *,
        kind: str,
        name: str,
        aliases: "list[str] | None" = None,
        source: str = "deterministic",
    ) -> None:
        """Upsert one entity row. ``source`` ∈ {deterministic, llm} (lane rule).

        Lane rule enforced here: an ``llm`` row never replaces an existing
        ``deterministic`` row for the same ``(paper_id, kind, name)`` — the
        deterministic dictionary is the closed vocabulary the retrieval
        entity-linker reads, and a Lane-B pass must not erode it.
        """
        self.connection.execute(
            "INSERT INTO lit_entities (paper_id, kind, name, aliases, source)"
            " VALUES (?, ?, ?, ?, ?)"
            " ON CONFLICT(paper_id, kind, name) DO UPDATE SET"
            " aliases = excluded.aliases, source = excluded.source"
            " WHERE NOT (lit_entities.source = 'deterministic'"
            "            AND excluded.source = 'llm')",
            (paper_id, kind, name, json.dumps(sorted(aliases or [])), source),
        )
        self.connection.commit()

    def put_result(
        self,
        paper_id: str,
        *,
        method: str,
        dataset: str,
        metric: str,
        value: float,
        span_quote: str,
        source: str = "llm",
    ) -> None:
        """Upsert one reported-result row. ``span_quote`` is the grounding proof."""
        self.connection.execute(
            "INSERT OR REPLACE INTO lit_results"
            " (paper_id, method, dataset, metric, value, span_quote, source)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (paper_id, method, dataset, metric, float(value), span_quote[:300], source),
        )
        self.connection.commit()

    def entity_count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM lit_entities").fetchone()[0])

    def result_count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM lit_results").fetchone()[0])

    # --- per-paper artifact dirs ---------------------------------------------

    def paper_dir(self, paper_id: str, *, create: bool = False) -> Path:
        """Deterministic artifact directory for a corpus paper.

        Sanitized id + short content hash: readable AND collision-free after
        sanitization (two DOIs may sanitize identically; the hash suffix keeps
        their directories distinct).
        """
        digest = hashlib.sha256(paper_id.encode("utf-8")).hexdigest()[:8]
        safe = _UNSAFE_DIR_CHARS_RE.sub("_", paper_id)[:60]
        path = self._root / _PAPERS_DIRNAME / f"{safe}-{digest}"
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def write_paper_meta(self, paper_id: str, meta: dict) -> None:
        """Atomic ``meta.json`` write into the paper's artifact dir. Fail-soft."""
        try:
            target = self.paper_dir(paper_id, create=True) / "meta.json"
            tmp = target.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
            tmp.replace(target)
        except Exception:  # noqa: BLE001 — artifact mirror must never block the build
            logger.debug("corpus: meta.json write failed for %s", paper_id, exc_info=True)


__all__ = [
    "CorpusStore",
    "RELATION_CITES",
    "corpus_root",
    "normalize_paper_id",
]
