"""Chunk + lexical (+ optional dense) index over corpus paper texts.

Everything lives inside ``corpus.db`` — one file, one writer, WAL — following
the survey's storage verdict:

  - ``lit_chunks``      — section-aware chunks (deterministic chunker below).
  - ``lit_chunks_fts``  — SQLite **FTS5** virtual table with native ``bm25()``,
    created behind a capability probe (FTS5 availability depends on the
    bundled sqlite3's compile flags); absent ⇒ retrieval falls back to the
    pure-Python BM25 in ``retrieval.py``.
  - ``lit_chunks_vec``  — OPTIONAL dense vectors via the ``sqlite-vec``
    loadable extension (``pip install openresearch[kg]``, pinned 0.1.x
    stable). Never loaded implicitly: vectors are only written when the
    caller passes ``embed_fn`` (production wires the existing 384-d
    all-MiniLM path; tests inject fixtures — the ONNX model download would be
    blocked by the socket-hermetic suite anyway).

Chunker determinism: pure function of the text — paragraph-packed windows of
``CHUNK_CHARS`` with a lightweight section-heading tracker. Two identical
builds produce byte-identical chunk rows (a determinism test enforces this).
"""

from __future__ import annotations

import logging
import re
import sqlite3
from typing import Callable, Sequence

from backend.services.knowledge.corpus.store import CorpusStore

logger = logging.getLogger(__name__)

CHUNK_CHARS = 1500
MAX_CHUNKS_PER_PAPER = 120
EMBED_DIM = 384  # all-MiniLM-L6-v2 — the repo's existing local embedding model

_HEADING_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*\s+)?"
    r"(abstract|introduction|related work|background|method(?:s|ology)?|"
    r"approach|experiments?|results?|evaluation|discussion|conclusion(?:s)?|"
    r"limitations|appendix|references)\b.{0,40}$",
    re.IGNORECASE,
)

_CHUNK_SCHEMA = """
CREATE TABLE IF NOT EXISTS lit_chunks (
    chunk_id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    section TEXT NOT NULL DEFAULT 'body',
    text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_lit_chunks_paper ON lit_chunks(paper_id);
"""


def chunk_text(text: str) -> list[tuple[str, str]]:
    """Deterministic section-aware chunking: ``[(section, chunk_text)]``.

    Paragraphs are packed into ≤``CHUNK_CHARS`` windows; a paragraph that
    looks like a section heading switches the section label for subsequent
    chunks. Oversized single paragraphs are hard-split. Bounded at
    ``MAX_CHUNKS_PER_PAPER``.
    """
    chunks: list[tuple[str, str]] = []
    section = "body"
    buffer: list[str] = []
    buffer_len = 0

    def _flush() -> None:
        nonlocal buffer, buffer_len
        if buffer:
            chunks.append((section, "\n\n".join(buffer)))
            buffer, buffer_len = [], 0

    for para in re.split(r"\n\s*\n", text or ""):
        para = para.strip()
        if not para or len(chunks) >= MAX_CHUNKS_PER_PAPER:
            continue
        heading = _HEADING_RE.match(para.splitlines()[0].strip())
        if heading and len(para) < 200:
            _flush()
            section = heading.group(1).lower()
            continue
        if buffer_len + len(para) > CHUNK_CHARS:
            _flush()
        while len(para) > CHUNK_CHARS:  # oversized paragraph: hard split
            chunks.append((section, para[:CHUNK_CHARS]))
            para = para[CHUNK_CHARS:]
            if len(chunks) >= MAX_CHUNKS_PER_PAPER:
                return chunks
        if para:
            buffer.append(para)
            buffer_len += len(para)
    _flush()
    return chunks[:MAX_CHUNKS_PER_PAPER]


def fts5_available(conn: sqlite3.Connection) -> bool:
    """Capability probe — FTS5 depends on the bundled sqlite3's compile flags."""
    try:
        conn.execute("CREATE VIRTUAL TABLE temp.__fts5_probe USING fts5(x)")
        conn.execute("DROP TABLE temp.__fts5_probe")
        return True
    except sqlite3.OperationalError:
        return False


def try_load_sqlite_vec(conn: sqlite3.Connection) -> bool:
    """Load the sqlite-vec extension iff installed AND loading is supported."""
    if not hasattr(conn, "enable_load_extension"):
        return False
    try:
        import sqlite_vec
    except Exception:  # noqa: BLE001 — optional extra absent
        return False
    try:
        conn.enable_load_extension(True)
        try:
            sqlite_vec.load(conn)
        finally:
            conn.enable_load_extension(False)
        return True
    except Exception:  # noqa: BLE001 — distro sqlite without loadable-extension support
        logger.debug("corpus index: sqlite-vec load failed", exc_info=True)
        return False


def build_chunk_index(
    corpus: CorpusStore,
    *,
    embed_fn: Callable[[Sequence[str]], Sequence[Sequence[float]]] | None = None,
) -> dict:
    """(Re)build ``lit_chunks`` (+ FTS5, + vectors when ``embed_fn`` given).

    Chunks every corpus paper that has a ``parsed_full_text.txt``; papers
    without full text contribute their abstract as a single chunk. Idempotent:
    per-paper delete-and-reinsert, so re-builds after corpus growth are safe.
    Returns a small stats dict for the build report/CLI.
    """
    conn = corpus.connection
    conn.executescript(_CHUNK_SCHEMA)
    has_fts = fts5_available(conn)
    if has_fts:
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS lit_chunks_fts USING fts5("
            "text, chunk_id UNINDEXED, paper_id UNINDEXED, section UNINDEXED)"
        )
    has_vec = try_load_sqlite_vec(conn) if embed_fn is not None else False
    if has_vec:
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS lit_chunks_vec USING vec0("
            f"embedding float[{EMBED_DIM}], +chunk_id text)"
        )

    stats = {"papers": 0, "chunks": 0, "fts5": has_fts, "vec": has_vec}
    for row in conn.execute(
        "SELECT id, abstract, fetched_level FROM lit_papers ORDER BY id"
    ).fetchall():
        paper_id = row["id"]
        text = ""
        if row["fetched_level"] == "fulltext":
            path = corpus.paper_dir(paper_id) / "parsed_full_text.txt"
            try:
                text = path.read_text(encoding="utf-8") if path.exists() else ""
            except OSError:
                text = ""
        chunks = chunk_text(text) if text else (
            [("abstract", row["abstract"])] if row["abstract"] else []
        )
        if not chunks:
            continue
        _reindex_paper(conn, paper_id, chunks, has_fts, has_vec, embed_fn)
        stats["papers"] += 1
        stats["chunks"] += len(chunks)
    conn.commit()
    return stats


def _reindex_paper(
    conn: sqlite3.Connection,
    paper_id: str,
    chunks: list[tuple[str, str]],
    has_fts: bool,
    has_vec: bool,
    embed_fn: Callable[[Sequence[str]], Sequence[Sequence[float]]] | None,
) -> None:
    old = [
        r["chunk_id"]
        for r in conn.execute(
            "SELECT chunk_id FROM lit_chunks WHERE paper_id = ?", (paper_id,)
        ).fetchall()
    ]
    conn.execute("DELETE FROM lit_chunks WHERE paper_id = ?", (paper_id,))
    if has_fts and old:
        conn.executemany(
            "DELETE FROM lit_chunks_fts WHERE chunk_id = ?", [(c,) for c in old]
        )
    if has_vec and old:
        # vec0 aux (+) columns can't be constrained in WHERE — scan rowids.
        old_set = set(old)
        stale = [
            (row[0],)
            for row in conn.execute("SELECT rowid, chunk_id FROM lit_chunks_vec")
            if row[1] in old_set
        ]
        if stale:
            conn.executemany("DELETE FROM lit_chunks_vec WHERE rowid = ?", stale)

    rows = [
        (f"{paper_id}#c{seq}", paper_id, seq, section, text)
        for seq, (section, text) in enumerate(chunks)
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO lit_chunks (chunk_id, paper_id, seq, section, text)"
        " VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    if has_fts:
        conn.executemany(
            "INSERT INTO lit_chunks_fts (text, chunk_id, paper_id, section)"
            " VALUES (?, ?, ?, ?)",
            [(text, cid, pid, section) for cid, pid, _seq, section, text in rows],
        )
    if has_vec and embed_fn is not None:
        try:
            import json as _json

            vectors = embed_fn([text for *_ignored, text in rows])
            conn.executemany(
                "INSERT OR REPLACE INTO lit_chunks_vec (embedding, chunk_id)"
                " VALUES (?, ?)",
                [
                    (_json.dumps(list(map(float, vec))), cid)
                    for (cid, *_rest), vec in zip(rows, vectors)
                ],
            )
        except Exception:  # noqa: BLE001 — dense lane is optional; lexical must survive
            logger.debug("corpus index: embedding pass failed for %s", paper_id, exc_info=True)


__all__ = [
    "CHUNK_CHARS",
    "EMBED_DIM",
    "MAX_CHUNKS_PER_PAPER",
    "build_chunk_index",
    "chunk_text",
    "fts5_available",
    "try_load_sqlite_vec",
]
