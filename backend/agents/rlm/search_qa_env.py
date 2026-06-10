"""Real multi-turn retrieval QA environment for SDAR Search-QA (2026-06-01).

The 2026-05-31 failure (`prj_09047604e591d969`).  Search-QA was reproduced as a
*closed-book* task — the prompt was literally ``Question:\n\nAnswer:`` with no
retrieval surface — so a 1.7B model could only fall back on parametric recall and
floored at ~0.05 token-F1 against the paper's 0.38–0.46.  The HotpotQA loader made
it worse: it discarded ``row["context"]`` entirely, throwing away the gold
supporting passages the model needed to read.

This module restores the paper's actual agentic loop: the policy issues
``search(<query>)`` actions, reads the top-k retrieved passages as an
*Observation*, optionally searches again, then commits an ``answer(<text>)`` whose
reward is the max-over-aliases SQuAD token-F1.  Two structural fixes land here:

* **HotpotQA contexts are kept.**  When a task carries ``contexts`` they seed a
  per-question candidate pool that the retriever can draw on even with no global
  index — directly undoing the loader bug.
* **Retrieval always returns something real.**  A three-tier :class:`Retriever`
  degrades gracefully: cached dense E5 over wiki-18 → BM25 over the candidate pool
  → a pure-python lexical-overlap ranker.  The grid never stalls on a cold index
  or a missing library, and the chosen backend is surfaced in ``last_info`` for
  the rubric/aggregator.

Copyable helper — mirror of the ``sdar_env_base.py`` / ``gpu_cell_runner.py``
pattern.  ``run_with_sdk`` copies this file flat into ``code/`` and the agent's
trainer imports it as a top-level module::

    from search_qa_env import SearchQAEnv, load_search_qa_tasks

So the import here is the FLAT ``from sdar_env_base import ...`` (NOT the
``backend.agents.rlm`` package path).  Every heavy dependency
(sentence-transformers, faiss, rank_bm25, datasets) is lazy-imported *inside* the
method that needs it, so the module imports cleanly on a venv that lacks them and
the unit test runs offline with injected fakes.  Fail-soft by construction: no
method raises on a malformed action, a failed retrieval, or a missing dependency —
a bad turn yields a nudge observation or a zero-reward terminal step, never an
exception that kills the cell (spec §0.3, §3).
"""

from __future__ import annotations

import os
import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from sdar_env_base import AgenticEnv, StepResult

__all__ = [
    "SearchQAEnv",
    "Retriever",
    "Passage",
    "normalize_answer",
    "token_f1",
    "exact_match",
    "load_search_qa_tasks",
]


# ---------------------------------------------------------------------------
# SQuAD-style scoring helpers (module-level — reused by the trainer/aggregator).
# ---------------------------------------------------------------------------

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(s: str) -> str:
    """SQuAD answer normalisation: lowercase, strip punctuation/articles, squash ws.

    The canonical reference implementation (Rajpurkar et al. 2016): lowercases,
    removes punctuation, removes the articles ``a``/``an``/``the``, and collapses
    runs of whitespace.  Used for both EM and token-F1 so the two agree on
    tokenisation.
    """
    if s is None:
        return ""
    text = str(s).lower()
    text = text.translate(_PUNCT_TABLE)
    text = _ARTICLES_RE.sub(" ", text)
    return " ".join(text.split())


def _f1_single(pred: str, gold: str) -> float:
    """Token-level F1 between one prediction and one gold (both pre-normalised)."""
    pred_tokens = pred.split()
    gold_tokens = gold.split()
    # SQuAD convention: if either side is empty, F1 is 1.0 iff both are empty
    # (e.g. both normalise to "yes"/"no" → handled above; both empty → match).
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def token_f1(pred: str, golds: Sequence[str] | str) -> float:
    """Max token-F1 of ``pred`` over a list of acceptable gold answers (aliases).

    Mirrors the SQuAD ``metric_max_over_ground_truths`` aggregation: a question
    may have several valid surface forms, so we score against the best.
    """
    if golds is None:
        return 0.0
    if isinstance(golds, str):
        golds = [golds]
    norm_pred = normalize_answer(pred)
    best = 0.0
    for gold in golds:
        score = _f1_single(norm_pred, normalize_answer(gold))
        if score > best:
            best = score
    return best


def exact_match(pred: str, golds: Sequence[str] | str) -> int:
    """1 if normalised ``pred`` equals any normalised gold, else 0 (SQuAD EM)."""
    if golds is None:
        return 0
    if isinstance(golds, str):
        golds = [golds]
    norm_pred = normalize_answer(pred)
    return int(any(norm_pred == normalize_answer(g) for g in golds))


# ---------------------------------------------------------------------------
# Retrieval backend — three tiers, always returns real passages.
# ---------------------------------------------------------------------------


@dataclass
class Passage:
    """One retrieved passage with its retrieval score (higher == more relevant)."""

    text: str
    score: float = 0.0
    title: str = ""


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokens for the lexical/BM25 paths (regex word boundaries)."""
    return re.findall(r"\w+", (text or "").lower())


class Retriever:
    """Three-tier retriever selected at construction from env vars + availability.

    Resolution order (spec §3):

    1. **Dense E5** — when ``SEARCH_QA_INDEX_DIR`` points at a loadable cached
       FAISS index + passage store and ``sentence_transformers`` + ``faiss`` import.
       Queries are encoded with ``intfloat/e5-base-v2`` under the ``"query: "``
       prefix the model was trained with.
    2. **BM25** — ``rank_bm25.BM25Okapi`` over the per-question candidate pool (the
       task's own ``contexts``, plus any global corpus passed in).  Chosen when the
       dense index is unavailable OR ``SEARCH_QA_RETRIEVER=bm25``.
    3. **Lexical overlap** — a dependency-free Jaccard-ish ranker over the same
       pool, used when ``rank_bm25`` is also absent.

    The backend is resolved lazily and *per query corpus*: the dense index is the
    same global store for every question, but BM25/overlap rebuild over each
    question's candidate pool (so HotpotQA's gold contexts are searchable even with
    no global index).  ``backend`` is one of ``"e5" | "bm25" | "overlap"`` and is
    surfaced to the env for ``last_info``.  Heavy imports happen inside the dense
    path only — constructing a :class:`Retriever` never imports them.
    """

    def __init__(
        self,
        *,
        index_dir: str | None = None,
        prefer: str | None = None,
        corpus: Sequence[str] | None = None,
        _encoder: Any = None,
        _faiss_index: Any = None,
        _passages: Sequence[str] | None = None,
    ) -> None:
        # ``prefer`` mirrors SEARCH_QA_RETRIEVER; "bm25" skips the dense tier.
        self._index_dir = index_dir
        self._prefer = (prefer or "").strip().lower() or None
        self._global_corpus: list[str] = [str(c) for c in (corpus or []) if c]
        self._logged = False

        # Dense state (populated lazily on first use, or injected by tests).
        self._encoder = _encoder
        self._faiss_index = _faiss_index
        self._passages: list[str] | None = (
            [str(p) for p in _passages] if _passages is not None else None
        )
        self._dense_ready: bool | None = (
            True if (_encoder is not None and _faiss_index is not None) else None
        )
        self._dense_disabled = self._prefer == "bm25"

        # The resolved backend label for the *most recent* retrieve() call.
        self.backend: str = "overlap"

    # --- public API ------------------------------------------------------

    def retrieve(self, query: str, *, pool: Sequence[str] | None = None, k: int = 3) -> list[Passage]:
        """Return up to ``k`` passages for ``query`` from the best available tier.

        ``pool`` is the per-question candidate pool (HotpotQA contexts etc.); it is
        merged with any global corpus for the BM25/overlap tiers.  Never raises —
        any backend failure degrades to the next tier, and the final tier is pure
        python, so the worst case still returns real passages (or ``[]`` only when
        there is genuinely nothing to search).
        """
        q = (query or "").strip()
        # 1) Dense E5 over the cached global index.
        if not self._dense_disabled and self._index_dir:
            try:
                hits = self._dense_retrieve(q, k=k)
                if hits is not None:
                    self.backend = "e5"
                    self._log_once()
                    return hits
            except Exception as exc:  # noqa: BLE001 — fail-soft to BM25/overlap
                self._dense_ready = False
                print(f"[search_qa] dense E5 retrieval failed ({exc!r}); falling back.")

        # Assemble the candidate pool for the lexical tiers.
        candidates = list(self._global_corpus)
        if pool:
            candidates.extend(str(p) for p in pool if p)
        # De-dup while preserving order (HotpotQA repeats title-prefixed paras).
        seen: set[str] = set()
        deduped = [c for c in candidates if not (c in seen or seen.add(c))]

        # 2) BM25 over the candidate pool.
        bm25_hits = self._bm25_retrieve(q, deduped, k=k)
        if bm25_hits is not None:
            self.backend = "bm25"
            self._log_once()
            return bm25_hits

        # 3) Pure-python lexical overlap (always available).
        self.backend = "overlap"
        self._log_once()
        return self._overlap_retrieve(q, deduped, k=k)

    # --- tier 1: dense E5 -------------------------------------------------

    def _ensure_dense(self) -> bool:
        """Lazily load the encoder + FAISS index + passage store.  Returns ready?

        Heavy imports (sentence_transformers, faiss) happen *here* only.  Any
        failure (missing lib, unreadable dir, malformed store) disables the dense
        tier permanently for this retriever and returns False.
        """
        if self._dense_ready is not None:
            return self._dense_ready
        try:
            import json
            import pickle

            from pathlib import Path as _Path

            from sentence_transformers import SentenceTransformer  # lazy
            import faiss  # lazy

            idx_dir = self._index_dir or ""
            root = _Path(idx_dir)

            def _find_first(patterns: list[str]) -> str | None:
                # Prefer an exact top-level match, then any recursive match — so a
                # snapshot_download that nests the index under a repo subdir, or
                # names it e.g. ``e5_Flat.index`` instead of ``index.faiss``, still
                # resolves.  This mirrors the env_cache provisioning gate, which
                # advertises E5 for ANY ``*.faiss``/``*.index`` under the dir; the
                # retriever must therefore load exactly what that gate accepts, or
                # the dense tier is advertised then silently falls back to BM25.
                for pat in patterns:
                    p = root / pat
                    if p.is_file():
                        return str(p)
                for pat in patterns:
                    hits = sorted(root.rglob(pat))
                    if hits:
                        return str(hits[0])
                return None

            # FAISS index: index.faiss → any *.faiss → any *.index. mmap-load so a
            # multi-GB index is shared across cell processes via the OS page cache
            # (one physical copy, not one per cell); fall back to a full read for an
            # index type that can't be memory-mapped.
            index_path = _find_first(["index.faiss", "*.faiss", "*.index"])
            if not index_path:
                raise FileNotFoundError(f"no FAISS index (*.faiss/*.index) under {idx_dir!r}")
            try:
                self._faiss_index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP)
            except Exception:  # noqa: BLE001 — not every index type mmaps; read fully
                self._faiss_index = faiss.read_index(index_path)

            # Passage store. A small index ships an explicit passages.{json,pkl,txt}
            # loaded into RAM. A wiki-scale corpus ships as a *.jsonl ({id, contents},
            # FAISS row i ↔ line i) accessed LAZILY by byte offset — so 21M passages
            # cost ~one seek per hit, not ~14 GB resident in every cell process.
            passages: Any = None
            json_path = _find_first(["passages.json"])
            pkl_path = _find_first(["passages.pkl"])
            txt_path = _find_first(["passages.txt"])
            if json_path:
                with open(json_path, "r", encoding="utf-8") as fh:
                    raw = json.load(fh)
                passages = [_as_passage_text(r) for r in raw]
            elif pkl_path:
                with open(pkl_path, "rb") as fh:
                    raw = pickle.load(fh)
                passages = [_as_passage_text(r) for r in raw]
            elif txt_path:
                with open(txt_path, "r", encoding="utf-8") as fh:
                    passages = [ln.rstrip("\n") for ln in fh]
            else:
                jsonl_path = _find_first(["*.jsonl"])
                if jsonl_path:
                    passages = _LazyJsonlStore(jsonl_path)
            if not passages:
                raise FileNotFoundError(f"no passage store under {idx_dir!r}")
            # Alignment guard: faiss row i must map to corpus line i. A row-count
            # mismatch means the index + corpus came from different snapshots — fail
            # to BM25 rather than silently serving wrong passages.
            _ntotal = getattr(self._faiss_index, "ntotal", None)
            if _ntotal is not None and len(passages) != _ntotal:
                raise ValueError(
                    f"dense index ntotal {_ntotal} != corpus passages {len(passages)} "
                    "— misaligned; falling back"
                )
            self._passages = passages

            # The query encoder MUST match the encoder the index was built with
            # (dimension + semantics) — configurable so a non-e5-base index works.
            # Default matches the prebuilt wiki-18 indexes (intfloat/e5-base-v2).
            if self._encoder is None:
                _enc = os.environ.get("SEARCH_QA_ENCODER", "").strip() or "intfloat/e5-base-v2"
                self._encoder = SentenceTransformer(_enc)
            self._dense_ready = True
        except Exception as exc:  # noqa: BLE001 — disable dense tier, fall back
            print(f"[search_qa] dense E5 index unavailable ({exc!r}); using fallback.")
            self._faiss_index = None  # drop the (possibly mmap'd) handle we won't use
            self._dense_ready = False
        return self._dense_ready

    def _dense_retrieve(self, query: str, *, k: int) -> list[Passage] | None:
        """E5 query encode + FAISS top-k.  Returns None if the dense tier is off."""
        if not self._ensure_dense():
            return None
        encoder = self._encoder
        index = self._faiss_index
        passages = self._passages or []
        if encoder is None or index is None or not passages:
            return None
        # E5 expects an explicit "query: " prefix at inference time.
        emb = encoder.encode(
            [f"query: {query}"], normalize_embeddings=True, convert_to_numpy=True
        )
        scores, ids = index.search(emb, max(1, k))
        out: list[Passage] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0 or idx >= len(passages):
                continue
            out.append(Passage(text=str(passages[idx]), score=float(score)))
        return out

    # --- tier 2: BM25 -----------------------------------------------------

    def _bm25_retrieve(self, query: str, pool: list[str], *, k: int) -> list[Passage] | None:
        """rank_bm25 over ``pool``.  Returns None when rank_bm25 is unavailable."""
        if not pool:
            return None
        try:
            from rank_bm25 import BM25Okapi  # lazy
        except Exception:  # noqa: BLE001 — let the caller drop to overlap
            return None
        try:
            tokenized = [_tokenize(p) for p in pool]
            bm25 = BM25Okapi(tokenized)
            scores = bm25.get_scores(_tokenize(query))
            ranked = sorted(
                range(len(pool)), key=lambda i: scores[i], reverse=True
            )[: max(1, k)]
            return [Passage(text=pool[i], score=float(scores[i])) for i in ranked]
        except Exception as exc:  # noqa: BLE001 — fail-soft to overlap
            print(f"[search_qa] BM25 retrieval failed ({exc!r}); using overlap.")
            return None

    # --- tier 3: lexical overlap (dependency-free) ------------------------

    def _overlap_retrieve(self, query: str, pool: list[str], *, k: int) -> list[Passage]:
        """Pure-python ranker: count of shared query tokens, length-normalised.

        Always available — this is the floor that guarantees the env returns real
        passages even with no libraries and no index.
        """
        if not pool:
            return []
        q_tokens = set(_tokenize(query))
        scored: list[Passage] = []
        for text in pool:
            doc_tokens = _tokenize(text)
            if not doc_tokens:
                scored.append(Passage(text=text, score=0.0))
                continue
            overlap = sum(1 for t in doc_tokens if t in q_tokens)
            # Normalise by doc length so a long passage with one hit doesn't win.
            score = overlap / (len(doc_tokens) ** 0.5)
            scored.append(Passage(text=text, score=score))
        scored.sort(key=lambda p: p.score, reverse=True)
        return scored[: max(1, k)]

    # --- diagnostics ------------------------------------------------------

    def _log_once(self) -> None:
        if not self._logged:
            print(f"[search_qa] retriever backend: {self.backend}")
            self._logged = True


def _as_passage_text(record: Any) -> str:
    """Coerce a passage-store record (str | {'text'|'contents'|'passage': ...}) to text."""
    if isinstance(record, str):
        return record
    if isinstance(record, dict):
        for key in ("text", "contents", "passage", "body"):
            if key in record and record[key]:
                title = record.get("title", "")
                body = str(record[key])
                return f"{title}. {body}".strip(". ") if title else body
    return str(record)


class _LazyJsonlStore:
    """Indexable, RAM-frugal view over a large ``{id, contents}`` JSONL corpus.

    FAISS row ``i`` corresponds to JSONL line ``i``. A byte-offset array (built once
    and cached next to the corpus as ``<name>.offsets.npy``) lets ``store[i]`` seek
    directly to line ``i`` and read just that passage, so a 21M-passage / ~14 GB wiki
    corpus costs ~168 MB of offsets + one disk seek per retrieved hit instead of
    loading the whole corpus into every cell process. Never raises on a bad line.
    """

    def __init__(self, jsonl_path: str, offsets_path: str | None = None) -> None:
        import numpy as _np  # lazy
        import threading

        self._path = str(jsonl_path)
        op = offsets_path or (self._path + ".offsets.npy")
        size_marker = self._path + ".offsets.size"
        cur_size = os.path.getsize(self._path)
        # Trust a cached offsets array ONLY if the corpus byte-size still matches —
        # a stale cache (corpus regenerated, offsets not) would seek to the wrong
        # lines and silently serve wrong passages. On any mismatch, rebuild.
        cached_ok = False
        if os.path.exists(op) and os.path.exists(size_marker):
            try:
                cached_ok = int(open(size_marker).read().strip()) == cur_size
            except Exception:  # noqa: BLE001
                cached_ok = False
        if cached_ok:
            self._offsets = _np.load(op, mmap_mode="r")
        else:
            self._offsets = self._build_offsets(self._path, op)
            try:
                with open(size_marker, "w") as _sf:
                    _sf.write(str(cur_size))
            except OSError:
                pass
        self._lock = threading.Lock()  # serialise seek+readline on the shared fd
        # Long-lived read handle for per-hit seeks (one fd per retriever instance).
        self._fh = open(self._path, "rb")  # noqa: SIM115

    @staticmethod
    def _build_offsets(path: str, offsets_path: str) -> Any:
        import numpy as _np  # lazy

        offsets: list[int] = []
        pos = 0
        with open(path, "rb") as fh:
            for line in fh:
                offsets.append(pos)
                pos += len(line)
        arr = _np.asarray(offsets, dtype=_np.int64)
        try:
            _np.save(offsets_path, arr)
        except OSError:
            pass
        return arr

    def __len__(self) -> int:
        return int(len(self._offsets))

    def __getitem__(self, i: int) -> str:
        import json  # lazy (cached after first call)

        if i < 0 or i >= len(self._offsets):
            return ""
        try:
            with self._lock:  # seek+readline must be atomic on the shared fd
                self._fh.seek(int(self._offsets[i]))
                line = self._fh.readline()
            return _as_passage_text(json.loads(line))
        except Exception:  # noqa: BLE001 — a malformed line never kills retrieval
            return ""


# ---------------------------------------------------------------------------
# Action parsing.
# ---------------------------------------------------------------------------


@dataclass
class ParsedAction:
    """Defensively-parsed model action: ``kind`` in {search, answer, none}."""

    kind: str  # "search" | "answer" | "none"
    payload: str = ""


def _strip_scaffolding(action: str) -> str:
    """Remove code fences, ``action:`` prefixes, and surrounding whitespace.

    Models wrap actions in ```` ```...``` ```` fences, prefix ``Action:`` /
    ``> ``, and add trailing chatter.  We pull out the meaningful command line so
    the grammar parser sees ``search(...)`` / ``answer(...)`` cleanly.
    """
    if not action:
        return ""
    text = str(action).strip()
    # Drop fenced blocks but keep their inner content.  A language tag is only a
    # tag when the fence is immediately followed by one (then a newline):
    # ```` ```python\n ````.  An inline ```` ```answer(42)``` ```` has NO newline,
    # so the bare-backticks replace below handles it — we must NOT let the
    # language-tag class greedily swallow the ``answer`` keyword.
    text = re.sub(r"```[a-zA-Z0-9_+-]*\n", "", text)
    text = text.replace("```", "")
    # Strip a leading "action:" / "thought:"-style prefix on the first line.
    text = re.sub(r"^\s*action\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*>\s*", "", text)
    return text.strip()


def _parse_action(action: str) -> ParsedAction:
    """Parse a model action into search/answer/none — case-insensitive, tolerant.

    Recognises ``search(<q>)``, ``search: <q>``, ``answer(<t>)``, ``answer: <t>``,
    and treats any other non-empty text as a bare final answer (the model's last
    line).  Empty / whitespace-only input → ``none`` (a wasted turn, not a crash).
    """
    cleaned = _strip_scaffolding(action)
    if not cleaned:
        return ParsedAction("none")

    # Scan every line so an action buried under reasoning is still found; the
    # first explicit search/answer wins.
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    for line in lines:
        m = re.match(r"^search\s*\((.*)\)\s*$", line, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return ParsedAction("search", m.group(1).strip().strip("\"'"))
        m = re.match(r"^search\s*:\s*(.+)$", line, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return ParsedAction("search", m.group(1).strip().strip("\"'"))
        m = re.match(r"^answer\s*\((.*)\)\s*$", line, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return ParsedAction("answer", m.group(1).strip().strip("\"'"))
        m = re.match(r"^answer\s*:\s*(.+)$", line, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return ParsedAction("answer", m.group(1).strip().strip("\"'"))

    # No explicit verb anywhere: treat the last non-empty line as a bare answer.
    return ParsedAction("answer", lines[-1].strip().strip("\"'"))


# ---------------------------------------------------------------------------
# The environment.
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT = (
    "You are answering an open-domain question by searching a knowledge base.\n"
    "On each turn emit exactly one action:\n"
    "  search(<query>)  - retrieve the top passages for <query>\n"
    "  answer(<text>)   - commit your final answer and end the episode\n"
    "Read the returned Observation passages, search again if needed, then answer.\n"
    "You have a limited number of turns, so answer once you have enough evidence."
)

_NUDGE_OBS = "Use search(<query>) or answer(<text>)."


class SearchQAEnv(AgenticEnv):
    """Multi-turn retrieval QA env for SDAR Search-QA (NQ-open + HotpotQA).

    Episode protocol (driven by the SDAR trainer / ``agentic_rollout``)::

        env.reset(seed=s, task={"question", "answers", "contexts", "source"})
        for _ in range(env.max_turns):           # max_turns = 6
            action = student.generate(env.build_student_prompt())
            res = env.step(action)               # search(...) or answer(...)
            if res.done: break
        reward = env.episode_reward()            # max-alias token-F1 of the answer

    ``search(<q>)`` retrieves top-3 passages (dense E5 → BM25 → overlap) and
    records them as an ``Observation``; ``answer(<t>)`` ends the episode with
    reward ``token_f1(t, answers)``.  Running out of turns without answering ends
    the episode at reward 0.0.  Defensive parsing means a malformed action only
    wastes a turn (nudge observation) — it never raises (spec §0.3).

    HotpotQA's gold ``contexts`` are kept and seed the per-question candidate pool,
    so the BM25/overlap tiers can retrieve the supporting passages even with no
    global dense index (the fix for the loader that dropped ``row["context"]``).
    """

    max_turns: int = 6

    def __init__(
        self,
        *,
        retriever: Retriever | None = None,
        corpus: Sequence[str] | None = None,
        top_k: int = 3,
    ) -> None:
        super().__init__()
        self._top_k = max(1, int(top_k))
        # An injected retriever (tests/trainer) wins; otherwise build one from the
        # env vars set by env_cache provisioning.
        if retriever is not None:
            self._retriever = retriever
        else:
            self._retriever = Retriever(
                index_dir=os.environ.get("SEARCH_QA_INDEX_DIR") or None,
                prefer=os.environ.get("SEARCH_QA_RETRIEVER") or None,
                corpus=corpus,
            )
        # Per-episode task state.
        self._question: str = ""
        self._answers: list[str] = []
        self._pool: list[str] = []
        self._source: str = ""
        self._n_search: int = 0
        self._answered: bool = False

    # --- the AgenticEnv contract -----------------------------------------

    def reset(self, *, seed: int | None = None, task: Any = None) -> str:
        """Start one QA episode; return the system prompt + question observation.

        ``task`` is ``{"question", "answers": [...], "contexts": [...]|None,
        "source": "nq"|"hotpotqa"}``.  Fail-soft: a missing/garbled task degrades
        to an empty question rather than raising — the first answer will simply
        score 0.0.
        """
        task = task or {}
        try:
            self._question = str(task.get("question", "") or "").strip()
            answers = task.get("answers") or []
            if isinstance(answers, str):
                answers = [answers]
            self._answers = [str(a) for a in answers if a is not None]
            contexts = task.get("contexts") or []
            if isinstance(contexts, str):
                contexts = [contexts]
            self._pool = [str(c) for c in contexts if c]
            self._source = str(task.get("source", "") or "")
        except Exception:  # noqa: BLE001 — never raise out of reset
            self._question, self._answers, self._pool, self._source = "", [], [], ""

        self._n_search = 0
        self._answered = False
        self._start_episode(system=_SYSTEM_PROMPT)

        observation = f"Question: {self._question}"
        self._record_obs(observation)
        return observation

    def step(self, action: str) -> StepResult:
        """Apply one model action.  Never raises (spec §0.3)."""
        self._record_act(action)

        try:
            parsed = _parse_action(action)
        except Exception:  # noqa: BLE001 — treat a parse blow-up as a nudge
            parsed = ParsedAction("none")

        # --- search --------------------------------------------------------
        if parsed.kind == "search":
            self._n_search += 1
            try:
                passages = self._retriever.retrieve(
                    parsed.payload, pool=self._pool, k=self._top_k
                )
            except Exception as exc:  # noqa: BLE001 — fail-soft to empty result
                print(f"[search_qa] retrieve raised ({exc!r}); returning no passages.")
                passages = []
            obs = self._format_passages(passages)
            self._record_obs(obs)
            # Out of turns after this search → terminal, no answer was committed.
            if self.turns_taken >= self.max_turns:
                return self._terminal_no_answer(obs)
            return StepResult(observation=obs, reward=0.0, done=False)

        # --- answer (explicit or bare final line) -------------------------
        if parsed.kind == "answer":
            self._answered = True
            reward = token_f1(parsed.payload, self._answers)
            em = exact_match(parsed.payload, self._answers)
            info = {
                "f1": float(reward),
                "em": int(em),
                "n_search": self._n_search,
                "answered": True,
                "retriever": self._retriever.backend,
                "source": self._source,
            }
            obs = f"Final answer recorded: {parsed.payload}"
            self._record_obs(obs)
            self._finish(reward, info=info)
            return StepResult(observation=obs, reward=reward, done=True, info=info)

        # --- unparseable: nudge, waste a turn -----------------------------
        self._record_obs(_NUDGE_OBS)
        if self.turns_taken >= self.max_turns:
            return self._terminal_no_answer(_NUDGE_OBS)
        return StepResult(observation=_NUDGE_OBS, reward=0.0, done=False)

    # --- helpers ----------------------------------------------------------

    def _terminal_no_answer(self, obs: str) -> StepResult:
        """End the episode at reward 0.0 (turn budget exhausted, no answer)."""
        info = {
            "f1": 0.0,
            "em": 0,
            "n_search": self._n_search,
            "answered": False,
            "retriever": self._retriever.backend,
            "source": self._source,
        }
        self._finish(0.0, info=info)
        return StepResult(observation=obs, reward=0.0, done=True, info=info)

    def _format_passages(self, passages: Sequence[Passage]) -> str:
        """Render retrieved passages into a single ``Observation:`` block."""
        if not passages:
            return "Observation: (no passages found)"
        lines = ["Observation:"]
        for i, p in enumerate(passages, start=1):
            text = (p.text or "").strip()
            lines.append(f"[{i}] {text}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Task loader — NQ-open + HotpotQA distractor, keeping HotpotQA contexts.
# ---------------------------------------------------------------------------


def _flatten_hotpot_context(context: Any) -> list[str]:
    """Flatten HotpotQA's ``context`` into paragraph strings.

    HotpotQA distractor ships ``context`` as ``{"title": [t,...], "sentences":
    [[s,...],...]}`` (HF ``datasets`` column) — or, on older dumps, a list of
    ``[title, [sentences...]]`` pairs.  Either way we join each article's sentence
    list into one ``"<title>. <sentences>"`` paragraph so the retriever can score
    whole passages (this is the data the old loader threw away).
    """
    paragraphs: list[str] = []
    if context is None:
        return paragraphs
    # HF column form: parallel "title" / "sentences" lists.
    if isinstance(context, dict):
        titles = context.get("title") or []
        sent_lists = context.get("sentences") or []
        for i, sents in enumerate(sent_lists):
            title = titles[i] if i < len(titles) else ""
            body = " ".join(s for s in (sents or []) if s)
            para = f"{title}. {body}".strip() if title else body.strip()
            if para:
                paragraphs.append(para)
        return paragraphs
    # List-of-pairs form: [[title, [sent, ...]], ...].
    if isinstance(context, (list, tuple)):
        for item in context:
            try:
                title, sents = item[0], item[1]
            except (TypeError, IndexError, KeyError):
                continue
            body = " ".join(s for s in (sents or []) if s)
            para = f"{title}. {body}".strip() if title else body.strip()
            if para:
                paragraphs.append(para)
    return paragraphs


def _coerce_answers(raw: Any) -> list[str]:
    """Normalise a dataset's answer field into a list of alias strings."""
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, dict):
        # SQuAD-style {"text": [...]} or HotpotQA single {"answer": ...} variants.
        for key in ("text", "answer", "aliases"):
            if key in raw and raw[key]:
                return _coerce_answers(raw[key])
        return []
    if isinstance(raw, (list, tuple)):
        out: list[str] = []
        for a in raw:
            out.extend(_coerce_answers(a))
        return out
    return [str(raw)]


def _load_dataset_resilient(repo_id: str, *args: Any, **kwargs: Any):
    """Load a HF dataset by its canonical ``owner/name`` id.

    Modern ``datasets`` rejects bare short names (``HfUriError: Repository id
    must be 'namespace/name'``), so callers MUST pass the namespaced id. We
    fall back to the legacy bare name (derived, never a literal) so a cache
    populated under the old id still resolves offline.
    """
    from datasets import load_dataset  # lazy

    try:
        return load_dataset(repo_id, *args, **kwargs)
    except Exception:
        bare = repo_id.split("/")[-1]
        if bare != repo_id:
            return load_dataset(bare, *args, **kwargs)
        raise


def _load_nq_open(n: int) -> list[dict[str, Any]]:
    """Load NQ-open validation rows as closed-pool tasks (no per-question context)."""
    ds = _load_dataset_resilient(
        "google-research-datasets/nq_open", split=f"validation[:{n}]"
    )
    tasks: list[dict[str, Any]] = []
    for row in ds:
        question = str(row.get("question", "") or "").strip()
        answers = _coerce_answers(row.get("answer"))
        if not question or not answers:
            continue
        tasks.append(
            {
                "question": question,
                "answers": answers,
                "contexts": None,  # NQ-open is open-domain; rely on the dense index
                "source": "nq",
            }
        )
    return tasks


def _load_hotpotqa(n: int) -> list[dict[str, Any]]:
    """Load HotpotQA distractor validation rows, KEEPING the gold context paragraphs."""
    ds = _load_dataset_resilient(
        "hotpotqa/hotpot_qa", "distractor", split=f"validation[:{n}]"
    )
    tasks: list[dict[str, Any]] = []
    for row in ds:
        question = str(row.get("question", "") or "").strip()
        answers = _coerce_answers(row.get("answer"))
        if not question or not answers:
            continue
        contexts = _flatten_hotpot_context(row.get("context"))
        tasks.append(
            {
                "question": question,
                "answers": answers,
                "contexts": contexts or None,
                "source": "hotpotqa",
            }
        )
    return tasks


def load_search_qa_tasks(
    n_per_source: int = 256,
    sources: Sequence[str] = ("nq", "hotpotqa"),
    *,
    loaders: dict[str, Callable[[int], list[dict[str, Any]]]] | None = None,
) -> list[dict[str, Any]]:
    """Load NQ-open + HotpotQA validation tasks (lazy ``datasets``, fail-soft).

    Returns a flat list of task dicts ``{"question", "answers", "contexts",
    "source"}``.  HotpotQA rows carry their gold supporting paragraphs in
    ``contexts`` (the loader-bug fix); NQ-open rows carry ``None`` (open-domain →
    rely on the dense index).  A source that fails to load (missing ``datasets``,
    no network, dataset-server error) contributes ``[]`` and prints a note —
    never raises (spec §3).  ``loaders`` is an injection seam for tests.
    """
    registry: dict[str, Callable[[int], list[dict[str, Any]]]] = loaders or {
        "nq": _load_nq_open,
        "hotpotqa": _load_hotpotqa,
    }
    tasks: list[dict[str, Any]] = []
    for source in sources:
        loader = registry.get(source)
        if loader is None:
            print(f"[search_qa] unknown source {source!r}; skipping.")
            continue
        try:
            loaded = loader(int(n_per_source))
            tasks.extend(loaded)
            print(f"[search_qa] loaded {len(loaded)} tasks from {source!r}.")
        except Exception as exc:  # noqa: BLE001 — fail-soft per source
            print(f"[search_qa] source {source!r} failed to load ({exc!r}); using [].")
    return tasks
