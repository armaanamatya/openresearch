"""Tests for the SDAR Search-QA agentic env (2026-06-01 full-scope envs, spec §3).

These exercise the real multi-turn retrieval QA loop that replaces the 2026-05-31
closed-book surrogate (which floored Search-QA at ~0.05 F1).  Everything runs on
the *base* venv with NO network and NO heavy deps (rank_bm25 / faiss /
sentence-transformers / datasets): the retriever and the dataset loaders are
injected / monkeypatched with fakes, and the module is imported via the FLAT
``sys.path`` insert that mirrors the runtime ``code/`` flat dir (the env modules
are copied flat, so they import each other as top-level modules).

Coverage (spec §3 test list):
  * action parsing — search / answer / garbage / code-fenced / prefixed variants
  * token_f1 / normalize_answer / exact_match on known SQuAD cases
  * a full FAKE-retriever episode: search → obs has passage → answer → reward ==
    expected F1 → done, info populated
  * max_turns exhaustion → reward 0.0, done
  * retriever fallback when dense + bm25 libs are absent → overlap, no raise
  * load_search_qa_tasks fail-soft (a source that raises → [] for that source)
"""

from __future__ import annotations

import sys

import pytest

# Mirror the runtime: env modules are copied FLAT into code/ and import each other
# as top-level modules, so the test imports them the same way. The flat dir is
# removed again right after import so it does NOT leak into the rest of the pytest
# session — a lingering entry gives package modules (e.g. rubric_guard) a second
# identity under a bare ``import``, breaking unrelated tests (rl_scaffold).
_RLM_DIR = str(  # repo-relative: the old hardcoded /home/sww35/... path
    __import__("pathlib").Path(__file__).resolve().parents[3] / "backend" / "agents" / "rlm"
)  # only collected on the author's machine (audit 2026-06-09)
sys.path.insert(0, _RLM_DIR)
try:
    import search_qa_env  # noqa: E402
    from search_qa_env import (  # noqa: E402
        Passage,
        Retriever,
        SearchQAEnv,
        exact_match,
        load_search_qa_tasks,
        normalize_answer,
        token_f1,
    )
finally:
    while _RLM_DIR in sys.path:
        sys.path.remove(_RLM_DIR)


# ---------------------------------------------------------------------------
# Fakes.
# ---------------------------------------------------------------------------


class FakeRetriever:
    """A retriever that returns canned passages and records what it was asked.

    Mirrors :class:`search_qa_env.Retriever`'s public surface (``retrieve`` +
    ``backend`` attribute) so it drops into ``SearchQAEnv(retriever=...)``.
    """

    def __init__(self, passages, backend="e5"):
        self._passages = list(passages)
        self.backend = backend
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query, *, pool=None, k=3):
        self.calls.append((query, k))
        return self._passages[:k]


# ---------------------------------------------------------------------------
# SQuAD scoring helpers.
# ---------------------------------------------------------------------------


def test_normalize_answer_squad_rules():
    assert normalize_answer("The Quick, Brown FOX!") == "quick brown fox"
    assert normalize_answer("  an  Apple ") == "apple"
    assert normalize_answer("U.S.A.") == "usa"
    assert normalize_answer(None) == ""


def test_token_f1_known_cases():
    # Exact match → 1.0.
    assert token_f1("Albert Einstein", ["albert einstein"]) == pytest.approx(1.0)
    # No overlap → 0.0.
    assert token_f1("Isaac Newton", ["Albert Einstein"]) == 0.0
    # Partial overlap: pred "the cat sat on", gold "the cat" →
    #   common normalised tokens {cat} (articles stripped) = 1
    #   pred tokens = [cat, sat] (=2), gold tokens = [cat] (=1)
    #   precision 1/2, recall 1/1 → F1 = 2*.5*1/(1.5) = 0.6667
    assert token_f1("the cat sat", ["the cat"]) == pytest.approx(2 / 3)
    # Max over aliases — best alias wins.
    assert token_f1("NYC", ["New York City", "NYC"]) == pytest.approx(1.0)
    assert token_f1("anything", None) == 0.0


def test_exact_match_known_cases():
    assert exact_match("The Cat", ["a cat"]) == 1
    assert exact_match("dog", ["cat", "dog"]) == 1
    assert exact_match("dog", ["cat"]) == 0
    assert exact_match("x", None) == 0


# ---------------------------------------------------------------------------
# Action parsing (via the private parser + observable env behaviour).
# ---------------------------------------------------------------------------


def test_parse_action_search_variants():
    p = search_qa_env._parse_action("search(quantum hall effect)")
    assert (p.kind, p.payload) == ("search", "quantum hall effect")

    p = search_qa_env._parse_action("Search: who painted the Mona Lisa")
    assert (p.kind, p.payload) == ("search", "who painted the Mona Lisa")

    # Code-fenced + capitalised verb.
    p = search_qa_env._parse_action("```\nSEARCH(eiffel tower height)\n```")
    assert (p.kind, p.payload) == ("search", "eiffel tower height")

    # Buried under a reasoning line — the explicit verb still wins.
    p = search_qa_env._parse_action("I should look this up.\nsearch(capital of France)")
    assert (p.kind, p.payload) == ("search", "capital of France")


def test_parse_action_answer_variants():
    p = search_qa_env._parse_action("answer(Paris)")
    assert (p.kind, p.payload) == ("answer", "Paris")

    p = search_qa_env._parse_action("Answer: Albert Einstein")
    assert (p.kind, p.payload) == ("answer", "Albert Einstein")

    # action: prefix + fenced.
    p = search_qa_env._parse_action("action: ```answer(42)```")
    assert (p.kind, p.payload) == ("answer", "42")

    # Bare final line (no verb) is treated as a final answer.
    p = search_qa_env._parse_action("The answer is clearly\nParis")
    assert p.kind == "answer"
    assert p.payload == "Paris"


def test_parse_action_garbage_is_none():
    assert search_qa_env._parse_action("").kind == "none"
    assert search_qa_env._parse_action("   \n  ").kind == "none"
    assert search_qa_env._parse_action("```\n\n```").kind == "none"


# ---------------------------------------------------------------------------
# Full episode with a FAKE retriever.
# ---------------------------------------------------------------------------


def test_full_episode_search_then_answer_scores_f1():
    passages = [
        Passage(text="Marie Curie discovered radium and polonium.", score=2.0),
        Passage(text="She won two Nobel Prizes.", score=1.0),
        Passage(text="Radioactivity research in Paris.", score=0.5),
    ]
    fake = FakeRetriever(passages, backend="e5")
    env = SearchQAEnv(retriever=fake)

    task = {
        "question": "Who discovered radium?",
        "answers": ["Marie Curie", "Curie"],
        "contexts": ["Marie Curie discovered radium and polonium."],
        "source": "hotpotqa",
    }
    initial = env.reset(seed=7, task=task)
    assert "Who discovered radium?" in initial
    assert not env.done

    # Turn 1: search → observation contains a retrieved passage.
    res = env.step("search(who discovered radium)")
    assert not res.done
    assert res.reward == 0.0
    assert "Observation:" in res.observation
    assert "Marie Curie discovered radium" in res.observation
    assert fake.calls and fake.calls[0][0] == "who discovered radium"
    # Top-k default is 3 — the env asked for 3.
    assert fake.calls[0][1] == 3

    # Turn 2: answer → terminal, reward == token_f1 over aliases (exact == 1.0).
    res = env.step("answer(Marie Curie)")
    assert res.done
    assert res.reward == pytest.approx(1.0)
    assert env.episode_reward() == pytest.approx(1.0)

    info = res.info
    assert info["f1"] == pytest.approx(1.0)
    assert info["em"] == 1
    assert info["n_search"] == 1
    assert info["answered"] is True
    assert info["retriever"] == "e5"
    assert info["source"] == "hotpotqa"
    # last_info mirrors the terminal info.
    assert env.last_info["f1"] == pytest.approx(1.0)
    # Transcript-rendered prompt includes the system grammar + the running turns.
    prompt = env.build_student_prompt()
    assert "search(<query>)" in prompt
    assert "Observation:" in prompt


def test_partial_answer_scores_partial_f1():
    fake = FakeRetriever([Passage(text="ctx")], backend="bm25")
    env = SearchQAEnv(retriever=fake)
    env.reset(seed=1, task={"question": "q", "answers": ["the cat"], "source": "nq"})
    res = env.step("answer: the cat sat")  # F1 = 2/3 (see helper test)
    assert res.done
    assert res.reward == pytest.approx(2 / 3)
    assert res.info["em"] == 0
    assert res.info["retriever"] == "bm25"


def test_bare_final_line_is_accepted_as_answer():
    fake = FakeRetriever([Passage(text="ctx")])
    env = SearchQAEnv(retriever=fake)
    env.reset(seed=1, task={"question": "q", "answers": ["paris"], "source": "nq"})
    res = env.step("Paris")  # no verb → bare answer
    assert res.done
    assert res.reward == pytest.approx(1.0)
    assert res.info["answered"] is True
    assert res.info["n_search"] == 0


def test_unparseable_action_nudges_without_reward_or_done():
    fake = FakeRetriever([Passage(text="ctx")])
    env = SearchQAEnv(retriever=fake)
    env.reset(seed=1, task={"question": "q", "answers": ["x"], "source": "nq"})
    # An empty action is unparseable → nudge, wastes a turn, not terminal.
    res = env.step("")
    assert not res.done
    assert res.reward == 0.0
    assert "search(<query>)" in res.observation
    assert env.turns_taken == 1


# ---------------------------------------------------------------------------
# max_turns exhaustion.
# ---------------------------------------------------------------------------


def test_max_turns_exhaustion_terminates_at_zero_reward():
    fake = FakeRetriever([Passage(text="some passage about things")], backend="overlap")
    env = SearchQAEnv(retriever=fake)
    env.reset(seed=3, task={"question": "q", "answers": ["never guessed"], "source": "nq"})

    # max_turns = 6: search six times, never answer → terminal at the 6th.
    last = None
    for i in range(SearchQAEnv.max_turns):
        last = env.step("search(stuff)")
    assert last is not None
    assert last.done
    assert last.reward == 0.0
    assert env.episode_reward() == 0.0
    assert last.info["answered"] is False
    assert last.info["n_search"] == SearchQAEnv.max_turns
    assert last.info["retriever"] == "overlap"
    assert env.turns_taken == SearchQAEnv.max_turns


def test_step_never_raises_on_garbage_after_reset():
    env = SearchQAEnv(retriever=FakeRetriever([]))
    env.reset(seed=0, task={"question": "q", "answers": ["a"], "source": "nq"})
    # Pathological inputs must not raise.
    for bad in [None, "", "```", "search()", "answer()", 12345]:
        res = env.step(bad)  # type: ignore[arg-type]
        assert isinstance(res.observation, str)


def test_reset_is_fail_soft_on_garbage_task():
    env = SearchQAEnv(retriever=FakeRetriever([]))
    # Garbled / missing task must not raise; answering then scores 0.0.
    obs = env.reset(seed=0, task=None)
    assert isinstance(obs, str)
    res = env.step("answer(whatever)")
    assert res.done
    assert res.reward == 0.0


# ---------------------------------------------------------------------------
# Retriever backend selection + fallback (no heavy libs, no network).
# ---------------------------------------------------------------------------


def test_retriever_falls_back_to_overlap_when_all_libs_absent(monkeypatch):
    """No SEARCH_QA_INDEX_DIR + rank_bm25 import fails → overlap, real passages."""
    monkeypatch.delenv("SEARCH_QA_INDEX_DIR", raising=False)
    monkeypatch.delenv("SEARCH_QA_RETRIEVER", raising=False)

    # Force rank_bm25 import to fail even if it ever gets installed.
    import builtins

    real_import = builtins.__import__

    def _blocking_import(name, *args, **kwargs):
        if name == "rank_bm25" or name.startswith("rank_bm25."):
            raise ImportError("blocked rank_bm25 for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocking_import)

    r = Retriever(index_dir=None, prefer=None)
    pool = [
        "The Eiffel Tower is located in Paris, France.",
        "The Great Wall of China is very long.",
        "Mount Everest is the tallest mountain on Earth.",
    ]
    hits = r.retrieve("where is the eiffel tower", pool=pool, k=2)
    assert r.backend == "overlap"
    assert len(hits) == 2
    # The most lexically-relevant passage ranks first.
    assert "Eiffel Tower" in hits[0].text
    # All returned items are real Passage objects with text.
    assert all(isinstance(h, Passage) and h.text for h in hits)


def test_retriever_prefer_bm25_skips_dense_even_with_index_dir(monkeypatch):
    """SEARCH_QA_RETRIEVER=bm25 must never touch the dense tier; falls to overlap
    when rank_bm25 is absent — and must not raise."""
    import builtins

    real_import = builtins.__import__

    def _blocking_import(name, *args, **kwargs):
        # If anything tries the dense libs, that's a bug — but also block bm25 so
        # we exercise the overlap floor without needing the lib installed.
        if name in {"rank_bm25", "faiss", "sentence_transformers"} or name.startswith(
            ("rank_bm25.", "faiss.", "sentence_transformers.")
        ):
            raise ImportError(f"blocked {name} for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocking_import)

    # index_dir is set, but prefer=bm25 must skip the dense path entirely.
    r = Retriever(index_dir="/nonexistent/index", prefer="bm25")
    hits = r.retrieve("anything", pool=["a relevant doc", "an irrelevant thing"], k=1)
    assert r.backend == "overlap"  # bm25 lib absent → overlap floor
    assert len(hits) == 1


def test_retriever_empty_pool_no_index_returns_empty_without_raising(monkeypatch):
    monkeypatch.delenv("SEARCH_QA_INDEX_DIR", raising=False)
    r = Retriever(index_dir=None, prefer=None)
    # Nothing to search and no global corpus → [] (real, not an exception).
    assert r.retrieve("q", pool=None, k=3) == []


def test_retriever_uses_injected_dense_seam():
    """A pre-built encoder + faiss index injected via the seam → backend 'e5'."""

    class FakeEncoder:
        def encode(self, texts, **kwargs):
            # Return a deterministic 1-D embedding per text (shape (n, 2)).
            return [[float(len(t)), 1.0] for t in texts]

    class FakeFaiss:
        def search(self, emb, k):
            # Always return ids [0, 1, ...] with descending scores.
            ids = list(range(k))
            scores = [float(k - i) for i in range(k)]
            return [scores], [ids]

    r = Retriever(
        index_dir="/whatever",
        _encoder=FakeEncoder(),
        _faiss_index=FakeFaiss(),
        _passages=["passage zero", "passage one", "passage two"],
    )
    hits = r.retrieve("query: who", k=2)
    assert r.backend == "e5"
    assert [h.text for h in hits] == ["passage zero", "passage one"]


# ---------------------------------------------------------------------------
# Task loader — fail-soft per source, HotpotQA contexts kept.
# ---------------------------------------------------------------------------


def test_load_tasks_fail_soft_on_failing_source():
    """A source whose loader raises contributes [] — never propagates the error."""

    def good_nq(n):
        return [{"question": "q1", "answers": ["a1"], "contexts": None, "source": "nq"}]

    def broken_hotpot(n):
        raise RuntimeError("simulated datasets/network failure")

    tasks = load_search_qa_tasks(
        n_per_source=5,
        sources=("nq", "hotpotqa"),
        loaders={"nq": good_nq, "hotpotqa": broken_hotpot},
    )
    # nq survived; hotpotqa failed soft → only the nq task remains.
    assert len(tasks) == 1
    assert tasks[0]["source"] == "nq"


def test_load_tasks_unknown_source_skipped():
    tasks = load_search_qa_tasks(sources=("does_not_exist",), loaders={})
    assert tasks == []


def test_flatten_hotpot_context_hf_column_form():
    """HotpotQA HF column form is flattened to '<title>. <sentences>' paragraphs —
    this is the data the old loader discarded (spec §3 fix)."""
    context = {
        "title": ["Marie Curie", "Radium"],
        "sentences": [
            ["She was a physicist.", " She discovered radium."],
            ["Radium is radioactive."],
        ],
    }
    paras = search_qa_env._flatten_hotpot_context(context)
    assert len(paras) == 2
    assert paras[0].startswith("Marie Curie.")
    assert "discovered radium" in paras[0]
    assert paras[1].startswith("Radium.")


def test_flatten_hotpot_context_list_of_pairs_form():
    context = [
        ["Eiffel Tower", ["It is in Paris.", " It is tall."]],
        ["Seine", ["A river in France."]],
    ]
    paras = search_qa_env._flatten_hotpot_context(context)
    assert len(paras) == 2
    assert paras[0].startswith("Eiffel Tower.")
    assert "Paris" in paras[0]


def test_flatten_hotpot_context_garbage_is_empty():
    assert search_qa_env._flatten_hotpot_context(None) == []
    assert search_qa_env._flatten_hotpot_context(123) == []
    # Malformed list entries are skipped, not raised on.
    assert search_qa_env._flatten_hotpot_context([None, 5, "x"]) == []


def test_hotpot_contexts_seed_pool_for_bm25_path(monkeypatch):
    """End-to-end: a HotpotQA task's contexts seed the candidate pool so the
    overlap/bm25 tier can retrieve them with no global index (the loader-bug fix)."""
    monkeypatch.delenv("SEARCH_QA_INDEX_DIR", raising=False)
    monkeypatch.delenv("SEARCH_QA_RETRIEVER", raising=False)

    # Block rank_bm25 so we land on the overlap floor deterministically.
    import builtins

    real_import = builtins.__import__

    def _blocking_import(name, *args, **kwargs):
        if name == "rank_bm25" or name.startswith("rank_bm25."):
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocking_import)

    # No injected retriever → env builds the real three-tier Retriever from env.
    env = SearchQAEnv()
    task = {
        "question": "Who discovered radium?",
        "answers": ["Marie Curie"],
        "contexts": [
            "Marie Curie. She discovered radium and polonium in Paris.",
            "Mount Everest. The tallest mountain on Earth.",
        ],
        "source": "hotpotqa",
    }
    env.reset(seed=0, task=task)
    res = env.step("search(who discovered radium)")
    assert not res.done
    # The gold context paragraph was retrieved from the per-question pool.
    assert "Marie Curie" in res.observation
    # And the backend resolved to the pure-python overlap floor (no libs/index).
    res2 = env.step("answer(Marie Curie)")
    assert res2.info["retriever"] == "overlap"
    assert res2.reward == pytest.approx(1.0)


def test_dense_loader_resolves_nonstandard_index_filename(tmp_path, monkeypatch):
    """The dense tier must load a FAISS index even when the downloaded repo names
    it something other than ``index.faiss`` and nests it in a subdir — env_cache
    advertises E5 for ANY ``*.faiss``/``*.index`` under the dir, so the retriever
    must load exactly what that gate accepts or dense is advertised then silently
    degrades to BM25 (the 2026-06-01 review MEDIUM). Heavy deps are faked so this
    runs on the base venv (no faiss / sentence-transformers installed)."""
    import json as _json
    import types

    # A repo whose index is named e5_Flat.index (NOT index.faiss), nested one level.
    sub = tmp_path / "wiki18"
    sub.mkdir()
    (sub / "e5_Flat.index").write_bytes(b"FAKE-FAISS")
    (sub / "passages.json").write_text(_json.dumps(["alpha passage", "beta passage"]))

    captured: dict = {}

    def _read_index(p):
        captured["index_path"] = p
        return "FAKE_INDEX_OBJ"

    fake_faiss = types.SimpleNamespace(read_index=_read_index)

    class _FakeST:
        def __init__(self, name):
            captured["model"] = name

        def encode(self, *a, **k):  # pragma: no cover - not exercised by _ensure_dense
            return [[0.0]]

    fake_st = types.ModuleType("sentence_transformers")
    fake_st.SentenceTransformer = _FakeST  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faiss", fake_faiss)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)

    r = Retriever(index_dir=str(tmp_path))
    assert r._ensure_dense() is True
    assert r._dense_ready is True
    assert captured["index_path"].endswith("e5_Flat.index")   # nonstandard name resolved
    assert r._passages == ["alpha passage", "beta passage"]    # nested passage store loaded
    assert captured["model"] == "intfloat/e5-base-v2"
