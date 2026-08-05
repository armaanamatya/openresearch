"""OFF+ON pairs for the literature primitives (20th/21st) + prompt section.

Flag: OPENRESEARCH_LITERATURE_SURVEY. Hermetic: corpus built in tmp dirs,
LLM = the counting FakeLlmClient from conftest, no network.
"""

from __future__ import annotations

import json
from pathlib import Path

from backend.agents.rlm.primitives import search_literature, survey_related_work
from backend.agents.rlm.system_prompt import build_system_prompt
from backend.services.knowledge.corpus.index import build_chunk_index
from backend.services.knowledge.corpus.store import CorpusStore, corpus_root


def _seed_corpus(runs_root: Path, project_id: str) -> CorpusStore:
    store = CorpusStore(corpus_root(runs_root))
    store.initialize()
    store.upsert_paper(
        "arxiv:2101.00001", title="Warmup Schedules in Agentic RL", fetched_level="fulltext"
    )
    store.upsert_paper("arxiv:2102.00002", title="A Cited Baseline", abstract="Baseline for ALFWorld tasks.")
    (store.paper_dir("arxiv:2101.00001", create=True) / "parsed_full_text.txt").write_text(
        "Methods\n\nWe use a cosine warmup schedule over 500 steps on ALFWorld.\n\n"
        "Results\n\nSuccess rate reaches 58.1 on ALFWorld.",
        encoding="utf-8",
    )
    store.add_relation("arxiv:2101.00001", "arxiv:2102.00002")
    store.connection.execute(
        "INSERT INTO lit_results (paper_id, method, dataset, metric, value, span_quote)"
        " VALUES ('arxiv:2101.00001', 'sdar', 'alfworld', 'success_rate', 58.1,"
        " 'Success rate reaches 58.1 on ALFWorld.')"
    )
    store.connection.commit()
    build_chunk_index(store)

    state = runs_root / project_id / "rlm_state"
    state.mkdir(parents=True, exist_ok=True)
    (state / "literature_spec.json").write_text(
        json.dumps(
            {
                "target": {"id": "arxiv:2605.15155"},
                "papers": [
                    {"id": "arxiv:2101.00001", "title": "Warmup Schedules in Agentic RL",
                     "relation": "direct_ref", "year": 2021},
                ],
            }
        ),
        encoding="utf-8",
    )
    return store


# --- OFF state ---------------------------------------------------------------


def test_both_primitives_disabled_without_flag(make_context, tmp_path, monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_LITERATURE_SURVEY", raising=False)
    ctx = make_context(tmp_path)
    assert search_literature(ctx=ctx) == {"status": "disabled"}
    assert survey_related_work("anything", ctx=ctx) == {"status": "disabled"}


def test_prompt_omits_literature_section_when_flag_off(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_LITERATURE_SURVEY", raising=False)
    from backend.agents.rlm.models import ROOT_MODELS

    prompt = build_system_prompt(
        context_metadata={"context": {"type": "str", "length": 10}},
        root_model=ROOT_MODELS["gpt-5"],
    )
    assert "RELATED-WORK CORPUS" not in prompt


# --- ON state ----------------------------------------------------------------


def test_prompt_includes_literature_section_when_flag_on(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_LITERATURE_SURVEY", "1")
    from backend.agents.rlm.models import ROOT_MODELS

    prompt = build_system_prompt(
        context_metadata={"context": {"type": "str", "length": 10}},
        root_model=ROOT_MODELS["gpt-5"],
    )
    assert prompt.count("RELATED-WORK CORPUS") == 1  # single append, no double
    assert "ADVISORY ONLY" in prompt


def test_search_literature_index_mode(make_context, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_LITERATURE_SURVEY", "1")
    ctx = make_context(tmp_path)
    _seed_corpus(tmp_path, ctx.project_id).close()

    result = search_literature(ctx=ctx)
    assert result["status"] == "ok" and result["kind"] == "index"
    assert result["papers"][0]["id"] == "arxiv:2101.00001"
    assert result["corpus_papers_total"] == 2
    assert "ADVISORY" in result["note"]


def test_search_literature_query_mode(make_context, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_LITERATURE_SURVEY", "1")
    ctx = make_context(tmp_path)
    _seed_corpus(tmp_path, ctx.project_id).close()

    result = search_literature(query="cosine warmup schedule", ctx=ctx)
    assert result["status"] == "ok" and result["kind"] == "search"
    assert result["hits"], "query mode must hit the methods chunk"
    assert result["hits"][0]["paper_id"] == "arxiv:2101.00001"
    assert result["hits"][0]["lane"] == "A"
    # Trace is written into the run's rlm_state for auditability.
    assert (ctx.project_dir / "rlm_state" / "literature_query_trace.json").exists()


def test_search_literature_results_lookup(make_context, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_LITERATURE_SURVEY", "1")
    ctx = make_context(tmp_path)
    _seed_corpus(tmp_path, ctx.project_id).close()

    result = search_literature(dataset="ALFWorld", ctx=ctx)
    assert result["status"] == "ok" and result["kind"] == "results"
    assert result["rows"][0]["metric"] == "success_rate"
    assert result["rows"][0]["value"] == 58.1


def test_search_literature_paper_read_bounded(make_context, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_LITERATURE_SURVEY", "1")
    ctx = make_context(tmp_path)
    _seed_corpus(tmp_path, ctx.project_id).close()

    result = search_literature(paper_id="arxiv:2101.00001", ctx=ctx)
    assert result["status"] == "ok" and result["kind"] == "paper"
    assert all(len(c["text"]) <= 800 for c in result["chunks"])

    missing = search_literature(paper_id="arxiv:9999.99999", ctx=ctx)
    assert missing["status"] == "error"


def test_survey_fans_out_one_call_per_paper(make_context, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_LITERATURE_SURVEY", "1")
    monkeypatch.delenv("OPENRESEARCH_ACCELERATOR", raising=False)
    ctx = make_context(tmp_path, llm_responses=["Uses cosine warmup [methods]."])
    _seed_corpus(tmp_path, ctx.project_id).close()

    result = survey_related_work("what warmup schedule is used?", k=3, ctx=ctx)
    assert result["status"] == "ok" and result["kind"] == "survey"
    # Retrieval reaches both corpus papers (query hit + citation expansion);
    # one bounded sub-call per surveyed paper on the planner client.
    assert len(ctx.llm_client.calls) == len(result["papers"]) >= 1
    assert result["llm_tier"] == "planner"
    assert len(result["digest"]) <= 4000
    assert "Warmup Schedules in Agentic RL" in result["digest"]
    for call in ctx.llm_client.calls:
        assert "excerpts only" in call["system"]


def test_survey_requires_question_and_clamps_k(make_context, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_LITERATURE_SURVEY", "1")
    ctx = make_context(tmp_path)
    _seed_corpus(tmp_path, ctx.project_id).close()

    assert survey_related_work("", ctx=ctx)["status"] == "error"
    result = survey_related_work("warmup", k=999, ctx=ctx)
    assert result["status"] == "ok"
    assert len(result["papers"]) <= 8


def test_survey_llm_failure_degrades_per_paper(make_context, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_LITERATURE_SURVEY", "1")
    ctx = make_context(tmp_path)
    _seed_corpus(tmp_path, ctx.project_id).close()

    class _Boom:
        def complete(self, *, system: str, user: str) -> str:
            raise RuntimeError("llm down")

    ctx.llm_client = _Boom()
    result = survey_related_work("warmup schedule", ctx=ctx)
    assert result["status"] == "ok"
    assert "sub-call failed" in result["digest"]
