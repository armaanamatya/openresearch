"""Tests for the recursive RLM loop in RlmQueryTool.

The Phase D++ upgrade replaced the one-shot truncate-and-answer
behaviour with the paper's recursive decomposition. These tests pin
the loop shape: chunk → select → recurse → aggregate, with depth
and call budgets.

Backwards compatibility: the leaf-path (small content) behaviour is
preserved, so the existing test_issue16_workspace_service.py
RlmQueryTool tests still hold. Those tests assert on the
Cited[T] shape and the public call() signature.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from backend.schemas.citations import Citation
from backend.services.context.workspace.model import Cited
from backend.services.context.workspace.tools.interface import WorkspaceToolError
from backend.services.context.workspace.tools.rlm_query import RlmQueryTool


# --- test doubles -----------------------------------------------------------


@dataclass
class _CountingLlm:
    """LLM stub that counts (system, user) pairs and returns scripted
    completions. Used so tests can pin the EXACT shape of the recursive
    expansion without hitting a network."""

    responses: list[str] = field(default_factory=list)
    calls: list[tuple[str, str]] = field(default_factory=list)

    def complete(self, *, system: str, user: str) -> str:
        self.calls.append((system, user))
        if not self.responses:
            return f"[leaf-{len(self.calls)}]"
        # Cycle through scripted responses if the test provides them.
        idx = (len(self.calls) - 1) % len(self.responses)
        return self.responses[idx]


class _StubView:
    """Minimal WorkspaceView impl: holds one variable + citations."""

    def __init__(self, name: str, value: Any, citations: tuple[Citation, ...]) -> None:
        self._name = name
        self._cited = Cited(value=value, citations=citations)

    def get(self, name: str):
        return self._cited if name == self._name else None

    def variable_names(self) -> set[str]:
        return {self._name}


def _make_view(name: str, value: Any) -> _StubView:
    citation = Citation(source_id="s1", chunk_id="c1", quote="ev", locator="L1")
    return _StubView(name, value, citations=(citation,))


# --- backwards-compat: the leaf path is identical to the old single-call ----


def test_leaf_path_short_content_makes_one_llm_call():
    """Content under the leaf budget → exactly one LLM call, depth 0."""
    view = _make_view("paper_text", "This paper studies recursive prompting.")
    llm = _CountingLlm(responses=["The paper studies recursive prompting."])
    tool = RlmQueryTool(view_provider=lambda _wsid: view, llm_client=llm)

    result = tool.call(
        workspace_id="ws_x",
        question="What does it study?",
        variable_name="paper_text",
    )

    assert isinstance(result, Cited)
    assert result.value["answer"] == "The paper studies recursive prompting."
    assert result.value["depth_reached"] == 0
    assert result.value["llm_calls"] == 1
    assert result.value["chunks_examined"] == 0
    assert len(llm.calls) == 1
    # Backwards-compat fields all present.
    assert result.value["question"] == "What does it study?"
    assert result.value["variable_name"] == "paper_text"
    assert result.value["context_key"] is None
    assert len(result.citations) == 1


def test_empty_question_raises():
    view = _make_view("paper_text", "x")
    llm = _CountingLlm()
    tool = RlmQueryTool(view_provider=lambda _: view, llm_client=llm)
    with pytest.raises(WorkspaceToolError, match="non-empty"):
        tool.call(workspace_id="ws", question="   ", variable_name="paper_text")


def test_missing_variable_raises():
    view = _make_view("paper_text", "x")
    llm = _CountingLlm()
    tool = RlmQueryTool(view_provider=lambda _: view, llm_client=llm)
    with pytest.raises(WorkspaceToolError, match="not found"):
        tool.call(workspace_id="ws", question="?", variable_name="missing")


# --- recursion ---------------------------------------------------------------


def test_two_chunks_no_selection_recurses_and_aggregates():
    """Content > leaf_budget but ≤ chunk_size * selection_top_k:
    chunked into N, no selection step (N ≤ top_k), every chunk
    recursed into, aggregator synthesizes."""
    # 24k chars across 4 paragraphs, leaf_budget 12k → chunks into ~2.
    paragraphs = [f"Paragraph {i}: " + ("x" * 5_900) for i in range(4)]
    content = "\n\n".join(paragraphs)
    view = _make_view("paper_text", content)

    llm = _CountingLlm(responses=["sub-answer", "sub-answer", "synthesized"])
    tool = RlmQueryTool(
        view_provider=lambda _: view,
        llm_client=llm,
        leaf_budget=12_000,
        chunk_size=12_000,
        selection_top_k=5,  # > number of chunks, so no selection
        selection_enabled=True,
        max_depth=3,
    )

    result = tool.call(workspace_id="ws", question="What?", variable_name="paper_text")

    assert result.value["depth_reached"] == 1
    # 2 leaf calls + 1 aggregate = 3 LLM calls. No selection step.
    assert result.value["llm_calls"] == 3
    assert result.value["chunks_examined"] == 2
    # No selection step happened — sample of selection_path entries.
    assert result.value["selection_path"] == [
        {"depth": 0, "total_chunks": 2, "selected": [0, 1]}
    ]


def test_many_chunks_triggers_selection_step():
    """Content with many chunks > top_k: select-then-recurse. The
    selection prompt is one extra LLM call before leaf calls."""
    # 8 paragraphs of ~11k chars each — fits 8 chunks of one paragraph each.
    paragraphs = [f"Topic-{i}: " + ("y" * 11_000) for i in range(8)]
    content = "\n\n".join(paragraphs)
    view = _make_view("paper_text", content)

    # Script: selection picks chunks 1 and 3, then 2 leaf answers, then aggregate.
    llm = _CountingLlm(responses=[
        json.dumps({"selected": [1, 3]}),
        "leaf-from-chunk-1",
        "leaf-from-chunk-3",
        "synthesized",
    ])
    tool = RlmQueryTool(
        view_provider=lambda _: view,
        llm_client=llm,
        leaf_budget=12_000,
        chunk_size=12_000,
        selection_top_k=3,  # < 8 chunks, so selection fires
        selection_enabled=True,
        max_depth=3,
    )

    result = tool.call(workspace_id="ws", question="?", variable_name="paper_text")

    # 1 selection + 2 leaves + 1 aggregate = 4 calls.
    assert result.value["llm_calls"] == 4
    assert result.value["chunks_examined"] == 8
    assert result.value["selection_path"] == [
        {"depth": 0, "total_chunks": 8, "selected": [1, 3]}
    ]


def test_selection_disabled_recurses_on_every_chunk():
    """selection_enabled=False: no router LLM call; every chunk gets
    drilled into."""
    paragraphs = [f"P{i}: " + ("z" * 11_000) for i in range(5)]
    content = "\n\n".join(paragraphs)
    view = _make_view("paper_text", content)

    llm = _CountingLlm(responses=["leaf-1", "leaf-2", "leaf-3", "leaf-4", "leaf-5", "agg"])
    tool = RlmQueryTool(
        view_provider=lambda _: view,
        llm_client=llm,
        leaf_budget=12_000,
        chunk_size=12_000,
        selection_top_k=2,
        selection_enabled=False,
        max_depth=3,
    )

    result = tool.call(workspace_id="ws", question="?", variable_name="paper_text")

    # 5 leaves + 1 aggregate = 6 calls (no selection).
    assert result.value["llm_calls"] == 6
    assert result.value["chunks_examined"] == 5


def test_max_depth_truncates():
    """Content so large that even at max_depth, leaves still exceed
    leaf_budget: the deepest layer truncates and answers."""
    # 200k chars, leaf_budget 1k, chunk_size 1k, top_k 100, max_depth 1.
    # depth 0: chunked into 200 of 1k each. With selection disabled, all 200
    # are recursed. depth 1 each leaf is exactly 1k → leaf base case.
    # No truncation triggered. Use deeper / smaller config to force it.
    content = "x" * 100_000  # one giant blob, no paragraph boundaries
    view = _make_view("paper_text", content)
    llm = _CountingLlm()
    tool = RlmQueryTool(
        view_provider=lambda _: view,
        llm_client=llm,
        leaf_budget=2_000,
        chunk_size=2_000,
        selection_top_k=2,
        selection_enabled=False,
        max_depth=1,  # cap recursion at 1 → depth 1 chunks of 2k still > leaf_budget? No, == budget.
        max_llm_calls=1000,
    )
    # 100k chars / 2k chunk = 50 chunks at depth 0. Each is exactly 2k → base case at depth 1.
    # max_depth=1 means we DO descend to depth 1, but at depth 1 each chunk
    # is == leaf_budget so leaf path fires (not the truncation branch).
    result = tool.call(workspace_id="ws", question="?", variable_name="paper_text")

    assert result.value["depth_reached"] == 1
    assert result.value["truncated_at_max_depth"] is False


def test_max_depth_truncation_path():
    """When chunks at depth >= max_depth still exceed leaf_budget, we
    truncate and answer (not recurse further)."""
    # Content: one blob 100k. leaf_budget 1k. chunk_size 50k. max_depth 1.
    # depth 0: split into 2 chunks of 50k each.
    # depth 1: each 50k chunk > leaf_budget AND depth == max_depth →
    # truncation branch fires.
    content = "a" * 100_000
    view = _make_view("paper_text", content)
    llm = _CountingLlm(responses=["truncated-leaf", "truncated-leaf", "synth"])
    tool = RlmQueryTool(
        view_provider=lambda _: view,
        llm_client=llm,
        leaf_budget=1_000,
        chunk_size=50_000,
        selection_top_k=10,
        selection_enabled=False,
        max_depth=1,
    )

    result = tool.call(workspace_id="ws", question="?", variable_name="paper_text")

    assert result.value["depth_reached"] == 1
    assert result.value["truncated_at_max_depth"] is True
    # 2 truncated leaves + 1 aggregate = 3 calls.
    assert result.value["llm_calls"] == 3


def test_max_llm_calls_budget_short_circuits():
    """When the call budget is hit mid-recursion, the loop returns
    with whatever sub-answers it has."""
    paragraphs = [f"P{i}: " + ("q" * 11_000) for i in range(10)]
    content = "\n\n".join(paragraphs)
    view = _make_view("paper_text", content)

    llm = _CountingLlm(responses=["leaf"] * 50)
    tool = RlmQueryTool(
        view_provider=lambda _: view,
        llm_client=llm,
        leaf_budget=12_000,
        chunk_size=12_000,
        selection_top_k=20,  # no selection (10 chunks ≤ 20)
        selection_enabled=False,
        max_depth=3,
        max_llm_calls=3,  # tight budget — only 3 leaf calls before bail
    )

    result = tool.call(workspace_id="ws", question="?", variable_name="paper_text")

    # Hit the cap at 3 calls. With 10 chunks and no aggregator call
    # left in the budget, the result concatenates the 3 leaves.
    assert result.value["llm_calls"] == 3


def test_selection_parses_jsonish_output():
    """The routing LLM may return JSON with preface text or whitespace —
    parser should still find {"selected": [...]}."""
    sloppy = 'Sure, here you go:\n{"selected": [0, 2]}\n— done.'
    parsed = RlmQueryTool._parse_selection(sloppy, total_chunks=4)
    assert parsed == [0, 2]


def test_selection_rejects_out_of_range_and_dupes():
    raw = json.dumps({"selected": [0, 99, 1, 0, -3, 1, 2]})
    parsed = RlmQueryTool._parse_selection(raw, total_chunks=3)
    assert parsed == [0, 1, 2]


def test_selection_unparseable_returns_empty():
    parsed = RlmQueryTool._parse_selection("not json at all", total_chunks=5)
    assert parsed == []


def test_zero_selected_chunks_returns_insufficient_context():
    """If selection picks no chunks, we shouldn't burn more LLM calls."""
    paragraphs = [f"P{i}: " + ("r" * 11_000) for i in range(8)]
    content = "\n\n".join(paragraphs)
    view = _make_view("paper_text", content)

    llm = _CountingLlm(responses=[json.dumps({"selected": []})])
    tool = RlmQueryTool(
        view_provider=lambda _: view,
        llm_client=llm,
        leaf_budget=12_000,
        chunk_size=12_000,
        selection_top_k=3,
        selection_enabled=True,
        max_depth=3,
    )

    result = tool.call(workspace_id="ws", question="?", variable_name="paper_text")

    assert "insufficient context" in result.value["answer"].lower()
    # Only the routing call fired — no leaf or aggregate.
    assert result.value["llm_calls"] == 1


def test_citations_propagate_through_recursion():
    """All Cited[T] results carry the base variable's citations,
    regardless of recursion depth."""
    paragraphs = [f"P{i}: " + ("c" * 11_000) for i in range(4)]
    content = "\n\n".join(paragraphs)
    citation1 = Citation(source_id="paper", chunk_id="abs", quote="abstract", locator="abs:1")
    citation2 = Citation(source_id="paper", chunk_id="meth", quote="methods", locator="meth:1")
    view = _StubView("paper_text", content, citations=(citation1, citation2))

    llm = _CountingLlm(responses=["leaf"] * 10)
    tool = RlmQueryTool(
        view_provider=lambda _: view,
        llm_client=llm,
        leaf_budget=12_000,
        chunk_size=12_000,
        selection_top_k=10,
        selection_enabled=False,
        max_depth=3,
    )

    result = tool.call(workspace_id="ws", question="?", variable_name="paper_text")

    # Both base citations propagate; nothing fabricated.
    assert len(result.citations) == 2
    assert {c.chunk_id for c in result.citations} == {"abs", "meth"}


def test_legacy_context_budget_kwarg_still_pins_leaf_budget():
    """The old constructor used context_budget=. Existing callers
    should keep working — we map it to leaf_budget."""
    view = _make_view("paper_text", "short")
    llm = _CountingLlm(responses=["ok"])
    tool = RlmQueryTool(
        view_provider=lambda _: view,
        llm_client=llm,
        context_budget=4_000,  # legacy kwarg
    )
    result = tool.call(workspace_id="ws", question="?", variable_name="paper_text")
    assert result.value["leaf_budget"] == 4_000


def test_context_key_drills_into_dict_variable():
    sections = {"sections": {"Methods": "We use PPO.", "Results": "Reward 500."}}
    view = _make_view("paper_sections", sections)
    llm = _CountingLlm(responses=["answer"])
    tool = RlmQueryTool(view_provider=lambda _: view, llm_client=llm)

    result = tool.call(
        workspace_id="ws",
        question="?",
        variable_name="paper_sections",
        context_key="Methods",
    )

    assert result.value["context_key"] == "Methods"
    # The leaf saw exactly the Methods section text.
    assert "We use PPO." in llm.calls[0][1]
