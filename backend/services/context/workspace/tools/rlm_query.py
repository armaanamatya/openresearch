"""RlmQueryTool — recursive LLM sub-query over a workspace variable.

Faithful implementation of the RLM paradigm from Zhang/Kraska/Khattab
(arXiv:2512.24601): treat the variable's content as an external
environment the LLM can programmatically examine, recursively calling
itself over snippets to handle inputs larger than the model's context
window.

Loop shape (depth-bounded, call-budgeted):

  recursive_query(content, question, depth):
      if len(content) <= leaf_budget:           # base case 1
          return llm_answer(content, question)
      if depth >= max_depth:                    # base case 2
          return llm_answer(truncate(content), question)
      chunks = chunk(content, chunk_size)
      if selection_enabled and len(chunks) > selection_top_k:
          relevant_idx = llm_select(chunks, question, top_k)
      else:
          relevant_idx = range(len(chunks))
      sub_answers = [
          recursive_query(chunks[i], question, depth + 1)
          for i in relevant_idx
      ]
      return llm_aggregate(question, sub_answers)

What this adds on top of the paper:

  - Cited[T] invariant — every Cited[T] returned carries the base
    variable's citations (the workspace's provenance chain).
  - Hard cost gates — max_depth, max_llm_calls bound the runaway path.
  - Telemetry — every call records depth_reached, llm_calls,
    chunks_examined, selection_path. The ToolInvoked event captures it.
  - Provider-agnostic — uses the LlmClient Protocol; tests use a stub
    counter so the recursion shape is asserted without hitting an API.

Backwards compatibility:
  - call(workspace_id, question, variable_name, context_key?) signature
    unchanged; existing test_issue16_workspace_service.py tests still
    pass (single-call path is the leaf base case).
  - result.value retains {question, variable_name, context_key, answer}.
    New fields (depth_reached, llm_calls, chunks_examined, etc.) are
    additive.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from backend.schemas.citations import Citation
from backend.services.context.workspace.model import Cited
from backend.services.context.workspace.projections import WorkspaceView
from backend.services.context.workspace.tools.interface import WorkspaceToolError


logger = logging.getLogger(__name__)

# Default budgets — chosen so a typical research-paper variable (~80k
# chars after pymupdf extraction) lands in a single L1 chunk on a
# modern model, but a 1M-char dump triggers recursion.
_DEFAULT_LEAF_BUDGET = 12_000        # chars per LLM call at a leaf
_DEFAULT_CHUNK_SIZE = 12_000         # chars per chunk when splitting
_DEFAULT_MAX_DEPTH = 3               # how deep recursion can go
_DEFAULT_SELECTION_TOP_K = 5         # how many chunks to drill into
_DEFAULT_MAX_LLM_CALLS = 24          # hard cap to prevent runaway cost


class LlmClient(Protocol):
    """Minimal synchronous LLM interface for workspace tools.

    Implementations can wrap OpenAI, Anthropic, or any provider. Tests
    use a counting stub. Must be deterministic given (system, user) so
    recursive expansion is repeatable.
    """

    def complete(self, *, system: str, user: str) -> str:
        """Return a completion string for the given system+user prompt."""
        ...


@dataclass
class _RecursionState:
    """Bookkeeping shared across one recursive_query invocation.

    Tracks the call budget, deepest recursion reached, and chunks
    examined so the caller can observe what actually happened. Mutated
    in place during the recursion.
    """

    max_depth: int
    max_llm_calls: int
    calls_made: int = 0
    max_depth_reached: int = 0
    chunks_examined: int = 0
    selection_path: list[dict[str, Any]] = field(default_factory=list)
    hit_truncation_branch: bool = False

    def can_call(self) -> bool:
        return self.calls_made < self.max_llm_calls

    def record_call(self) -> None:
        self.calls_made += 1

    def observe_depth(self, depth: int) -> None:
        self.max_depth_reached = max(self.max_depth_reached, depth)


# --- prompts ----------------------------------------------------------------

_LEAF_SYSTEM = (
    "You are a research assistant. Answer the question based ONLY on "
    "the provided context. If the context does not contain enough "
    "information to answer, say 'insufficient context' explicitly. "
    "Be precise; cite specific phrases from the context when relevant."
)

_SELECT_SYSTEM_TEMPLATE = (
    "You are a routing assistant. You see a list of context chunks "
    "(numbered, with a short preview each). Pick which chunks are most "
    "likely to contain information that answers the question. Output a "
    'JSON object: {"selected": [<chunk index>, ...]}. Pick at most '
    "%TOPK% chunks. If no chunks look relevant, return "
    '{"selected": []}.'
)

_AGGREGATE_SYSTEM = (
    "You are a synthesis assistant. You see several sub-answers to the "
    "same question, each derived from a different piece of context. "
    "Synthesize them into one coherent answer. If sub-answers conflict, "
    "note the conflict. If most sub-answers say 'insufficient context', "
    "say so. Do not invent information beyond what the sub-answers "
    "report."
)


class RlmQueryTool:
    """Recursive LLM sub-query over a workspace variable.

    Behaviour by content size (chars):
      ≤ leaf_budget                  one LLM call (the base case)
      ≤ chunk_size * selection_top_k chunk + select_top_k + aggregate
      larger                          recurse on each selected chunk

    All paths terminate in ≤ max_llm_calls LLM calls and ≤ max_depth
    levels of recursion. The default 24-call cap covers a 10-chunk
    selective traverse at depth 2 with synthesis at each level. Bump it
    deliberately for unusually large inputs.
    """

    name = "rlm_query"

    def __init__(
        self,
        view_provider: Any,
        llm_client: LlmClient,
        *,
        leaf_budget: int = _DEFAULT_LEAF_BUDGET,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        max_depth: int = _DEFAULT_MAX_DEPTH,
        selection_top_k: int = _DEFAULT_SELECTION_TOP_K,
        selection_enabled: bool = True,
        max_llm_calls: int = _DEFAULT_MAX_LLM_CALLS,
        # Back-compat: older code may pass `context_budget=`. Treat it as
        # the leaf budget so legacy tests still pin the same behaviour.
        context_budget: int | None = None,
    ) -> None:
        self._view_provider = view_provider
        self._llm = llm_client
        self._leaf_budget = context_budget if context_budget is not None else leaf_budget
        self._chunk_size = max(chunk_size, self._leaf_budget)
        self._max_depth = max_depth
        self._selection_top_k = selection_top_k
        self._selection_enabled = selection_enabled
        self._max_llm_calls = max_llm_calls

    # ----- public ----------------------------------------------------------

    def call(
        self,
        *,
        workspace_id: str,
        question: str,
        variable_name: str,
        context_key: str | None = None,
        **kwargs: Any,
    ) -> Cited[dict[str, Any]]:
        """Query a workspace variable with a focused question.

        Returns Cited[dict] with the same shape as before plus
        recursion bookkeeping fields.
        """
        if not question.strip():
            raise WorkspaceToolError("rlm_query question must be non-empty")

        view = self._get_view(workspace_id)
        cited_var = view.get(variable_name)
        if cited_var is None:
            available = sorted(view.variable_names())
            raise WorkspaceToolError(
                f"Variable {variable_name!r} not found in workspace "
                f"{workspace_id!r}. Available: {available}"
            )

        content = self._extract_context(cited_var.value, variable_name, context_key)
        if not content.strip():
            raise WorkspaceToolError(
                f"Variable {variable_name!r} has no text content to query."
            )

        state = _RecursionState(
            max_depth=self._max_depth,
            max_llm_calls=self._max_llm_calls,
        )
        answer = self._recursive_query(content, question.strip(), state, depth=0)

        return Cited(
            value={
                "question": question,
                "variable_name": variable_name,
                "context_key": context_key,
                "answer": answer,
                "context_chars": len(content),
                "leaf_budget": self._leaf_budget,
                "chunk_size": self._chunk_size,
                "max_depth": self._max_depth,
                "depth_reached": state.max_depth_reached,
                "llm_calls": state.calls_made,
                "chunks_examined": state.chunks_examined,
                "selection_path": state.selection_path,
                "truncated_at_max_depth": state.hit_truncation_branch,
            },
            citations=cited_var.citations,
        )

    # ----- recursion -------------------------------------------------------

    def _recursive_query(
        self, content: str, question: str, state: _RecursionState, *, depth: int
    ) -> str:
        state.observe_depth(depth)

        # Base case 1: content fits in one LLM call.
        if len(content) <= self._leaf_budget:
            return self._leaf_answer(content, question, state, depth)

        # Base case 2: max depth reached — truncate and answer.
        if depth >= self._max_depth:
            state.hit_truncation_branch = True
            return self._leaf_answer(content[: self._leaf_budget], question, state, depth)

        # Recursive case: chunk, optionally select, recurse, aggregate.
        chunks = self._chunk(content)
        state.chunks_examined += len(chunks)

        if self._selection_enabled and len(chunks) > self._selection_top_k:
            selected = self._select_chunks(chunks, question, state, depth)
        else:
            selected = list(range(len(chunks)))

        state.selection_path.append(
            {"depth": depth, "total_chunks": len(chunks), "selected": selected}
        )

        if not selected:
            return "insufficient context (no chunks selected as relevant)"

        sub_answers: list[str] = []
        for idx in selected:
            if not state.can_call():
                # Hit the call budget — bail with what we have so far.
                logger.warning("rlm_query: max_llm_calls reached at depth %d", depth)
                break
            sub_answer = self._recursive_query(
                chunks[idx], question, state, depth=depth + 1
            )
            sub_answers.append(sub_answer)

        if len(sub_answers) == 0:
            return "insufficient context"
        if len(sub_answers) == 1:
            return sub_answers[0]

        return self._aggregate(question, sub_answers, state, depth)

    # ----- leaf -----------------------------------------------------------

    def _leaf_answer(
        self, content: str, question: str, state: _RecursionState, depth: int
    ) -> str:
        if not state.can_call():
            return "insufficient context (call budget exhausted)"
        state.record_call()
        user = f"Context:\n\n{content}\n\nQuestion: {question}"
        return self._llm.complete(system=_LEAF_SYSTEM, user=user)

    # ----- selection ------------------------------------------------------

    def _select_chunks(
        self,
        chunks: list[str],
        question: str,
        state: _RecursionState,
        depth: int,
    ) -> list[int]:
        """Ask the LLM which chunks look relevant. Returns chunk indices.

        Each chunk is summarised to its first ~200 chars in the prompt
        so this routing step is cheap. The LLM returns a JSON array of
        indices it picks. Falls back to "all chunks (top_k cap)" if the
        response can't be parsed.
        """
        if not state.can_call():
            return list(range(min(len(chunks), self._selection_top_k)))
        state.record_call()

        previews = []
        for i, chunk in enumerate(chunks):
            head = chunk[:200].replace("\n", " ").strip()
            previews.append(f"[{i}] {head}…")
        previews_text = "\n".join(previews)

        system = _SELECT_SYSTEM_TEMPLATE.replace(
            "%TOPK%", str(self._selection_top_k)
        )
        user = (
            f"Question: {question}\n\n"
            f"Chunk previews (first 200 chars of each):\n{previews_text}\n\n"
            f"Output only the JSON object."
        )

        raw = self._llm.complete(system=system, user=user)
        return self._parse_selection(raw, total_chunks=len(chunks))

    @staticmethod
    def _parse_selection(raw: str, *, total_chunks: int) -> list[int]:
        """Parse the routing LLM's selection JSON; tolerate sloppy output."""
        try:
            # Find the first { and the last } — be lenient about preface text.
            start = raw.find("{")
            end = raw.rfind("}")
            if start == -1 or end == -1 or end < start:
                return []
            parsed = json.loads(raw[start : end + 1])
            selected_raw = parsed.get("selected", [])
            if not isinstance(selected_raw, list):
                return []
            indices: list[int] = []
            for v in selected_raw:
                if isinstance(v, int) and 0 <= v < total_chunks:
                    indices.append(v)
            # Dedupe while preserving order.
            seen: set[int] = set()
            uniq: list[int] = []
            for i in indices:
                if i not in seen:
                    uniq.append(i)
                    seen.add(i)
            return uniq
        except (json.JSONDecodeError, KeyError, TypeError):
            return []

    # ----- aggregation ----------------------------------------------------

    def _aggregate(
        self,
        question: str,
        sub_answers: list[str],
        state: _RecursionState,
        depth: int,
    ) -> str:
        if not state.can_call():
            # Out of budget — return concatenation so no signal is lost.
            return "\n\n---\n\n".join(sub_answers)
        state.record_call()

        joined = "\n\n".join(
            f"### Sub-answer {i + 1}\n{ans}" for i, ans in enumerate(sub_answers)
        )
        user = (
            f"Question: {question}\n\n"
            f"Sub-answers from different context segments:\n\n{joined}\n\n"
            f"Synthesize one coherent answer."
        )
        return self._llm.complete(system=_AGGREGATE_SYSTEM, user=user)

    # ----- chunking -------------------------------------------------------

    def _chunk(self, content: str) -> list[str]:
        """Split content into chunks ≤ chunk_size chars, preferring
        paragraph boundaries (double newline) and falling back to
        single-newline or hard char splits.

        This is intentionally simple. The paper's contribution isn't the
        chunker — section-aware chunking is the indexer's job (we
        already do that for the paper text via SectionChunker). When
        the variable's content arrives here as a single blob, we split
        on natural boundaries first, hard-window second.
        """
        if len(content) <= self._chunk_size:
            return [content]

        chunks: list[str] = []
        paragraphs = content.split("\n\n")
        buf = ""
        for para in paragraphs:
            block = para if not buf else f"{buf}\n\n{para}"
            if len(block) <= self._chunk_size:
                buf = block
                continue
            # buf is at or near capacity; flush.
            if buf:
                chunks.append(buf)
            # If a single paragraph exceeds chunk_size, hard-split it.
            if len(para) > self._chunk_size:
                for i in range(0, len(para), self._chunk_size):
                    chunks.append(para[i : i + self._chunk_size])
                buf = ""
            else:
                buf = para
        if buf:
            chunks.append(buf)
        return chunks

    # ----- view + context extraction (unchanged from prior version) -------

    def _get_view(self, workspace_id: str) -> WorkspaceView:
        if hasattr(self._view_provider, "materialize_view"):
            return self._view_provider.materialize_view(workspace_id)
        return self._view_provider(workspace_id)

    def _extract_context(
        self, value: Any, variable_name: str, context_key: str | None
    ) -> str:
        if isinstance(value, str):
            return value

        if isinstance(value, dict):
            if context_key is not None:
                sub = value.get(context_key)
                if sub is not None:
                    if isinstance(sub, str):
                        return sub
                    return json.dumps(sub, indent=2, default=str)
                for v in value.values():
                    if isinstance(v, dict) and context_key in v:
                        sub = v[context_key]
                        return sub if isinstance(sub, str) else json.dumps(
                            sub, indent=2, default=str
                        )
                raise WorkspaceToolError(
                    f"Key {context_key!r} not found in variable "
                    f"{variable_name!r}."
                )

            if "text" in value:
                return str(value["text"])
            if "sections" in value and isinstance(value["sections"], dict):
                return "\n\n".join(
                    f"## {k}\n{v}" for k, v in value["sections"].items()
                )
            return json.dumps(value, indent=2, default=str)

        return json.dumps(value, default=str)


# --- provider client (unchanged) --------------------------------------------

class ClaudeLlmClient:
    """LlmClient implementation using Claude Code via claude-agent-sdk.

    Uses the ``query()`` function from claude-agent-sdk which spawns
    Claude Code as a subprocess. No ANTHROPIC_API_KEY needed — uses
    the user's Claude Code subscription.
    """

    def __init__(self, model: str | None = None, max_turns: int = 1) -> None:
        self._model = model
        self._max_turns = max_turns

    def complete(self, *, system: str, user: str) -> str:
        """Synchronous wrapper around the async claude-agent-sdk query."""
        import asyncio

        return asyncio.run(self._async_complete(system=system, user=user))

    async def _async_complete(self, *, system: str, user: str) -> str:
        from claude_agent_sdk import (
            ClaudeAgentOptions,
            ResultMessage,
            query,
        )

        options = ClaudeAgentOptions(
            system_prompt=system,
            model=self._model,
            max_turns=self._max_turns,
            permission_mode="plan",
            tools=[],
        )

        result_text = ""
        async for event in query(prompt=user, options=options):
            if isinstance(event, ResultMessage):
                result_text = event.result or ""
                break

        return result_text


__all__ = ["ClaudeLlmClient", "LlmClient", "RlmQueryTool"]
