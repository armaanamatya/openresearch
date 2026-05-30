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
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from backend.schemas.citations import Citation
from backend.services.context.workspace.model import Cited
from backend.services.context.workspace.projections import WorkspaceView
from backend.services.context.workspace.tools._retry import with_429_backoff
from backend.services.context.workspace.tools.interface import WorkspaceToolError


logger = logging.getLogger(__name__)


def _bundled_claude_child_pids() -> set[int]:
    """Return PIDs of bundled-claude subprocesses descended from this process.

    BUG-NEW-044 helper: ``claude_agent_sdk`` spawns its bundled CLI via
    ``asyncio.create_subprocess_exec``; the subprocess is internal to the SDK
    and not exposed to callers. To kill a wedged child after a timeout we
    discover it by walking child PIDs and matching the bundled-binary path.

    Uses stdlib ``pgrep -P`` (recursive via two-pass walk) — no psutil
    dependency. Fail-soft: returns an empty set on any error, since the
    caller only uses this for best-effort cleanup.
    """
    try:
        my_pid = os.getpid()
        # Recursively walk descendants: BFS so we get every level (the SDK
        # subprocess can itself spawn workers).
        descendants: set[int] = set()
        frontier = {my_pid}
        for _ in range(8):  # bound depth to prevent runaway
            if not frontier:
                break
            next_frontier: set[int] = set()
            for parent in frontier:
                try:
                    out = subprocess.run(
                        ["pgrep", "-P", str(parent)],
                        capture_output=True, text=True, timeout=2,
                    )
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    continue
                for line in out.stdout.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        child = int(line)
                    except ValueError:
                        continue
                    if child in descendants or child == my_pid:
                        continue
                    descendants.add(child)
                    next_frontier.add(child)
            frontier = next_frontier
        # Filter to bundled-claude only by checking COMM.
        bundled: set[int] = set()
        for pid in descendants:
            try:
                out = subprocess.run(
                    ["ps", "-p", str(pid), "-o", "command="],
                    capture_output=True, text=True, timeout=2,
                )
                if "claude_agent_sdk/_bundled/claude" in out.stdout:
                    bundled.add(pid)
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
        return bundled
    except Exception:  # noqa: BLE001 — best-effort discovery
        return set()

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


# --- sub-RLM stall handling (Phase 1 / FM-001/002/007) ----------------------
#
# The bundled claude-agent-sdk stream has NO read-idle timeout (Context7:
# `query()` "continues indefinitely if no ResultMessage"). The previous fix
# bounded a sub-call only by a TOTAL wall-clock cap and returned "" on timeout —
# indistinguishable from a real empty answer (FM-002). We add a per-event
# read-idle timeout and surface a non-empty, self-describing sentinel so the
# root can retry instead of treating a dead socket as the answer.

SUB_RLM_STALL_SENTINEL = "[SUB_RLM_STALL]"


def _stall_message(idle_s: float) -> str:
    """Root-facing surface for a stalled sub-call (rlm's complete() contract is -> str)."""
    return (
        f"{SUB_RLM_STALL_SENTINEL} the sub-query stalled (no stream bytes for "
        f"{idle_s:.0f}s) and was aborted. Retry with a smaller slice or fewer "
        f"concurrent sub-calls; do NOT treat this as the answer."
    )


class _SubRlmReadIdleTimeout(Exception):
    """Raised inside the worker when the SDK stream is idle past read_idle_s."""

    def __init__(self, idle_s: float):
        super().__init__(f"sub-RLM stream idle > {idle_s:.0f}s")
        self.idle_s = idle_s


def _read_idle_default() -> float:
    """Default per-event read-idle bound; <=0 disables (env REPROLAB_SUBRLM_READ_IDLE_S)."""
    try:
        return float(os.environ.get("REPROLAB_SUBRLM_READ_IDLE_S", "120") or "120")
    except (TypeError, ValueError):
        return 120.0


# --- provider client --------------------------------------------------------

class ClaudeLlmClient:
    """LlmClient implementation using Claude Code via claude-agent-sdk.

    Uses the ``query()`` function from claude-agent-sdk which spawns
    Claude Code as a subprocess. No ANTHROPIC_API_KEY needed — uses
    the user's Claude Code subscription.

    Token usage is captured from ``ResultMessage.usage`` on every call and
    stored on ``_last_usage`` (a dict in the CostLedgerEntry.from_usage shape).
    Callers that need the token counts for cost-ledger recording can read
    ``client._last_usage`` immediately after ``complete()`` returns.
    """

    def __init__(
        self,
        model: str | None = None,
        max_turns: int = 1,
        stall_event_sink: Callable[[dict], None] | None = None,
    ) -> None:
        self._model = model
        self._max_turns = max_turns
        # Last-call token usage — populated by _async_complete, consumed by
        # callers (e.g. binding.py's _ledger) for cost-ledger recording.
        # Defaults to all-zeros so callers can always read it safely.
        self._last_usage: dict[str, int] = _ZERO_USAGE.copy()
        # Phase 2 wires this to the dashboard emit; default None = log only.
        self._stall_event_sink = stall_event_sink
        # Set by complete() before each scheduling so _async_complete can read it.
        self._read_idle_s: float = _read_idle_default()

    @with_429_backoff
    def complete(self, *, system: str, user: str, read_idle_s: float | None = None) -> str:
        """Synchronous wrapper around the async claude-agent-sdk query.

        ``read_idle_s`` (Phase 1) bounds the stream by *idle* time between events,
        not total duration — a half-open/stalled socket is aborted after
        ``read_idle_s`` seconds with no bytes (default ``REPROLAB_SUBRLM_READ_IDLE_S``
        = 120; ``<=0`` disables). On a stall the wedged child is killed and a
        non-empty ``SUB_RLM_STALL_SENTINEL`` string is returned (NOT ``""``) so the
        root can distinguish a dead socket from a real empty answer (FM-002).

        Always thread-isolated: the bundled claude-agent-sdk has a reliable
        nested-generator ``aclose()`` race (Defect 1) and a separate futex
        hang in ``transport.close()`` (Defect 2) — see
        ``docs/superpowers/specs/2026-05-22-sdk-aclose-investigation.md``.
        Running ``asyncio.run`` in a dedicated worker thread with
        ``shutdown(wait=False)`` contains both defects: the SDK's loop-bound
        async generators are created and torn down inside the worker's own
        loop, never the caller's, and an abandoned worker is left to GC at
        process exit. Mirrors the rdr ``_run_sdk_in_thread`` fix
        (commit 33c787d).

        Token usage is captured from ``ResultMessage.usage`` and stored on
        ``self._last_usage`` for the caller to retrieve.
        """
        import asyncio
        import concurrent.futures

        # BUG-NEW-044 (2026-05-29): the TOTAL wall-clock backstop. Successful
        # sub-RLM calls were 1–7 min; 600s is the generous ceiling. Phase 1 adds
        # a per-event READ-IDLE bound (below) that fires first on a stalled stream
        # — total-time alone cannot tell a dead socket from a slow-but-healthy one.
        _timeout_s = 600.0
        self._read_idle_s = read_idle_s if read_idle_s is not None else _read_idle_default()

        coro_factory = lambda: self._async_complete(system=system, user=user)

        # Snapshot bundled-claude children BEFORE the call so we can SIGKILL the
        # wedged one afterwards: ``ex.shutdown(wait=False)`` does NOT kill the
        # asyncio.run worker thread or its spawned subprocess, and Python threads
        # aren't killable from outside, so without an explicit kill a wedged
        # ``claude`` child sits in ``kevent64`` holding the OAuth slot for hours.
        _pre_pids = _bundled_claude_child_pids()

        ex = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="rlm-query-sdk-worker"
        )
        try:
            future = ex.submit(lambda: asyncio.run(coro_factory()))
            try:
                text, usage = future.result(timeout=_timeout_s)
            except _SubRlmReadIdleTimeout as stall:
                # Phase 1: a pure stall (no salvageable text) past the read-idle
                # bound. Kill the wedged child, surface a typed sentinel.
                self._kill_wedged_children(_pre_pids)
                self._notify_stall(stall.idle_s)
                logger.warning(
                    "rlm_query: SUB_RLM_STALL read-idle %.0fs — killed wedged "
                    "child(ren); returning sentinel so the root can retry.",
                    stall.idle_s,
                )
                return _stall_message(stall.idle_s)
            except concurrent.futures.TimeoutError:
                # Total-time backstop. Same handling — return the sentinel, not "".
                self._kill_wedged_children(_pre_pids)
                self._notify_stall(_timeout_s)
                logger.warning(
                    "rlm_query: SUB_RLM_STALL total %.0fs — killed wedged "
                    "child(ren); returning sentinel.",
                    _timeout_s,
                )
                return _stall_message(_timeout_s)
            self._last_usage = usage
            return text
        finally:
            ex.shutdown(wait=False)

    def _kill_wedged_children(self, pre_pids: set[int]) -> None:
        """SIGKILL bundled-claude children spawned by THIS call (fail-soft, cross-platform).

        The SDK spawns the CLI outside our process group, so we can't ``killpg`` a
        group we own; we diff post-vs-pre child PIDs and SIGKILL the new ones, with a
        best-effort ``killpg`` for any grandchildren. Non-POSIX / pgrep-less hosts
        no-op. Never raises (D3).
        """
        import os as _os
        import signal as _signal
        import time as _time

        if not hasattr(_os, "kill"):
            return
        try:
            wedged = _bundled_claude_child_pids() - pre_pids
        except Exception:  # noqa: BLE001 — best-effort discovery
            return
        for pid in wedged:
            try:
                try:  # best-effort grandchildren via the child's own group
                    _os.killpg(_os.getpgid(pid), _signal.SIGKILL)
                except (OSError, ProcessLookupError, AttributeError):
                    pass
                _os.kill(pid, _signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
        if wedged:
            _time.sleep(0.2)  # let the OS reap zombies before the next snapshot

    def _notify_stall(self, idle_s: float) -> None:
        """Best-effort: push a sub_rlm_stalled event dict to the sink (Phase 2 wires it)."""
        if self._stall_event_sink is None:
            return
        try:
            self._stall_event_sink({
                "event": "sub_rlm_stalled",
                "model": self._model or "unknown",
                "idle_seconds": float(idle_s),
            })
        except Exception:  # noqa: BLE001 — observability never blocks
            logger.debug("rlm_query: stall_event_sink raised", exc_info=True)

    async def _async_complete(self, *, system: str, user: str) -> tuple[str, dict[str, int]]:
        """Return (result_text, usage_dict) from the SDK stream.

        Usage dict has the CostLedgerEntry.from_usage shape:
        input_tokens, output_tokens, cache_creation_input_tokens,
        cache_read_input_tokens, reasoning_tokens.
        """
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ResultMessage,
            query,
        )
        from backend.services.pricing.token_accumulator import TokenAccumulator

        # BUG-NEW-038 (2026-05-29): isolate from the user's outer claude-agent-sdk
        # config. Without `setting_sources=[]` and `mcp_servers={}`, the SDK
        # inherits ~/.claude/settings.json, the user's MCP servers, and any plan
        # mode they have on. The inner root model then sees those tools in its
        # inventory and refuses to write REPL code ("I have no execution surface,
        # only Google Drive / Sentry / Context7 / etc."). `permission_mode="plan"`
        # also leaked through and told the model "plan mode active — block all
        # execution," which is the wrong signal for a max_turns=1 text generation
        # call. Pin to a clean, isolated, no-tools text completion.
        options = ClaudeAgentOptions(
            system_prompt=system,
            model=self._model,
            max_turns=self._max_turns,
            # "default" (NOT "plan"): in plan mode Claude Code expects to present
            # a plan via ExitPlanMode rather than emit a direct answer, so with
            # tools=[] the ResultMessage.result comes back EMPTY for any prompt
            # that asks the model to *do work* — e.g. the RLM root's "write
            # Python calling these primitives". That produced empty root
            # iterations and an immediate death-spiral (verified 2026-05-29:
            # "plan" -> result_len=0; "default" -> a real ```python block).
            # tools=[] already forbids any tool use / file edit, so "default" is
            # side-effect-free here.
            permission_mode="default",
            tools=[],
            mcp_servers={},
            setting_sources=[],
        )

        result_text = ""
        assistant_parts: list[str] = []
        assistant_usages: list = []
        result_usage = None
        acc = TokenAccumulator()
        # Consume the SDK stream and BREAK as soon as the ResultMessage arrives.
        # We deliberately do NOT drain to exhaustion and do NOT call agen.aclose()
        # ourselves. The bundled claude-agent-sdk has a transport.close() futex
        # hang (Defect 2 — docs/superpowers/specs/2026-05-22-sdk-aclose-investigation.md):
        # draining awaits a next message that never arrives after the subprocess
        # exits, and an explicit aclose() can trip that futex hang. Either wedges
        # this worker thread, and complete()'s future.result() has NO timeout, so
        # the whole run hangs. Letting asyncio.run() tear the suspended generator
        # down only hits the benign Defect-1 "aclose(): already running" race,
        # which is logged and harmless. (Verified 2026-05-29: drain + explicit
        # aclose -> 20-min wedge, 0 iterations; break + no explicit aclose -> clean.)
        # Phase 1: read-idle loop. We advance the async generator one event at a
        # time, each bounded by ``read_idle_s`` of *idle* (no-bytes) time. A stalled
        # / half-open stream raises ``_SubRlmReadIdleTimeout`` so complete() can kill
        # the child and surface the sentinel. ``read_idle_s <= 0`` disables the bound
        # (plain ``__anext__`` with no per-event timeout). We still do NOT drain to
        # exhaustion or call aclose() (Defect 2 futex hang) — break on ResultMessage.
        import asyncio

        read_idle_s = self._read_idle_s
        agen = query(prompt=user, options=options)
        aiter = agen.__aiter__()
        try:
            while True:
                try:
                    if read_idle_s and read_idle_s > 0:
                        event = await asyncio.wait_for(
                            aiter.__anext__(), timeout=read_idle_s
                        )
                    else:
                        event = await aiter.__anext__()
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError as exc:
                    raise _SubRlmReadIdleTimeout(read_idle_s) from exc
                if isinstance(event, AssistantMessage):
                    # Salvage streamed assistant text + usage: the SDK RAISES on
                    # an error ResultMessage (commonly "Reached maximum number of
                    # turns (1)") even though the model's answer already streamed
                    # here. Recovering it keeps the RLM root loop alive instead of
                    # failing the whole run.
                    for block in (getattr(event, "content", None) or []):
                        txt = getattr(block, "text", None)
                        if txt:
                            assistant_parts.append(txt)
                    usage = getattr(event, "usage", None)
                    if usage is not None:
                        assistant_usages.append(usage)
                elif isinstance(event, ResultMessage):
                    result_text = event.result or ""
                    if event.usage is not None:
                        result_usage = event.usage
                    break
        except _SubRlmReadIdleTimeout:
            # Pure stall with no salvageable text → propagate so complete() returns
            # the sentinel. Partial text already streamed → keep it (partial > stall).
            if not assistant_parts:
                raise
            logger.warning(
                "rlm_query: read-idle %.0fs but %d assistant part(s) salvaged",
                read_idle_s, len(assistant_parts),
            )
        except Exception as exc:  # noqa: BLE001 — salvage over crash
            logger.warning(
                "rlm_query: claude-agent-sdk stream raised (%s); salvaging "
                "%d assistant text part(s)",
                exc,
                len(assistant_parts),
            )

        # Text: prefer a clean ResultMessage; else the salvaged assistant text.
        if not result_text and assistant_parts:
            result_text = "".join(assistant_parts)
        # Usage: ResultMessage is authoritative; fall back to summing the
        # per-AssistantMessage usages only when no ResultMessage arrived (avoids
        # double-counting the cumulative ResultMessage total).
        if result_usage is not None:
            acc.absorb_usage(result_usage)
        else:
            for usage in assistant_usages:
                acc.absorb_usage(usage)

        return result_text, acc.as_dict()


# Zero-usage sentinel — used as the default for _last_usage before any call.
_ZERO_USAGE: dict[str, int] = {
    "input_tokens": 0,
    "output_tokens": 0,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
    "reasoning_tokens": 0,
}


__all__ = ["ClaudeLlmClient", "LlmClient", "RlmQueryTool"]
