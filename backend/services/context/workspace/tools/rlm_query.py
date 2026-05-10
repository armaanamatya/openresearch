"""RlmQueryTool — recursive LLM sub-query over a workspace variable.

This is the core Layer 1 capability from the RLM paper (arXiv:2512.24601):
instead of stuffing all context into a prompt, the agent issues focused
sub-queries against specific context segments. Each sub-query uses ~2-3k
tokens vs 95k+ for naive prompt stuffing.

The tool:
  1. Loads the target variable from the workspace view
  2. Truncates to a context budget (default 4000 chars)
  3. Calls the LLM with a focused question + context
  4. Returns Cited[dict] with the answer and source citations
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from backend.schemas.citations import Citation
from backend.services.context.workspace.model import Cited
from backend.services.context.workspace.projections import WorkspaceView
from backend.services.context.workspace.tools.interface import WorkspaceToolError


_DEFAULT_CONTEXT_BUDGET = 4000


class LlmClient(Protocol):
    """Minimal synchronous LLM interface for workspace tools.

    Implementations can wrap OpenAI, Anthropic, or any provider.
    Tests use a simple stub.
    """

    def complete(self, *, system: str, user: str) -> str:
        """Return a completion string for the given system+user prompt."""
        ...


class RlmQueryTool:
    """Recursive LLM sub-query over a workspace variable's content.

    Unlike SemanticSearchTool (keyword retrieval) or LookupTool (exact
    source lookup), this tool uses an LLM to reason about a specific
    context segment and answer a focused question.
    """

    name = "rlm_query"

    def __init__(
        self,
        view_provider: Any,
        llm_client: LlmClient,
        context_budget: int = _DEFAULT_CONTEXT_BUDGET,
    ) -> None:
        self._view_provider = view_provider
        self._llm = llm_client
        self._context_budget = context_budget

    def _get_view(self, workspace_id: str) -> WorkspaceView:
        if hasattr(self._view_provider, "materialize_view"):
            return self._view_provider.materialize_view(workspace_id)
        return self._view_provider(workspace_id)

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

        Args:
            workspace_id: Target workspace.
            question: The question to answer.
            variable_name: Which variable to use as context.
            context_key: Optional key to drill into a dict variable
                         (e.g., "Methods" for paper_sections).
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

        # Extract context text from the variable value.
        context_text = self._extract_context(
            cited_var.value, variable_name, context_key
        )
        if not context_text.strip():
            raise WorkspaceToolError(
                f"Variable {variable_name!r} has no text content to query."
            )

        # Truncate to budget.
        truncated = context_text[: self._context_budget]

        system_prompt = (
            "You are a research assistant. Answer the question based ONLY on "
            "the provided context. If the context does not contain enough "
            "information, say so explicitly. Be precise and cite specific "
            "parts of the context."
        )
        user_prompt = (
            f"Context (from variable '{variable_name}'"
            + (f", key '{context_key}'" if context_key else "")
            + f"):\n\n{truncated}\n\n"
            f"Question: {question}"
        )

        answer = self._llm.complete(system=system_prompt, user=user_prompt)

        return Cited(
            value={
                "question": question,
                "variable_name": variable_name,
                "context_key": context_key,
                "answer": answer,
                "context_chars": len(truncated),
                "truncated": len(context_text) > self._context_budget,
            },
            citations=cited_var.citations,
        )

    def _extract_context(
        self, value: Any, variable_name: str, context_key: str | None
    ) -> str:
        """Extract a text string from a variable's value payload."""
        if isinstance(value, str):
            return value

        if isinstance(value, dict):
            # If context_key specified, drill into that key.
            if context_key is not None:
                # Try the key directly (e.g., paper_sections["Methods"]).
                sub = value.get(context_key)
                if sub is not None:
                    if isinstance(sub, str):
                        return sub
                    return json.dumps(sub, indent=2, default=str)
                # Try nested dict (e.g., paper_sections.sections["Methods"]).
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

            # No context_key: extract the most text-like field.
            if "text" in value:
                return str(value["text"])
            if "sections" in value and isinstance(value["sections"], dict):
                return "\n\n".join(
                    f"## {k}\n{v}" for k, v in value["sections"].items()
                )
            # Fallback: serialize the whole dict.
            return json.dumps(value, indent=2, default=str)

        return json.dumps(value, default=str)


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
