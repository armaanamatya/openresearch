"""InspectVariableTool — retrieve a variable's full value and citations.

Agents use this to drill into a specific variable discovered via
ListVariablesTool. Returns the complete Cited[dict] for the variable.
"""

from __future__ import annotations

from typing import Any

from backend.services.context.workspace.model import Cited
from backend.services.context.workspace.projections import WorkspaceView
from backend.services.context.workspace.tools.interface import WorkspaceToolError


class InspectVariableTool:
    """Retrieve a workspace variable's value and all citations."""

    name = "inspect_variable"

    def __init__(self, view_provider: Any) -> None:
        self._view_provider = view_provider

    def _get_view(self, workspace_id: str) -> WorkspaceView:
        if hasattr(self._view_provider, "materialize_view"):
            return self._view_provider.materialize_view(workspace_id)
        return self._view_provider(workspace_id)

    def call(
        self, *, workspace_id: str, variable_name: str, **kwargs: Any
    ) -> Cited[dict[str, Any]]:
        view = self._get_view(workspace_id)
        cited_var = view.get(variable_name)
        if cited_var is None:
            available = sorted(view.variable_names())
            raise WorkspaceToolError(
                f"Variable {variable_name!r} not found in workspace "
                f"{workspace_id!r}. Available: {available}"
            )
        scope = view.get_scope(variable_name)
        value: dict[str, Any] = {
            "variable_name": variable_name,
            "scope": scope.value if scope else "private_to_parent",
            "value": cited_var.value,
            "citation_count": len(cited_var.citations),
        }
        return Cited(value=value, citations=cited_var.citations)


__all__ = ["InspectVariableTool"]
