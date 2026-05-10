"""ListVariablesTool — enumerate all variables in a workspace.

Agents use this to discover what context is available before drilling
into specific variables with InspectVariableTool or RlmQueryTool.
"""

from __future__ import annotations

from typing import Any

from backend.schemas.citations import Citation
from backend.services.context.workspace.model import Cited
from backend.services.context.workspace.projections import WorkspaceView
from backend.services.context.workspace.tools.interface import WorkspaceToolError


class ListVariablesTool:
    """List all variables in a workspace with metadata."""

    name = "list_variables"

    def __init__(self, view_provider: Any) -> None:
        """Accept a callable(workspace_id) -> WorkspaceView or a
        WorkspaceAppService (duck-typed via materialize_view)."""
        self._view_provider = view_provider

    def _get_view(self, workspace_id: str) -> WorkspaceView:
        if hasattr(self._view_provider, "materialize_view"):
            return self._view_provider.materialize_view(workspace_id)
        return self._view_provider(workspace_id)

    def call(self, *, workspace_id: str, **kwargs: Any) -> Cited[dict[str, Any]]:
        view = self._get_view(workspace_id)
        if not view.variable_names():
            raise WorkspaceToolError(
                f"Workspace {workspace_id!r} has no variables."
            )

        entries: list[dict[str, Any]] = []
        citations: list[Citation] = []
        for name in sorted(view.variable_names()):
            cited_var = view.get(name)
            if cited_var is None:
                continue
            scope = view.get_scope(name)
            entry = {
                "variable_name": name,
                "citation_count": len(cited_var.citations),
                "scope": scope.value if scope else "private_to_parent",
            }
            # Add value type hint based on payload keys.
            if isinstance(cited_var.value, dict):
                entry["keys"] = sorted(cited_var.value.keys())
            entries.append(entry)
            # Use first citation from each variable as evidence.
            citations.append(cited_var.citations[0])

        return Cited(
            value={
                "workspace_id": workspace_id,
                "variable_count": len(entries),
                "variables": entries,
            },
            citations=tuple(citations),
        )


__all__ = ["ListVariablesTool"]
