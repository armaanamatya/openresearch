"""WorkspaceProjection — materializes Cited[T] variables from events.

Replays a workspace's event stream into a `WorkspaceView` whose
`variables` dict maps name -> Cited[Any]. Materialization is the
4th defense layer of the citation invariant (spec §5.6): even an
adversarial event in storage cannot produce a `Cited[T]` without
non-empty citations because the constructor raises.
"""

from __future__ import annotations

from typing import Any

from backend.schemas.citations import Citation
from backend.schemas.scope import Scope
from backend.services.context.workspace.model import Cited


class WorkspaceView:
    """Read-side materialization of a workspace.

    Tracks variables as Cited[Any] with per-variable scope metadata.
    """

    def __init__(self, workspace_id: str) -> None:
        self._workspace_id = workspace_id
        self._variables: dict[str, Cited[Any]] = {}
        self._scopes: dict[str, Scope] = {}
        self._is_ready = False

    @property
    def workspace_id(self) -> str:
        return self._workspace_id

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def get(self, name: str) -> Cited[Any] | None:
        return self._variables.get(name)

    def get_scope(self, name: str) -> Scope | None:
        return self._scopes.get(name)

    def variable_names(self) -> list[str]:
        return list(self._variables.keys())

    @property
    def variables(self) -> dict[str, Cited[Any]]:
        # Defensive copy — callers shouldn't mutate the projection.
        return dict(self._variables)

    @property
    def variable_count(self) -> int:
        return len(self._variables)

    # --- Apply (used by the projection) -----------------------------------

    def _apply_variable(
        self,
        name: str,
        value_payload: dict[str, Any],
        citations_payload: list[dict[str, Any]] | tuple[dict[str, Any], ...],
        scope: Scope = Scope.private_to_parent,
    ) -> None:
        cites = tuple(Citation.model_validate(c) for c in citations_payload)
        # Cited.__post_init__ raises CitationMissingError on empty —
        # but the event's NonEmptyCitations validator already rejected
        # that on the way IN. Belt + suspenders.
        self._variables[name] = Cited.from_payload(value=value_payload, citations=cites)
        self._scopes[name] = scope

    def _promote_variable(self, name: str, new_scope: Scope) -> None:
        self._scopes[name] = new_scope

    def _set_ready(self) -> None:
        self._is_ready = True


class WorkspaceProjection:
    """Tracks WorkspaceViews keyed by workspace_id."""

    def __init__(self) -> None:
        self._views: dict[str, WorkspaceView] = {}

    def view(self, workspace_id: str) -> WorkspaceView:
        view = self._views.get(workspace_id)
        if view is None:
            view = WorkspaceView(workspace_id)
            self._views[workspace_id] = view
        return view

    def apply_workspace_created(self, workspace_id: str) -> None:
        # Initialize the view; no variables yet.
        self.view(workspace_id)

    def apply_variable_loaded(
        self,
        *,
        workspace_id: str,
        variable_name: str,
        value_payload: dict[str, Any],
        citations_payload: list[dict[str, Any]] | tuple[dict[str, Any], ...],
        scope: Scope = Scope.private_to_parent,
    ) -> None:
        self.view(workspace_id)._apply_variable(
            variable_name, value_payload, citations_payload, scope
        )

    def apply_variable_enriched(
        self,
        *,
        workspace_id: str,
        variable_name: str,
        value_payload: dict[str, Any],
        citations_payload: list[dict[str, Any]] | tuple[dict[str, Any], ...],
        scope: Scope = Scope.private_to_parent,
    ) -> None:
        # Same shape as loaded — just overwrites.
        self.view(workspace_id)._apply_variable(
            variable_name, value_payload, citations_payload, scope
        )

    def apply_variable_promoted(
        self,
        *,
        workspace_id: str,
        variable_name: str,
        new_scope: Scope,
    ) -> None:
        self.view(workspace_id)._promote_variable(variable_name, new_scope)

    def apply_workspace_ready(self, workspace_id: str) -> None:
        self.view(workspace_id)._set_ready()


__all__ = ["WorkspaceProjection", "WorkspaceView"]
