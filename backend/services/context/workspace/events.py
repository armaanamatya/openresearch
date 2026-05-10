"""Workspace domain events (#16).

Per spec §5.5 the workspace events carry typed Scope, task_id,
parent_task_id, branch_id (cross-team #11 / #10 references). For the
slice we keep them optional at the schema level — the bridge to the
blackboard ships in a follow-up.
"""

from __future__ import annotations

from typing import Any, ClassVar

from backend.messaging.event import DomainEvent, register_event
from backend.schemas.citations import Citation, NonEmptyCitations
from backend.schemas.scope import Scope


@register_event
class WorkspaceCreated(DomainEvent):
    event_type: ClassVar[str] = "workspace_created"
    schema_version: ClassVar[int] = 1
    workspace_id: str
    project_id: str
    agent_name: str
    parent_workspace_id: str | None = None
    branch_id: str | None = None
    task_id: str | None = None


@register_event
class VariableLoaded(DomainEvent):
    event_type: ClassVar[str] = "variable_loaded"
    schema_version: ClassVar[int] = 1
    workspace_id: str
    variable_name: str
    value_payload: dict[str, Any]
    citations: NonEmptyCitations
    scope: Scope = Scope.private_to_parent
    source_agent: str | None = None


@register_event
class VariableEnriched(DomainEvent):
    """Emitted when an agent updates / enhances a previously-loaded variable.

    Distinct from VariableLoaded so the dashboard can render enrichment
    as a separate event class. Same invariants apply.
    """

    event_type: ClassVar[str] = "variable_enriched"
    schema_version: ClassVar[int] = 1
    workspace_id: str
    variable_name: str
    value_payload: dict[str, Any]
    citations: NonEmptyCitations
    scope: Scope = Scope.private_to_parent
    enriched_by: str | None = None


@register_event
class CitationAttached(DomainEvent):
    """An agent decision is recorded with its supporting citations.
    Not coupled to any single variable — covers e.g. verifier
    findings, reproduction-contract decisions."""

    event_type: ClassVar[str] = "citation_attached"
    schema_version: ClassVar[int] = 1
    workspace_id: str
    decision_id: str
    decision_payload: dict[str, Any]
    citations: NonEmptyCitations


@register_event
class ToolInvoked(DomainEvent):
    """A WorkspaceTool was called and produced a Cited[Any]. Recorded
    as a fact in the event stream so replay reproduces tool history."""

    event_type: ClassVar[str] = "tool_invoked"
    schema_version: ClassVar[int] = 1
    workspace_id: str
    tool_name: str
    arguments: dict[str, Any]
    result_payload: dict[str, Any]
    citations: NonEmptyCitations
    duration_ms: int = 0


@register_event
class WorkspaceReady(DomainEvent):
    event_type: ClassVar[str] = "workspace_ready"
    schema_version: ClassVar[int] = 1
    workspace_id: str
    variable_count: int


@register_event
class WorkspaceClosed(DomainEvent):
    event_type: ClassVar[str] = "workspace_closed"
    schema_version: ClassVar[int] = 1
    workspace_id: str
    reason: str


@register_event
class VariablePromoted(DomainEvent):
    """Emitted when a variable's scope is promoted (e.g., private → branch → global).

    Per spec §5.5, supports the three-level visibility model for
    cross-agent variable sharing.
    """

    event_type: ClassVar[str] = "variable_promoted"
    schema_version: ClassVar[int] = 1
    workspace_id: str
    variable_name: str
    old_scope: Scope
    new_scope: Scope
    promoted_by: str | None = None


__all__ = [
    "CitationAttached",
    "ToolInvoked",
    "VariableEnriched",
    "VariableLoaded",
    "VariablePromoted",
    "WorkspaceClosed",
    "WorkspaceCreated",
    "WorkspaceReady",
]
