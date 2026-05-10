"""WorkspaceAppService — builds workspaces, preloads variables, manages lifecycle.

Preloads three variables from indexed sources:
  - `paper_text`: full paper text concatenated from all section chunks
  - `paper_sections`: dict mapping section locator -> section text
  - `claim_map`: one entry per section with source_id, title, excerpt

Also provides `enrich_variable()` for progressive enrichment,
`close_workspace()` for lifecycle cleanup, and `promote_variable()`
for scope transitions.
"""

from __future__ import annotations

import hashlib
from typing import Any

from pydantic import ConfigDict

from backend.eventstore.interface import EventStore
from backend.messaging.command import Command
from backend.messaging.envelope import (
    AggregateId,
    CorrelationId,
    EventEnvelope,
    make_envelope,
    new_correlation_id,
)
from backend.messaging.event import DomainEvent, resolve_event_class
from backend.schemas.citations import Citation, NonEmptyCitations
from backend.schemas.scope import Scope
from backend.services.context.indexer.projections import SourcesProjection
from backend.services.context.indexer.service import IndexerAppService
from backend.services.context.indexer.aggregate import IndexAggregate, IndexState
from backend.services.context.indexer.model import SourceKind
from backend.services.context.workspace.aggregate import (
    InvalidWorkspaceTransition,
    WorkspaceAggregate,
    WorkspaceState,
)
from backend.services.context.workspace.events import (
    VariableEnriched,
    VariableLoaded,
    VariablePromoted,
    WorkspaceClosed,
    WorkspaceCreated,
    WorkspaceReady,
)
from backend.services.context.workspace.projections import (
    WorkspaceProjection,
    WorkspaceView,
)


_CLAIM_MAP_QUOTE_TRUNCATE = 240


class BuildWorkspace(Command):
    model_config = ConfigDict(frozen=True)
    project_id: str
    agent_name: str = "default"
    workspace_id: str | None = None
    """Optional: caller can pin a workspace_id for deterministic e2e
    tests. When None, the service derives one from project + agent."""


class WorkspaceError(Exception):
    pass


def _workspace_id_for(project_id: str, agent_name: str) -> str:
    h = hashlib.sha256(f"workspace:{project_id}:{agent_name}".encode())
    return f"ws_{h.hexdigest()[:16]}"


def _workspace_aggregate_id(workspace_id: str) -> AggregateId:
    return AggregateId(workspace_id)


def _index_aggregate_id(project_id: str) -> AggregateId:
    return AggregateId(f"{project_id}:index")


class WorkspaceAppService:
    def __init__(self, store: EventStore, indexer: IndexerAppService) -> None:
        self._store = store
        self._indexer = indexer

    # --- Public ------------------------------------------------------------

    def build_workspace(
        self,
        cmd: BuildWorkspace,
        *,
        correlation_id: CorrelationId | None = None,
    ) -> str:
        """Build a workspace for `project_id`. Returns the workspace_id.

        - Validates the index is INDEXED.
        - Creates the workspace aggregate (idempotent on re-issue).
        - Preloads `paper_text`, `paper_sections`, and `claim_map`.
        - Emits WorkspaceReady.
        """
        cid = correlation_id or new_correlation_id()
        workspace_id = cmd.workspace_id or _workspace_id_for(
            cmd.project_id, cmd.agent_name
        )

        # Guard: index must be ready.
        index = self._load_index_aggregate(cmd.project_id)
        if index.state is not IndexState.INDEXED:
            raise WorkspaceError(
                f"Cannot build workspace for {cmd.project_id!r}: index is in state "
                f"{index.state.value!r}; must be {IndexState.INDEXED.value!r}."
            )

        agg = self._load_workspace_aggregate(workspace_id)
        if agg.state is WorkspaceState.READY:
            return workspace_id  # idempotent — already built
        if agg.state is WorkspaceState.LOADING:
            # Re-issue mid-load: complete the load.
            pass

        # Step 1: WorkspaceCreated (only if NEW).
        if agg.state is WorkspaceState.NEW:
            try:
                events = list(
                    agg.handle_create(
                        project_id=cmd.project_id, agent_name=cmd.agent_name
                    )
                )
            except InvalidWorkspaceTransition as exc:
                raise WorkspaceError(str(exc)) from exc
            self._append(agg, workspace_id, events, cid)

        # Step 2: build the SourcesProjection by replaying the index events.
        proj = SourcesProjection()
        self._indexer.project_into_projection(cmd.project_id, proj)

        sections = sorted(
            (s for s in proj.list_sources(cmd.project_id)
             if s.kind is SourceKind.paper_section),
            key=lambda s: s.id,
        )

        if not sections:
            # Empty project — emit a structural stub so workspace is valid.
            stub_citation = Citation(
                source_id=f"project:{cmd.project_id}",
                chunk_id=None,
                quote=f"<project {cmd.project_id}: no indexed sections>",
                locator=cmd.project_id,
                confidence=0.5,
            )
            stub_event = VariableLoaded(
                workspace_id=workspace_id,
                variable_name="paper_text",
                value_payload={"text": "", "project_id": cmd.project_id},
                citations=(stub_citation,),
                scope=Scope.private_to_parent,
                source_agent="workspace_service",
            )
            self._append(agg, workspace_id, [stub_event], cid)
        else:
            # Step 3a: preload `paper_text` — full concatenated text.
            paper_text_parts: list[str] = []
            text_citations: list[Citation] = []
            for src in sections:
                chunks = proj.chunks_for_source(src.id)
                section_text = " ".join(c.text for c in chunks)
                paper_text_parts.append(section_text)
                if chunks:
                    text_citations.append(Citation(
                        source_id=src.id,
                        chunk_id=chunks[0].id,
                        quote=chunks[0].text[:_CLAIM_MAP_QUOTE_TRUNCATE],
                        locator=src.locator,
                        confidence=1.0,
                    ))
            if text_citations:
                self._append(agg, workspace_id, [VariableLoaded(
                    workspace_id=workspace_id,
                    variable_name="paper_text",
                    value_payload={
                        "text": "\n\n".join(paper_text_parts),
                        "project_id": cmd.project_id,
                    },
                    citations=tuple(text_citations),
                    scope=Scope.private_to_parent,
                    source_agent="workspace_service",
                )], cid)

            # Step 3b: preload `paper_sections` — dict of locator -> text.
            sections_dict: dict[str, str] = {}
            section_citations: list[Citation] = []
            for src in sections:
                chunks = proj.chunks_for_source(src.id)
                section_text = " ".join(c.text for c in chunks)
                sections_dict[src.locator] = section_text
                if chunks:
                    section_citations.append(Citation(
                        source_id=src.id,
                        chunk_id=chunks[0].id,
                        quote=chunks[0].text[:_CLAIM_MAP_QUOTE_TRUNCATE],
                        locator=src.locator,
                        confidence=1.0,
                    ))
            if section_citations:
                self._append(agg, workspace_id, [VariableLoaded(
                    workspace_id=workspace_id,
                    variable_name="paper_sections",
                    value_payload={
                        "sections": sections_dict,
                        "project_id": cmd.project_id,
                    },
                    citations=tuple(section_citations),
                    scope=Scope.private_to_parent,
                    source_agent="workspace_service",
                )], cid)

            # Step 3c: preload `claim_map` — one entry per section source.
            claim_map = self._build_claim_map(cmd.project_id, proj)
            citations = self._claim_map_citations(proj)
            self._append(agg, workspace_id, [VariableLoaded(
                workspace_id=workspace_id,
                variable_name="claim_map",
                value_payload=claim_map,
                citations=citations,
                scope=Scope.private_to_parent,
                source_agent="workspace_service",
            )], cid)

        # Step 4: WorkspaceReady.
        ready = WorkspaceReady(
            workspace_id=workspace_id, variable_count=agg.variable_count
        )
        self._append(agg, workspace_id, [ready], cid)
        return workspace_id

    def enrich_variable(
        self,
        *,
        workspace_id: str,
        variable_name: str,
        value_payload: dict[str, Any],
        citations: NonEmptyCitations,
        enriched_by: str,
        scope: Scope = Scope.private_to_parent,
        correlation_id: CorrelationId | None = None,
    ) -> None:
        """Add or overwrite a workspace variable via progressive enrichment.

        Called by the orchestrator when an agent completes and its
        structured outputs need to become queryable variables for
        downstream agents.
        """
        cid = correlation_id or new_correlation_id()
        agg = self._load_workspace_aggregate(workspace_id)
        if agg.state not in (WorkspaceState.LOADING, WorkspaceState.READY):
            raise WorkspaceError(
                f"Cannot enrich variable in workspace {workspace_id!r}: "
                f"state is {agg.state.value!r}, must be 'loading' or 'ready'."
            )
        event = VariableEnriched(
            workspace_id=workspace_id,
            variable_name=variable_name,
            value_payload=value_payload,
            citations=citations,
            scope=scope,
            enriched_by=enriched_by,
        )
        self._append(agg, workspace_id, [event], cid)

    def close_workspace(
        self,
        *,
        workspace_id: str,
        reason: str = "completed",
        correlation_id: CorrelationId | None = None,
    ) -> None:
        """Close a workspace. Idempotent on already-closed workspaces."""
        cid = correlation_id or new_correlation_id()
        agg = self._load_workspace_aggregate(workspace_id)
        if agg.state is WorkspaceState.CLOSED:
            return  # idempotent
        if agg.state is WorkspaceState.NEW:
            raise WorkspaceError(
                f"Cannot close workspace {workspace_id!r}: it was never created."
            )
        event = WorkspaceClosed(workspace_id=workspace_id, reason=reason)
        self._append(agg, workspace_id, [event], cid)

    def promote_variable(
        self,
        *,
        workspace_id: str,
        variable_name: str,
        new_scope: Scope,
        promoted_by: str | None = None,
        correlation_id: CorrelationId | None = None,
    ) -> None:
        """Promote a variable's scope (e.g., private_to_parent -> branch_shared)."""
        cid = correlation_id or new_correlation_id()
        agg = self._load_workspace_aggregate(workspace_id)
        if agg.state not in (WorkspaceState.LOADING, WorkspaceState.READY):
            raise WorkspaceError(
                f"Cannot promote variable in workspace {workspace_id!r}: "
                f"state is {agg.state.value!r}."
            )
        # Determine current scope from the view.
        view = self.materialize_view(workspace_id)
        if view.get(variable_name) is None:
            raise WorkspaceError(
                f"Variable {variable_name!r} not found in workspace {workspace_id!r}."
            )
        old_scope = view.get_scope(variable_name) or Scope.private_to_parent
        if old_scope == new_scope:
            return  # no-op
        event = VariablePromoted(
            workspace_id=workspace_id,
            variable_name=variable_name,
            old_scope=old_scope,
            new_scope=new_scope,
            promoted_by=promoted_by,
        )
        self._append(agg, workspace_id, [event], cid)

    def materialize_view(self, workspace_id: str) -> WorkspaceView:
        """Replay this workspace's events into a fresh WorkspaceView."""
        proj = WorkspaceProjection()
        for stored in self._store.load(_workspace_aggregate_id(workspace_id)):
            if stored.event_type == "workspace_created":
                proj.apply_workspace_created(workspace_id)
            elif stored.event_type == "variable_loaded":
                proj.apply_variable_loaded(
                    workspace_id=workspace_id,
                    variable_name=stored.payload["variable_name"],
                    value_payload=stored.payload["value_payload"],
                    citations_payload=stored.payload["citations"],
                    scope=Scope(stored.payload.get("scope", Scope.private_to_parent.value)),
                )
            elif stored.event_type == "variable_enriched":
                proj.apply_variable_enriched(
                    workspace_id=workspace_id,
                    variable_name=stored.payload["variable_name"],
                    value_payload=stored.payload["value_payload"],
                    citations_payload=stored.payload["citations"],
                    scope=Scope(stored.payload.get("scope", Scope.private_to_parent.value)),
                )
            elif stored.event_type == "variable_promoted":
                proj.apply_variable_promoted(
                    workspace_id=workspace_id,
                    variable_name=stored.payload["variable_name"],
                    new_scope=Scope(stored.payload["new_scope"]),
                )
            elif stored.event_type == "workspace_ready":
                proj.apply_workspace_ready(workspace_id)
        return proj.view(workspace_id)

    def get_state(self, workspace_id: str) -> WorkspaceState:
        return self._load_workspace_aggregate(workspace_id).state

    # --- Internal: claim_map preload ---------------------------------------

    def _build_claim_map(
        self, project_id: str, proj: SourcesProjection
    ) -> dict[str, Any]:
        """Construct a deterministic claim_map dict from sources."""
        entries: list[dict[str, Any]] = []
        sections = sorted(
            (s for s in proj.list_sources(project_id)
             if s.kind is SourceKind.paper_section),
            key=lambda s: s.id,
        )
        for src in sections:
            chunks = proj.chunks_for_source(src.id)
            text_excerpt = (
                chunks[0].text[:_CLAIM_MAP_QUOTE_TRUNCATE]
                if chunks
                else ""
            )
            entries.append(
                {
                    "source_id": src.id,
                    "title": src.locator,
                    "excerpt": text_excerpt,
                }
            )
        return {
            "project_id": project_id,
            "entries": entries,
        }

    def _claim_map_citations(self, proj: SourcesProjection) -> NonEmptyCitations:
        """One citation per section source, with the section's first chunk
        text as the evidence-grade quote (Codex feedback)."""
        citations: list[Citation] = []
        for src in sorted(proj.list_sources(), key=lambda s: s.id):
            if src.kind is not SourceKind.paper_section:
                continue
            chunks = proj.chunks_for_source(src.id)
            if chunks:
                quote = chunks[0].text[:_CLAIM_MAP_QUOTE_TRUNCATE]
                chunk_id: str | None = chunks[0].id
            else:
                quote = f"<source: {src.locator}>"
                chunk_id = None
            citations.append(
                Citation(
                    source_id=src.id,
                    chunk_id=chunk_id,
                    quote=quote,
                    locator=src.locator,
                    confidence=1.0,
                )
            )
        # Pydantic NonEmptyCitations rejects empty — caller checks first.
        return tuple(citations)

    # --- Internal: aggregate / append helpers -----------------------------

    def _load_workspace_aggregate(self, workspace_id: str) -> WorkspaceAggregate:
        agg = WorkspaceAggregate.empty(workspace_id)
        for stored in self._store.load(_workspace_aggregate_id(workspace_id)):
            cls = resolve_event_class(stored.event_type, stored.schema_version)
            agg.apply(stored.into(cls))
        return agg

    def _load_index_aggregate(self, project_id: str) -> IndexAggregate:
        agg = IndexAggregate.empty(project_id)
        for stored in self._store.load(_index_aggregate_id(project_id)):
            cls = resolve_event_class(stored.event_type, stored.schema_version)
            agg.apply(stored.into(cls))
        return agg

    def _append(
        self,
        agg: WorkspaceAggregate,
        workspace_id: str,
        events: list[DomainEvent],
        correlation_id: CorrelationId,
    ) -> None:
        envelopes: list[EventEnvelope] = [
            make_envelope(
                source="context.workspace.service",
                correlation_id=correlation_id,
            )
            for _ in events
        ]
        self._store.append(
            aggregate_id=_workspace_aggregate_id(workspace_id),
            aggregate_type="workspace",
            events=events,
            expected_version=agg.version,
            envelopes=envelopes,
        )
        agg.apply_all(events)


__all__ = ["BuildWorkspace", "WorkspaceAppService", "WorkspaceError"]
