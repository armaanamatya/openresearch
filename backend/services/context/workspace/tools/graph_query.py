"""Workspace graph_query tool backed by KnowledgeGraphService."""

from __future__ import annotations

from typing import Any

from backend.schemas.citations import Citation
from backend.services.context.graph.service import KnowledgeGraphService
from backend.services.context.workspace.model import Cited
from backend.services.context.workspace.tools.interface import WorkspaceToolError


class GraphQueryTool:
    """Expose structural graph lookup as a cited workspace tool."""

    name = "graph_query"

    def __init__(self, graph: KnowledgeGraphService) -> None:
        self._graph = graph

    def call(
        self,
        *,
        workspace_id: str,
        entity_type: str,
        project_id: str | None = None,
        **relationships: Any,
    ) -> Cited[dict[str, Any]]:
        del workspace_id
        result = self._graph.query(entity_type, project_id=project_id, **relationships)
        if not result.nodes:
            raise WorkspaceToolError("graph_query found no matching graph nodes")

        citations = tuple(
            Citation(
                source_id=f"knowledge_graph:{node.project_id}",
                chunk_id=node.id,
                quote=f"{node.kind} {node.name}",
                locator=f"{node.path}:{node.start_line or 1}" if node.path else node.name,
                confidence=0.9,
            )
            for node in result.nodes
        )
        return Cited(
            value={
                "query": result.query,
                "results": [
                    {
                        "id": node.id,
                        "kind": node.kind,
                        "name": node.name,
                        "path": node.path,
                        "start_line": node.start_line,
                        "end_line": node.end_line,
                        "metadata": node.metadata,
                    }
                    for node in result.nodes
                ],
            },
            citations=citations,
        )


__all__ = ["GraphQueryTool"]
