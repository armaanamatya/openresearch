"""Knowledge graph models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


NodeKind = Literal[
    "module",
    "class",
    "function",
    "method",
    "external_symbol",
    "config_file",
    "paper_concept",
]

EdgeKind = Literal[
    "defines",
    "imports",
    "calls",
    "inherits",
    "sets",
    "mentions",
    "semantically_similar_to",
]


class GraphNode(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    project_id: str
    kind: NodeKind
    name: str
    path: str = ""
    start_line: int | None = None
    end_line: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    project_id: str
    source_id: str
    target_id: str
    kind: EdgeKind
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphQueryResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: dict[str, Any]
    nodes: tuple[GraphNode, ...] = ()
    edges: tuple[GraphEdge, ...] = ()


__all__ = [
    "EdgeKind",
    "GraphEdge",
    "GraphNode",
    "GraphQueryResult",
    "NodeKind",
]
