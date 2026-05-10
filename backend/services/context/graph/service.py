"""SQLite-backed knowledge graph service."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from backend.persistence.database import Database
from backend.services.context.graph.model import GraphEdge, GraphNode, GraphQueryResult


def graph_node_id(
    *,
    project_id: str,
    kind: str,
    name: str,
    path: str = "",
    start_line: int | None = None,
) -> str:
    h = hashlib.sha256()
    h.update(f"node:{project_id}:{kind}:{path}:{name}:{start_line or 0}".encode())
    return f"kgn_{h.hexdigest()[:20]}"


def graph_edge_id(
    *,
    project_id: str,
    source_id: str,
    target_id: str,
    kind: str,
    discriminator: str = "",
) -> str:
    h = hashlib.sha256()
    h.update(f"edge:{project_id}:{source_id}:{target_id}:{kind}:{discriminator}".encode())
    return f"kge_{h.hexdigest()[:20]}"


class KnowledgeGraphService:
    """Stores and queries structural relationships.

    Query support is deliberately small and matches the PRD examples:
    `graph_query("function", calls="train")`,
    `graph_query("module", imports="torch")`, and generic metadata filters.
    """

    def __init__(self, db: Database) -> None:
        self._db = db
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        self._db.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS knowledge_graph_nodes (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                name TEXT NOT NULL,
                path TEXT NOT NULL DEFAULT '',
                start_line INTEGER,
                end_line INTEGER,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS knowledge_graph_edges (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_kgn_project_kind
                ON knowledge_graph_nodes(project_id, kind);
            CREATE INDEX IF NOT EXISTS idx_kgn_project_name
                ON knowledge_graph_nodes(project_id, name);
            CREATE INDEX IF NOT EXISTS idx_kge_project_kind
                ON knowledge_graph_edges(project_id, kind);
            CREATE INDEX IF NOT EXISTS idx_kge_source
                ON knowledge_graph_edges(source_id);
            CREATE INDEX IF NOT EXISTS idx_kge_target
                ON knowledge_graph_edges(target_id);
            """
        )
        self._db.connection.commit()

    def clear_project(self, project_id: str) -> None:
        self._db.connection.execute(
            "DELETE FROM knowledge_graph_edges WHERE project_id = ?", (project_id,)
        )
        self._db.connection.execute(
            "DELETE FROM knowledge_graph_nodes WHERE project_id = ?", (project_id,)
        )
        self._db.connection.commit()

    def upsert_node(self, node: GraphNode) -> None:
        self._db.connection.execute(
            """
            INSERT OR REPLACE INTO knowledge_graph_nodes
                (id, project_id, kind, name, path, start_line, end_line, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                node.id,
                node.project_id,
                node.kind,
                node.name,
                node.path,
                node.start_line,
                node.end_line,
                json.dumps(node.metadata, sort_keys=True),
            ),
        )
        self._db.connection.commit()

    def upsert_edge(self, edge: GraphEdge) -> None:
        self._db.connection.execute(
            """
            INSERT OR REPLACE INTO knowledge_graph_edges
                (id, project_id, source_id, target_id, kind, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                edge.id,
                edge.project_id,
                edge.source_id,
                edge.target_id,
                edge.kind,
                json.dumps(edge.metadata, sort_keys=True),
            ),
        )
        self._db.connection.commit()

    def add_nodes(self, nodes: Iterable[GraphNode]) -> None:
        for node in nodes:
            self.upsert_node(node)

    def add_edges(self, edges: Iterable[GraphEdge]) -> None:
        for edge in edges:
            self.upsert_edge(edge)

    def get_node(self, node_id: str) -> GraphNode | None:
        row = self._db.connection.execute(
            "SELECT * FROM knowledge_graph_nodes WHERE id = ?", (node_id,)
        ).fetchone()
        return _node_from_row(row) if row is not None else None

    def list_nodes(
        self,
        *,
        project_id: str | None = None,
        kind: str | None = None,
        name: str | None = None,
    ) -> tuple[GraphNode, ...]:
        clauses: list[str] = []
        params: list[Any] = []
        if project_id is not None:
            clauses.append("project_id = ?")
            params.append(project_id)
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        if name is not None:
            clauses.append("name = ?")
            params.append(name)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._db.connection.execute(
            f"SELECT * FROM knowledge_graph_nodes {where} ORDER BY path, start_line, name",
            tuple(params),
        ).fetchall()
        return tuple(_node_from_row(row) for row in rows)

    def list_edges(
        self,
        *,
        project_id: str | None = None,
        kind: str | None = None,
        source_id: str | None = None,
        target_id: str | None = None,
    ) -> tuple[GraphEdge, ...]:
        clauses: list[str] = []
        params: list[Any] = []
        if project_id is not None:
            clauses.append("project_id = ?")
            params.append(project_id)
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        if source_id is not None:
            clauses.append("source_id = ?")
            params.append(source_id)
        if target_id is not None:
            clauses.append("target_id = ?")
            params.append(target_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._db.connection.execute(
            f"SELECT * FROM knowledge_graph_edges {where} ORDER BY kind, source_id, target_id",
            tuple(params),
        ).fetchall()
        return tuple(_edge_from_row(row) for row in rows)

    def query(
        self,
        entity_type: str,
        *,
        project_id: str | None = None,
        calls: str | None = None,
        imports: str | None = None,
        name: str | None = None,
        path_contains: str | None = None,
        **metadata_filters: Any,
    ) -> GraphQueryResult:
        nodes = list(self.list_nodes(project_id=project_id, kind=entity_type, name=name))
        if path_contains is not None:
            nodes = [node for node in nodes if path_contains in node.path]
        if metadata_filters:
            nodes = [
                node
                for node in nodes
                if all(node.metadata.get(k) == v for k, v in metadata_filters.items())
            ]

        selected_edges: list[GraphEdge] = []
        if calls is not None:
            nodes, selected_edges = self._filter_by_relation(
                nodes, relation="calls", target_name=calls
            )
        if imports is not None:
            nodes, import_edges = self._filter_by_relation(
                nodes, relation="imports", target_name=imports
            )
            selected_edges.extend(import_edges)

        return GraphQueryResult(
            query={
                "entity_type": entity_type,
                "project_id": project_id,
                "calls": calls,
                "imports": imports,
                "name": name,
                "path_contains": path_contains,
                **metadata_filters,
            },
            nodes=tuple(nodes),
            edges=tuple(selected_edges),
        )

    def graph_query(self, entity_type: str, **relationships: Any) -> list[dict[str, Any]]:
        """Convenience shape for Context REPL-style `graph_query()` calls."""
        result = self.query(entity_type, **relationships)
        return [
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
        ]

    def ingest_python_repo(
        self,
        *,
        project_id: str,
        repo_root: Path,
        clear_existing: bool = True,
    ) -> tuple[int, int]:
        from backend.services.context.graph.ast_builder import PythonAstGraphBuilder

        if clear_existing:
            self.clear_project(project_id)
        nodes, edges = PythonAstGraphBuilder(project_id=project_id, repo_root=repo_root).build()
        self.add_nodes(nodes)
        self.add_edges(edges)
        return len(nodes), len(edges)

    def _filter_by_relation(
        self,
        nodes: list[GraphNode],
        *,
        relation: str,
        target_name: str,
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        if not nodes:
            return [], []
        node_by_id = {node.id: node for node in nodes}
        edges = self.list_edges(kind=relation)
        target = target_name.split(".")[0] if relation == "imports" else target_name
        matched_edges: list[GraphEdge] = []
        matched_nodes: dict[str, GraphNode] = {}
        for edge in edges:
            if edge.source_id not in node_by_id:
                continue
            target_node = self.get_node(edge.target_id)
            edge_target = edge.metadata.get("target_name")
            if target_node and target_node.name == target:
                matched_edges.append(edge)
                matched_nodes[edge.source_id] = node_by_id[edge.source_id]
            elif edge_target == target or edge_target == target_name:
                matched_edges.append(edge)
                matched_nodes[edge.source_id] = node_by_id[edge.source_id]
        return list(matched_nodes.values()), matched_edges


def _node_from_row(row: Any) -> GraphNode:
    return GraphNode(
        id=row["id"],
        project_id=row["project_id"],
        kind=row["kind"],
        name=row["name"],
        path=row["path"],
        start_line=row["start_line"],
        end_line=row["end_line"],
        metadata=json.loads(row["metadata_json"] or "{}"),
    )


def _edge_from_row(row: Any) -> GraphEdge:
    return GraphEdge(
        id=row["id"],
        project_id=row["project_id"],
        source_id=row["source_id"],
        target_id=row["target_id"],
        kind=row["kind"],
        metadata=json.loads(row["metadata_json"] or "{}"),
    )


__all__ = [
    "KnowledgeGraphService",
    "graph_edge_id",
    "graph_node_id",
]
