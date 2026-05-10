"""Knowledge graph services for structural code navigation.

Phase 2 starts with deterministic Python AST extraction. A future Graphify
adapter can populate the same store with multi-language and semantic edges.
"""

from backend.services.context.graph.ast_builder import PythonAstGraphBuilder
from backend.services.context.graph.model import GraphEdge, GraphNode, GraphQueryResult
from backend.services.context.graph.service import KnowledgeGraphService

__all__ = [
    "GraphEdge",
    "GraphNode",
    "GraphQueryResult",
    "KnowledgeGraphService",
    "PythonAstGraphBuilder",
]
