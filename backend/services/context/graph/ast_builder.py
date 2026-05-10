"""Deterministic Python AST graph builder.

This is the Phase 2 local substitute for Graphify's structural pass. It
extracts modules, imports, classes, functions, methods, inheritance, and call
relationships without using an LLM.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable

from backend.services.context.graph.model import GraphEdge, GraphNode
from backend.services.context.graph.service import graph_edge_id, graph_node_id


_EXCLUDED_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "node_modules",
    "venv",
    ".venv",
}


class PythonAstGraphBuilder:
    def __init__(self, *, project_id: str, repo_root: Path) -> None:
        self.project_id = project_id
        self.repo_root = repo_root.resolve()
        self._nodes: dict[str, GraphNode] = {}
        self._edges: dict[str, GraphEdge] = {}
        self._symbol_index: dict[tuple[str, str], str] = {}

    def build(self) -> tuple[tuple[GraphNode, ...], tuple[GraphEdge, ...]]:
        for path in self._iter_python_files():
            self._visit_file(path)
        return tuple(self._nodes.values()), tuple(self._edges.values())

    def _iter_python_files(self) -> Iterable[Path]:
        for path in sorted(self.repo_root.rglob("*.py")):
            rel_parts = path.relative_to(self.repo_root).parts
            if any(part in _EXCLUDED_DIRS for part in rel_parts):
                continue
            yield path

    def _visit_file(self, path: Path) -> None:
        rel = path.relative_to(self.repo_root).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        except (SyntaxError, UnicodeDecodeError):
            return

        module_name = _module_name_from_path(rel)
        module = self._node(
            kind="module",
            name=module_name,
            path=rel,
            start_line=1,
            end_line=getattr(tree, "end_lineno", None),
            metadata={"file_name": path.name},
        )

        visitor = _FileVisitor(builder=self, module_node=module, rel_path=rel)
        visitor.visit(tree)

    def _node(
        self,
        *,
        kind: str,
        name: str,
        path: str = "",
        start_line: int | None = None,
        end_line: int | None = None,
        metadata: dict | None = None,
    ) -> GraphNode:
        node_id = graph_node_id(
            project_id=self.project_id,
            kind=kind,
            name=name,
            path=path,
            start_line=start_line,
        )
        node = GraphNode(
            id=node_id,
            project_id=self.project_id,
            kind=kind,
            name=name,
            path=path,
            start_line=start_line,
            end_line=end_line,
            metadata=metadata or {},
        )
        self._nodes[node.id] = node
        if kind in {"class", "function", "method", "module", "external_symbol"}:
            self._symbol_index.setdefault((kind, name), node.id)
            self._symbol_index.setdefault(("any", name), node.id)
        return node

    def _external_symbol(self, name: str) -> GraphNode:
        node_id = graph_node_id(
            project_id=self.project_id,
            kind="external_symbol",
            name=name,
        )
        existing = self._nodes.get(node_id)
        if existing is not None:
            return existing
        node = GraphNode(
            id=node_id,
            project_id=self.project_id,
            kind="external_symbol",
            name=name,
            metadata={"external": True},
        )
        self._nodes[node.id] = node
        self._symbol_index.setdefault(("external_symbol", name), node.id)
        self._symbol_index.setdefault(("any", name), node.id)
        return node

    def _edge(
        self,
        *,
        source: GraphNode,
        target: GraphNode,
        kind: str,
        metadata: dict | None = None,
        discriminator: str = "",
    ) -> None:
        edge_id = graph_edge_id(
            project_id=self.project_id,
            source_id=source.id,
            target_id=target.id,
            kind=kind,
            discriminator=discriminator,
        )
        self._edges[edge_id] = GraphEdge(
            id=edge_id,
            project_id=self.project_id,
            source_id=source.id,
            target_id=target.id,
            kind=kind,
            metadata=metadata or {},
        )

    def _resolve_or_external(self, name: str) -> GraphNode:
        node_id = self._symbol_index.get(("any", name))
        if node_id is not None:
            return self._nodes[node_id]
        return self._external_symbol(name)


class _FileVisitor(ast.NodeVisitor):
    def __init__(
        self,
        *,
        builder: PythonAstGraphBuilder,
        module_node: GraphNode,
        rel_path: str,
    ) -> None:
        self.builder = builder
        self.module_node = module_node
        self.rel_path = rel_path
        self.scope_stack: list[GraphNode] = [module_node]

    @property
    def scope(self) -> GraphNode:
        return self.scope_stack[-1]

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            target_name = alias.name.split(".")[0]
            target = self.builder._external_symbol(target_name)
            self.builder._edge(
                source=self.module_node,
                target=target,
                kind="imports",
                metadata={"target_name": target_name, "imported": alias.name},
                discriminator=f"{node.lineno}:{alias.name}",
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if not node.module:
            return
        target_name = node.module.split(".")[0]
        target = self.builder._external_symbol(target_name)
        self.builder._edge(
            source=self.module_node,
            target=target,
            kind="imports",
            metadata={
                "target_name": target_name,
                "imported": node.module,
                "level": node.level,
                "names": [alias.name for alias in node.names],
            },
            discriminator=f"{node.lineno}:{node.module}",
        )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        class_node = self.builder._node(
            kind="class",
            name=node.name,
            path=self.rel_path,
            start_line=node.lineno,
            end_line=getattr(node, "end_lineno", None),
            metadata={"qualified_name": _qualified_name(self.scope_stack, node.name)},
        )
        self.builder._edge(
            source=self.scope,
            target=class_node,
            kind="defines",
            discriminator=f"{node.lineno}:{node.name}",
        )
        for base in node.bases:
            base_name = _call_name(base)
            if base_name:
                target = self.builder._resolve_or_external(base_name)
                self.builder._edge(
                    source=class_node,
                    target=target,
                    kind="inherits",
                    metadata={"target_name": base_name},
                    discriminator=f"{node.lineno}:{base_name}",
                )
        self.scope_stack.append(class_node)
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        parent_is_class = self.scope.kind == "class"
        kind = "method" if parent_is_class else "function"
        func_node = self.builder._node(
            kind=kind,
            name=node.name,
            path=self.rel_path,
            start_line=node.lineno,
            end_line=getattr(node, "end_lineno", None),
            metadata={
                "qualified_name": _qualified_name(self.scope_stack, node.name),
                "async": isinstance(node, ast.AsyncFunctionDef),
            },
        )
        self.builder._edge(
            source=self.scope,
            target=func_node,
            kind="defines",
            discriminator=f"{node.lineno}:{node.name}",
        )
        self.scope_stack.append(func_node)
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        target_name = _call_name(node.func)
        if target_name:
            target = self.builder._resolve_or_external(target_name)
            self.builder._edge(
                source=self.scope,
                target=target,
                kind="calls",
                metadata={"target_name": target_name, "line": node.lineno},
                discriminator=f"{node.lineno}:{target_name}",
            )
        self.generic_visit(node)


def _module_name_from_path(rel_path: str) -> str:
    suffix = rel_path[:-3] if rel_path.endswith(".py") else rel_path
    return suffix.replace("/", ".")


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call):
        return _call_name(node.func)
    if isinstance(node, ast.Subscript):
        return _call_name(node.value)
    return None


def _qualified_name(scope_stack: list[GraphNode], name: str) -> str:
    parts = [node.name for node in scope_stack if node.kind != "module"]
    return ".".join([*parts, name]) if parts else name


__all__ = ["PythonAstGraphBuilder"]
