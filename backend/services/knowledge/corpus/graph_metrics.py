"""Deterministic bibliometric graph metrics over the citation edge table.

Design (survey-validated, 2026-08-03): the citation graph is ground truth —
free, deterministic, and the only graph structure the deployed scholarly-RAG
systems (PaperQA2, CG-RAG) actually use. This module computes, per target:

  Always (pure Python over the edge list — no dependencies):
  - ``in_degree``       — in-corpus citation count
  - ``co_citation``     — # papers citing both this paper and the target
  - ``coupling``        — bibliographic coupling: # shared references with target

  When ``networkx`` is importable (optional ``[graph]`` extra, pure-python):
  - ``pagerank``        — citation PageRank over the corpus graph
  - ``ppr``             — personalized PageRank seeded at the TARGET — the
                          HippoRAG retrieval mechanism applied to a
                          ground-truth graph. Fixed seed vector + fixed
                          ``alpha`` ⇒ fully deterministic.

NetworkX is an analytics scratchpad, never persisted (the LightRAG
pickle-graph anti-pattern) — SQLite stays the authority; results land in
``lit_graph_metrics`` via ``CorpusStore.put_graph_metrics``. Everything here
is fail-soft: missing networkx or an odd graph degrades to the pure-Python
subset, never raises.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_PPR_ALPHA = 0.85  # damping; fixed for determinism — never expose as a knob


def compute_graph_metrics(
    edges: list[tuple[str, str]], target_id: str
) -> dict[str, dict[str, float]]:
    """Per-paper metric dict for every node in ``edges`` (excluding the target).

    ``edges`` are ``(src, dst)`` = "src cites dst" pairs, deterministically
    ordered by the caller (``CorpusStore.edges``).
    """
    if not edges:
        return {}

    cites: dict[str, set[str]] = {}   # paper -> set(papers it cites)
    cited_by: dict[str, set[str]] = {}  # paper -> set(papers citing it)
    nodes: set[str] = set()
    for src, dst in edges:
        cites.setdefault(src, set()).add(dst)
        cited_by.setdefault(dst, set()).add(src)
        nodes.add(src)
        nodes.add(dst)

    target_refs = cites.get(target_id, set())
    target_citers = cited_by.get(target_id, set())

    metrics: dict[str, dict[str, float]] = {}
    for node in sorted(nodes):
        if node == target_id:
            continue
        node_refs = cites.get(node, set())
        node_citers = cited_by.get(node, set())
        metrics[node] = {
            "in_degree": float(len(node_citers)),
            "co_citation": float(len(node_citers & target_citers)),
            "coupling": float(len(node_refs & target_refs)),
        }

    _add_networkx_metrics(metrics, edges, target_id)
    return metrics


def _add_networkx_metrics(
    metrics: dict[str, dict[str, float]],
    edges: list[tuple[str, str]],
    target_id: str,
) -> None:
    """PageRank + target-seeded PPR when networkx is available. Fail-soft."""
    try:
        import networkx as nx
    except Exception:  # noqa: BLE001 — optional extra absent: degrade silently
        return
    try:
        graph = nx.DiGraph()
        graph.add_edges_from(edges)
        pagerank = nx.pagerank(graph, alpha=_PPR_ALPHA)
        ppr: dict[str, float] = {}
        if target_id in graph:
            # PPR flows along citation direction from the target; on an
            # undirected view it also reaches the target's citers — use the
            # undirected view so both graph directions contribute (fixed
            # seed = {target: 1.0} ⇒ deterministic).
            ppr = nx.pagerank(
                graph.to_undirected(as_view=False),
                alpha=_PPR_ALPHA,
                personalization={target_id: 1.0},
            )
        for node, per_paper in metrics.items():
            if node in pagerank:
                per_paper["pagerank"] = float(pagerank[node])
            if node in ppr:
                per_paper["ppr"] = float(ppr[node])
    except Exception:  # noqa: BLE001 — a broken analytics pass must not break the build
        logger.debug("graph_metrics: networkx pass failed", exc_info=True)


def graph_boost(per_paper: dict[str, float]) -> float:
    """Bounded ranking boost from a paper's metric dict (never the sole ranker).

    Deterministic and capped: co-citation/coupling contribute 0.05 per shared
    neighbour up to 0.15 each; PPR (when present) contributes up to 0.15 scaled
    by 5 (corpus PPR mass on a ~100-node graph is O(0.01–0.1)). Total ≤ 0.45 —
    strictly under the 0.5 adjacency-tier gap (`direct_ref` 3.0 vs `s2_ref`
    2.5), so graph evidence can reorder within a tier but never fabricate a
    tier jump.
    """
    boost = min(0.15, 0.05 * per_paper.get("co_citation", 0.0))
    boost += min(0.15, 0.05 * per_paper.get("coupling", 0.0))
    boost += min(0.15, 5.0 * per_paper.get("ppr", 0.0))
    return round(min(0.45, boost), 6)


__all__ = ["compute_graph_metrics", "graph_boost"]
