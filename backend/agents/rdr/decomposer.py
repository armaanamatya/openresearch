"""Phase-1 Rubric Decomposer — cuts the official PaperBench rubric tree into
ordered, dependency-sorted ``WorkCluster``s.

Public API::

    from backend.agents.rdr.decomposer import decompose
    clusters = decompose(rubric_tree)

See ``docs/superpowers/specs/2026-05-22-rubric-driven-harness-design.md`` §5.
"""

from __future__ import annotations

import re
from collections import OrderedDict

from backend.agents.rdr.models import TASK_CATEGORY_ORDER, RubricLeaf, WorkCluster

# ---------------------------------------------------------------------------
# Citation parsing
# ---------------------------------------------------------------------------

# Each pattern group captures the canonical reference type and its designator.
# Order: Section/Appendix/Table/Figure — longest alternatives first per group.
_CITE_PATTERN = re.compile(
    r"""
    (?:
        (?P<section>Section)\s+(?P<sec_num>\d+(?:\.\d+)*)           # Section 5 / Section 5.2
      | (?P<appendix>Appendix)\s+(?P<app_num>[A-Za-z0-9]+(?:\.\d+)*)  # Appendix E.1 / Appendix A.3.1
      | (?P<table>Table)\s+(?P<tbl_num>\d+)                          # Table 2
      | (?P<figure>Fig(?:ure|\.?)?)\s+(?P<fig_num>\d+)               # Figure 3 / Fig 3 / Fig. 3
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _parse_citations(text: str) -> list[str]:
    """Return deduplicated, order-preserving list of canonical citation strings."""
    seen: dict[str, None] = OrderedDict()  # used as ordered set
    for m in _CITE_PATTERN.finditer(text):
        if m.group("section"):
            key = f"Section {m.group('sec_num')}"
        elif m.group("appendix"):
            key = f"Appendix {m.group('app_num')}"
        elif m.group("table"):
            key = f"Table {m.group('tbl_num')}"
        else:  # figure
            key = f"Figure {m.group('fig_num')}"
        seen[key] = None
    return list(seen)


# ---------------------------------------------------------------------------
# Leaf extraction
# ---------------------------------------------------------------------------

_DEFAULT_CATEGORY = TASK_CATEGORY_ORDER[0]  # "Code Development"


def _is_leaf(node: dict) -> bool:
    return not node.get("sub_tasks")


def _extract_leaves(node: dict) -> list[RubricLeaf]:
    """DFS collection of all ``RubricLeaf`` objects in *node*'s subtree."""
    if _is_leaf(node):
        cat = node.get("task_category") or _DEFAULT_CATEGORY
        return [
            RubricLeaf(
                id=node["id"],
                requirements=node["requirements"],
                weight=float(node["weight"]),
                task_category=cat,
                paper_citations=_parse_citations(node["requirements"]),
            )
        ]
    leaves: list[RubricLeaf] = []
    for child in node["sub_tasks"]:
        leaves.extend(_extract_leaves(child))
    return leaves


def _leaf_count(node: dict) -> int:
    """Count leaves in *node*'s subtree without building objects."""
    if _is_leaf(node):
        return 1
    return sum(_leaf_count(c) for c in node["sub_tasks"])


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def _cluster_candidates(node: dict, max_leaves: int) -> list[dict]:
    """Return the list of rubric nodes that become cluster roots.

    Algorithm:
    - If the node's subtree has ≤ max_leaves → this node is one cluster.
    - Otherwise → recurse into each child, collecting that child's clusters.
    - A single leaf is always exactly one cluster (leaf_count == 1 ≤ any cap).
    """
    count = _leaf_count(node)
    if count <= max_leaves:
        return [node]
    # Need to split — recurse into children.
    result: list[dict] = []
    for child in node["sub_tasks"]:
        result.extend(_cluster_candidates(child, max_leaves))
    return result


# ---------------------------------------------------------------------------
# Dominant category helpers
# ---------------------------------------------------------------------------


def _dominant_category(leaves: list[RubricLeaf]) -> str:
    """Return the task_category with the greatest summed leaf weight; break ties
    by earliest position in TASK_CATEGORY_ORDER."""
    weights: dict[str, float] = {cat: 0.0 for cat in TASK_CATEGORY_ORDER}
    for leaf in leaves:
        weights[leaf.task_category] = weights.get(leaf.task_category, 0.0) + leaf.weight
    # Sort by (-weight, position_in_order) — the first entry wins
    best = min(
        TASK_CATEGORY_ORDER,
        key=lambda cat: (-weights.get(cat, 0.0), TASK_CATEGORY_ORDER.index(cat)),
    )
    return best


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def decompose(
    rubric_tree: dict,
    *,
    max_leaves_per_cluster: int = 12,
) -> list[WorkCluster]:
    """Cut the official PaperBench rubric.json tree into ordered, dependency-sorted
    work-clusters.

    Each top-level child of *rubric_tree* is the starting point for the
    splitting heuristic.  No cluster will have more than *max_leaves_per_cluster*
    leaves.  Every leaf in the tree belongs to exactly one returned cluster
    (strict partition).

    The returned list is sorted by ``(TASK_CATEGORY_ORDER.index(dominant_category),
    -weight)``.  ``depends_on`` is derived from the category topology:
    Code Execution clusters depend on all Code Development clusters; Result
    Analysis clusters depend on all Code Development + Code Execution clusters.
    """
    # 1. Collect cluster-node candidates from each top-level child (never make
    #    the root itself a cluster even if it has ≤ cap leaves).
    candidate_nodes: list[dict] = []
    for top_child in rubric_tree.get("sub_tasks", []):
        candidate_nodes.extend(_cluster_candidates(top_child, max_leaves_per_cluster))

    # 2. Build WorkCluster objects (unsorted, no depends_on yet).
    clusters: list[WorkCluster] = []
    for node in candidate_nodes:
        leaves = _extract_leaves(node)
        dom_cat = _dominant_category(leaves)
        # union of leaf citations, deduplicated, order-preserving
        seen_cites: dict[str, None] = OrderedDict()
        for leaf in leaves:
            for cite in leaf.paper_citations:
                seen_cites[cite] = None
        clusters.append(
            WorkCluster(
                id=node["id"],
                title=node["requirements"],
                leaves=leaves,
                dominant_category=dom_cat,
                weight=sum(l.weight for l in leaves),
                paper_citations=list(seen_cites),
            )
        )

    # 3. Sort: primary = category order, secondary = descending weight.
    clusters.sort(
        key=lambda c: (TASK_CATEGORY_ORDER.index(c.dominant_category), -c.weight)
    )

    # 4. Assign depends_on based purely on category topology.
    cd_ids = [c.id for c in clusters if c.dominant_category == "Code Development"]
    ce_ids = [c.id for c in clusters if c.dominant_category == "Code Execution"]

    for c in clusters:
        if c.dominant_category == "Code Execution":
            c.depends_on = list(cd_ids)
        elif c.dominant_category == "Result Analysis":
            c.depends_on = list(cd_ids) + list(ce_ids)
        # Code Development: depends_on stays as default []

    return clusters
