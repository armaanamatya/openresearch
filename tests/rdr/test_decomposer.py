"""Tests for the Phase-1 Rubric Decomposer (``backend/agents/rdr/decomposer.py``)."""

from __future__ import annotations

from backend.agents.rdr.decomposer import decompose
from backend.agents.rdr.models import TASK_CATEGORY_ORDER


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_leaf_ids(rubric_tree: dict) -> set[str]:
    """Walk the raw rubric dict and return the set of all leaf node ids."""
    sub = rubric_tree.get("sub_tasks", [])
    if not sub:
        return {rubric_tree["id"]}
    ids: set[str] = set()
    for child in sub:
        ids |= _collect_leaf_ids(child)
    return ids


def _leaf_count(rubric_tree: dict) -> int:
    return len(_collect_leaf_ids(rubric_tree))


def _make_node(
    node_id: str,
    requirements: str,
    weight: float,
    task_category: str | None,
    children: list[dict],
) -> dict:
    return {
        "id": node_id,
        "requirements": requirements,
        "weight": weight,
        "sub_tasks": children,
        "task_category": task_category,
        "finegrained_task_category": None,
    }


def _make_leaf(node_id: str, requirements: str, weight: float, task_category: str) -> dict:
    return _make_node(node_id, requirements, weight, task_category, [])


def _synthetic_tree(*, n_leaves: int, category: str = "Code Development") -> dict:
    """Root with a single mid-level node containing *n_leaves* uniform leaves."""
    leaves = [
        _make_leaf(f"L{i}", f"Implement leaf {i}", 1.0, category)
        for i in range(n_leaves)
    ]
    mid = _make_node("MID", "Mid-level node", float(n_leaves), None, leaves)
    return _make_node("ROOT", "Root", float(n_leaves), None, [mid])


# ---------------------------------------------------------------------------
# Partition tests
# ---------------------------------------------------------------------------


def test_partition_real_fixture_all_leaves_covered(paperbench_rubric: dict) -> None:
    """Every leaf in the tree appears in exactly one cluster — no drops, no dups."""
    all_leaf_ids = _collect_leaf_ids(paperbench_rubric)
    clusters = decompose(paperbench_rubric)

    # gather ids from clusters
    cluster_leaf_ids: list[str] = []
    for c in clusters:
        for leaf in c.leaves:
            cluster_leaf_ids.append(leaf.id)

    assert set(cluster_leaf_ids) == all_leaf_ids, "Leaf ids don't match tree"
    # strict partition — no duplicates
    assert len(cluster_leaf_ids) == len(set(cluster_leaf_ids)), "Duplicate leaf in clusters"


def test_partition_counts_match(paperbench_rubric: dict) -> None:
    """Sum of cluster leaf counts equals the tree leaf count (computed dynamically)."""
    expected = _leaf_count(paperbench_rubric)
    clusters = decompose(paperbench_rubric)
    actual = sum(len(c.leaves) for c in clusters)
    assert actual == expected


# ---------------------------------------------------------------------------
# Cap tests
# ---------------------------------------------------------------------------


def test_cap_default_no_cluster_exceeds_12(paperbench_rubric: dict) -> None:
    clusters = decompose(paperbench_rubric)
    for c in clusters:
        assert len(c.leaves) <= 12, (
            f"Cluster '{c.id}' has {len(c.leaves)} leaves, exceeds default cap of 12"
        )


def test_cap_small_cap_5_no_cluster_exceeds_5(paperbench_rubric: dict) -> None:
    clusters = decompose(paperbench_rubric, max_leaves_per_cluster=5)
    for c in clusters:
        assert len(c.leaves) <= 5, (
            f"Cluster '{c.id}' has {len(c.leaves)} leaves, exceeds cap of 5"
        )


def test_cap_small_still_full_partition(paperbench_rubric: dict) -> None:
    """Partition invariant holds regardless of cap."""
    all_leaf_ids = _collect_leaf_ids(paperbench_rubric)
    clusters = decompose(paperbench_rubric, max_leaves_per_cluster=5)
    cluster_leaf_ids = [leaf.id for c in clusters for leaf in c.leaves]
    assert set(cluster_leaf_ids) == all_leaf_ids
    assert len(cluster_leaf_ids) == len(set(cluster_leaf_ids))


# ---------------------------------------------------------------------------
# Splitting behaviour on synthetic trees
# ---------------------------------------------------------------------------


def test_splitting_large_node_is_split() -> None:
    """A mid-level node with > cap leaves must be split into multiple clusters."""
    tree = _synthetic_tree(n_leaves=14)
    clusters = decompose(tree, max_leaves_per_cluster=12)
    assert len(clusters) > 1, "Expected node with 14 leaves to split"
    for c in clusters:
        assert len(c.leaves) <= 12


def test_splitting_small_node_stays_one_cluster() -> None:
    """A mid-level node with ≤ cap leaves must remain a single cluster."""
    tree = _synthetic_tree(n_leaves=8)
    clusters = decompose(tree, max_leaves_per_cluster=12)
    assert len(clusters) == 1, "Expected 8-leaf node to stay as one cluster"
    assert len(clusters[0].leaves) == 8


def test_splitting_exact_cap_stays_one_cluster() -> None:
    """A node with exactly cap leaves must not be split."""
    tree = _synthetic_tree(n_leaves=12)
    clusters = decompose(tree, max_leaves_per_cluster=12)
    assert len(clusters) == 1


def test_splitting_leaf_is_its_own_cluster() -> None:
    """A top-level-child that is itself a leaf (sub_tasks=[]) forms a 1-leaf cluster."""
    single_leaf = _make_leaf("ONLY", "A lone leaf", 1.0, "Code Execution")
    root = _make_node("ROOT", "Root", 1.0, None, [single_leaf])
    clusters = decompose(root)
    assert len(clusters) == 1
    assert clusters[0].leaves[0].id == "ONLY"


# ---------------------------------------------------------------------------
# Ordering tests
# ---------------------------------------------------------------------------


def test_ordering_category_order_respected(paperbench_rubric: dict) -> None:
    """All Code Development clusters precede all Code Execution clusters,
    which precede all Result Analysis clusters."""
    clusters = decompose(paperbench_rubric)
    cat_indices = [TASK_CATEGORY_ORDER.index(c.dominant_category) for c in clusters]
    assert cat_indices == sorted(cat_indices), (
        "Clusters are not sorted by TASK_CATEGORY_ORDER"
    )


def test_ordering_within_category_non_increasing_weight(paperbench_rubric: dict) -> None:
    """Within each category band, clusters are ordered by descending weight."""
    clusters = decompose(paperbench_rubric)
    for cat in TASK_CATEGORY_ORDER:
        band = [c for c in clusters if c.dominant_category == cat]
        weights = [c.weight for c in band]
        assert weights == sorted(weights, reverse=True), (
            f"Clusters in '{cat}' not sorted by descending weight"
        )


# ---------------------------------------------------------------------------
# Citation parsing tests
# ---------------------------------------------------------------------------


def test_citations_section_numeric() -> None:
    leaf = _make_leaf("L1", "Implement as described in Section 5.", 1.0, "Code Development")
    root = _make_node("ROOT", "Root", 1.0, None, [leaf])
    clusters = decompose(root)
    assert "Section 5" in clusters[0].leaves[0].paper_citations


def test_citations_section_decimal() -> None:
    leaf = _make_leaf("L1", "See Section 2.2 for details.", 1.0, "Code Development")
    root = _make_node("ROOT", "Root", 1.0, None, [leaf])
    clusters = decompose(root)
    assert "Section 2.2" in clusters[0].leaves[0].paper_citations


def test_citations_appendix_letter() -> None:
    leaf = _make_leaf("L1", "Following Appendix E.1", 1.0, "Code Development")
    root = _make_node("ROOT", "Root", 1.0, None, [leaf])
    clusters = decompose(root)
    assert "Appendix E.1" in clusters[0].leaves[0].paper_citations


def test_citations_appendix_dotted_number() -> None:
    leaf = _make_leaf("L1", "See Appendix A.3.1 for full details.", 1.0, "Code Development")
    root = _make_node("ROOT", "Root", 1.0, None, [leaf])
    clusters = decompose(root)
    assert "Appendix A.3.1" in clusters[0].leaves[0].paper_citations


def test_citations_table() -> None:
    leaf = _make_leaf("L1", "Values listed in Table 2.", 1.0, "Code Development")
    root = _make_node("ROOT", "Root", 1.0, None, [leaf])
    clusters = decompose(root)
    assert "Table 2" in clusters[0].leaves[0].paper_citations


def test_citations_figure_variants() -> None:
    """Both 'Figure N' and 'Fig. N' and 'Fig N' are matched."""
    for req, expected in [
        ("See Figure 3 for an illustration.", "Figure 3"),
        ("See Fig. 3 for an illustration.", "Figure 3"),
        ("See Fig 3 for an illustration.", "Figure 3"),
    ]:
        leaf = _make_leaf("L1", req, 1.0, "Code Development")
        root = _make_node("ROOT", "Root", 1.0, None, [leaf])
        clusters = decompose(root)
        assert expected in clusters[0].leaves[0].paper_citations, (
            f"Expected '{expected}' in citations for requirement: '{req}'"
        )


def test_citations_case_insensitive() -> None:
    leaf = _make_leaf("L1", "As described in section 5.2.", 1.0, "Code Development")
    root = _make_node("ROOT", "Root", 1.0, None, [leaf])
    clusters = decompose(root)
    assert "Section 5.2" in clusters[0].leaves[0].paper_citations


def test_citations_deduplicated_order_preserving() -> None:
    """Duplicate references in a leaf's text appear only once."""
    leaf = _make_leaf(
        "L1",
        "See Section 5 and Section 5 again, also Table 2.",
        1.0,
        "Code Development",
    )
    root = _make_node("ROOT", "Root", 1.0, None, [leaf])
    clusters = decompose(root)
    cites = clusters[0].leaves[0].paper_citations
    assert cites.count("Section 5") == 1
    # order preserved: Section 5 first, then Table 2
    assert cites == ["Section 5", "Table 2"]


def test_citations_cluster_union_of_leaves(paperbench_rubric: dict) -> None:
    """Each cluster's paper_citations is the deduplicated union of its leaves'."""
    clusters = decompose(paperbench_rubric)
    for c in clusters:
        expected_union: list[str] = []
        seen: set[str] = set()
        for leaf in c.leaves:
            for cite in leaf.paper_citations:
                if cite not in seen:
                    expected_union.append(cite)
                    seen.add(cite)
        assert c.paper_citations == expected_union, (
            f"Cluster '{c.id}' paper_citations mismatch"
        )


def test_citations_real_fixture_appendix_e1(paperbench_rubric: dict) -> None:
    """The first top-level area (tasks implemented as per Appendix E.1) must carry
    Appendix E.1 somewhere in its cluster(s)."""
    clusters = decompose(paperbench_rubric)
    # The first top-level area id is bb0c35d6
    target_node_id = paperbench_rubric["sub_tasks"][0]["id"]
    matching = [c for c in clusters if c.id == target_node_id]
    # It has 9 leaves, under cap — so exactly one cluster with that id
    assert len(matching) == 1
    # The cluster *title* is from the node that cites Appendix E.1
    assert "Appendix E.1" in matching[0].title or any(
        "Appendix E.1" in leaf.requirements for leaf in matching[0].leaves
    ) or "Appendix E.1" in matching[0].paper_citations


# ---------------------------------------------------------------------------
# Dependencies tests
# ---------------------------------------------------------------------------


def test_dependencies_code_dev_has_empty_depends_on(paperbench_rubric: dict) -> None:
    clusters = decompose(paperbench_rubric)
    for c in clusters:
        if c.dominant_category == "Code Development":
            assert c.depends_on == [], (
                f"Code Development cluster '{c.id}' should have empty depends_on"
            )


def test_dependencies_result_analysis_depends_on_all_cd_and_ce(
    paperbench_rubric: dict,
) -> None:
    clusters = decompose(paperbench_rubric)
    cd_ids = {c.id for c in clusters if c.dominant_category == "Code Development"}
    ce_ids = {c.id for c in clusters if c.dominant_category == "Code Execution"}
    for c in clusters:
        if c.dominant_category == "Result Analysis":
            assert cd_ids.issubset(set(c.depends_on)), (
                f"Result Analysis cluster '{c.id}' missing Code Development deps"
            )
            assert ce_ids.issubset(set(c.depends_on)), (
                f"Result Analysis cluster '{c.id}' missing Code Execution deps"
            )


def test_dependencies_code_execution_depends_on_all_cd(paperbench_rubric: dict) -> None:
    clusters = decompose(paperbench_rubric)
    cd_ids = {c.id for c in clusters if c.dominant_category == "Code Development"}
    for c in clusters:
        if c.dominant_category == "Code Execution":
            assert cd_ids.issubset(set(c.depends_on)), (
                f"Code Execution cluster '{c.id}' missing Code Development deps"
            )


def test_dependencies_code_execution_not_in_result_analysis_depends_on_without_ce() -> None:
    """Synthetic: a tree with only CD and RA; RA depends on all CD, no CE ids."""
    leaves_cd = [_make_leaf(f"CD{i}", f"CD leaf {i}", 1.0, "Code Development") for i in range(3)]
    leaves_ra = [_make_leaf(f"RA{i}", f"RA leaf {i}", 1.0, "Result Analysis") for i in range(3)]
    cd_group = _make_node("G_CD", "CD group", 3.0, None, leaves_cd)
    ra_group = _make_node("G_RA", "RA group", 3.0, None, leaves_ra)
    root = _make_node("ROOT", "Root", 6.0, None, [cd_group, ra_group])
    clusters = decompose(root)
    cd_ids = {c.id for c in clusters if c.dominant_category == "Code Development"}
    for c in clusters:
        if c.dominant_category == "Result Analysis":
            assert cd_ids.issubset(set(c.depends_on))


# ---------------------------------------------------------------------------
# Dominant category tests
# ---------------------------------------------------------------------------


def test_dominant_category_weight_weighted() -> None:
    """A cluster with one heavy Code Execution leaf and three light Code Development
    leaves gets dominant_category == 'Code Execution'."""
    heavy = _make_leaf("HEAVY", "Heavy leaf", 10.0, "Code Execution")
    light1 = _make_leaf("L1", "Light leaf 1", 1.0, "Code Development")
    light2 = _make_leaf("L2", "Light leaf 2", 1.0, "Code Development")
    light3 = _make_leaf("L3", "Light leaf 3", 1.0, "Code Development")
    group = _make_node("G", "Mixed group", 13.0, None, [heavy, light1, light2, light3])
    root = _make_node("ROOT", "Root", 13.0, None, [group])
    clusters = decompose(root)
    assert len(clusters) == 1
    assert clusters[0].dominant_category == "Code Execution"


def test_dominant_category_tie_broken_by_order() -> None:
    """Equal total weight → earliest category in TASK_CATEGORY_ORDER wins."""
    cd_leaf = _make_leaf("CD", "CD", 5.0, "Code Development")
    ce_leaf = _make_leaf("CE", "CE", 5.0, "Code Execution")
    group = _make_node("G", "Tied group", 10.0, None, [cd_leaf, ce_leaf])
    root = _make_node("ROOT", "Root", 10.0, None, [group])
    clusters = decompose(root)
    assert clusters[0].dominant_category == "Code Development"


def test_dominant_category_null_leaf_defaults_to_code_development() -> None:
    """Leaves with null/missing task_category default to Code Development."""
    leaf = _make_leaf("L", "A leaf", 1.0, None)  # type: ignore[arg-type]
    group = _make_node("G", "Group", 1.0, None, [leaf])
    root = _make_node("ROOT", "Root", 1.0, None, [group])
    clusters = decompose(root)
    assert clusters[0].leaves[0].task_category == "Code Development"
    assert clusters[0].dominant_category == "Code Development"


# ---------------------------------------------------------------------------
# Weight tests
# ---------------------------------------------------------------------------


def test_cluster_weight_is_sum_of_leaf_weights() -> None:
    leaves = [
        _make_leaf("L1", "Leaf 1", 2.5, "Code Development"),
        _make_leaf("L2", "Leaf 2", 1.5, "Code Development"),
    ]
    group = _make_node("G", "Group", 4.0, None, leaves)
    root = _make_node("ROOT", "Root", 4.0, None, [group])
    clusters = decompose(root)
    assert abs(clusters[0].weight - 4.0) < 1e-9


def test_cluster_weight_coerced_to_float(paperbench_rubric: dict) -> None:
    """All cluster weights are Python floats (rubric json may give ints)."""
    clusters = decompose(paperbench_rubric)
    for c in clusters:
        assert isinstance(c.weight, float)
        for leaf in c.leaves:
            assert isinstance(leaf.weight, float)


# ---------------------------------------------------------------------------
# Leaf order test
# ---------------------------------------------------------------------------


def test_leaf_order_is_depth_first() -> None:
    """Leaves within a cluster appear in DFS (tree) order."""
    l1 = _make_leaf("L1", "First", 1.0, "Code Development")
    l2 = _make_leaf("L2", "Second", 1.0, "Code Development")
    l3 = _make_leaf("L3", "Third", 1.0, "Code Development")
    # Build a two-level hierarchy: group has child1 (with L1, L2) and L3
    child1 = _make_node("C1", "Child1", 2.0, None, [l1, l2])
    group = _make_node("G", "Group", 3.0, None, [child1, l3])
    root = _make_node("ROOT", "Root", 3.0, None, [group])
    clusters = decompose(root)
    assert len(clusters) == 1
    assert [leaf.id for leaf in clusters[0].leaves] == ["L1", "L2", "L3"]


# ---------------------------------------------------------------------------
# Determinism tests
# ---------------------------------------------------------------------------


def test_determinism_same_input_same_output(paperbench_rubric: dict) -> None:
    """decompose is pure: two calls on identical input produce equal results."""
    result_a = decompose(paperbench_rubric)
    result_b = decompose(paperbench_rubric)
    assert len(result_a) == len(result_b)
    for a, b in zip(result_a, result_b):
        assert a.id == b.id
        assert a.title == b.title
        assert a.dominant_category == b.dominant_category
        assert abs(a.weight - b.weight) < 1e-9
        assert a.depends_on == b.depends_on
        assert a.paper_citations == b.paper_citations
        assert [l.id for l in a.leaves] == [l.id for l in b.leaves]


def test_determinism_does_not_mutate_input(paperbench_rubric: dict) -> None:
    """decompose does not modify the input dict."""
    import copy
    original = copy.deepcopy(paperbench_rubric)
    decompose(paperbench_rubric)
    assert paperbench_rubric == original


# ---------------------------------------------------------------------------
# Cluster id and title tests
# ---------------------------------------------------------------------------


def test_cluster_id_is_node_id() -> None:
    """The cluster id must equal the rubric node's id field."""
    leaf = _make_leaf("L1", "Do something", 1.0, "Code Development")
    group = _make_node("MY_NODE_ID", "Do the thing", 1.0, None, [leaf])
    root = _make_node("ROOT", "Root", 1.0, None, [group])
    clusters = decompose(root)
    assert clusters[0].id == "MY_NODE_ID"


def test_cluster_title_is_node_requirements() -> None:
    leaf = _make_leaf("L1", "Do something", 1.0, "Code Development")
    group = _make_node("GID", "The group requirements text", 1.0, None, [leaf])
    root = _make_node("ROOT", "Root", 1.0, None, [group])
    clusters = decompose(root)
    assert clusters[0].title == "The group requirements text"


# ---------------------------------------------------------------------------
# Root-not-single-cluster invariant
# ---------------------------------------------------------------------------


def test_never_root_as_single_cluster(paperbench_rubric: dict) -> None:
    """The root itself must never be the sole cluster (regardless of cap)."""
    total = _leaf_count(paperbench_rubric)
    clusters = decompose(paperbench_rubric, max_leaves_per_cluster=total + 1000)
    root_id = paperbench_rubric["id"]
    assert not (len(clusters) == 1 and clusters[0].id == root_id), (
        "Root was returned as a single cluster — must always recurse into children"
    )
