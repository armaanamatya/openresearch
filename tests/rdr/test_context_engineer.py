"""Tests for the Phase-2 Context Engineer (``backend/agents/rdr/context_engineer.py``)."""

from __future__ import annotations

import copy

import pytest

from backend.agents.rdr.context_engineer import build_context, estimate_tokens
from backend.agents.rdr.decomposer import decompose
from backend.agents.rdr.models import Artifacts, RubricLeaf, WorkCluster

# ---------------------------------------------------------------------------
# Shared synthetic helpers
# ---------------------------------------------------------------------------

_SIMPLE_PAPER = """\
# Abstract

This paper presents a method.

## 1 Introduction

We introduce the work here.

## 2 Background

Prior art is discussed in this section.

### 2.1 Related Work

This subsection covers related work in detail.

## 5 Experiments

We ran experiments comparing baseline to ours. Table 2 and Figure 3 are shown.

## 5.2 Ablation Study

Here we ablate each component.

## Appendix E Full Details

This appendix contains detailed derivations.

## Appendix E.1 Extended Results

Extended result tables and figures are here.
"""


def _make_leaf(
    leaf_id: str,
    requirements: str,
    weight: float = 1.0,
    task_category: str = "Code Development",
    paper_citations: list[str] | None = None,
) -> RubricLeaf:
    return RubricLeaf(
        id=leaf_id,
        requirements=requirements,
        weight=weight,
        task_category=task_category,
        paper_citations=paper_citations or [],
    )


def _make_cluster(
    cluster_id: str = "CL1",
    title: str = "Test cluster",
    leaves: list[RubricLeaf] | None = None,
    dominant_category: str = "Code Development",
    weight: float = 3.0,
    depends_on: list[str] | None = None,
    paper_citations: list[str] | None = None,
) -> WorkCluster:
    lf = leaves or [_make_leaf("L1", "Do something", 3.0)]
    return WorkCluster(
        id=cluster_id,
        title=title,
        leaves=lf,
        dominant_category=dominant_category,
        weight=weight,
        depends_on=depends_on or [],
        paper_citations=paper_citations or [],
    )


def _make_artifacts(cluster_id: str, files: dict[str, str], notes: str = "") -> Artifacts:
    return Artifacts(cluster_id=cluster_id, files=files, notes=notes)


# ---------------------------------------------------------------------------
# 1. Leaf contract — verbatim requirements, weights present
# ---------------------------------------------------------------------------


def test_leaf_contract_contains_all_requirements() -> None:
    """Every leaf's requirements text appears verbatim in leaf_contract."""
    leaves = [
        _make_leaf("L1", "Implement the forward pass exactly as described.", 2.0),
        _make_leaf("L2", "Reproduce Table 3 results within 0.5% error.", 3.0),
        _make_leaf("L3", "Log the training loss to a file.", 1.5),
    ]
    cluster = _make_cluster(leaves=leaves, weight=6.5)
    ctx = build_context(cluster, paper="", artifacts={})
    for leaf in leaves:
        assert leaf.requirements in ctx.leaf_contract, (
            f"Requirements for leaf {leaf.id!r} not found verbatim in leaf_contract"
        )


def test_leaf_contract_shows_weights() -> None:
    """Each leaf's weight is shown in the contract."""
    leaves = [
        _make_leaf("L1", "First requirement", 2.5),
        _make_leaf("L2", "Second requirement", 1.5),
    ]
    cluster = _make_cluster(leaves=leaves, weight=4.0)
    ctx = build_context(cluster, paper="", artifacts={})
    assert "2.5" in ctx.leaf_contract
    assert "1.5" in ctx.leaf_contract


def test_leaf_contract_shows_total_weight_and_count() -> None:
    leaves = [_make_leaf(f"L{i}", f"Req {i}", 2.0) for i in range(3)]
    cluster = _make_cluster(leaves=leaves, weight=6.0, title="My Cluster")
    ctx = build_context(cluster, paper="", artifacts={})
    assert "My Cluster" in ctx.leaf_contract
    assert "3 requirements" in ctx.leaf_contract
    assert "6.0" in ctx.leaf_contract


def test_leaf_contract_single_leaf_grammar() -> None:
    """Singular 'requirement' when there is exactly one leaf."""
    cluster = _make_cluster(leaves=[_make_leaf("L1", "Just one req", 1.0)], weight=1.0)
    ctx = build_context(cluster, paper="", artifacts={})
    assert "1 requirement" in ctx.leaf_contract
    assert "requirements" not in ctx.leaf_contract


def test_leaf_contract_index_format() -> None:
    """Leaves are indexed [1], [2], … in contract."""
    leaves = [_make_leaf(f"L{i}", f"Req {i}", 1.0) for i in range(1, 4)]
    cluster = _make_cluster(leaves=leaves, weight=3.0)
    ctx = build_context(cluster, paper="", artifacts={})
    for i in range(1, 4):
        assert f"[{i}]" in ctx.leaf_contract


# ---------------------------------------------------------------------------
# 2. Citation resolution
# ---------------------------------------------------------------------------


def test_citation_section_5_resolves() -> None:
    """A cluster citing 'Section 5' gets the '5 Experiments' section body."""
    leaf = _make_leaf("L1", "As per Section 5.", 1.0, paper_citations=["Section 5"])
    cluster = _make_cluster(
        leaves=[leaf], paper_citations=["Section 5"]
    )
    ctx = build_context(cluster, paper=_SIMPLE_PAPER, artifacts={})
    assert len(ctx.paper_sections) >= 1
    headings = [cs.heading for cs in ctx.paper_sections]
    # The resolved heading should contain "5"
    assert any("5" in h for h in headings), f"Expected section 5, got {headings}"
    texts = [cs.text for cs in ctx.paper_sections]
    assert any("experiments" in t.lower() for t in texts)


def test_citation_section_52_resolves() -> None:
    """Cite 'Section 5.2' → the '5.2 Ablation Study' subsection."""
    leaf = _make_leaf("L1", "See Section 5.2.", 1.0, paper_citations=["Section 5.2"])
    cluster = _make_cluster(leaves=[leaf], paper_citations=["Section 5.2"])
    ctx = build_context(cluster, paper=_SIMPLE_PAPER, artifacts={})
    headings = [cs.heading for cs in ctx.paper_sections]
    assert any("5.2" in h for h in headings), f"Expected 5.2 section, got {headings}"


def test_citation_appendix_e1_resolves() -> None:
    """Cite 'Appendix E.1' → the Appendix E.1 section body."""
    leaf = _make_leaf("L1", "Following Appendix E.1.", 1.0, paper_citations=["Appendix E.1"])
    cluster = _make_cluster(leaves=[leaf], paper_citations=["Appendix E.1"])
    ctx = build_context(cluster, paper=_SIMPLE_PAPER, artifacts={})
    headings = [cs.heading for cs in ctx.paper_sections]
    assert any("E.1" in h for h in headings), f"Expected Appendix E.1, got {headings}"


def test_citation_table_resolves_to_containing_section() -> None:
    """Cite 'Table 2' → the section whose body contains 'Table 2'."""
    leaf = _make_leaf("L1", "Table 2 results.", 1.0, paper_citations=["Table 2"])
    cluster = _make_cluster(leaves=[leaf], paper_citations=["Table 2"])
    ctx = build_context(cluster, paper=_SIMPLE_PAPER, artifacts={})
    texts = [cs.text for cs in ctx.paper_sections]
    assert any("Table 2" in t for t in texts), "Expected section containing 'Table 2'"


def test_citation_figure_resolves_to_containing_section() -> None:
    """Cite 'Figure 3' → the section whose body contains 'Figure 3'."""
    leaf = _make_leaf("L1", "Figure 3 shows.", 1.0, paper_citations=["Figure 3"])
    cluster = _make_cluster(leaves=[leaf], paper_citations=["Figure 3"])
    ctx = build_context(cluster, paper=_SIMPLE_PAPER, artifacts={})
    texts = [cs.text for cs in ctx.paper_sections]
    assert any("Figure 3" in t for t in texts), "Expected section containing 'Figure 3'"


def test_citation_deduplication() -> None:
    """Same section cited by two leaves appears only once in paper_sections."""
    leaves = [
        _make_leaf("L1", "See Section 5.", 1.0, paper_citations=["Section 5"]),
        _make_leaf("L2", "Also Section 5.", 1.0, paper_citations=["Section 5"]),
    ]
    cluster = _make_cluster(leaves=leaves, paper_citations=["Section 5"])
    ctx = build_context(cluster, paper=_SIMPLE_PAPER, artifacts={})
    headings = [cs.heading for cs in ctx.paper_sections]
    section5_headings = [h for h in headings if "5" in h and "5.2" not in h]
    # Should appear at most once.
    assert len(section5_headings) <= 1, "Section 5 duplicated in paper_sections"


def test_citation_unresolvable_skipped() -> None:
    """A citation for a non-existent section is silently skipped."""
    leaf = _make_leaf("L1", "See Section 99.", 1.0, paper_citations=["Section 99"])
    cluster = _make_cluster(leaves=[leaf], paper_citations=["Section 99"])
    ctx = build_context(cluster, paper=_SIMPLE_PAPER, artifacts={})
    headings = [cs.heading for cs in ctx.paper_sections]
    assert not any("99" in h for h in headings)


def test_citation_empty_paper_yields_empty_sections() -> None:
    leaf = _make_leaf("L1", "See Section 5.", 1.0, paper_citations=["Section 5"])
    cluster = _make_cluster(leaves=[leaf], paper_citations=["Section 5"])
    ctx = build_context(cluster, paper="", artifacts={})
    assert ctx.paper_sections == []


def test_citation_section_number_prefix_not_greedy() -> None:
    """Citing 'Section 5' must NOT match the '5.2 Ablation' subsection."""
    leaf = _make_leaf("L1", "See Section 5.", 1.0, paper_citations=["Section 5"])
    cluster = _make_cluster(leaves=[leaf], paper_citations=["Section 5"])
    ctx = build_context(cluster, paper=_SIMPLE_PAPER, artifacts={})
    headings = [cs.heading for cs in ctx.paper_sections]
    assert not any("5.2" in h for h in headings), (
        "'Section 5' citation matched '5.2' subsection"
    )


# ---------------------------------------------------------------------------
# 3. Dependency closure
# ---------------------------------------------------------------------------


def test_dependency_closure_includes_only_declared_deps() -> None:
    """With two prior clusters and depends_on naming just one, only that one's
    files appear in dependency_artifacts."""
    art_a = _make_artifacts("A", {"src/model.py": "class Model: pass"})
    art_b = _make_artifacts("B", {"src/train.py": "def train(): pass"})
    cluster = _make_cluster(depends_on=["A"])
    ctx = build_context(cluster, paper="", artifacts={"A": art_a, "B": art_b})
    assert "src/model.py" in ctx.dependency_artifacts
    assert "src/train.py" not in ctx.dependency_artifacts


def test_dependency_closure_merges_files() -> None:
    """When two clusters are in depends_on, their files are both present."""
    art_a = _make_artifacts("A", {"src/model.py": "class Model: pass"})
    art_b = _make_artifacts("B", {"src/train.py": "def train(): pass"})
    cluster = _make_cluster(depends_on=["A", "B"])
    ctx = build_context(cluster, paper="", artifacts={"A": art_a, "B": art_b})
    assert "src/model.py" in ctx.dependency_artifacts
    assert "src/train.py" in ctx.dependency_artifacts


def test_dependency_closure_missing_id_skipped() -> None:
    """A depends_on id not in artifacts is silently skipped."""
    cluster = _make_cluster(depends_on=["MISSING"])
    ctx = build_context(cluster, paper="", artifacts={})
    assert ctx.dependency_artifacts == {}


def test_dependency_closure_empty_depends_on() -> None:
    """An empty depends_on yields an empty dependency_artifacts."""
    art = _make_artifacts("OTHER", {"file.py": "code"})
    cluster = _make_cluster(depends_on=[])
    ctx = build_context(cluster, paper="", artifacts={"OTHER": art})
    assert ctx.dependency_artifacts == {}


# ---------------------------------------------------------------------------
# 4. Prior feedback (repair pass)
# ---------------------------------------------------------------------------


def test_prior_feedback_none_when_no_prior_scores() -> None:
    cluster = _make_cluster()
    ctx = build_context(cluster, paper="", artifacts={}, prior_scores=None)
    assert ctx.prior_feedback is None


def test_prior_feedback_weak_leaf_contains_justification() -> None:
    """A leaf scoring 0.1 → its justification appears verbatim in prior_feedback."""
    leaf = _make_leaf("L1", "Reproduce Table 3.", 2.0)
    cluster = _make_cluster(leaves=[leaf])
    prior_scores = {
        "overall_score": 0.1,
        "leaf_count": 1,
        "graded": 1,
        "leaf_scores": [
            {"id": "L1", "score": 0.1, "justification": "Table 3 not present in output."}
        ],
    }
    ctx = build_context(cluster, paper="", artifacts={}, prior_scores=prior_scores)
    assert ctx.prior_feedback is not None
    assert "Table 3 not present in output." in ctx.prior_feedback


def test_prior_feedback_weak_leaf_contains_requirements() -> None:
    """A weak leaf's requirements text also appears in prior_feedback."""
    leaf = _make_leaf("L1", "Reproduce Table 3 within 1% error.", 2.0)
    cluster = _make_cluster(leaves=[leaf])
    prior_scores = {
        "overall_score": 0.1,
        "leaf_count": 1,
        "graded": 1,
        "leaf_scores": [{"id": "L1", "score": 0.2, "justification": "Missing table."}],
    }
    ctx = build_context(cluster, paper="", artifacts={}, prior_scores=prior_scores)
    assert "Reproduce Table 3 within 1% error." in ctx.prior_feedback


def test_prior_feedback_adequate_leaves_still_non_none() -> None:
    """When all leaves of this cluster score ≥ 0.5, prior_feedback is still set
    (tells agent to re-attempt)."""
    leaf = _make_leaf("L1", "Implement forward pass.", 1.0)
    cluster = _make_cluster(leaves=[leaf])
    prior_scores = {
        "overall_score": 0.8,
        "leaf_count": 1,
        "graded": 1,
        "leaf_scores": [{"id": "L1", "score": 0.9, "justification": "Good."}],
    }
    ctx = build_context(cluster, paper="", artifacts={}, prior_scores=prior_scores)
    assert ctx.prior_feedback is not None
    assert ctx.prior_feedback != ""


def test_prior_feedback_ignores_other_clusters_leaves() -> None:
    """Leaf scores for a different cluster's leaf do NOT appear in this cluster's feedback."""
    leaf = _make_leaf("L1", "My leaf", 1.0)
    cluster = _make_cluster(leaves=[leaf])
    prior_scores = {
        "overall_score": 0.0,
        "leaf_count": 2,
        "graded": 2,
        "leaf_scores": [
            {"id": "OTHER_LEAF", "score": 0.0, "justification": "Should not appear."},
        ],
    }
    ctx = build_context(cluster, paper="", artifacts={}, prior_scores=prior_scores)
    # Since L1 (this cluster's leaf) has no entry, all leaves score adequately
    assert ctx.prior_feedback is not None
    assert "Should not appear." not in ctx.prior_feedback


# ---------------------------------------------------------------------------
# 5. Lexical fallback (semantic: citation)
# ---------------------------------------------------------------------------


def test_semantic_fallback_when_no_citations() -> None:
    """A cluster whose leaves have no citations still gets a paper_section via
    lexical retrieval — the citation key starts with 'semantic:'."""
    leaf = _make_leaf(
        "L1",
        "Implement the experiment training loop and log results.",
        1.0,
        paper_citations=[],
    )
    cluster = _make_cluster(leaves=[leaf], paper_citations=[])
    ctx = build_context(cluster, paper=_SIMPLE_PAPER, artifacts={})
    assert any(
        cs.citation.startswith("semantic:") for cs in ctx.paper_sections
    ), "Expected at least one semantic: fallback section"


def test_semantic_fallback_not_applied_when_all_leaves_have_citations() -> None:
    """When every leaf has explicit citations, no semantic: fallback is added."""
    leaf = _make_leaf(
        "L1",
        "See Section 2.",
        1.0,
        paper_citations=["Section 2"],
    )
    cluster = _make_cluster(leaves=[leaf], paper_citations=["Section 2"])
    ctx = build_context(cluster, paper=_SIMPLE_PAPER, artifacts={})
    semantic = [cs for cs in ctx.paper_sections if cs.citation.startswith("semantic:")]
    assert len(semantic) == 0, "Should not add semantic fallback when all leaves have citations"


def test_semantic_fallback_empty_paper_no_error() -> None:
    """Lexical fallback on an empty paper produces no sections (not an error)."""
    leaf = _make_leaf("L1", "Implement training loop.", 1.0, paper_citations=[])
    cluster = _make_cluster(leaves=[leaf])
    ctx = build_context(cluster, paper="", artifacts={})
    assert ctx.paper_sections == []


# ---------------------------------------------------------------------------
# 6. Token budget
# ---------------------------------------------------------------------------


def test_token_budget_respected_with_tiny_budget() -> None:
    """With a tiny budget, estimated window tokens must be within budget, and
    leaf_contract must remain fully intact."""
    leaves = [_make_leaf(f"L{i}", f"Implement requirement number {i} in full.", 1.0) for i in range(5)]
    cluster = _make_cluster(leaves=leaves, weight=5.0, paper_citations=["Section 5"])
    big_dep = _make_artifacts("DEP", {"big.py": "x = 1\n" * 2000})
    cluster2 = _make_cluster(cluster_id="CL2", depends_on=["DEP"])
    cluster2.depends_on = ["DEP"]
    cluster_with_dep = WorkCluster(
        id="CL_TEST",
        title="Test cluster",
        leaves=leaves,
        dominant_category="Code Development",
        weight=5.0,
        depends_on=["DEP"],
        paper_citations=["Section 5"],
    )
    tiny_budget = 210  # well below the content we're passing in (pre-trim ~3200+ tokens)
    ctx = build_context(
        cluster_with_dep,
        paper=_SIMPLE_PAPER,
        artifacts={"DEP": big_dep},
        token_budget=tiny_budget,
    )
    # Estimate actual total
    total = estimate_tokens(ctx.leaf_contract)
    for cs in ctx.paper_sections:
        total += estimate_tokens(cs.text)
    for v in ctx.dependency_artifacts.values():
        total += estimate_tokens(v)
    total += estimate_tokens(ctx.prior_feedback or "")
    total += estimate_tokens(ctx.working_summary)

    assert total <= tiny_budget, f"Budget {tiny_budget} exceeded: got {total}"
    # leaf_contract must be fully intact
    for leaf in leaves:
        assert leaf.requirements in ctx.leaf_contract


def test_leaf_contract_never_trimmed() -> None:
    """No matter how tiny the budget, leaf_contract is preserved verbatim."""
    leaves = [_make_leaf("L1", "A" * 10000, 1.0)]  # very long requirement
    cluster = _make_cluster(leaves=leaves, weight=1.0)
    ctx = build_context(cluster, paper="", artifacts={}, token_budget=10)
    assert "A" * 10000 in ctx.leaf_contract


# ---------------------------------------------------------------------------
# 7. Working summary
# ---------------------------------------------------------------------------


def test_working_summary_empty_when_no_artifacts() -> None:
    cluster = _make_cluster()
    ctx = build_context(cluster, paper="", artifacts={})
    assert ctx.working_summary == ""


def test_working_summary_lists_all_file_paths() -> None:
    art_a = _make_artifacts("A", {"src/a.py": "pass", "src/b.py": "pass"})
    art_b = _make_artifacts("B", {"src/c.py": "pass"})
    cluster = _make_cluster()
    ctx = build_context(cluster, paper="", artifacts={"A": art_a, "B": art_b})
    for path in ("src/a.py", "src/b.py", "src/c.py"):
        assert path in ctx.working_summary


def test_working_summary_includes_notes() -> None:
    art = _make_artifacts("CL_PREV", {"f.py": "code"}, notes="Trained for 10 epochs.")
    cluster = _make_cluster()
    ctx = build_context(cluster, paper="", artifacts={"CL_PREV": art})
    assert "Trained for 10 epochs." in ctx.working_summary


def test_working_summary_paths_sorted() -> None:
    art = _make_artifacts("A", {"z.py": "pass", "a.py": "pass", "m.py": "pass"})
    cluster = _make_cluster()
    ctx = build_context(cluster, paper="", artifacts={"A": art})
    # All paths should appear in sorted order
    paths_in_summary = [p for p in ["a.py", "m.py", "z.py"] if p in ctx.working_summary]
    positions = [ctx.working_summary.index(p) for p in paths_in_summary]
    assert positions == sorted(positions)


# ---------------------------------------------------------------------------
# 8. Determinism
# ---------------------------------------------------------------------------


def test_determinism_identical_calls_produce_equal_output() -> None:
    leaves = [
        _make_leaf("L1", "Implement the LSTM encoder.", 2.0, paper_citations=["Section 5"]),
        _make_leaf("L2", "Run ablation on learning rate.", 1.0),
    ]
    cluster = _make_cluster(leaves=leaves, paper_citations=["Section 5"])
    art = _make_artifacts("PREV", {"model.py": "class LSTM: pass"}, notes="Done.")
    prior_scores = {
        "overall_score": 0.3,
        "leaf_count": 2,
        "graded": 2,
        "leaf_scores": [
            {"id": "L1", "score": 0.3, "justification": "Encoder incomplete."},
            {"id": "L2", "score": 0.8, "justification": "Fine."},
        ],
    }
    kwargs = dict(
        paper=_SIMPLE_PAPER,
        artifacts={"PREV": art},
        prior_scores=prior_scores,
        token_budget=10_000,
    )
    ctx_a = build_context(cluster, **kwargs)
    ctx_b = build_context(cluster, **kwargs)

    assert ctx_a.leaf_contract == ctx_b.leaf_contract
    assert ctx_a.working_summary == ctx_b.working_summary
    assert ctx_a.prior_feedback == ctx_b.prior_feedback
    assert ctx_a.dependency_artifacts == ctx_b.dependency_artifacts
    assert len(ctx_a.paper_sections) == len(ctx_b.paper_sections)
    for a_sec, b_sec in zip(ctx_a.paper_sections, ctx_b.paper_sections):
        assert a_sec.citation == b_sec.citation
        assert a_sec.heading == b_sec.heading
        assert a_sec.text == b_sec.text


def test_determinism_does_not_mutate_cluster(paperbench_rubric: dict) -> None:
    """build_context does not modify the WorkCluster or its leaves."""
    clusters = decompose(paperbench_rubric)
    cluster = clusters[0]
    original_leaves = copy.deepcopy(cluster.leaves)
    original_citations = list(cluster.paper_citations)
    build_context(cluster, paper=_SIMPLE_PAPER, artifacts={})
    assert cluster.paper_citations == original_citations
    assert [lf.id for lf in cluster.leaves] == [lf.id for lf in original_leaves]


# ---------------------------------------------------------------------------
# 9. Integration with real paperbench fixture
# ---------------------------------------------------------------------------


def test_paperbench_clusters_all_get_context(paperbench_rubric: dict) -> None:
    """Every cluster from decompose() can produce an AgentContext without error."""
    clusters = decompose(paperbench_rubric)
    for cluster in clusters:
        ctx = build_context(cluster, paper=_SIMPLE_PAPER, artifacts={})
        assert ctx.leaf_contract
        assert ctx.cluster is cluster
        assert ctx.prior_feedback is None


def test_paperbench_leaf_contract_completeness(paperbench_rubric: dict) -> None:
    """For every cluster, every leaf's requirements text is in the leaf_contract."""
    clusters = decompose(paperbench_rubric)
    for cluster in clusters:
        ctx = build_context(cluster, paper=_SIMPLE_PAPER, artifacts={})
        for leaf in cluster.leaves:
            assert leaf.requirements in ctx.leaf_contract, (
                f"Leaf {leaf.id!r} requirements missing from contract of cluster {cluster.id!r}"
            )


def test_paperbench_dependency_chain_works(paperbench_rubric: dict) -> None:
    """Simulate a run: build context for Code Execution clusters using Code
    Development artifacts; dependency_artifacts should contain those files."""
    clusters = decompose(paperbench_rubric)
    cd_clusters = [c for c in clusters if c.dominant_category == "Code Development"]
    ce_clusters = [c for c in clusters if c.dominant_category == "Code Execution"]

    if not cd_clusters or not ce_clusters:
        pytest.skip("No CD or CE clusters in fixture")

    # Simulate finished CD artifacts.
    artifacts: dict[str, Artifacts] = {}
    for c in cd_clusters:
        artifacts[c.id] = Artifacts(
            cluster_id=c.id,
            files={f"src/{c.id}.py": f"# {c.id} code"},
        )

    for ce_cluster in ce_clusters[:2]:  # test a sample
        ctx = build_context(ce_cluster, paper=_SIMPLE_PAPER, artifacts=artifacts)
        # CE depends on all CD — so dependency_artifacts should contain CD files.
        for cd in cd_clusters:
            assert f"src/{cd.id}.py" in ctx.dependency_artifacts, (
                f"CE cluster {ce_cluster.id!r} missing dep file for {cd.id!r}"
            )


# ---------------------------------------------------------------------------
# 10. estimate_tokens
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 11. Nested section body — parent extends through subsections
# ---------------------------------------------------------------------------

_NESTED_PAPER = """\
## 5 Experiments

We ran experiments overall.

### 5.1 Setup

The experimental setup involved GPU clusters.

### 5.2 Results

Our results show significant improvements in accuracy.

## 6 Conclusion

We conclude the paper here.
"""


def test_parent_section_body_includes_subsections() -> None:
    """Citing 'Section 5' on a paper with ## 5 / ### 5.1 / ### 5.2 must
    return a CitedSection whose text contains content from BOTH 5.1 and 5.2."""
    leaf = _make_leaf("L1", "Reproduce Section 5 results.", 1.0, paper_citations=["Section 5"])
    cluster = _make_cluster(leaves=[leaf], paper_citations=["Section 5"])
    ctx = build_context(cluster, paper=_NESTED_PAPER, artifacts={})

    # Find the CitedSection for Section 5 (heading contains "5" but not "5.1" or "5.2")
    sec5 = next(
        (cs for cs in ctx.paper_sections if "5" in cs.heading and "5.1" not in cs.heading and "5.2" not in cs.heading),
        None,
    )
    assert sec5 is not None, f"Section 5 not resolved; got headings: {[cs.heading for cs in ctx.paper_sections]}"
    # Body must contain text from both subsections
    assert "GPU clusters" in sec5.text, "Section 5 body missing 5.1 content ('GPU clusters')"
    assert "significant improvements in accuracy" in sec5.text, "Section 5 body missing 5.2 content ('significant improvements in accuracy')"


def test_estimate_tokens_empty() -> None:
    assert estimate_tokens("") == 0


def test_estimate_tokens_basic() -> None:
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 8) == 2
    assert estimate_tokens("a" * 100) == 25


# ---------------------------------------------------------------------------
# FIX 4: _trim_to_budget closure correctness — section trimming reduces usage
# ---------------------------------------------------------------------------


def test_token_budget_multi_section_trimming_loop() -> None:
    """_trim_to_budget must keep trimming paper sections iteratively until the
    budget is met — the closure must see updated (trimmed) text, not the original.

    We construct a scenario where a single halving is not enough:
    - leaf_contract: ~50 tokens
    - section body: 2000 chars → 500 tokens; half = 250 tokens; quarter = 125 tokens
    - tiny budget = 200 tokens (well above leaf_contract alone but below one section)
    """
    from backend.agents.rdr.context_engineer import _trim_to_budget, estimate_tokens
    from backend.agents.rdr.models import CitedSection

    leaf_contract = "A" * 200  # ~50 tokens
    big_body = "B" * 2000  # 500 tokens
    section = CitedSection(citation="Section 1", heading="1 Methods", text=big_body)

    # Budget: enough for leaf_contract (50 tokens) + a small bit of section body.
    # leaf_contract alone = 50, big_body = 500 → total without trimming = 550.
    # budget = 200 → section body must be trimmed to ≤ 150 tokens → ≤ 600 chars.
    budget = 200

    trimmed_sections, dep, summary = _trim_to_budget(
        leaf_contract=leaf_contract,
        paper_sections=[section],
        dependency_artifacts={},
        prior_feedback=None,
        working_summary="",
        token_budget=budget,
    )

    # Compute actual usage after trimming
    total = estimate_tokens(leaf_contract)
    for cs in trimmed_sections:
        total += estimate_tokens(cs.text)

    assert total <= budget, (
        f"Budget {budget} exceeded after _trim_to_budget: got {total} tokens"
    )


def test_token_budget_multi_section_many_sections() -> None:
    """With many large sections and a tight budget, the iterative loop must trim
    each section multiple times until the budget is met.  The closure-captures-
    name bug would cause this to overshoot the budget.
    """
    from backend.agents.rdr.context_engineer import _trim_to_budget, estimate_tokens
    from backend.agents.rdr.models import CitedSection

    leaf_contract = "L" * 40  # ~10 tokens
    sections = [
        CitedSection(
            citation=f"Section {i}",
            heading=f"{i} Section",
            text="X" * 1600,  # 400 tokens each
        )
        for i in range(1, 6)  # 5 sections → 2000 tokens total
    ]
    # Budget: 10 (contract) + 90 (headroom) = 100 tokens total
    budget = 100

    trimmed_sections, _, _ = _trim_to_budget(
        leaf_contract=leaf_contract,
        paper_sections=sections,
        dependency_artifacts={},
        prior_feedback=None,
        working_summary="",
        token_budget=budget,
    )

    total = estimate_tokens(leaf_contract)
    for cs in trimmed_sections:
        total += estimate_tokens(cs.text)

    assert total <= budget, (
        f"Budget {budget} exceeded with many sections: got {total} tokens"
    )
