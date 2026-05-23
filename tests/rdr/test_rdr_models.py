"""Smoke tests for the ``rdr`` data contracts (``backend/agents/rdr/models.py``)."""

from __future__ import annotations

import dataclasses

import pytest

from backend.agents.rdr.models import (
    AgentContext,
    Artifacts,
    CitedSection,
    RdrResult,
    RubricLeaf,
    TASK_CATEGORY_ORDER,
    WorkCluster,
)


def _leaf(leaf_id: str = "L1") -> RubricLeaf:
    return RubricLeaf(
        id=leaf_id,
        requirements="Implement X as described in Section 5.",
        weight=1.0,
        task_category="Code Development",
        paper_citations=["Section 5"],
    )


def test_rubric_leaf_is_frozen() -> None:
    leaf = _leaf()
    assert leaf.weight == 1.0
    assert leaf.paper_citations == ["Section 5"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        leaf.weight = 2.0  # type: ignore[misc]


def test_cited_section_is_frozen() -> None:
    sec = CitedSection(citation="Section 5", heading="5 Experiments", text="body")
    assert sec.citation == "Section 5"
    with pytest.raises(dataclasses.FrozenInstanceError):
        sec.text = "other"  # type: ignore[misc]


def test_work_cluster_holds_leaves_with_list_defaults() -> None:
    cluster = WorkCluster(
        id="C1",
        title="Core method",
        leaves=[_leaf("L1"), _leaf("L2")],
        dominant_category="Code Development",
        weight=2.0,
    )
    assert len(cluster.leaves) == 2
    assert cluster.depends_on == []
    assert cluster.paper_citations == []


def test_artifacts_failsoft_defaults() -> None:
    art = Artifacts(cluster_id="C1")
    assert art.files == {}
    assert art.commands == []
    assert art.failed is False
    assert art.error == ""


def test_agent_context_construction() -> None:
    cluster = WorkCluster(
        id="C1",
        title="t",
        leaves=[_leaf()],
        dominant_category="Code Development",
        weight=1.0,
    )
    actx = AgentContext(
        cluster=cluster,
        leaf_contract="- L1 (w=1.0): Implement X",
        paper_sections=[CitedSection("Section 5", "5 Experiments", "body")],
    )
    assert actx.prior_feedback is None
    assert actx.dependency_artifacts == {}
    assert actx.working_summary == ""


def test_rdr_result_construction() -> None:
    res = RdrResult(project_id="p", status="completed", rubric_score=0.5)
    assert res.status == "completed"
    assert res.clusters_total == 0
    assert res.clusters_failed == 0
    assert res.repair_iterations == 0
    assert res.cost_usd is None


def test_task_category_order_is_canonical() -> None:
    assert TASK_CATEGORY_ORDER == (
        "Code Development",
        "Code Execution",
        "Result Analysis",
    )
