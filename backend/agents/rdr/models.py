"""Data contracts for the rubric-driven reproduction harness (``rdr``).

These dataclasses are the stable interface between the harness components:
the Decomposer cuts the official PaperBench rubric tree into ``WorkCluster``s
of ``RubricLeaf``s; the Context Engineer turns a cluster into an
``AgentContext``; the Reproduction Agent returns ``Artifacts``; the Controller
returns an ``RdrResult``.

See ``docs/superpowers/specs/2026-05-22-rubric-driven-harness-design.md`` §4.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

# Canonical task-category dependency order: Code Development must precede Code
# Execution, which must precede Result Analysis. The Decomposer sorts clusters
# by this order; ``index()`` on it is the sort key.
TASK_CATEGORY_ORDER: tuple[str, ...] = (
    "Code Development",
    "Code Execution",
    "Result Analysis",
)


@dataclass(frozen=True)
class RubricLeaf:
    """One gradable leaf of the official PaperBench rubric tree.

    ``requirements`` is the verbatim rubric text — the gradable contract the
    leaf scorer judges against, and the spine of the agent's context window.
    ``paper_citations`` are reference strings ("Section 5", "Appendix E.1")
    parsed out of ``requirements`` by the Decomposer.
    """

    id: str
    requirements: str
    weight: float
    task_category: str  # Code Development | Code Execution | Result Analysis
    paper_citations: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CitedSection:
    """A slice of the paper retrieved into a cluster's context window.

    ``citation`` is the reference it answers ("Section 5", "Appendix E.1", or
    "semantic:<query>" for the retrieval fallback). ``heading`` is the resolved
    section heading ("" when unknown). ``text`` is the section body.
    """

    citation: str
    heading: str
    text: str


@dataclass
class WorkCluster:
    """An agent-sized unit of reproduction work — a coherent mid-level rubric
    subtree. Every leaf in ``leaves`` is a Controller obligation: attempted,
    scored, and repaired if weak."""

    id: str
    title: str  # the cluster node's `requirements` text
    leaves: list[RubricLeaf]
    dominant_category: str
    weight: float  # sum of leaf weights
    depends_on: list[str] = field(default_factory=list)  # cluster ids
    paper_citations: list[str] = field(default_factory=list)  # union of leaf citations


@dataclass
class Artifacts:
    """What one Reproduction Agent invocation produced for one cluster.

    ``files`` maps repo-relative paths to file content; the Controller merges
    them into the assembled project. ``commands`` are run commands the cluster
    contributed. ``failed`` is the fail-soft flag — an agent error yields
    ``failed=True`` with empty ``files`` and a populated ``error``.
    """

    cluster_id: str
    files: dict[str, str] = field(default_factory=dict)
    commands: list[str] = field(default_factory=list)
    notes: str = ""
    failed: bool = False
    error: str = ""


@dataclass
class AgentContext:
    """The precisely-engineered context window for one Reproduction Agent
    invocation. Built deterministically by the Context Engineer (design §6).

    ``leaf_contract`` is the verbatim cluster leaves + weights, formatted as
    the agent's gradable contract. ``prior_feedback`` is populated only on a
    repair pass — the failed leaves plus the leaf scorer's justifications.
    """

    cluster: WorkCluster
    leaf_contract: str
    paper_sections: list[CitedSection]
    dependency_artifacts: dict[str, str] = field(default_factory=dict)
    prior_feedback: str | None = None
    working_summary: str = ""
    # BES (2026-06-07): when set, the agent writes into this per-candidate scratch
    # dir instead of the shared project_dir/code — so N competing candidates build
    # + score in isolation. None => shared code dir (today's path). String-typed
    # annotation under `from __future__ import annotations` (no runtime import).
    candidate_code_dir: "Path | None" = None


@dataclass
class RdrResult:
    """The outcome of a full ``rdr`` run — returned by ``run_pipeline_rdr``."""

    project_id: str
    status: str  # completed | partial | failed
    rubric_score: float | None = None
    clusters_total: int = 0
    clusters_failed: int = 0
    repair_iterations: int = 0
    final_report_path: str | None = None
    cost_usd: float | None = None


__all__ = [
    "TASK_CATEGORY_ORDER",
    "RubricLeaf",
    "CitedSection",
    "WorkCluster",
    "Artifacts",
    "AgentContext",
    "RdrResult",
]

# Re-exported for callers that catch frozen-dataclass mutation.
FrozenInstanceError = dataclasses.FrozenInstanceError
