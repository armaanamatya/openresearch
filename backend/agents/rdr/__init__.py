"""Rubric-driven reproduction harness (``rdr``).

A deterministic Python controller decomposes the official PaperBench rubric
into work-clusters and dispatches scoped Claude reproduction agents — one per
cluster, each with a precisely-engineered context window — then scores the
result against the exact rubric and repairs weak clusters in a capped loop.

``rdr`` is one of three supported reproduction modes — peer to ``rlm``
(default hybrid: RDR phase 1 + RLM phase 2) and ``rlm-pure`` (pure RLM
escape hatch). The legacy ``sdk`` / ``offline`` paths were removed in
PR #72 (the 14-stage pipeline cleanup).
"""

from backend.agents.rdr.models import (
    AgentContext,
    Artifacts,
    CitedSection,
    RdrResult,
    RubricLeaf,
    TASK_CATEGORY_ORDER,
    WorkCluster,
)

__all__ = [
    "TASK_CATEGORY_ORDER",
    "RubricLeaf",
    "CitedSection",
    "WorkCluster",
    "Artifacts",
    "AgentContext",
    "RdrResult",
]
