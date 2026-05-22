"""Rubric-driven reproduction harness (``rdr``).

A deterministic Python controller decomposes the official PaperBench rubric
into work-clusters and dispatches scoped Claude reproduction agents — one per
cluster, each with a precisely-engineered context window — then scores the
result against the exact rubric and repairs weak clusters in a capped loop.

``rdr`` is an additive, opt-in run mode; ``rlm`` / ``sdk`` / ``offline`` are
untouched. See ``docs/superpowers/specs/2026-05-22-rubric-driven-harness-design.md``.
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
