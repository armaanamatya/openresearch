"""Rubric accounting and upstream PaperBench judge command construction."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


SIMPLE_JUDGE_COMPLETER_CONFIG = (
    "preparedness_turn_completer.oai_completions_turn_completer:"
    "OpenAICompletionsTurnCompleter.Config"
)
SIMPLE_JUDGE_MODEL = "o3-mini-2025-01-31"
SIMPLE_JUDGE_REASONING_EFFORT = "high"


@dataclass(frozen=True)
class RubricCategoryWeight:
    """Weighted contribution of a category in a PaperBench rubric."""

    category: str
    weight: float
    leaf_count: int

    @property
    def percent(self) -> float:
        return self.weight * 100.0


@dataclass(frozen=True)
class RubricSummary:
    """PaperBench rubric shape and leaf-weight accounting."""

    node_count: int
    leaf_count: int
    max_depth: int
    task_category_weights: dict[str, RubricCategoryWeight] = field(default_factory=dict)
    finegrained_category_weights: dict[str, RubricCategoryWeight] = field(default_factory=dict)

    def category_weight(self, category: str) -> float:
        item = self.task_category_weights.get(category)
        return item.weight if item else 0.0

    def finegrained_weight(self, category: str) -> float:
        item = self.finegrained_category_weights.get(category)
        return item.weight if item else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_count": self.node_count,
            "leaf_count": self.leaf_count,
            "max_depth": self.max_depth,
            "task_category_weights": {
                key: {
                    "weight": value.weight,
                    "percent": value.percent,
                    "leaf_count": value.leaf_count,
                }
                for key, value in self.task_category_weights.items()
            },
            "finegrained_category_weights": {
                key: {
                    "weight": value.weight,
                    "percent": value.percent,
                    "leaf_count": value.leaf_count,
                }
                for key, value in self.finegrained_category_weights.items()
            },
        }


@dataclass(frozen=True)
class PaperBenchJudgeCommand:
    """A deterministic invocation of the upstream PaperBench judge.

    The command targets PaperBench's public ``paperbench.scripts.run_judge``
    entrypoint.  The simple judge requires the caller to pass the same upstream
    completer config used for the comparison being reported.
    """

    frontier_evals_dir: Path
    submission_path: Path
    paper_id: str
    out_dir: Path
    judge: str = "simple"
    max_depth: int = 999
    code_only: bool = False
    resources_provided: bool = False
    completer_config: str | None = SIMPLE_JUDGE_COMPLETER_CONFIG
    completer_model: str | None = SIMPLE_JUDGE_MODEL
    completer_reasoning_effort: str | None = SIMPLE_JUDGE_REASONING_EFFORT

    def argv(self) -> list[str]:
        args = [
            "uv",
            "run",
            "python",
            "-m",
            "paperbench.scripts.run_judge",
            f"submission_path={self.submission_path}",
            f"paper_id={self.paper_id}",
            f"judge={self.judge}",
            f"max_depth={self.max_depth}",
            f"out_dir={self.out_dir}",
            f"code_only={str(self.code_only)}",
            f"resources_provided={str(self.resources_provided)}",
        ]
        if self.completer_config is not None:
            args.append(f"completer_config={self.completer_config}")
        if self.completer_model is not None:
            args.append(f"completer_config.model={self.completer_model}")
        if self.completer_reasoning_effort is not None:
            args.append(
                f"completer_config.reasoning_effort={self.completer_reasoning_effort}"
            )
        return args

    @property
    def cwd(self) -> Path:
        return self.frontier_evals_dir


def summarize_rubric(rubric: dict[str, Any]) -> RubricSummary:
    """Summarize a PaperBench rubric using upstream recursive weight semantics."""

    nodes = list(_iter_nodes(rubric, depth=0, normalized_weight=1.0))
    task_weights: dict[str, tuple[float, int]] = {}
    fine_weights: dict[str, tuple[float, int]] = {}
    for node, depth, normalized_weight in nodes:
        children = _children(node)
        if children:
            continue
        task_category = str(node.get("task_category") or "Uncategorized")
        fine_category = str(node.get("finegrained_task_category") or "Uncategorized")
        task_weights[task_category] = _add_weight(
            task_weights.get(task_category), normalized_weight
        )
        fine_weights[fine_category] = _add_weight(
            fine_weights.get(fine_category), normalized_weight
        )

    return RubricSummary(
        node_count=len(nodes),
        leaf_count=sum(1 for node, _depth, _weight in nodes if not _children(node)),
        max_depth=max((depth for _node, depth, _weight in nodes), default=0),
        task_category_weights=_category_map(task_weights),
        finegrained_category_weights=_category_map(fine_weights),
    )


def code_development_ceiling(rubric: dict[str, Any]) -> float:
    """Return the full-rubric score ceiling for a code-only perfect submission."""

    return summarize_rubric(rubric).category_weight("Code Development")


def mean_standard_error(values: Iterable[float]) -> tuple[float, float, int]:
    """Return mean, sample-standard-error, and n using PaperBench's convention."""

    scores = [float(value) for value in values]
    if not scores:
        raise ValueError("mean_standard_error requires at least one value")
    mean = sum(scores) / len(scores)
    if len(scores) == 1:
        return mean, 0.0, 1
    variance = sum((score - mean) ** 2 for score in scores) / (len(scores) - 1)
    return mean, math.sqrt(variance) / math.sqrt(len(scores)), len(scores)


def _iter_nodes(
    node: dict[str, Any],
    *,
    depth: int,
    normalized_weight: float,
) -> Iterable[tuple[dict[str, Any], int, float]]:
    yield node, depth, normalized_weight
    children = _children(node)
    if not children:
        return
    total_weight = sum(float(child.get("weight", 0.0) or 0.0) for child in children)
    if total_weight <= 0:
        child_weight = 0.0
        for child in children:
            yield from _iter_nodes(child, depth=depth + 1, normalized_weight=child_weight)
        return
    for child in children:
        child_weight = normalized_weight * float(child.get("weight", 0.0) or 0.0) / total_weight
        yield from _iter_nodes(child, depth=depth + 1, normalized_weight=child_weight)


def _children(node: dict[str, Any]) -> list[dict[str, Any]]:
    value = node.get("sub_tasks") or []
    return [child for child in value if isinstance(child, dict)]


def _add_weight(current: tuple[float, int] | None, weight: float) -> tuple[float, int]:
    if current is None:
        return weight, 1
    return current[0] + weight, current[1] + 1


def _category_map(values: dict[str, tuple[float, int]]) -> dict[str, RubricCategoryWeight]:
    return {
        key: RubricCategoryWeight(category=key, weight=weight, leaf_count=count)
        for key, (weight, count) in sorted(values.items())
    }
