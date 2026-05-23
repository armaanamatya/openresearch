"""Deterministic stub primitives for the RLM orchestrator.

Development and integration aid — ships as internal backend code (not in
`tests/`) so that `run.py`'s `_resolve_custom_tools` can exercise the full
orchestrator loop before Issue #59's real `binding.build_custom_tools` lands.

Each stub matches the root-visible signature from spec §5 exactly and returns
deterministically-shaped data matching the §5 return column.  Stubs do not
contact any external service and do not depend on `ctx` beyond accepting it in
the factory call.

Usage:
    tools = build_stub_custom_tools(ctx)
    # pass `tools` as `custom_tools` to `rlm.RLM(...)`

Spec: §13 (2026-05-21-rlm-phase3-orchestrator-design.md).
"""

from __future__ import annotations

import logging
from typing import Any

from backend.agents.rlm.context import RunContext
from backend.agents.schemas import (
    EnvironmentSpec,
    ImprovementHypothesis,
    ReproductionContract,
    TrainingRecipe,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Individual stub callables (§5 signatures)
# ---------------------------------------------------------------------------


def _understand_section(text_slice: str) -> dict:
    """Stub for `understand_section(text_slice: str) -> dict`.

    Returns a partial PaperClaimMap shape — only the keys the RLM root
    can observe from this primitive (datasets, metrics, training_recipe,
    hardware_clues, ambiguities).  Keys not in PaperClaimMap would drift
    the root's REPL code away from what the real primitive returns (T20/I7).
    """
    return {
        "datasets": [{"name": "stub-dataset"}],
        "metrics": [{"name": "accuracy"}],
        "training_recipe": TrainingRecipe().model_dump(),
        "hardware_clues": ["stub GPU clue"],
        "ambiguities": [],
    }


def _extract_hyperparameters(text_slice: str) -> dict:
    """Stub for `extract_hyperparameters(text_slice: str) -> dict`.

    Returns a TrainingRecipe.model_dump() with placeholder values so keys
    cannot drift from the real schema (T20 / review I7).
    """
    return TrainingRecipe(
        optimizer="adam",
        learning_rate="1e-4",
        batch_size="32",
        epochs_or_steps="10",
        scheduler="",
    ).model_dump()


def _detect_environment(method_spec: dict) -> dict:
    """Stub for `detect_environment(method_spec: dict) -> dict`.

    Returns an EnvironmentSpec.model_dump() with placeholder values so keys
    cannot drift from the real schema (T20 / review I7).
    """
    return EnvironmentSpec(
        dockerfile="FROM python:3.11-slim\n",
        python_version="3.11",
        framework="pytorch",
        pip_packages={"numpy": "latest"},
    ).model_dump()


def _build_environment(env_spec: dict) -> dict:
    """Stub for `build_environment(env_spec: dict) -> dict`.

    Returns ``{ok, image_tag, error, attempts}`` shape.
    """
    return {
        "ok": True,
        "image_tag": "stub:latest",
        "error": "",
        "attempts": 1,
    }


def _plan_reproduction(method_spec: dict, env_spec: dict) -> dict:
    """Stub for `plan_reproduction(method_spec: dict, env_spec: dict) -> dict`.

    Returns a ReproductionContract.model_dump() with placeholder values so keys
    cannot drift from the real schema (T20 / review I7).
    """
    return ReproductionContract(
        reproduction_definition="Train and evaluate the stub baseline.",
        smoke_test_plan="python train.py --epochs 1",
        full_run_plan="python train.py",
        expected_outputs=["model.pt", "results.json"],
        dataset_plan="Use stub dataset.",
        evaluation_plan="python evaluate.py",
        verification_checklist=["accuracy metric produced"],
    ).model_dump()


def _implement_baseline(plan: dict) -> str:
    """Stub for `implement_baseline(plan: dict) -> str`.

    Returns a fixed code-dir path string.
    """
    return "/stub/code"


def _run_experiment(code_path: str, env_id: str) -> dict:
    """Stub for `run_experiment(code_path: str, env_id: str) -> dict`.

    Returns ``{success, metrics, logs}`` — the documented real return shape
    (T20 / review I7).
    """
    return {
        "success": True,
        "metrics": {"placeholder_metric": 1.0},
        "logs": "stub run",
    }


def _verify_against_rubric(results: dict, rubric: dict) -> dict:
    """Stub for `verify_against_rubric(results: dict, rubric: dict) -> dict`.

    Returns the shape matching the real primitive's documented return:
    overall_score, meets_target, target_score, rubric_source, leaf_count,
    graded, degraded, weak_leaves, leaf_scores (T20 / review I7).
    """
    return {
        "overall_score": 0.5,
        "meets_target": False,
        "target_score": 0.5,
        "rubric_source": "paperbench_bundle",
        "leaf_count": 1,
        "graded": 1,
        "degraded": False,
        "weak_leaves": [],
        "leaf_scores": [],
    }


def _propose_improvements(
    current_results: Any,
    rubric_scores: Any,
    k: int | None = None,
) -> list[dict]:
    """Stub for `propose_improvements(current_results, rubric_scores, k=None) -> list[dict]`.

    Returns a list of ImprovementHypothesis.model_dump() items so keys cannot
    drift from the real schema (T20 / review I7).
    """
    candidates = [
        ImprovementHypothesis(
            path_id="stub-path-1",
            hypothesis="Tune the learning rate schedule.",
            rationale="A lower lr schedule often improves convergence.",
            expected_outcome="Higher accuracy metric.",
            compute_estimate="~1x baseline",
            expected_value_score=0.6,
            category="hyperparameter",
        ).model_dump(),
        ImprovementHypothesis(
            path_id="stub-path-2",
            hypothesis="Increase training epochs.",
            rationale="More training typically reduces underfitting.",
            expected_outcome="Lower training loss.",
            compute_estimate="~2x baseline",
            expected_value_score=0.5,
            category="hyperparameter",
        ).model_dump(),
    ]
    if k is not None:
        candidates = candidates[:k]
    return candidates


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

# Mapping: primitive name -> (callable, description)
_STUB_ENTRIES: list[tuple[str, Any, str]] = [
    (
        "understand_section",
        _understand_section,
        "Extract claims, datasets, and metrics from a section-sized text slice.",
    ),
    (
        "extract_hyperparameters",
        _extract_hyperparameters,
        "Extract training hyperparameters (lr, batch size, optimizer, …) from a text slice.",
    ),
    (
        "detect_environment",
        _detect_environment,
        "Infer Python/CUDA/framework versions and required packages from a method_spec.",
    ),
    (
        "build_environment",
        _build_environment,
        "Build a Docker image for an env_spec; returns {ok, image_tag, error, attempts}.",
    ),
    (
        "plan_reproduction",
        _plan_reproduction,
        "Generate a step-by-step reproduction plan from a method_spec and env_spec.",
    ),
    (
        "implement_baseline",
        _implement_baseline,
        "Generate the baseline code from a reproduction plan; returns the code-dir path.",
    ),
    (
        "run_experiment",
        _run_experiment,
        "Execute the baseline in the sandbox; returns {success, metrics, logs}.",
    ),
    (
        "verify_against_rubric",
        _verify_against_rubric,
        "Score experimental results against a PaperBench-style rubric.",
    ),
    (
        "propose_improvements",
        _propose_improvements,
        "Propose k improvement candidates ranked by expected rubric uplift.",
    ),
]


def build_stub_custom_tools(ctx: RunContext) -> dict[str, dict]:
    """Return the 9 deterministic stub primitives as an RLM `custom_tools` dict.

    Each entry has the ``{"tool": callable, "description": str}`` shape that
    `rlm.RLM(custom_tools=...)` requires.

    The `ctx` argument is accepted for API-compatibility with
    `binding.build_custom_tools` — stub callables do not close over it.

    Args:
        ctx: Run-scoped context (accepted for signature parity; not used).

    Returns:
        A dict with exactly 9 entries, one per primitive in spec §5.
    """
    logger.debug(
        "stub_primitives: building stub custom tools for project_id=%s",
        getattr(ctx, "project_id", "?"),
    )
    return {
        name: {"tool": tool, "description": description}
        for name, tool, description in _STUB_ENTRIES
    }
