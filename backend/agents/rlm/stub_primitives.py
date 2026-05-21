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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Individual stub callables (§5 signatures)
# ---------------------------------------------------------------------------


def _understand_section(text_slice: str) -> dict:
    """Stub for `understand_section(text_slice: str) -> dict`.

    Returns a minimal fixed partial-claim map (PaperClaimMap shape).
    """
    return {
        "claims": [
            {
                "id": "stub-claim-1",
                "text": "Stub claim extracted from section.",
                "metric": "accuracy",
                "value": 0.0,
                "section": "stub",
            }
        ],
        "datasets": ["stub-dataset"],
        "metrics": ["accuracy"],
        "source_slice_len": len(text_slice),
    }


def _extract_hyperparameters(text_slice: str) -> dict:
    """Stub for `extract_hyperparameters(text_slice: str) -> dict`.

    Returns a minimal fixed TrainingRecipe shape.
    """
    return {
        "learning_rate": 1e-4,
        "batch_size": 32,
        "epochs": 10,
        "optimizer": "adam",
        "framework": "pytorch",
        "source_slice_len": len(text_slice),
    }


def _detect_environment(method_spec: dict) -> dict:
    """Stub for `detect_environment(method_spec: dict) -> dict`.

    Returns a minimal fixed EnvironmentSpec shape.
    """
    return {
        "python_version": "3.10",
        "cuda_version": "11.8",
        "packages": ["torch==2.0.0", "numpy>=1.24"],
        "base_image": "pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime",
        "env_type": "docker",
        "source_spec_keys": list(method_spec.keys()) if isinstance(method_spec, dict) else [],
    }


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

    Returns a minimal fixed ReproductionContract shape.
    """
    return {
        "steps": [
            {"id": "step-1", "action": "train", "description": "Stub training step."},
            {"id": "step-2", "action": "evaluate", "description": "Stub evaluation step."},
        ],
        "entry_point": "train.py",
        "eval_command": "python evaluate.py",
        "expected_outputs": ["model.pt", "results.json"],
        "spec_keys_used": list(method_spec.keys()) if isinstance(method_spec, dict) else [],
    }


def _implement_baseline(plan: dict) -> str:
    """Stub for `implement_baseline(plan: dict) -> str`.

    Returns a fixed code-dir path string.
    """
    return "/stub/code"


def _run_experiment(code_path: str, env_id: str) -> dict:
    """Stub for `run_experiment(code_path: str, env_id: str) -> dict`.

    Returns ``{success, metrics, logs}`` shape.
    """
    return {
        "success": True,
        "metrics": {},
        "logs": "stub",
    }


def _verify_against_rubric(results: dict, rubric: dict) -> dict:
    """Stub for `verify_against_rubric(results: dict, rubric: dict) -> dict`.

    Returns a minimal fixed RubricVerification shape.
    """
    return {
        "overall_score": 0.5,
        "meets_target": False,
        "areas": [
            {"name": "accuracy", "score": 0.5, "notes": "stub score"},
        ],
        "verdict": "partial",
        "details": "Stub rubric verification — no real evaluation performed.",
    }


def _propose_improvements(
    current_results: Any,
    rubric_scores: Any,
    k: int | None = None,
) -> list[dict]:
    """Stub for `propose_improvements(current_results, rubric_scores, k=None) -> list[dict]`.

    Returns a fixed list of improvement candidate dicts.
    """
    candidates = [
        {
            "tag": "stub-improvement-1",
            "description": "Tune learning rate schedule.",
            "target_rubric_area": "accuracy",
            "estimated_uplift": 0.05,
        },
        {
            "tag": "stub-improvement-2",
            "description": "Increase training epochs.",
            "target_rubric_area": "accuracy",
            "estimated_uplift": 0.03,
        },
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
