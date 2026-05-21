"""Primitive function registry exposed to the RLM REPL.

Each primitive wraps an existing stage agent's core function (see
`docs/rlm-pivot-mapping.md` §1) and:
  - Emits a `primitive_call` SSE event for the live-iteration UI
  - Updates `cost_ledger.jsonl`
  - Returns a structured dict the root can store in REPL variables

**Algorithm-2 guard (brief §7.7, mapping doc §1 invariant):**
No primitive signature accepts `paper_text` / `supplementary_text` /
`repo_files` as a whole-corpus argument. Primitives take slices and
structured specs only. The root assembles slices with REPL code and
`llm_query`/`rlm_query` against constructed slices.

Phase 2 (#59) implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from backend.agents.rlm.context import RunContext


def understand_section(text_slice: str, *, ctx: "RunContext") -> dict:
    """Extract datasets/metrics/training-recipe/hardware/ambiguities from a slice.

    Wraps the *title-agnostic* heuristic helpers in
    `backend/agents/paper_understanding.py`. Returns a PARTIAL PaperClaimMap
    dict — `core_contribution`, `claims`, `model_architecture` and
    `evaluation_protocol` need section titles and are left for the root model
    to extract with `llm_query` over `context` (design decision D5).
    """
    from backend.agents.paper_understanding import (
        _extract_datasets, _extract_metrics, _extract_training_recipe,
        _extract_hardware, _extract_ambiguities,
    )
    sections = {"_": text_slice}
    return {
        "datasets": [d.model_dump() for d in _extract_datasets(sections)],
        "metrics": [m.model_dump() for m in _extract_metrics(sections)],
        "training_recipe": _extract_training_recipe(sections).model_dump(),
        "hardware_clues": _extract_hardware(sections),
        "ambiguities": [a.model_dump() for a in _extract_ambiguities(sections)],
    }


def extract_hyperparameters(text_slice: str) -> dict:
    """Extract hyperparameters from a slice (typically the training-recipe section).

    Wraps `backend/agents/paper_understanding.py::_extract_training_recipe`.
    """
    raise NotImplementedError("Phase 3 (#60) — wrap paper_understanding._extract_training_recipe")


def detect_environment(method_spec: dict) -> dict:
    """Infer Python/CUDA/framework versions and pip packages from a method_spec.

    Wraps `backend/agents/environment_detective.py::run_offline` core logic.
    Returns a dict matching the `EnvironmentSpec` schema shape.
    """
    raise NotImplementedError("Phase 3 (#60) — wrap environment_detective.run_offline")


def build_environment(env_spec: dict) -> dict:
    """Build the Docker image for an env_spec, with repair-on-failure.

    Wraps the existing Docker build-and-repair loop (Track 4). The internal
    retry loop (`environment_build_max_attempts`) lives inside the primitive
    — the root sees one call, gets one result.
    """
    raise NotImplementedError("Phase 3 (#60) — wrap env build-and-repair loop")


def plan_reproduction(method_spec: dict, env_spec: dict) -> dict:
    """Generate a reproduction plan from structured specs (NOT from paper_text)."""
    raise NotImplementedError("Phase 3 (#60) — wrap reproduction-planner agent")


def implement_baseline(plan: dict) -> str:
    """Generate the baseline code from a reproduction plan; return code_path."""
    raise NotImplementedError("Phase 3 (#60) — wrap baseline_implementation.run_offline")


def run_experiment(code_path: str, env_id: str) -> dict:
    """Execute the baseline in the sandbox; return metrics.

    Sandbox state (Docker image, RunPod pod) lives outside the REPL. This
    primitive is a sync wrapper that schedules the async sandbox call on
    the orchestrator's event loop and blocks until it returns.
    See `docs/rlm-pivot-mapping.md` §6.5.
    """
    raise NotImplementedError("Phase 3 (#60) — wrap experiment_runner.run_with_runtime")


def verify_against_rubric(results: dict, rubric: dict) -> dict:
    """Score `results` against the PaperBench-style `rubric`.

    Wraps the existing rubric-verifier agent. Replaces fixed Gate 1/2/3
    checkpoints — the root calls this when it judges appropriate.
    """
    raise NotImplementedError("Phase 3 (#60) — wrap rubric-verifier agent")


def propose_improvements(
    current_results: dict,
    rubric_scores: dict,
    k: int | None = None,
) -> list[dict]:
    """Propose paper-specific improvements with proposer-assigned free-form tags.

    Variable-length list. NOT the hardcoded 5-category taxonomy from the
    old `improvement_orchestrator`. The proposer's system prompt is
    rewritten in Phase 3 (brief FM#4 validation: run on 3 papers, assert
    candidate lists differ in count or content).
    """
    raise NotImplementedError("Phase 3 (#60) — wrap rewritten improvement-orchestrator")


def set_final(report: dict) -> None:
    """Convenience: bind `report` to a REPL variable and emit FINAL_VAR(report).

    The root can also assign and emit the tag manually; this primitive is
    sugar for the common case.
    """
    raise NotImplementedError("Phase 2 (#59) — REPL binding helper")


PRIMITIVE_REGISTRY: dict[str, Callable[..., Any]] = {
    "understand_section": understand_section,
    "extract_hyperparameters": extract_hyperparameters,
    "detect_environment": detect_environment,
    "build_environment": build_environment,
    "plan_reproduction": plan_reproduction,
    "implement_baseline": implement_baseline,
    "run_experiment": run_experiment,
    "verify_against_rubric": verify_against_rubric,
    "propose_improvements": propose_improvements,
    "set_final": set_final,
}
