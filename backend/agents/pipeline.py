"""ReproLab Pipeline — dispatch shim for the RLM execution path."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.agents.execution import DEFAULT_SANDBOX_MODE


# ---------------------------------------------------------------------------
# RLM mode (Phase 3, issue #60)
# ---------------------------------------------------------------------------


async def run_pipeline_rlm(
    project_id: str,
    runs_root: Path,
    workspace_claim_map: dict[str, Any],
    *,
    model: str | None = None,
    provider: str | None = None,
    runtime: Any | None = None,
    run_budget: Any | None = None,
    sandbox_mode: Any = DEFAULT_SANDBOX_MODE,
    seed: int | None = None,
    execution_profile: Any | None = None,
    attempt_id: str | None = None,
    run_group_id: str | None = None,
    workspace_service: Any | None = None,
    workspace_id: str | None = None,
) -> Any:
    """Thin dispatch wrapper — delegates to ``backend.agents.rlm.run.run_pipeline_rlm``.

    Keeps ``pipeline.py`` as the uniform dispatch home for the RLM run mode.
    Returns an :class:`~backend.agents.rlm.run.RLMRunResult`.
    """
    from backend.agents.rlm.run import run_pipeline_rlm as _run_rlm

    return await _run_rlm(
        project_id,
        runs_root,
        workspace_claim_map,
        model=model,
        provider=provider,
        runtime=runtime,
        run_budget=run_budget,
        sandbox_mode=sandbox_mode,
        seed=seed,
        execution_profile=execution_profile,
        attempt_id=attempt_id,
        run_group_id=run_group_id,
        workspace_service=workspace_service,
        workspace_id=workspace_id,
    )
