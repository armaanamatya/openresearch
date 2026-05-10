"""Seeded PaperBench pipeline scheduling helpers."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.agents.execution import DEFAULT_SANDBOX_MODE, ExecutionProfile, SandboxMode
from backend.agents.orchestrator import PipelineState
from backend.agents.pipeline import run_pipeline_sdk
from backend.agents.runtime import AgentRuntime, ProviderName
from backend.evals.paperbench.score import mean_standard_error


@dataclass(frozen=True)
class PaperBenchAttemptConfig:
    project_id: str
    runs_root: Path
    workspace_claim_map: dict[str, Any]
    seed: int
    attempt_id: str
    run_group_id: str
    model: str | None = None
    provider: ProviderName | str | None = None
    verification_provider: ProviderName | str | None = None
    runtime: AgentRuntime | None = None
    verification_runtime: AgentRuntime | None = None
    user_hints: list[str] | None = None
    n_improvement_paths: int = 3
    execution_profile: ExecutionProfile | None = None
    sandbox_mode: SandboxMode | str = DEFAULT_SANDBOX_MODE
    blacklist_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class PaperBenchAttemptResult:
    attempt_id: str
    seed: int
    project_id: str
    state: PipelineState
    elapsed_seconds: float


async def run_seeded_pipeline_attempts(
    attempts: list[PaperBenchAttemptConfig],
    *,
    max_parallel: int = 1,
    resume: bool = True,
) -> list[PaperBenchAttemptResult]:
    """Run one pipeline attempt per seed with bounded parallelism."""

    if max_parallel < 1:
        raise ValueError("max_parallel must be >= 1")
    semaphore = asyncio.Semaphore(max_parallel)

    async def _run_one(config: PaperBenchAttemptConfig) -> PaperBenchAttemptResult:
        async with semaphore:
            t0 = time.time()
            state = await run_pipeline_sdk(
                config.project_id,
                config.runs_root,
                config.workspace_claim_map,
                model=config.model,
                provider=config.provider,
                verification_provider=config.verification_provider,
                runtime=config.runtime,
                verification_runtime=config.verification_runtime,
                user_hints=config.user_hints,
                n_improvement_paths=config.n_improvement_paths,
                resume=resume,
                execution_profile=config.execution_profile,
                sandbox_mode=config.sandbox_mode,
                seed=config.seed,
                attempt_id=config.attempt_id,
                run_group_id=config.run_group_id,
                blacklist_terms=config.blacklist_terms,
            )
            return PaperBenchAttemptResult(
                attempt_id=config.attempt_id,
                seed=config.seed,
                project_id=config.project_id,
                state=state,
                elapsed_seconds=time.time() - t0,
            )

    return await asyncio.gather(*(_run_one(config) for config in attempts))


def summarize_attempt_scores(scores: list[float]) -> dict[str, float | int]:
    mean, standard_error, n = mean_standard_error(scores)
    return {"mean": mean, "standard_error": standard_error, "n": n}
