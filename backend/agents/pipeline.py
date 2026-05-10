"""ReproLab End-to-End Pipeline — ties all agents together.

Two execution modes:
  1. ``run_pipeline_sdk()``  — Full LLM-powered pipeline via the configured
     agent SDK provider. This is the PRIMARY mode. Works with ANY paper.

  2. ``run_pipeline_offline()`` — Deterministic demo pipeline (no LLM).
     Uses heuristic extractors and pre-built PPO implementation for testing.
     Only works for papers similar to PPO CartPole-v1.

Both modes produce the same output structure and use the same schemas.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

from backend.agents.schemas import (
    GateDecision,
    GateStatus,
    ResearchMap,
)
from backend.agents.orchestrator import PipelineStage, PipelineState
from backend.agents.execution import (
    ExecutionProfile,
    SandboxMode,
    ensure_sandbox_mode_available,
    resolve_sandbox_mode,
)
from backend.agents.resilience import RunBudget
from backend.agents.runtime import AgentRuntime, ProviderName

logger = logging.getLogger(__name__)


async def run_pipeline_sdk(
    project_id: str,
    runs_root: Path,
    workspace_claim_map: dict[str, Any],
    *,
    model: str | None = None,
    provider: ProviderName | str | None = None,
    verification_provider: ProviderName | str | None = None,
    runtime: AgentRuntime | None = None,
    verification_runtime: AgentRuntime | None = None,
    user_hints: list[str] | None = None,
    n_improvement_paths: int = 3,
    resume: bool = True,
    execution_profile: ExecutionProfile | None = None,
    run_budget: RunBudget | None = None,
    sandbox_mode: SandboxMode | str = SandboxMode.docker,
    seed: int | None = None,
    attempt_id: str | None = None,
    run_group_id: str | None = None,
    blacklist_terms: tuple[str, ...] = (),
    workspace_service: Any | None = None,
    workspace_id: str | None = None,
) -> PipelineState:
    """Run the full pipeline using the configured agent SDK provider.

    This is the primary execution mode. Every agent call goes through
    the provider runtime to analyze, generate, and verify dynamically.
    """
    from backend.agents.orchestrator import ReproLabOrchestrator

    resolved_sandbox_mode = resolve_sandbox_mode(sandbox_mode, pipeline_mode="sdk")
    orchestrator = ReproLabOrchestrator(
        project_id=project_id,
        runs_root=runs_root,
        model=model,
        provider=provider,
        verification_provider=verification_provider,
        runtime=runtime,
        verification_runtime=verification_runtime,
        execution_profile=execution_profile,
        run_budget=run_budget,
        sandbox_mode=resolved_sandbox_mode,
        seed=seed,
        attempt_id=attempt_id,
        run_group_id=run_group_id,
        blacklist_terms=blacklist_terms,
        workspace_service=workspace_service,
        workspace_id=workspace_id,
    )
    return await orchestrator.run(
        resume=resume,
        user_hints=user_hints,
        n_improvement_paths=n_improvement_paths,
    )


def run_pipeline_offline(
    project_id: str,
    runs_root: Path,
    workspace_claim_map: dict[str, Any],
    *,
    user_hints: list[str] | None = None,
    n_improvement_paths: int = 3,
    execution_profile: ExecutionProfile | None = None,
    sandbox_mode: SandboxMode | str = SandboxMode.simulate,
    seed: int | None = None,
    attempt_id: str | None = None,
    run_group_id: str | None = None,
    blacklist_terms: tuple[str, ...] = (),
    workspace_service: Any | None = None,
    workspace_id: str | None = None,
) -> PipelineState:
    """Run the full pipeline WITHOUT an LLM (deterministic demo mode).

    Uses heuristic extractors and pre-built implementations. Useful for:
    - Testing the pipeline flow
    - CI/CD validation
    - Offline demos

    NOTE: The offline extractors are generic (work with any paper sections),
    but the baseline implementation template is PPO-specific. For other papers,
    use run_pipeline_sdk() which generates code dynamically.
    """
    from backend.agents.paper_understanding import run_offline as paper_understanding
    from backend.agents.environment_detective import run_offline as env_detective
    from backend.agents.baseline_implementation import run_offline as baseline_impl
    from backend.agents.experiment_runner import run_offline as experiment_run
    from backend.agents.experiment_runner import run_with_local_process
    from backend.agents.experiment_runner import run_with_runpod
    from backend.agents.experiment_runner import run_with_runtime as experiment_run_docker
    from backend.agents.verification import run_gate_offline, run_improvement_gate_offline
    from backend.agents.improvement import select_hypotheses_offline, run_path_offline

    from backend.schemas.citations import Citation

    runs = Path(runs_root)
    profile = execution_profile or ExecutionProfile.from_mode("efficient")
    resolved_sandbox_mode = resolve_sandbox_mode(sandbox_mode, pipeline_mode="offline")
    ensure_sandbox_mode_available(resolved_sandbox_mode)
    state = PipelineState(
        project_id=project_id,
        seed=seed,
        attempt_id=attempt_id,
        run_group_id=run_group_id,
        blacklist_terms=list(blacklist_terms),
    )

    def _enrich(variable_name: str, value: dict, agent_id: str) -> None:
        if workspace_service is None or workspace_id is None:
            return
        try:
            cite = Citation(
                source_id=f"agent:{agent_id}",
                chunk_id=None,
                quote=f"Output from {agent_id} for project {project_id}",
                locator=f"{agent_id}@{project_id}",
                confidence=0.9,
            )
            workspace_service.enrich_variable(
                workspace_id=workspace_id,
                variable_name=variable_name,
                value_payload=value,
                citations=(cite,),
                enriched_by=agent_id,
            )
        except Exception:
            logger.warning("Failed to enrich workspace: %s", variable_name, exc_info=True)

    # --- Step 1: Paper Understanding ---
    print(f"[1/9] Paper Understanding Agent", file=sys.stderr)
    state.paper_claim_map = paper_understanding(
        project_id, runs, workspace_claim_map,
    )
    state.stage = PipelineStage.PAPER_UNDERSTOOD
    for amb in state.paper_claim_map.ambiguities:
        state.assumption_ledger.append(amb.model_dump())
    _enrich("paper_claim_map_agent", state.paper_claim_map.model_dump(), "paper-understanding")
    print(f"      {len(state.paper_claim_map.ambiguities)} ambiguities detected", file=sys.stderr)

    # --- Step 2: Artifact Discovery (simplified offline) ---
    print(f"[2/9] Artifact Discovery Agent", file=sys.stderr)
    state.artifact_index = {
        "artifacts": [],
        "recommended_repo": None,
        "dataset_links": [],
        "note": "Offline mode: no web search performed",
    }
    state.stage = PipelineStage.ARTIFACTS_DISCOVERED

    # --- Step 3: Environment Detective ---
    print(f"[3/9] Environment Detective Agent", file=sys.stderr)
    state.environment_spec = env_detective(
        project_id, runs, state.paper_claim_map, state.artifact_index,
    )
    for assumption in state.environment_spec.assumptions:
        state.assumption_ledger.append(assumption.model_dump())
    state.stage = PipelineStage.ENVIRONMENT_BUILT
    _enrich("environment_spec", state.environment_spec.model_dump(), "environment-detective")
    print(f"      Dockerfile generated: Python {state.environment_spec.python_version}, "
          f"{state.environment_spec.framework}=={state.environment_spec.framework_version}", file=sys.stderr)

    # --- Step 4: Reproduction Planner (simplified offline) ---
    print(f"[4/9] Reproduction Planner", file=sys.stderr)
    from backend.agents.schemas import ReproductionContract
    state.reproduction_contract = ReproductionContract(
        reproduction_definition="Same algorithm, same dataset, same specifications where discoverable.",
        smoke_test_plan="Run for 1000 timesteps, verify reward > 0.",
        full_run_plan="Run for 500k timesteps on CartPole-v1, evaluate over 100 episodes.",
        expected_outputs=["metrics.json", "plots/reward_curve.png", "logs/run.log", "commands.log", "provenance.json"],
        evaluation_plan="Mean reward over 100 evaluation episodes.",
    )
    state.stage = PipelineStage.PLAN_CREATED

    # --- Gate 1: Plan Verification ---
    print(f"[Gate 1] Plan Verification", file=sys.stderr)
    # In offline mode, auto-pass gate 1 (plan is deterministic)
    state.gate_1 = GateDecision(gate="gate_1", passed=True, status=GateStatus.verified)
    state.decision_log.append("gate_1: verified (offline mode)")
    state.stage = PipelineStage.GATE_1_PASSED
    state.save_checkpoint(runs)
    print(f"      PASSED", file=sys.stderr)

    # --- Step 5: Baseline Implementation ---
    print(f"[5/9] Baseline Implementation Agent", file=sys.stderr)
    state.baseline_result = baseline_impl(
        project_id, runs, state.paper_claim_map, state.environment_spec,
        state.reproduction_contract, state.artifact_index,
    )
    state.stage = PipelineStage.BASELINE_IMPLEMENTED
    _enrich("baseline_result", state.baseline_result.model_dump(), "baseline-implementation")
    print(f"      Mode: {state.baseline_result.mode}, "
          f"assumptions applied: {state.baseline_result.assumptions_applied}", file=sys.stderr)

    # --- Step 6: Experiment Runner ---
    print(f"[6/9] Experiment Runner Agent", file=sys.stderr)
    if resolved_sandbox_mode is SandboxMode.docker:
        import anyio

        async def _run_docker_experiment():
            return await experiment_run_docker(
                project_id,
                runs,
                state.baseline_result,
                state.reproduction_contract,
                command_timeout=profile.command_timeout_seconds,
                network_disabled=profile.sandbox_network_disabled,
                memory_limit=profile.sandbox_memory_limit,
                cpus=profile.sandbox_cpus,
                platform=profile.sandbox_platform,
                gpu_mode=profile.gpu_mode.value,
                extra_environment=profile.sandbox_environment,
            )

        state.experiment_artifacts = anyio.run(_run_docker_experiment)
    elif resolved_sandbox_mode is SandboxMode.local:
        import anyio

        async def _run_local_experiment():
            return await run_with_local_process(
                project_id,
                runs,
                state.baseline_result,
                state.reproduction_contract,
                command_timeout=profile.command_timeout_seconds,
                gpu_mode=profile.gpu_mode.value,
                extra_environment=profile.sandbox_environment,
            )

        state.experiment_artifacts = anyio.run(_run_local_experiment)
    elif resolved_sandbox_mode is SandboxMode.runpod:
        import anyio

        async def _run_runpod_experiment():
            return await run_with_runpod(
                project_id,
                runs,
                state.baseline_result,
                state.reproduction_contract,
                command_timeout=profile.command_timeout_seconds,
            )

        state.experiment_artifacts = anyio.run(_run_runpod_experiment)
    else:
        state.experiment_artifacts = experiment_run(
            project_id, runs, state.baseline_result, state.reproduction_contract,
        )
    state.stage = PipelineStage.BASELINE_RUN
    _enrich("experiment_artifacts", state.experiment_artifacts.model_dump(), "experiment-runner")
    print(f"      Success: {state.experiment_artifacts.success}, "
          f"mean_reward: {state.experiment_artifacts.metrics.get('mean_reward', 'N/A')}", file=sys.stderr)

    # --- Gate 2: Baseline Verification ---
    print(f"[Gate 2] Baseline Verification", file=sys.stderr)
    code_dir = runs / project_id / "code"
    gate2_report = run_gate_offline(
        "gate_2",
        state.paper_claim_map,
        state.baseline_result,
        state.experiment_artifacts,
        code_dir=code_dir,
    )
    state.gate_2 = GateDecision(
        gate="gate_2",
        passed=gate2_report.status in (GateStatus.verified, GateStatus.verified_with_caveats),
        status=gate2_report.status,
    )
    state.decision_log.append(gate2_report.decision_log_entry)
    state.stage = PipelineStage.GATE_2_PASSED
    state.save_checkpoint(runs)
    print(f"      {gate2_report.status.value} (avg score: "
          f"{sum(s.score for s in gate2_report.verifier_scores)/len(gate2_report.verifier_scores):.2f})", file=sys.stderr)

    if not state.gate_2.passed:
        print(f"      GATE 2 FAILED — stopping pipeline", file=sys.stderr)
        return state

    # --- Step 7: Improvement Orchestrator ---
    print(f"[7/9] Improvement Orchestrator", file=sys.stderr)
    state.improvement_hypotheses = select_hypotheses_offline(
        state.paper_claim_map,
        state.experiment_artifacts.metrics,
        user_hints=user_hints,
        n_paths=n_improvement_paths,
    )
    state.stage = PipelineStage.IMPROVEMENTS_SELECTED
    for h in state.improvement_hypotheses:
        print(f"      → {h.path_id}: {h.hypothesis[:60]}...", file=sys.stderr)

    # --- Step 8: Path Agents ---
    print(f"[8/9] Running {len(state.improvement_hypotheses)} Improvement Path Agents", file=sys.stderr)
    for hypothesis in state.improvement_hypotheses:
        result = run_path_offline(
            project_id, runs, hypothesis,
            state.experiment_artifacts.metrics,
        )
        state.path_results.append(result)
        status_str = "✓" if result.success else "✗"
        reward = result.metrics.get("mean_reward", "N/A")
        print(f"      {status_str} {hypothesis.path_id}: reward={reward}", file=sys.stderr)
    state.stage = PipelineStage.IMPROVEMENTS_RUN

    # --- Gate 3: Improvement Verification ---
    print(f"[Gate 3] Improvement Verification", file=sys.stderr)
    gate3_report = run_improvement_gate_offline(
        state.path_results,
        state.paper_claim_map,
        state.experiment_artifacts.metrics,
    )
    state.gate_3 = GateDecision(
        gate="gate_3",
        passed=gate3_report.status in (GateStatus.verified, GateStatus.verified_with_caveats),
        status=gate3_report.status,
    )
    state.decision_log.append(gate3_report.decision_log_entry)
    state.stage = PipelineStage.GATE_3_PASSED
    state.save_checkpoint(runs)
    print(f"      {gate3_report.status.value}", file=sys.stderr)

    # --- Step 9: Research Map ---
    print(f"[9/9] Generating Research Map", file=sys.stderr)
    successful = [p for p in state.path_results if p.success and p.metrics.get("improvement", 0) > 0]
    regressions = [p for p in state.path_results if p.success and p.metrics.get("improvement", 0) < 0]
    failed = [p for p in state.path_results if not p.success]

    state.research_map = ResearchMap(
        baseline_summary=(
            f"PPO CartPole-v1 baseline: mean_reward="
            f"{state.experiment_artifacts.metrics.get('mean_reward', 'N/A')} "
            f"({state.gate_2.status.value})"
        ),
        promising_directions=[
            f"{p.path_id}: {p.hypothesis} (reward={p.metrics.get('mean_reward', '?')})"
            for p in successful
        ],
        dead_ends=[
            f"{p.path_id}: {p.hypothesis} (reward={p.metrics.get('mean_reward', '?')})"
            for p in regressions
        ],
        inconclusive=[
            f"{p.path_id}: {p.hypothesis} ({p.failure_notes})"
            for p in failed
        ],
        next_experiments=[
            "Combine best improvement with baseline",
            "Run full 500k timesteps (not reduced)",
            "Test on additional environments (Hopper, Walker2d)",
        ],
        overall_reproducibility_assessment=(
            f"Baseline: {state.gate_2.status.value}. "
            f"Improvements: {len(successful)} promising, "
            f"{len(regressions)} dead ends, {len(failed)} failed."
        ),
    )
    state.stage = PipelineStage.RESEARCH_MAP_GENERATED

    # Write final outputs
    out_dir = runs / project_id
    (out_dir / "research_map.json").write_text(state.research_map.model_dump_json(indent=2))
    (out_dir / "assumption_ledger.json").write_text(json.dumps(state.assumption_ledger, indent=2))
    (out_dir / "decision_log.json").write_text(json.dumps(state.decision_log, indent=2))

    _enrich("research_map", state.research_map.model_dump(), "research-map-generator")
    _enrich("assumption_ledger", {"entries": state.assumption_ledger}, "orchestrator")
    _enrich("decision_log", {"entries": state.decision_log}, "orchestrator")

    state.stage = PipelineStage.COMPLETE
    state.save_checkpoint(runs)

    if workspace_service is not None and workspace_id is not None:
        try:
            workspace_service.close_workspace(
                workspace_id=workspace_id, reason="pipeline_complete"
            )
        except Exception:
            logger.warning("Failed to close workspace", exc_info=True)

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Pipeline complete for {project_id}", file=sys.stderr)
    print(f"  Baseline: {state.experiment_artifacts.metrics.get('mean_reward', 'N/A')} reward", file=sys.stderr)
    print(f"  Improvements: {len(successful)} promising, {len(regressions)} dead ends", file=sys.stderr)
    print(f"  Assumptions: {len(state.assumption_ledger)}", file=sys.stderr)
    print(f"  Output: {out_dir}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    return state
