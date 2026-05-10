"""ReproLab Root Orchestrator — drives the full reproduction pipeline.

The orchestrator uses a hybrid approach:
  - Python code drives the pipeline sequence and manages state
  - Each agent step invokes the configured provider runtime
  - Structured outputs are parsed and fed into the next agent's prompt
  - Checkpoints are saved to the event store after each gate

Usage:
    orchestrator = ReproLabOrchestrator(project_id, runs_root)
    result = await orchestrator.run(source_pdf="demo_paper.pdf")
"""

from __future__ import annotations

import enum
import json
import logging
import sys
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.agents.registry import AGENT_REGISTRY
from backend.agents.execution import (
    ExecutionProfile,
    SandboxMode,
    ensure_sandbox_mode_available,
    resolve_sandbox_mode,
)
from backend.agents.runtime import (
    AgentRuntime,
    AgentRuntimeSpec,
    ProviderName,
    RuntimeGuard,
    make_runtime,
)
from backend.agents.resilience import ProviderHealthMonitor, RunBudget, RunCostLedger
from backend.agents.resilience.engine import (
    RuntimeKwargs,
    default_recovery_policy,
    run_agent_with_resilience,
)
from backend.agents.schemas import (
    AgentOutput,
    BaselineResult,
    EnvironmentSpec,
    ExperimentArtifacts,
    GateDecision,
    GateStatus,
    ImprovementHypothesis,
    PaperClaimMap,
    PathResult,
    ReproductionContract,
    ResearchMap,
    VerificationReport,
)
from backend.agents.structured_output import append_structured_output_instruction
from backend.agents.telemetry import (
    AgentTelemetryRecorder,
)
from backend.hermes_audit import (
    HermesAuditReport,
    HermesAuditScope,
    HermesAuditService,
    HermesAuditStatus,
    HermesAuditStorage,
    HermesInterventionType,
    NousHermesClient,
    build_checkpoint_audit_payload,
    build_step_audit_payload,
)
from backend.schemas.citations import Citation

logger = logging.getLogger(__name__)

@dataclass
class AgentExecutionTrace:
    """Trace metadata captured for one agent invocation."""

    agent_id: str
    output_text: str
    trace_text: str
    tool_calls: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0


class PipelineStage(str, enum.Enum):
    """Stages in the reproduction pipeline."""

    INGESTED = "ingested"
    PAPER_UNDERSTOOD = "paper_understood"
    ARTIFACTS_DISCOVERED = "artifacts_discovered"
    ENVIRONMENT_BUILT = "environment_built"
    PLAN_CREATED = "plan_created"
    GATE_1_PASSED = "gate_1_passed"
    BASELINE_IMPLEMENTED = "baseline_implemented"
    BASELINE_RUN = "baseline_run"
    GATE_2_PASSED = "gate_2_passed"
    IMPROVEMENTS_SELECTED = "improvements_selected"
    IMPROVEMENTS_RUN = "improvements_run"
    GATE_3_PASSED = "gate_3_passed"
    RESEARCH_MAP_GENERATED = "research_map_generated"
    COMPLETE = "complete"


@dataclass
class PipelineState:
    """Mutable state that accumulates as the pipeline progresses."""

    project_id: str
    stage: PipelineStage = PipelineStage.INGESTED
    paper_claim_map: PaperClaimMap | None = None
    artifact_index: dict[str, Any] | None = None
    environment_spec: EnvironmentSpec | None = None
    reproduction_contract: ReproductionContract | None = None
    gate_1: GateDecision | None = None
    baseline_result: BaselineResult | None = None
    experiment_artifacts: ExperimentArtifacts | None = None
    gate_2: GateDecision | None = None
    improvement_hypotheses: list[ImprovementHypothesis] = field(default_factory=list)
    path_results: list[PathResult] = field(default_factory=list)
    gate_3: GateDecision | None = None
    research_map: ResearchMap | None = None
    assumption_ledger: list[dict[str, Any]] = field(default_factory=list)
    decision_log: list[str] = field(default_factory=list)
    hermes_step_reports: dict[str, list[HermesAuditReport]] = field(default_factory=dict)
    hermes_checkpoint_reports: dict[str, list[HermesAuditReport]] = field(default_factory=dict)
    hermes_interventions: list[dict[str, Any]] = field(default_factory=list)
    seed: int | None = None
    attempt_id: str | None = None
    run_group_id: str | None = None
    blacklist_terms: list[str] = field(default_factory=list)

    def save_checkpoint(self, runs_root: Path) -> Path:
        """Persist pipeline state to disk for crash-resume."""
        checkpoint_dir = runs_root / self.project_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / "pipeline_state.json"
        data = {
            "project_id": self.project_id,
            "stage": self.stage.value,
            "assumption_ledger": self.assumption_ledger,
            "decision_log": self.decision_log,
            "seed": self.seed,
            "attempt_id": self.attempt_id,
            "run_group_id": self.run_group_id,
            "blacklist_terms": self.blacklist_terms,
        }
        if self.paper_claim_map:
            data["paper_claim_map"] = self.paper_claim_map.model_dump()
        if self.environment_spec:
            data["environment_spec"] = self.environment_spec.model_dump()
        if self.reproduction_contract:
            data["reproduction_contract"] = self.reproduction_contract.model_dump()
        if self.baseline_result:
            data["baseline_result"] = self.baseline_result.model_dump()
        if self.experiment_artifacts:
            data["experiment_artifacts"] = self.experiment_artifacts.model_dump()
        if self.research_map:
            data["research_map"] = self.research_map.model_dump()
        if self.gate_1:
            data["gate_1"] = self.gate_1.model_dump()
        if self.gate_2:
            data["gate_2"] = self.gate_2.model_dump()
        if self.gate_3:
            data["gate_3"] = self.gate_3.model_dump()
        if self.improvement_hypotheses:
            data["improvement_hypotheses"] = [h.model_dump() for h in self.improvement_hypotheses]
        if self.path_results:
            data["path_results"] = [r.model_dump() for r in self.path_results]
        if self.hermes_step_reports:
            data["hermes_step_reports"] = {
                key: [report.model_dump() for report in reports]
                for key, reports in self.hermes_step_reports.items()
            }
        if self.hermes_checkpoint_reports:
            data["hermes_checkpoint_reports"] = {
                key: [report.model_dump() for report in reports]
                for key, reports in self.hermes_checkpoint_reports.items()
            }
        if self.hermes_interventions:
            data["hermes_interventions"] = self.hermes_interventions
        path.write_text(json.dumps(data, indent=2))
        logger.info("Checkpoint saved: stage=%s path=%s", self.stage.value, path)
        return path

    @classmethod
    def load_checkpoint(cls, runs_root: Path, project_id: str) -> PipelineState | None:
        """Load pipeline state from disk if a checkpoint exists."""
        path = runs_root / project_id / "pipeline_state.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        state = cls(project_id=data["project_id"])
        state.stage = PipelineStage(data["stage"])
        state.assumption_ledger = data.get("assumption_ledger", [])
        state.decision_log = data.get("decision_log", [])
        state.seed = data.get("seed")
        state.attempt_id = data.get("attempt_id")
        state.run_group_id = data.get("run_group_id")
        state.blacklist_terms = data.get("blacklist_terms", [])
        if "paper_claim_map" in data:
            state.paper_claim_map = PaperClaimMap(**data["paper_claim_map"])
        if "environment_spec" in data:
            state.environment_spec = EnvironmentSpec(**data["environment_spec"])
        if "reproduction_contract" in data:
            state.reproduction_contract = ReproductionContract(**data["reproduction_contract"])
        if "baseline_result" in data:
            state.baseline_result = BaselineResult(**data["baseline_result"])
        if "experiment_artifacts" in data:
            state.experiment_artifacts = ExperimentArtifacts(**data["experiment_artifacts"])
        if "research_map" in data:
            state.research_map = ResearchMap(**data["research_map"])
        if "gate_1" in data:
            state.gate_1 = GateDecision(**data["gate_1"])
        if "gate_2" in data:
            state.gate_2 = GateDecision(**data["gate_2"])
        if "gate_3" in data:
            state.gate_3 = GateDecision(**data["gate_3"])
        if "improvement_hypotheses" in data:
            state.improvement_hypotheses = [
                ImprovementHypothesis(**h) for h in data["improvement_hypotheses"]
            ]
        if "path_results" in data:
            state.path_results = [PathResult(**r) for r in data["path_results"]]
        if "hermes_step_reports" in data:
            state.hermes_step_reports = {
                key: [HermesAuditReport(**report) for report in reports]
                for key, reports in data["hermes_step_reports"].items()
            }
        if "hermes_checkpoint_reports" in data:
            state.hermes_checkpoint_reports = {
                key: [HermesAuditReport(**report) for report in reports]
                for key, reports in data["hermes_checkpoint_reports"].items()
            }
        state.hermes_interventions = data.get("hermes_interventions", [])
        logger.info("Checkpoint loaded: stage=%s", state.stage.value)
        return state


class ReproLabOrchestrator:
    """Drives the full ReproLab pipeline using the configured agent runtime.

    Each pipeline step:
      1. Builds a prompt with context from previous steps
      2. Invokes the provider runtime targeting the appropriate agent
      3. Parses the structured output
      4. Updates pipeline state
      5. Saves a checkpoint after verification gates
    """

    def __init__(
        self,
        project_id: str,
        runs_root: Path,
        *,
        model: str | None = None,
        max_turns_per_agent: int | None = None,
        permission_mode: str = "bypassPermissions",
        provider: ProviderName | str | None = None,
        verification_provider: ProviderName | str | None = None,
        runtime: AgentRuntime | None = None,
        verification_runtime: AgentRuntime | None = None,
        claude_limit_fallback_runtime: AgentRuntime | None = None,
        execution_profile: ExecutionProfile | None = None,
        run_budget: RunBudget | None = None,
        sandbox_mode: SandboxMode | str = SandboxMode.docker,
        hermes_audit_service: HermesAuditService | None = None,
        seed: int | None = None,
        attempt_id: str | None = None,
        run_group_id: str | None = None,
        blacklist_terms: tuple[str, ...] = (),
        workspace_service: Any | None = None,
        workspace_id: str | None = None,
    ) -> None:
        self.project_id = project_id
        self.runs_root = Path(runs_root)
        self.model = model
        self.execution_profile = execution_profile or ExecutionProfile.from_mode(
            "efficient"
        )
        self.max_turns_per_agent = (
            max_turns_per_agent
            if max_turns_per_agent is not None
            else self.execution_profile.max_turns_per_agent
        )
        self.heavy_agent_max_turns = self.execution_profile.heavy_agent_max_turns
        self.permission_mode = permission_mode
        self.sandbox_mode = resolve_sandbox_mode(sandbox_mode, pipeline_mode="sdk")
        self.seed = seed
        self.attempt_id = attempt_id
        self.run_group_id = run_group_id
        self.blacklist_terms = tuple(term.strip() for term in blacklist_terms if term.strip())
        if self.sandbox_mode is SandboxMode.simulate:
            raise ValueError(
                "SDK pipeline does not support simulated experiment execution. "
                "Use the offline pipeline for deterministic simulation or select docker/local/runpod."
            )
        self._runtime = runtime or make_runtime(provider)
        self._verification_runtime = (
            verification_runtime
            or (make_runtime(verification_provider) if verification_provider else self._runtime)
        )
        self._fallback_runtimes: dict[ProviderName, AgentRuntime] = {}
        if claude_limit_fallback_runtime is not None:
            self._fallback_runtimes[
                claude_limit_fallback_runtime.provider_name
            ] = claude_limit_fallback_runtime
        self._project_dir = self.runs_root / project_id
        self._project_dir.mkdir(parents=True, exist_ok=True)
        self._telemetry = AgentTelemetryRecorder(
            self._project_dir / "agent_telemetry.jsonl"
        )
        ledger_path = self._project_dir / "cost_ledger.jsonl"
        self._cost_ledger = RunCostLedger.load_jsonl(
            ledger_path,
            project_id=project_id,
            attach_path=True,
        )
        self._run_budget = run_budget or RunBudget()
        self._provider_health = ProviderHealthMonitor()
        self._pipeline_started_at = datetime.now(timezone.utc)
        self._fallback_summary_path = self._project_dir / "fallback_summary.json"
        self._latest_agent_traces: dict[str, AgentExecutionTrace] = {}
        self._hermes_audit_service = hermes_audit_service or HermesAuditService(
            client=NousHermesClient(runs_root=self.runs_root),
            storage=HermesAuditStorage(self.runs_root, project_id),
        )
        self._workspace_service = workspace_service
        self._workspace_id = workspace_id

    # Agents that write code / run experiments need more turns
    _HEAVY_AGENTS = {"baseline-implementation", "improvement-path", "experiment-runner"}
    _OUTPUT_MODELS = {
        "paper-understanding": PaperClaimMap,
        "environment-detective": EnvironmentSpec,
        "reproduction-planner": ReproductionContract,
        "baseline-implementation": BaselineResult,
        "experiment-runner": ExperimentArtifacts,
        "supervisor-verifier": VerificationReport,
        "improvement-path": PathResult,
    }

    def _build_runtime_spec(
        self,
        agent_id: str,
        *,
        runtime: AgentRuntime,
        cwd: str | Path | None = None,
        max_turns: int | None,
    ) -> AgentRuntimeSpec:
        spec = AGENT_REGISTRY[agent_id]
        provider = runtime.provider_name
        guard = RuntimeGuard(
            blocked_terms=self.blacklist_terms,
            max_tool_calls=self.execution_profile.max_tool_calls_per_agent,
        )
        sub_agents = tuple(
            replace(sub_spec.to_runtime_spec(provider), guard=guard)
            for sub_id, sub_spec in AGENT_REGISTRY.items()
            if sub_id != agent_id
        )
        runtime_spec = spec.to_runtime_spec(
            provider,
            model_override=self.model,
            max_turns=max_turns,
            working_directory=Path(cwd or self._project_dir),
            sub_agents=sub_agents,
        )
        return replace(
            runtime_spec,
            permission_mode=self.permission_mode,
            guard=guard,
        )

    async def _invoke_agent(
        self,
        agent_id: str,
        task_prompt: str,
        *,
        cwd: str | Path | None = None,
        max_turns: int | None = None,
        _runtime_override: AgentRuntime | None = None,
        _allow_claude_limit_fallback: bool = True,
        _structured_prompt: bool = False,
    ) -> str:
        """Invoke a single agent via the SDK and return its final text output."""
        primary_runtime = _runtime_override or self._runtime_for_agent(agent_id)
        if max_turns is None:
            max_turns = (
                self.heavy_agent_max_turns
                if agent_id in self._HEAVY_AGENTS
                else self.max_turns_per_agent
            )

        task_prompt = self._append_run_controls(task_prompt)
        if not _structured_prompt:
            task_prompt = append_structured_output_instruction(
                task_prompt,
                self._OUTPUT_MODELS.get(agent_id),
            )

        cwd_path = Path(cwd or self._project_dir)
        chain = self._provider_chain(primary_runtime.provider_name)
        policy = default_recovery_policy(chain=chain, health=self._provider_health)

        def runtime_for(provider: ProviderName) -> AgentRuntime:
            return self._runtime_for_provider(provider, primary_runtime=primary_runtime)

        def build_runtime_spec(
            runtime: AgentRuntime,
            attempt_max_turns: int | None,
        ) -> AgentRuntimeSpec:
            return self._build_runtime_spec(
                agent_id,
                runtime=runtime,
                cwd=cwd_path,
                max_turns=attempt_max_turns,
            )

        result_obj = await run_agent_with_resilience(
            agent_id=agent_id,
            base_prompt=task_prompt,
            primary_provider=primary_runtime.provider_name,
            runtime_for=runtime_for,
            chain=chain,
            policy=policy,
            health=self._provider_health,
            ledger=self._cost_ledger,
            budget=self._run_budget,
            runtime_kwargs=RuntimeKwargs(
                cwd=cwd_path,
                max_turns=max_turns,
                wall_clock_seconds=self.execution_profile.agent_wall_clock_seconds,
                build_runtime_spec=build_runtime_spec,
                telemetry=self._telemetry,
                run_started_at=self._pipeline_started_at,
                salvage_validator=lambda text: self._partial_output_validates(
                    agent_id,
                    text,
                ),
                summary_path=self._fallback_summary_path,
            ),
        )
        result = result_obj.output_text
        if not result.strip():
            print(
                f"  [{agent_id}] WARNING: empty output",
                file=sys.stderr,
                flush=True,
            )
        logger.info("Agent %s completed (%d chars output)", agent_id, len(result))
        self._latest_agent_traces[agent_id] = AgentExecutionTrace(
            agent_id=agent_id,
            output_text=result,
            trace_text=result_obj.trace_text,
            tool_calls=result_obj.tool_calls,
            elapsed_seconds=result_obj.elapsed_seconds,
        )
        return result

    def _append_run_controls(self, prompt: str) -> str:
        controls: list[str] = []
        if self.seed is not None:
            controls.append(
                f"Use random seed {self.seed} for scripts, configs, data splits, and experiments."
            )
        if self.attempt_id or self.run_group_id:
            controls.append(
                "Run metadata: "
                f"attempt_id={self.attempt_id or 'unset'}, "
                f"run_group_id={self.run_group_id or 'unset'}."
            )
        if self.blacklist_terms:
            controls.append(
                "Do not access, fetch, clone, download, or copy from these blocked resources: "
                + ", ".join(self.blacklist_terms)
            )
        if not controls:
            return prompt
        return prompt + "\n\nRuntime controls:\n- " + "\n- ".join(controls)

    def _runtime_for_agent(self, agent_id: str) -> AgentRuntime:
        if agent_id == "supervisor-verifier":
            return self._verification_runtime
        return self._runtime

    def _provider_chain(self, primary: ProviderName) -> list[ProviderName]:
        other: ProviderName = "openai" if primary == "anthropic" else "anthropic"
        chain: list[ProviderName] = []
        for provider in (primary, other):
            if provider not in chain:
                chain.append(provider)
        return chain

    def _runtime_for_provider(
        self,
        provider: ProviderName,
        *,
        primary_runtime: AgentRuntime,
    ) -> AgentRuntime:
        if primary_runtime.provider_name == provider:
            return primary_runtime
        if self._runtime.provider_name == provider:
            return self._runtime
        if self._verification_runtime.provider_name == provider:
            return self._verification_runtime
        cached = self._fallback_runtimes.get(provider)
        if cached is not None:
            return cached
        runtime = make_runtime(provider)
        self._fallback_runtimes[provider] = runtime
        return runtime

    def _partial_output_validates(self, agent_id: str, text: str) -> bool:
        model = self._OUTPUT_MODELS.get(agent_id)
        if model is None:
            return bool(text.strip())
        try:
            data = self._extract_json(text)
            if agent_id == "supervisor-verifier":
                data = self._normalize_verifier_scores(data)
            elif agent_id == "reproduction-planner":
                data = self._normalize_reproduction_contract(data)
            model(**data)
            return True
        except Exception:
            return False

    def _enrich_workspace(
        self,
        variable_name: str,
        value_payload: dict[str, Any],
        agent_id: str,
    ) -> None:
        """Write an agent's structured output back to the workspace as a variable.

        No-op if workspace integration is not configured.
        """
        if self._workspace_service is None or self._workspace_id is None:
            return
        try:
            citation = Citation(
                source_id=f"agent:{agent_id}",
                chunk_id=None,
                quote=f"Output from {agent_id} agent for project {self.project_id}",
                locator=f"{agent_id}@{self.project_id}",
                confidence=0.9,
            )
            self._workspace_service.enrich_variable(
                workspace_id=self._workspace_id,
                variable_name=variable_name,
                value_payload=value_payload,
                citations=(citation,),
                enriched_by=agent_id,
            )
            logger.info(
                "Workspace enriched: %s from %s", variable_name, agent_id
            )
        except Exception:
            logger.warning(
                "Failed to enrich workspace variable %s from %s",
                variable_name,
                agent_id,
                exc_info=True,
            )

    def _close_workspace(self, reason: str = "pipeline_complete") -> None:
        """Close the workspace when pipeline finishes. No-op if not configured."""
        if self._workspace_service is None or self._workspace_id is None:
            return
        try:
            self._workspace_service.close_workspace(
                workspace_id=self._workspace_id, reason=reason
            )
            logger.info("Workspace %s closed: %s", self._workspace_id, reason)
        except Exception:
            logger.warning(
                "Failed to close workspace %s",
                self._workspace_id,
                exc_info=True,
            )

    def _normalize_verifier_scores(self, data: dict[str, Any]) -> dict[str, Any]:
        """Normalize LLM-generated verification data to match schema expectations."""
        if "verifier_scores" in data:
            for vs in data["verifier_scores"]:
                # LLM sometimes uses "verifier" instead of "verifier_name"
                if "verifier" in vs and "verifier_name" not in vs:
                    vs["verifier_name"] = vs.pop("verifier")
                # LLM sometimes returns 0-100 scores instead of 0.0-1.0
                if "score" in vs and isinstance(vs["score"], (int, float)) and vs["score"] > 1.0:
                    vs["score"] = vs["score"] / 100.0
                if "severity" not in vs or not vs["severity"]:
                    vs["severity"] = "medium" if vs.get("mismatches") else "low"
        return data

    def _normalize_reproduction_contract(self, data: dict[str, Any]) -> dict[str, Any]:
        """Normalize planner output so it can be parsed into ReproductionContract."""
        if (
            "expected_outputs" in data
            and isinstance(data["expected_outputs"], list)
            and data["expected_outputs"]
            and isinstance(data["expected_outputs"][0], dict)
        ):
            normalized_outputs: list[str] = []
            for item in data["expected_outputs"]:
                if isinstance(item, dict):
                    normalized_outputs.append(
                        item.get("path")
                        or item.get("name")
                        or item.get("label")
                        or json.dumps(item, sort_keys=True)
                    )
                else:
                    normalized_outputs.append(str(item))
            data["expected_outputs"] = normalized_outputs
        return data

    def _extract_json(self, text: str, fallback_file: str | None = None) -> dict[str, Any]:
        """Extract JSON from agent output, handling markdown fences.

        If the agent wrote JSON to a file instead of returning it inline,
        falls back to reading the file from disk.
        """
        import re

        # Try to find JSON in code fences first
        fence_match = re.search(r"```(?:json)?\s*\n(.*?)\n\s*```", text, re.DOTALL)
        if fence_match:
            try:
                return json.loads(fence_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find a top-level JSON object in text
        brace_start = text.find("{")
        if brace_start >= 0:
            depth = 0
            for i in range(brace_start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[brace_start : i + 1])
                        except json.JSONDecodeError:
                            break

        # Fallback: check if agent wrote to the expected file on disk
        if fallback_file:
            fpath = Path(fallback_file)
            if fpath.exists():
                logger.info("Reading agent output from file: %s", fpath)
                return json.loads(fpath.read_text())
            # Also check if agent used relative path from its cwd
            # (creates nested runs/project_id/runs/project_id/file)
            nested = self._project_dir / fpath.name
            if nested.exists():
                logger.info("Reading agent output from nested file: %s", nested)
                return json.loads(nested.read_text())
            # Search recursively for the file
            for found in self._project_dir.rglob(fpath.name):
                logger.info("Reading agent output from found file: %s", found)
                return json.loads(found.read_text())

        raise ValueError(f"No JSON found in agent output: {text[:200]}")

    def _state_snapshot_for_audit(self, state: PipelineState) -> dict[str, Any]:
        snapshot: dict[str, Any] = {
            "project_id": state.project_id,
            "stage": state.stage.value,
            "assumption_ledger": state.assumption_ledger,
            "decision_log": state.decision_log,
        }
        if state.paper_claim_map:
            snapshot["paper_claim_map"] = state.paper_claim_map.model_dump()
        if state.environment_spec:
            snapshot["environment_spec"] = state.environment_spec.model_dump()
        if state.reproduction_contract:
            snapshot["reproduction_contract"] = state.reproduction_contract.model_dump()
        if state.baseline_result:
            snapshot["baseline_result"] = state.baseline_result.model_dump()
        if state.experiment_artifacts:
            snapshot["experiment_artifacts"] = state.experiment_artifacts.model_dump()
        if state.path_results:
            snapshot["path_results"] = [result.model_dump() for result in state.path_results]
        if state.research_map:
            snapshot["research_map"] = state.research_map.model_dump()
        return snapshot

    def _artifact_paths_for_state(self, state: PipelineState) -> list[str]:
        paths: list[str] = []
        if state.baseline_result:
            for path in [state.baseline_result.code_path, state.baseline_result.dockerfile_path]:
                if path:
                    paths.append(path)
        if state.experiment_artifacts:
            for path in [
                state.experiment_artifacts.log_path,
                state.experiment_artifacts.commands_log_path,
                state.experiment_artifacts.provenance_path,
                *state.experiment_artifacts.plots,
            ]:
                if path:
                    paths.append(path)
        for result in state.path_results:
            paths.extend([plot for plot in result.plots if plot])
        return paths

    def _append_hermes_report(self, state: PipelineState, report: HermesAuditReport) -> None:
        collection = (
            state.hermes_step_reports
            if report.scope == HermesAuditScope.step
            else state.hermes_checkpoint_reports
        )
        collection.setdefault(report.target, []).append(report)
        if report.recommended_intervention != HermesInterventionType.annotate:
            state.hermes_interventions.append(
                {
                    "target": report.target,
                    "scope": report.scope.value,
                    "action": report.recommended_intervention.value,
                    "reason": report.summary,
                    "status": report.status.value,
                }
            )

    def _downgrade_gate_status(self, status: GateStatus) -> GateStatus:
        if status == GateStatus.verified:
            return GateStatus.verified_with_caveats
        if status == GateStatus.verified_with_caveats:
            return GateStatus.partial_reproduction
        if status == GateStatus.partial_reproduction:
            return GateStatus.failed_reproduction
        return status

    def _apply_checkpoint_report_to_gate(
        self,
        state: PipelineState,
        report: HermesAuditReport,
        gate_decision: GateDecision,
    ) -> GateDecision:
        if report.status != HermesAuditStatus.unsupported:
            return gate_decision
        if report.recommended_intervention in {
            HermesInterventionType.downgrade_claim,
            HermesInterventionType.suppress_publication,
            HermesInterventionType.escalate_human,
        }:
            downgraded = self._downgrade_gate_status(gate_decision.status)
            gate_decision.status = downgraded
            gate_decision.passed = downgraded in (
                GateStatus.verified,
                GateStatus.verified_with_caveats,
            )
            gate_decision.blocking_issues.extend(report.unsupported_claims or [report.summary])
        return gate_decision

    def _apply_research_map_intervention(
        self,
        state: PipelineState,
        report: HermesAuditReport,
    ) -> None:
        if not state.research_map:
            return
        if report.recommended_intervention != HermesInterventionType.suppress_publication:
            return
        remaining: list[str] = []
        moved: list[str] = []
        for direction in state.research_map.promising_directions:
            if any(claim.lower() in direction.lower() for claim in report.unsupported_claims):
                moved.append(direction)
            else:
                remaining.append(direction)
        state.research_map.promising_directions = remaining
        for direction in moved:
            if direction not in state.research_map.inconclusive:
                state.research_map.inconclusive.append(f"Hermes suppressed: {direction}")
        if report.summary:
            state.research_map.overall_reproducibility_assessment = (
                state.research_map.overall_reproducibility_assessment + f" Hermes note: {report.summary}"
            ).strip()

    def _step_completion_message(
        self,
        target_stage: PipelineStage,
        state: PipelineState,
    ) -> str:
        if target_stage is PipelineStage.GATE_1_PASSED and state.gate_1 and not state.gate_1.passed:
            return f"  ! Gate 1 evaluated: {state.gate_1.status.value}"
        if target_stage is PipelineStage.GATE_2_PASSED and state.gate_2 and not state.gate_2.passed:
            return f"  ! Gate 2 evaluated: {state.gate_2.status.value}"
        if target_stage is PipelineStage.GATE_3_PASSED and state.gate_3 and not state.gate_3.passed:
            return f"  ! Gate 3 evaluated: {state.gate_3.status.value}"
        return f"  OK Completed: {state.stage.value}"

    def _audit_step(
        self,
        state: PipelineState,
        *,
        target: str,
        structured_output: dict[str, Any],
    ) -> HermesAuditReport:
        trace = self._latest_agent_traces.get(target)
        payload = build_step_audit_payload(
            project_id=self.project_id,
            target=target,
            state_snapshot=self._state_snapshot_for_audit(state),
            structured_output=structured_output,
            trace_text=trace.trace_text if trace else "",
            artifact_paths=self._artifact_paths_for_state(state),
        )
        report = self._hermes_audit_service.audit(
            scope=HermesAuditScope.step,
            target=target,
            payload=payload,
        )
        self._append_hermes_report(state, report)
        return report

    def _audit_checkpoint(
        self,
        state: PipelineState,
        *,
        target: str,
        evidence_bundle: dict[str, Any],
        trace_text: str = "",
    ) -> HermesAuditReport:
        payload = build_checkpoint_audit_payload(
            project_id=self.project_id,
            target=target,
            state_snapshot=self._state_snapshot_for_audit(state),
            evidence_bundle=evidence_bundle,
            trace_text=trace_text,
            artifact_paths=self._artifact_paths_for_state(state),
        )
        report = self._hermes_audit_service.audit(
            scope=HermesAuditScope.checkpoint,
            target=target,
            payload=payload,
        )
        self._append_hermes_report(state, report)
        return report

    async def run_paper_understanding(self, state: PipelineState) -> PipelineState:
        """Step 1: Paper Understanding Agent."""
        logger.info("[1/9] Running Paper Understanding Agent")
        out_file = self._project_dir / "paper_claim_map.json"
        prompt = (
            f"Analyze the paper for project {self.project_id}.\n"
            f"The parsed paper content is in: {self._project_dir}\n"
            f"Read the parsed sections and extract the full PaperClaimMap.\n"
            f"Return the JSON in your response AND write it to {out_file}"
        )
        output = await self._invoke_agent("paper-understanding", prompt)
        data = self._extract_json(output, fallback_file=str(out_file))
        state.paper_claim_map = PaperClaimMap(**data)
        # Merge ambiguities into assumption ledger
        for amb in state.paper_claim_map.ambiguities:
            state.assumption_ledger.append(amb.model_dump())
        self._audit_step(
            state,
            target="paper-understanding",
            structured_output=state.paper_claim_map.model_dump(),
        )
        self._enrich_workspace(
            "paper_claim_map_agent",
            state.paper_claim_map.model_dump(),
            "paper-understanding",
        )
        state.stage = PipelineStage.PAPER_UNDERSTOOD
        return state

    async def run_artifact_discovery(self, state: PipelineState) -> PipelineState:
        """Step 2: Artifact Discovery Agent."""
        logger.info("[2/9] Running Artifact Discovery Agent")
        claim_map_json = state.paper_claim_map.model_dump_json(indent=2) if state.paper_claim_map else "{}"
        prompt = (
            f"Find external artifacts for project {self.project_id}.\n"
            f"Paper claim map:\n```json\n{claim_map_json}\n```\n"
            f"Write artifact_index.json to {self._project_dir}/"
        )
        output = await self._invoke_agent("artifact-discovery", prompt)
        state.artifact_index = self._extract_json(
            output, fallback_file=str(self._project_dir / "artifact_index.json"),
        )
        self._audit_step(
            state,
            target="artifact-discovery",
            structured_output=state.artifact_index,
        )
        self._enrich_workspace(
            "artifact_index", state.artifact_index, "artifact-discovery"
        )
        state.stage = PipelineStage.ARTIFACTS_DISCOVERED
        return state

    async def run_environment_detective(self, state: PipelineState) -> PipelineState:
        """Step 3: Environment Detective Agent."""
        logger.info("[3/9] Running Environment Detective Agent")
        context = {
            "paper_claim_map": state.paper_claim_map.model_dump() if state.paper_claim_map else {},
            "artifact_index": state.artifact_index or {},
        }
        prompt = (
            f"Build the Docker environment for project {self.project_id}.\n"
            f"Context:\n```json\n{json.dumps(context, indent=2)}\n```\n"
            f"Write Dockerfile and environment_spec.json to {self._project_dir}/"
        )
        output = await self._invoke_agent("environment-detective", prompt)
        data = self._extract_json(
            output, fallback_file=str(self._project_dir / "environment_spec.json"),
        )
        state.environment_spec = EnvironmentSpec(**data)
        # Merge environment assumptions
        for assumption in state.environment_spec.assumptions:
            state.assumption_ledger.append(assumption.model_dump())
        self._audit_step(
            state,
            target="environment-detective",
            structured_output=state.environment_spec.model_dump(),
        )
        self._enrich_workspace(
            "environment_spec",
            state.environment_spec.model_dump(),
            "environment-detective",
        )
        state.stage = PipelineStage.ENVIRONMENT_BUILT
        return state

    async def run_reproduction_planner(self, state: PipelineState) -> PipelineState:
        """Step 4: Reproduction Planner."""
        logger.info("[4/9] Running Reproduction Planner")
        context = {
            "paper_claim_map": state.paper_claim_map.model_dump() if state.paper_claim_map else {},
            "environment_spec": state.environment_spec.model_dump() if state.environment_spec else {},
            "assumption_ledger": state.assumption_ledger,
        }
        prompt = (
            f"Create the reproduction contract for project {self.project_id}.\n"
            f"Context:\n```json\n{json.dumps(context, indent=2)}\n```\n"
            f"Write reproduction_contract.json to {self._project_dir}/"
        )
        output = await self._invoke_agent("reproduction-planner", prompt)
        data = self._normalize_reproduction_contract(
            self._extract_json(
                output, fallback_file=str(self._project_dir / "reproduction_contract.json"),
            )
        )
        state.reproduction_contract = ReproductionContract(**data)
        self._audit_step(
            state,
            target="reproduction-planner",
            structured_output=state.reproduction_contract.model_dump(),
        )
        self._enrich_workspace(
            "reproduction_contract",
            state.reproduction_contract.model_dump(),
            "reproduction-planner",
        )
        state.stage = PipelineStage.PLAN_CREATED
        return state

    async def run_gate_1(self, state: PipelineState) -> PipelineState:
        """Gate 1: Plan Verification."""
        logger.info("[Gate 1] Running Plan Verification")
        context = {
            "paper_claim_map": state.paper_claim_map.model_dump() if state.paper_claim_map else {},
            "reproduction_contract": state.reproduction_contract.model_dump() if state.reproduction_contract else {},
            "environment_spec": state.environment_spec.model_dump() if state.environment_spec else {},
            "assumption_ledger": state.assumption_ledger,
        }
        prompt = (
            f"Verify the reproduction plan for project {self.project_id}.\n"
            f"This is Gate 1: Plan Verification.\n"
            f"Context:\n```json\n{json.dumps(context, indent=2)}\n```\n"
            f"Run all 4 verifiers and produce a final gate decision."
        )
        output = await self._invoke_agent("supervisor-verifier", prompt)
        data = self._normalize_verifier_scores(self._extract_json(output))
        report = VerificationReport(**data)
        state.gate_1 = GateDecision(
            gate="gate_1",
            passed=report.status in (GateStatus.verified, GateStatus.verified_with_caveats),
            status=report.status,
        )
        checkpoint_report = self._audit_checkpoint(
            state,
            target="gate_1",
            evidence_bundle=context | {"verification_report": report.model_dump()},
            trace_text=output,
        )
        state.gate_1 = self._apply_checkpoint_report_to_gate(state, checkpoint_report, state.gate_1)
        state.decision_log.append(report.decision_log_entry)
        self._enrich_workspace(
            "gate_1", state.gate_1.model_dump(), "supervisor-verifier"
        )
        state.stage = PipelineStage.GATE_1_PASSED
        state.save_checkpoint(self.runs_root)
        return state

    async def run_baseline_implementation(self, state: PipelineState) -> PipelineState:
        """Step 5: Baseline Implementation Agent."""
        logger.info("[5/9] Running Baseline Implementation Agent")
        context = {
            "paper_claim_map": state.paper_claim_map.model_dump() if state.paper_claim_map else {},
            "reproduction_contract": state.reproduction_contract.model_dump() if state.reproduction_contract else {},
            "environment_spec": state.environment_spec.model_dump() if state.environment_spec else {},
            "artifact_index": state.artifact_index or {},
            "assumption_ledger": state.assumption_ledger,
        }
        code_dir = self._project_dir / "code"
        code_dir.mkdir(parents=True, exist_ok=True)
        prompt = (
            f"Implement the baseline for project {self.project_id}.\n"
            f"Write code to {code_dir}\n"
            f"Context:\n```json\n{json.dumps(context, indent=2)}\n```"
        )
        output = await self._invoke_agent(
            "baseline-implementation", prompt, cwd=code_dir,
        )
        data = self._extract_json(
            output, fallback_file=str(self._project_dir / "baseline_result.json"),
        )
        state.baseline_result = BaselineResult(**data)
        self._audit_step(
            state,
            target="baseline-implementation",
            structured_output=state.baseline_result.model_dump(),
        )
        self._enrich_workspace(
            "baseline_result",
            state.baseline_result.model_dump(),
            "baseline-implementation",
        )
        state.stage = PipelineStage.BASELINE_IMPLEMENTED
        return state

    async def run_experiment(self, state: PipelineState) -> PipelineState:
        """Step 6: Experiment Runner Agent."""
        logger.info("[6/9] Running Experiment Runner Agent")
        if state.baseline_result is None:
            raise ValueError("Cannot run experiment before baseline implementation")
        from backend.agents.experiment_runner import (
            run_with_local_process,
            run_with_runpod,
            run_with_runtime,
        )

        if self.sandbox_mode is SandboxMode.local:
            state.experiment_artifacts = await run_with_local_process(
                self.project_id,
                self.runs_root,
                state.baseline_result,
                state.reproduction_contract,
                command_timeout=self.execution_profile.command_timeout_seconds,
                gpu_mode=self.execution_profile.gpu_mode.value,
                extra_environment=self.execution_profile.sandbox_environment,
            )
        elif self.sandbox_mode is SandboxMode.runpod:
            state.experiment_artifacts = await run_with_runpod(
                self.project_id,
                self.runs_root,
                state.baseline_result,
                state.reproduction_contract,
                command_timeout=self.execution_profile.command_timeout_seconds,
            )
        else:
            state.experiment_artifacts = await run_with_runtime(
                self.project_id,
                self.runs_root,
                state.baseline_result,
                state.reproduction_contract,
                command_timeout=self.execution_profile.command_timeout_seconds,
                network_disabled=self.execution_profile.sandbox_network_disabled,
                memory_limit=self.execution_profile.sandbox_memory_limit,
                cpus=self.execution_profile.sandbox_cpus,
                platform=self.execution_profile.sandbox_platform,
                gpu_mode=self.execution_profile.gpu_mode.value,
                extra_environment=self.execution_profile.sandbox_environment,
            )
        self._audit_step(
            state,
            target="experiment-runner",
            structured_output=state.experiment_artifacts.model_dump(),
        )
        self._enrich_workspace(
            "experiment_artifacts",
            state.experiment_artifacts.model_dump(),
            "experiment-runner",
        )
        state.stage = PipelineStage.BASELINE_RUN
        return state

    async def run_gate_2(self, state: PipelineState) -> PipelineState:
        """Gate 2: Baseline Verification."""
        logger.info("[Gate 2] Running Baseline Verification")
        context = {
            "paper_claim_map": state.paper_claim_map.model_dump() if state.paper_claim_map else {},
            "baseline_result": state.baseline_result.model_dump() if state.baseline_result else {},
            "experiment_artifacts": state.experiment_artifacts.model_dump() if state.experiment_artifacts else {},
            "assumption_ledger": state.assumption_ledger,
        }
        prompt = (
            f"Verify the baseline reproduction for project {self.project_id}.\n"
            f"This is Gate 2: Baseline Verification.\n"
            f"Artifacts are in {self._project_dir / 'baseline'}\n"
            f"Context:\n```json\n{json.dumps(context, indent=2)}\n```\n"
            f"Run all 4 verifiers and produce a final gate decision."
        )
        output = await self._invoke_agent("supervisor-verifier", prompt)
        data = self._normalize_verifier_scores(self._extract_json(output))
        report = VerificationReport(**data)
        state.gate_2 = GateDecision(
            gate="gate_2",
            passed=report.status in (GateStatus.verified, GateStatus.verified_with_caveats),
            status=report.status,
        )
        checkpoint_report = self._audit_checkpoint(
            state,
            target="gate_2",
            evidence_bundle=context | {"verification_report": report.model_dump()},
            trace_text=output,
        )
        state.gate_2 = self._apply_checkpoint_report_to_gate(state, checkpoint_report, state.gate_2)
        state.decision_log.append(report.decision_log_entry)
        self._enrich_workspace(
            "gate_2", state.gate_2.model_dump(), "supervisor-verifier"
        )
        state.stage = PipelineStage.GATE_2_PASSED
        state.save_checkpoint(self.runs_root)
        return state

    async def run_improvements(
        self,
        state: PipelineState,
        *,
        user_hints: list[str] | None = None,
        n_paths: int = 3,
    ) -> PipelineState:
        """Steps 7-8: Improvement Orchestrator + Path Agents."""
        logger.info("[7/9] Running Improvement Orchestrator")
        context = {
            "paper_claim_map": state.paper_claim_map.model_dump() if state.paper_claim_map else {},
            "experiment_artifacts": state.experiment_artifacts.model_dump() if state.experiment_artifacts else {},
            "baseline_result": state.baseline_result.model_dump() if state.baseline_result else {},
            "assumption_ledger": state.assumption_ledger,
        }
        hints_str = ""
        if user_hints:
            hints_str = f"\nUser hints: {', '.join(user_hints)}"
        prompt = (
            f"Select {n_paths} improvement hypotheses for project {self.project_id}.{hints_str}\n"
            f"Context:\n```json\n{json.dumps(context, indent=2)}\n```"
        )
        output = await self._invoke_agent("improvement-orchestrator", prompt)
        data = self._extract_json(output)
        hypotheses_raw = data.get("hypotheses", [])
        state.improvement_hypotheses = [
            ImprovementHypothesis(**h) for h in hypotheses_raw
        ]
        hypotheses_payload = {"hypotheses": [hypothesis.model_dump() for hypothesis in state.improvement_hypotheses]}
        self._audit_step(
            state,
            target="improvement-orchestrator",
            structured_output=hypotheses_payload,
        )
        self._enrich_workspace(
            "improvement_hypotheses",
            hypotheses_payload,
            "improvement-orchestrator",
        )
        state.stage = PipelineStage.IMPROVEMENTS_SELECTED

        # Run each path agent
        logger.info("[8/9] Running %d Improvement Path Agents", len(state.improvement_hypotheses))
        for hypothesis in state.improvement_hypotheses:
            path_dir = self._prepare_improvement_workspace(state, hypothesis)
            path_prompt = (
                f"Execute improvement hypothesis for project {self.project_id}.\n"
                f"Work in: {path_dir}\n"
                f"Baseline code is in: {self._project_dir / 'code'}\n"
                f"Hypothesis:\n```json\n{hypothesis.model_dump_json(indent=2)}\n```\n"
                f"Environment:\n```json\n{state.environment_spec.model_dump_json(indent=2) if state.environment_spec else '{}'}\n```"
            )
            path_output = await self._invoke_agent(
                "improvement-path", path_prompt, cwd=path_dir,
            )
            try:
                path_data = self._extract_json(path_output)
                path_result = PathResult(**path_data)
                state.path_results.append(path_result)
                self._audit_step(
                    state,
                    target=f"improvement-path:{hypothesis.path_id}",
                    structured_output=path_result.model_dump(),
                )
            except (ValueError, Exception) as exc:
                logger.warning("Path %s failed to parse: %s", hypothesis.path_id, exc)
                state.path_results.append(
                    PathResult(
                        path_id=hypothesis.path_id,
                        hypothesis=hypothesis.hypothesis,
                        failure_notes=str(exc),
                        success=False,
                    )
                )
        state.stage = PipelineStage.IMPROVEMENTS_RUN
        self._enrich_workspace(
            "path_results",
            {"results": [r.model_dump() for r in state.path_results]},
            "improvement-path",
        )
        return state

    def _prepare_improvement_workspace(
        self,
        state: PipelineState,
        hypothesis: ImprovementHypothesis,
    ) -> Path:
        """Create an isolated workspace for one improvement path.

        Phase 2 uses git worktrees when the baseline code is a git repository.
        Generated code paths are not always repos yet, so non-git baselines keep
        the existing isolated directory behavior.
        """
        fallback = self._project_dir / "improvements" / hypothesis.path_id
        baseline_code = (
            Path(state.baseline_result.code_path)
            if state.baseline_result and state.baseline_result.code_path
            else self._project_dir / "code"
        )
        if not baseline_code.is_absolute():
            baseline_code = Path.cwd() / baseline_code

        if not self._is_git_repo(baseline_code):
            fallback.mkdir(parents=True, exist_ok=True)
            return fallback

        from backend.services.worktrees import GitWorktreeError, GitWorktreeManager

        manager = GitWorktreeManager(worktrees_root=self._project_dir / "worktrees")
        spec = manager.spec_for(
            project_id=self.project_id,
            path_id=hypothesis.path_id,
            slug=hypothesis.hypothesis,
        )
        if spec.worktree_path.exists():
            return spec.worktree_path
        try:
            info = manager.create(repo_root=baseline_code, spec=spec)
            state.decision_log.append(
                f"worktree:{hypothesis.path_id}: {info.branch} -> {info.path}"
            )
            return spec.worktree_path
        except GitWorktreeError as exc:
            logger.warning(
                "Falling back to directory workspace for %s: %s",
                hypothesis.path_id,
                exc,
            )
            state.decision_log.append(
                f"worktree_fallback:{hypothesis.path_id}: {exc}"
            )
            fallback.mkdir(parents=True, exist_ok=True)
            return fallback

    def _is_git_repo(self, path: Path) -> bool:
        import subprocess

        result = subprocess.run(
            ("git", "-C", str(path), "rev-parse", "--is-inside-work-tree"),
            check=False,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0 and result.stdout.strip() == "true"

    async def run_gate_3(self, state: PipelineState) -> PipelineState:
        """Gate 3: Improvement Verification + Research Map."""
        logger.info("[Gate 3] Running Improvement Verification")
        context = {
            "paper_claim_map": state.paper_claim_map.model_dump() if state.paper_claim_map else {},
            "path_results": [r.model_dump() for r in state.path_results],
            "baseline_result": state.baseline_result.model_dump() if state.baseline_result else {},
            "experiment_artifacts": state.experiment_artifacts.model_dump() if state.experiment_artifacts else {},
        }
        prompt = (
            f"Verify all improvement paths for project {self.project_id}.\n"
            f"This is Gate 3: Improvement Verification.\n"
            f"Also generate the final Research Map.\n"
            f"Context:\n```json\n{json.dumps(context, indent=2)}\n```"
        )
        output = await self._invoke_agent("supervisor-verifier", prompt)
        data = self._normalize_verifier_scores(self._extract_json(output))
        report = VerificationReport(**data)
        state.gate_3 = GateDecision(
            gate="gate_3",
            passed=report.status in (GateStatus.verified, GateStatus.verified_with_caveats),
            status=report.status,
        )
        checkpoint_report = self._audit_checkpoint(
            state,
            target="gate_3",
            evidence_bundle=context | {"verification_report": report.model_dump()},
            trace_text=output,
        )
        state.gate_3 = self._apply_checkpoint_report_to_gate(state, checkpoint_report, state.gate_3)
        state.decision_log.append(report.decision_log_entry)
        self._enrich_workspace(
            "gate_3", state.gate_3.model_dump(), "supervisor-verifier"
        )
        state.stage = PipelineStage.GATE_3_PASSED
        state.save_checkpoint(self.runs_root)
        return state

    async def generate_research_map(self, state: PipelineState) -> PipelineState:
        """Step 9: Generate final Research Map."""
        logger.info("[9/9] Generating Research Map")
        context = {
            "paper_claim_map": state.paper_claim_map.model_dump() if state.paper_claim_map else {},
            "experiment_artifacts": state.experiment_artifacts.model_dump() if state.experiment_artifacts else {},
            "path_results": [r.model_dump() for r in state.path_results],
            "gate_2": state.gate_2.model_dump() if state.gate_2 else {},
            "gate_3": state.gate_3.model_dump() if state.gate_3 else {},
            "assumption_ledger": state.assumption_ledger,
            "decision_log": state.decision_log,
        }
        prompt = (
            f"Generate the final Research Map for project {self.project_id}.\n"
            f"Summarize: baseline results, promising directions, dead ends, and next experiments.\n"
            f"Context:\n```json\n{json.dumps(context, indent=2)}\n```\n"
            f"Write research_map.json to {self._project_dir}/"
        )
        output = await self._invoke_agent("supervisor-verifier", prompt)
        data = self._extract_json(
            output, fallback_file=str(self._project_dir / "research_map.json"),
        )
        state.research_map = ResearchMap(**data)
        self._audit_step(
            state,
            target="research_map_generated",
            structured_output=state.research_map.model_dump(),
        )
        checkpoint_report = self._audit_checkpoint(
            state,
            target="research_map_generated",
            evidence_bundle=context | {"research_map": state.research_map.model_dump()},
            trace_text=output,
        )
        self._apply_research_map_intervention(state, checkpoint_report)
        self._enrich_workspace(
            "research_map",
            state.research_map.model_dump(),
            "research-map-generator",
        )
        self._enrich_workspace(
            "assumption_ledger",
            {"entries": state.assumption_ledger},
            "orchestrator",
        )
        self._enrich_workspace(
            "decision_log",
            {"entries": state.decision_log},
            "orchestrator",
        )
        state.stage = PipelineStage.RESEARCH_MAP_GENERATED
        # Write final artifacts
        (self._project_dir / "research_map.json").write_text(
            state.research_map.model_dump_json(indent=2)
        )
        (self._project_dir / "assumption_ledger.json").write_text(
            json.dumps(state.assumption_ledger, indent=2)
        )
        (self._project_dir / "decision_log.json").write_text(
            json.dumps(state.decision_log, indent=2)
        )
        state.stage = PipelineStage.COMPLETE
        state.save_checkpoint(self.runs_root)
        return state

    async def run(
        self,
        *,
        resume: bool = True,
        user_hints: list[str] | None = None,
        n_improvement_paths: int = 3,
    ) -> PipelineState:
        """Run the full pipeline, resuming from the last checkpoint if available."""
        state: PipelineState | None = None
        if resume:
            state = PipelineState.load_checkpoint(self.runs_root, self.project_id)
        if state is None:
            state = PipelineState(
                project_id=self.project_id,
                seed=self.seed,
                attempt_id=self.attempt_id,
                run_group_id=self.run_group_id,
                blacklist_terms=list(self.blacklist_terms),
            )
        else:
            state.seed = self.seed if self.seed is not None else state.seed
            state.attempt_id = self.attempt_id or state.attempt_id
            state.run_group_id = self.run_group_id or state.run_group_id
            if self.blacklist_terms:
                state.blacklist_terms = list(self.blacklist_terms)

        stages_order = list(PipelineStage)
        if stages_order.index(state.stage) < stages_order.index(PipelineStage.BASELINE_RUN):
            ensure_sandbox_mode_available(self.sandbox_mode)

        # Define the pipeline as a sequence of (stage_threshold, step_fn) pairs.
        # Each step only runs if the pipeline hasn't passed that stage yet.
        pipeline: list[tuple[PipelineStage, Any]] = [
            (PipelineStage.PAPER_UNDERSTOOD, self.run_paper_understanding),
            (PipelineStage.ARTIFACTS_DISCOVERED, self.run_artifact_discovery),
            (PipelineStage.ENVIRONMENT_BUILT, self.run_environment_detective),
            (PipelineStage.PLAN_CREATED, self.run_reproduction_planner),
            (PipelineStage.GATE_1_PASSED, self.run_gate_1),
            (PipelineStage.BASELINE_IMPLEMENTED, self.run_baseline_implementation),
            (PipelineStage.BASELINE_RUN, self.run_experiment),
            (PipelineStage.GATE_2_PASSED, self.run_gate_2),
        ]

        current_idx = stages_order.index(state.stage)

        for target_stage, step_fn in pipeline:
            target_idx = stages_order.index(target_stage)
            if current_idx >= target_idx:
                print(f"  >> Skipping {target_stage.value} (already at {state.stage.value})", file=sys.stderr, flush=True)
                continue
            print(f"\n{'='*50}", file=sys.stderr, flush=True)
            print(f"  > Starting: {target_stage.value}", file=sys.stderr, flush=True)
            print(f"{'='*50}", file=sys.stderr, flush=True)
            try:
                state = await step_fn(state)
            except Exception as exc:
                print(f"  X FAILED: {target_stage.value} -- {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
                logger.exception("Step %s failed", target_stage.value)
                raise
            print(self._step_completion_message(target_stage, state), file=sys.stderr, flush=True)
            current_idx = stages_order.index(state.stage)

            # Check gate results
            if state.gate_1 and not state.gate_1.passed:
                print(f"  X Gate 1 FAILED: {state.gate_1.status.value}", file=sys.stderr, flush=True)
                return state
            if state.gate_2 and not state.gate_2.passed:
                print(f"  X Gate 2 FAILED: {state.gate_2.status.value}", file=sys.stderr, flush=True)
                return state

        # Improvement phase
        if current_idx < stages_order.index(PipelineStage.IMPROVEMENTS_RUN):
            state = await self.run_improvements(
                state, user_hints=user_hints, n_paths=n_improvement_paths,
            )
            current_idx = stages_order.index(state.stage)

        if current_idx < stages_order.index(PipelineStage.GATE_3_PASSED):
            state = await self.run_gate_3(state)
            current_idx = stages_order.index(state.stage)

        if current_idx < stages_order.index(PipelineStage.RESEARCH_MAP_GENERATED):
            state = await self.generate_research_map(state)

        self._close_workspace("pipeline_complete")
        logger.info("Pipeline complete for project %s", self.project_id)
        return state
