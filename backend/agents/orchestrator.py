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

from backend.agents.dependency_verifier import verify_dockerfile
from backend.agents.registry import AGENT_REGISTRY
from backend.agents.report_generator import (
    generate_final_report,
    write_final_report,
)
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
    CompositionAttempt,
    CompositionPhase,
    EnvironmentSpec,
    ExperimentArtifacts,
    FinalReport,
    GateDecision,
    GateStatus,
    ImprovementHypothesis,
    ImprovementRound,
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
    COMPOSITION_TESTED = "composition_tested"
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
    improvement_rounds: list[ImprovementRound] = field(default_factory=list)
    composition_phase: CompositionPhase | None = None
    gate_3: GateDecision | None = None
    research_map: ResearchMap | None = None
    final_report: FinalReport | None = None
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
        if self.final_report:
            data["final_report"] = self.final_report.model_dump()
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
        if self.improvement_rounds:
            data["improvement_rounds"] = [r.model_dump() for r in self.improvement_rounds]
        if self.composition_phase:
            data["composition_phase"] = self.composition_phase.model_dump()
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
        if "final_report" in data:
            state.final_report = FinalReport(**data["final_report"])
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
        if "improvement_rounds" in data:
            state.improvement_rounds = [
                ImprovementRound(**r) for r in data["improvement_rounds"]
            ]
        if "composition_phase" in data:
            state.composition_phase = CompositionPhase(**data["composition_phase"])
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
        collected_text: list[str] = []
        started_at = utc_now_iso()
        trace_lines: list[str] = []
        tool_calls: list[str] = []
        t0 = time.time()
        msg_count = 0
        success = True
        error_message = ""
        usage: dict[str, Any] = {}
        tool_call_count = 0
        fallback_runtime: AgentRuntime | None = None
        print(f"  [{agent_id}] starting...", file=sys.stderr, flush=True)

        wall_clock_seconds = self.execution_profile.agent_wall_clock_seconds
        try:
            # Wall-clock cap on the entire agent invocation. Catches stuck
            # runs even when the SDK happily streams forever (e.g. infinite
            # tool-call loop). asyncio.timeout requires Python 3.11+.
            timeout_ctx = (
                asyncio.timeout(wall_clock_seconds)
                if wall_clock_seconds is not None
                else _NullAsyncContext()
            )
            async with timeout_ctx:
                async for event in runtime.run_agent(
                    agent=runtime_spec,
                    user_input=task_prompt,
                ):
                    elapsed = time.time() - t0
                    msg_count += 1
                    if isinstance(event, StreamText):
                        collected_text.append(event.text)
                        trace_lines.append(event.text)
                        snippet = event.text[:120].replace("\n", " ").strip()
                        if snippet:
                            print(
                                f"  [{agent_id}] ({elapsed:.0f}s) {snippet}...",
                                file=sys.stderr,
                                flush=True,
                            )
                    elif isinstance(event, StreamToolCall):
                        tool_call_count += 1
                        if (
                            runtime_spec.guard.max_tool_calls is not None
                            and tool_call_count > runtime_spec.guard.max_tool_calls
                        ):
                            raise AgentLimitExceeded(
                                agent_id=agent_id,
                                kind="tool_calls",
                                limit_value=runtime_spec.guard.max_tool_calls,
                                elapsed_seconds=elapsed,
                                partial_output="".join(collected_text),
                            )
                        tool_info = event.tool_name
                        inp = event.tool_input or {}
                        if "file_path" in inp:
                            tool_info += f" {inp['file_path']}"
                        elif "command" in inp:
                            cmd = str(inp["command"])[:80]
                            tool_info += f" `{cmd}`"
                        elif "pattern" in inp:
                            tool_info += f" {inp['pattern']}"
                        tool_calls.append(tool_info)
                        trace_lines.append(f"tool: {tool_info}")
                        print(
                            f"  [{agent_id}] ({elapsed:.0f}s) tool: {tool_info}",
                            file=sys.stderr,
                            flush=True,
                        )
                    elif isinstance(event, StreamUsage):
                        usage = coerce_usage(event.as_dict())
                        usage["provider"] = runtime.provider_name
                        usage["model"] = runtime_spec.model
                        print(
                            f"  [{agent_id}] completed in {elapsed:.0f}s ({msg_count} events, {sum(len(t) for t in collected_text)} chars)",
                            file=sys.stderr,
                            flush=True,
                        )
        except TimeoutError as exc:
            # asyncio.timeout fired — wrap as a typed limit error.
            raise AgentLimitExceeded(
                agent_id=agent_id,
                kind="wall_clock",
                limit_value=wall_clock_seconds or 0,
                elapsed_seconds=time.time() - t0,
                partial_output="".join(collected_text),
            ) from exc
        except Exception as exc:
            success = False
            error_message = f"{type(exc).__name__}: {exc}"
            usage.setdefault("provider", runtime.provider_name)
            usage.setdefault("model", runtime_spec.model)
            # Detect the SDK's "Reached maximum number of turns (N)" message
            # and convert to a typed AgentLimitExceeded so callers can react
            # programmatically instead of string-matching exception text.
            if not isinstance(exc, AgentLimitExceeded):
                limit_match = _TURN_LIMIT_RE.search(str(exc))
                if limit_match:
                    raise AgentLimitExceeded(
                        agent_id=agent_id,
                        kind="turns",
                        limit_value=int(limit_match.group(1)),
                        elapsed_seconds=time.time() - t0,
                        partial_output="".join(collected_text),
                    ) from exc
            if _allow_claude_limit_fallback:
                fallback_runtime = self._claude_limit_fallback_for(
                    runtime,
                    exc,
                    "\n".join(trace_lines + collected_text),
                )
            if fallback_runtime is None:
                raise
            print(
                f"  [{agent_id}] Claude limit detected; retrying with OpenAI...",
                file=sys.stderr,
                flush=True,
            )
        finally:
            self._telemetry.append(
                AgentInvocationRecord(
                    agent_id=agent_id,
                    model=runtime_spec.model,
                    started_at=started_at,
                    finished_at=utc_now_iso(),
                    duration_seconds=time.time() - t0,
                    message_count=msg_count,
                    output_chars=sum(len(text) for text in collected_text),
                    success=success,
                    error_message=error_message,
                    tool_calls=tool_calls,
                    usage=usage,
                )
            )
        if fallback_runtime is not None:
            return await self._invoke_agent(
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
        trace = AgentExecutionTrace(
            agent_id=agent_id,
            output_text=result,
            trace_text=result_obj.trace_text,
            tool_calls=result_obj.tool_calls,
            elapsed_seconds=result_obj.elapsed_seconds,
        )
        self._latest_agent_traces[agent_id] = trace
        self._persist_trace(trace)
        return result

    def _persist_trace(self, trace: AgentExecutionTrace) -> None:
        """Write agent trace to traces/ directory for post-hoc inspection."""
        try:
            traces_dir = self._project_dir / "traces"
            traces_dir.mkdir(parents=True, exist_ok=True)
            trace_data = {
                "agent_id": trace.agent_id,
                "elapsed_seconds": trace.elapsed_seconds,
                "tool_calls": trace.tool_calls,
                "output_text": trace.output_text,
                "trace_text": trace.trace_text,
            }
            (traces_dir / f"{trace.agent_id}.json").write_text(
                json.dumps(trace_data, indent=2)
            )
        except OSError as exc:
            logger.warning("Could not persist trace for %s: %s", trace.agent_id, exc)

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

    _MAX_DEP_VERIFY_RETRIES = 2

    async def _verify_dockerfile_deps(
        self,
        state: PipelineState,
        agent_id: str,
    ) -> PipelineState:
        """Verify Dockerfile dependencies and re-run agent if hallucinations found.

        Scans the Dockerfile for git SHAs, PyPI versions, and repo URLs,
        verifying each one actually exists. If failures are found, re-invokes
        the agent with feedback about what was wrong.
        """
        dockerfile_path = self._project_dir / "Dockerfile"
        if not dockerfile_path.exists():
            return state

        for attempt in range(self._MAX_DEP_VERIFY_RETRIES):
            report = await verify_dockerfile(dockerfile_path)

            if not report.has_failures:
                if report.checks:
                    logger.info(
                        "Dependency verification passed: %d checks OK",
                        len(report.checks),
                    )
                    state.decision_log.append(
                        f"Dependency verification passed after {agent_id}: "
                        f"{len(report.checks)} dependencies verified."
                    )
                return state

            # Log failures
            failure_summary = "; ".join(f.summary() for f in report.failures)
            logger.warning(
                "Dependency verification FAILED (attempt %d/%d): %s",
                attempt + 1,
                self._MAX_DEP_VERIFY_RETRIES,
                failure_summary,
            )
            state.decision_log.append(
                f"Dependency verification failed after {agent_id} "
                f"(attempt {attempt + 1}): {failure_summary}"
            )

            # Re-invoke the agent with feedback
            feedback = report.feedback_prompt()
            context = {
                "paper_claim_map": state.paper_claim_map.model_dump() if state.paper_claim_map else {},
                "artifact_index": state.artifact_index or {},
            }
            if state.environment_spec:
                context["environment_spec"] = state.environment_spec.model_dump()
            fix_prompt = (
                f"Fix the Dockerfile for project {self.project_id}.\n"
                f"The dependency verifier found the following problems:\n\n"
                f"{feedback}\n\n"
                f"Read the current Dockerfile at {dockerfile_path}, fix ALL "
                f"the issues listed above, and write the corrected Dockerfile "
                f"back to the same path.\n"
                f"Context:\n```json\n{json.dumps(context, indent=2)}\n```\n"
                f"Return the updated environment_spec JSON."
            )
            print(
                f"  [dep-verify] Attempt {attempt + 1}: "
                f"{len(report.failures)} failures, re-running {agent_id}...",
                file=sys.stderr,
                flush=True,
            )
            output = await self._invoke_agent(agent_id, fix_prompt)
            try:
                data = self._extract_json(
                    output,
                    fallback_file=str(self._project_dir / "environment_spec.json"),
                )
                state.environment_spec = EnvironmentSpec(**data)
            except Exception:
                logger.warning(
                    "Could not parse env spec from fix attempt; "
                    "Dockerfile may still have been updated on disk."
                )

        # After all retries, log remaining failures but don't block
        final_report = await verify_dockerfile(dockerfile_path)
        if final_report.has_failures:
            remaining = "; ".join(f.summary() for f in final_report.failures)
            state.decision_log.append(
                f"WARNING: {len(final_report.failures)} dependency issues "
                f"persist after {self._MAX_DEP_VERIFY_RETRIES} fix attempts: "
                f"{remaining}"
            )
            logger.warning(
                "Dependency issues persist after retries: %s", remaining
            )

        return state

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
        if state.improvement_rounds:
            snapshot["improvement_rounds"] = [r.model_dump() for r in state.improvement_rounds]
        if state.composition_phase:
            snapshot["composition_phase"] = state.composition_phase.model_dump()
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
        # --- Dependency verification guardrail ---
        state = await self._verify_dockerfile_deps(state, "environment-detective")

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
        # --- Dependency verification guardrail ---
        state = await self._verify_dockerfile_deps(state, "baseline-implementation")

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

    # ------------------------------------------------------------------
    # Parallel improvement-path execution
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Per-path checkpointing
    # ------------------------------------------------------------------

    def _path_results_dir(self) -> Path:
        d = self._project_dir / "path_results"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _save_path_result_to_disk(self, result: PathResult) -> None:
        """Persist a single path result so it survives a crash."""
        path = self._path_results_dir() / f"{result.path_id}.json"
        path.write_text(result.model_dump_json(indent=2))

    def _load_completed_path_results(self) -> dict[str, PathResult]:
        """Load all previously-completed path results from disk."""
        results: dict[str, PathResult] = {}
        results_dir = self._project_dir / "path_results"
        if not results_dir.exists():
            return results
        for path_file in results_dir.glob("*.json"):
            try:
                data = json.loads(path_file.read_text())
                result = PathResult(**data)
                results[result.path_id] = result
            except Exception:
                logger.warning("Skipping corrupt path result: %s", path_file)
        return results

    _RATE_LIMIT_MAX_RETRIES = 3
    _RATE_LIMIT_BASE_DELAY = 30  # seconds

    async def _run_single_improvement_path(
        self,
        hypothesis: ImprovementHypothesis,
        state: PipelineState,
        semaphore: asyncio.Semaphore,
    ) -> PathResult:
        """Execute one improvement path, respecting the concurrency semaphore.

        Retries with exponential back-off when a Claude subscription rate
        limit is detected.  After exhausting retries the path is marked
        failed so the pipeline can continue.
        """
        path_dir = self._prepare_improvement_workspace(state, hypothesis)
        path_prompt = (
            f"Execute improvement hypothesis for project {self.project_id}.\n"
            f"Work in: {path_dir}\n"
            f"Baseline code is in: {self._project_dir / 'code'}\n"
            f"Hypothesis:\n```json\n{hypothesis.model_dump_json(indent=2)}\n```\n"
            f"Environment:\n```json\n"
            f"{state.environment_spec.model_dump_json(indent=2) if state.environment_spec else '{}'}"
            f"\n```"
        )

        last_exc: Exception | None = None
        for attempt in range(1, self._RATE_LIMIT_MAX_RETRIES + 1):
            async with semaphore:
                try:
                    path_output = await self._invoke_agent(
                        "improvement-path", path_prompt, cwd=path_dir,
                    )
                    path_data = self._extract_json(path_output)
                    result = PathResult(**path_data)
                    self._save_path_result_to_disk(result)
                    self._audit_step(
                        state,
                        target=f"improvement-path:{hypothesis.path_id}",
                        structured_output=result.model_dump(),
                    )
                    return result
                except Exception as exc:
                    last_exc = exc
                    if not _looks_like_claude_limit_failure(exc):
                        break  # non-rate-limit error → don't retry
                    if attempt < self._RATE_LIMIT_MAX_RETRIES:
                        delay = self._RATE_LIMIT_BASE_DELAY * (2 ** (attempt - 1))
                        print(
                            f"  [{hypothesis.path_id}] rate-limited (attempt {attempt}/"
                            f"{self._RATE_LIMIT_MAX_RETRIES}), retrying in {delay}s...",
                            file=sys.stderr,
                            flush=True,
                        )
                        await asyncio.sleep(delay)

        # All retries exhausted or non-retriable error
        logger.warning(
            "Path %s failed: %s", hypothesis.path_id, last_exc,
        )
        result = PathResult(
            path_id=hypothesis.path_id,
            hypothesis=hypothesis.hypothesis,
            failure_notes=str(last_exc),
            success=False,
        )
        self._save_path_result_to_disk(result)
        return result

    async def run_improvements(
        self,
        state: PipelineState,
        *,
        user_hints: list[str] | None = None,
        n_paths: int = 3,
    ) -> PipelineState:
        """Steps 7-8: Iterative Improvement Rounds.

        Runs up to ``max_improvement_rounds`` rounds of parallel improvement
        paths.  After each round the best-performing path becomes the baseline
        for the next round.  Stops early when improvement falls below
        ``improvement_convergence_pct`` or no path improves over baseline.

        Path agents within each round run in parallel, bounded by
        ``execution_profile.max_concurrent_agents``.
        """
        max_rounds = self.execution_profile.max_improvement_rounds
        convergence_threshold = self.execution_profile.improvement_convergence_pct

        # Track the "current best" metrics — starts as the original baseline
        current_baseline_metrics: dict[str, Any] = (
            state.experiment_artifacts.metrics if state.experiment_artifacts else {}
        )
        current_baseline_path_id: str | None = None  # None = original baseline

        for round_num in range(1, max_rounds + 1):
            is_first_round = round_num == 1
            logger.info(
                "[7/9] Running Improvement Orchestrator (round %d/%d)",
                round_num,
                max_rounds,
            )
            print(
                f"\n{'='*50}\n"
                f"  > Improvement Round {round_num}/{max_rounds}"
                f" (baseline: {current_baseline_path_id or 'original'})\n"
                f"{'='*50}",
                file=sys.stderr,
                flush=True,
            )

            # --- Select and run hypotheses ---
            use_adaptive = self.execution_profile.adaptive_selection
            if use_adaptive:
                # Adaptive mode: generate larger pool, run in batches
                pool_size = max(
                    n_paths,
                    int(n_paths * self.execution_profile.hypothesis_pool_multiplier),
                )
                pool = await self._generate_hypothesis_pool(
                    state,
                    pool_size=pool_size,
                    round_num=round_num,
                    user_hints=user_hints,
                )
                # For round N>1 with prior context, use the round-N prompt
                # to generate the pool instead (handled inside the pool method)
                hypotheses = pool[:n_paths]  # track nominal set
                state.improvement_hypotheses = pool  # store full pool
                state.stage = PipelineStage.IMPROVEMENTS_SELECTED

                logger.info(
                    "[8/9] Adaptive mode: pool=%d, running %d (round %d)",
                    len(pool), n_paths, round_num,
                )
                print(
                    f"  [round {round_num}] Adaptive: pool of {len(pool)} "
                    f"candidates, running {n_paths} in batches",
                    file=sys.stderr,
                    flush=True,
                )
                results = await self._run_adaptive_batches(
                    state,
                    pool=pool,
                    n_to_run=n_paths,
                    round_num=round_num,
                    baseline_metrics=current_baseline_metrics,
                )
            else:
                # Standard mode: select exactly N, run all in parallel
                hypotheses = await self._select_round_hypotheses(
                    state,
                    round_num=round_num,
                    n_paths=n_paths,
                    user_hints=user_hints,
                    current_baseline_path_id=current_baseline_path_id,
                    current_baseline_metrics=current_baseline_metrics,
                )
                state.improvement_hypotheses = hypotheses
                state.stage = PipelineStage.IMPROVEMENTS_SELECTED

                # Run path agents in parallel, skipping already-completed
                concurrency = self.execution_profile.max_concurrent_agents
                completed = self._load_completed_path_results()
                remaining_hypotheses = [
                    h for h in hypotheses if h.path_id not in completed
                ]
                n_total = len(hypotheses)
                n_skip = n_total - len(remaining_hypotheses)
                logger.info(
                    "[8/9] Running %d Improvement Path Agents (round %d, concurrency=%d, skipping %d completed)",
                    len(remaining_hypotheses),
                    round_num,
                    concurrency,
                    n_skip,
                )
                if n_skip > 0:
                    print(
                        f"  [round {round_num}] Resuming: {n_skip}/{n_total} paths "
                        f"already completed, launching {len(remaining_hypotheses)} remaining",
                        file=sys.stderr,
                        flush=True,
                    )
                else:
                    print(
                        f"  [round {round_num}] Launching {n_total} paths "
                        f"(max {concurrency} concurrent)",
                        file=sys.stderr,
                        flush=True,
                    )

                semaphore = asyncio.Semaphore(concurrency)
                tasks = [
                    self._run_single_improvement_path(hypothesis, state, semaphore)
                    for hypothesis in remaining_hypotheses
                ]
                new_results = list(await asyncio.gather(*tasks))

                # Merge: completed from disk + newly run, in hypothesis order
                results = []
                for h in hypotheses:
                    if h.path_id in completed and h.path_id not in {r.path_id for r in new_results}:
                        results.append(completed[h.path_id])
                    else:
                        match = next((r for r in new_results if r.path_id == h.path_id), None)
                        if match:
                            results.append(match)

            # --- Evaluate round: find the best path ---
            best_path, best_metrics, improvement_pct = self._evaluate_round(
                results, current_baseline_metrics,
            )
            converged = (
                best_path is None
                or (improvement_pct is not None and improvement_pct < convergence_threshold)
            )

            # Record the round
            round_record = ImprovementRound(
                round_number=round_num,
                baseline_path_id=current_baseline_path_id,
                baseline_metrics=current_baseline_metrics,
                hypotheses=hypotheses,
                path_results=results,
                best_path_id=best_path.path_id if best_path else None,
                best_metrics=best_metrics,
                improvement_pct=improvement_pct,
                converged=converged,
            )
            state.improvement_rounds.append(round_record)

            # Accumulate all path results across rounds
            state.path_results.extend(results)

            print(
                f"  [round {round_num}] Best: {best_path.path_id if best_path else 'none'}"
                f" | improvement: {improvement_pct:.2f}%"
                if improvement_pct is not None
                else f"  [round {round_num}] No improvement",
                file=sys.stderr,
                flush=True,
            )

            # Checkpoint after each round
            state.save_checkpoint(self.runs_root)

            # --- Convergence check ---
            if converged:
                state.decision_log.append(
                    f"Improvement converged at round {round_num}/{max_rounds}: "
                    f"best improvement {improvement_pct:.2f}% < threshold {convergence_threshold}%"
                    if improvement_pct is not None
                    else f"No path improved over baseline at round {round_num}/{max_rounds}"
                )
                print(
                    f"  [round {round_num}] Converged — stopping improvement loop",
                    file=sys.stderr,
                    flush=True,
                )
                break

            # Promote winner as next round's baseline
            current_baseline_path_id = best_path.path_id  # type: ignore[union-attr]
            current_baseline_metrics = best_metrics
            state.decision_log.append(
                f"Round {round_num}: promoted {current_baseline_path_id} "
                f"as baseline for round {round_num + 1} "
                f"(+{improvement_pct:.2f}% over previous baseline)"
            )

        state.stage = PipelineStage.IMPROVEMENTS_RUN
        self._enrich_workspace(
            "path_results",
            {"results": [r.model_dump() for r in state.path_results]},
            "improvement-path",
        )
        self._enrich_workspace(
            "improvement_rounds",
            {"rounds": [r.model_dump() for r in state.improvement_rounds]},
            "improvement-orchestrator",
        )
        return state

    async def _select_round_hypotheses(
        self,
        state: PipelineState,
        *,
        round_num: int,
        n_paths: int,
        user_hints: list[str] | None,
        current_baseline_path_id: str | None,
        current_baseline_metrics: dict[str, Any],
    ) -> list[ImprovementHypothesis]:
        """Select improvement hypotheses for a given round.

        Round 1 uses the standard prompt.  Subsequent rounds use a
        round-aware prompt that includes prior round results so the
        orchestrator can learn from what worked and avoid repeats.
        """
        context = {
            "paper_claim_map": state.paper_claim_map.model_dump() if state.paper_claim_map else {},
            "experiment_artifacts": state.experiment_artifacts.model_dump() if state.experiment_artifacts else {},
            "baseline_result": state.baseline_result.model_dump() if state.baseline_result else {},
            "assumption_ledger": state.assumption_ledger,
        }
        hints_str = ""
        if user_hints:
            hints_str = f"\nUser hints: {', '.join(user_hints)}"

        if round_num == 1:
            prompt = (
                f"Select {n_paths} improvement hypotheses for project {self.project_id}.{hints_str}\n"
                f"Context:\n```json\n{json.dumps(context, indent=2)}\n```"
            )
        else:
            prior_summary = self._build_prior_rounds_summary(state.improvement_rounds)
            from backend.agents.prompts.improvement import IMPROVEMENT_ORCHESTRATOR_ROUND_N_PROMPT
            round_prompt = IMPROVEMENT_ORCHESTRATOR_ROUND_N_PROMPT.format(
                round_number=round_num,
                prev_round=round_num - 1,
                prior_rounds_summary=prior_summary,
                current_baseline_path_id=current_baseline_path_id or "original",
                current_baseline_metrics=json.dumps(current_baseline_metrics, indent=2),
            )
            prompt = (
                f"{round_prompt}\n{hints_str}\n"
                f"Select {n_paths} NEW hypotheses for project {self.project_id}.\n"
                f"Context:\n```json\n{json.dumps(context, indent=2)}\n```"
            )

        output = await self._invoke_agent("improvement-orchestrator", prompt)
        data = self._extract_json(output)
        hypotheses_raw = data.get("hypotheses", [])
        hypotheses = [ImprovementHypothesis(**h) for h in hypotheses_raw]
        hypotheses_payload = {"hypotheses": [h.model_dump() for h in hypotheses]}
        self._audit_step(
            state,
            target=f"improvement-orchestrator:round_{round_num}",
            structured_output=hypotheses_payload,
        )
        self._enrich_workspace(
            f"improvement_hypotheses_round_{round_num}",
            hypotheses_payload,
            "improvement-orchestrator",
        )
        return hypotheses

    async def _generate_hypothesis_pool(
        self,
        state: PipelineState,
        *,
        pool_size: int,
        round_num: int,
        user_hints: list[str] | None,
    ) -> list[ImprovementHypothesis]:
        """Generate a larger pool of scored hypotheses for adaptive selection."""
        from backend.agents.prompts.improvement import ADAPTIVE_POOL_GENERATION_PROMPT

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
            ADAPTIVE_POOL_GENERATION_PROMPT.format(pool_size=pool_size)
            + f"{hints_str}\n"
            f"Generate {pool_size} candidate hypotheses for project {self.project_id}.\n"
            f"Context:\n```json\n{json.dumps(context, indent=2)}\n```"
        )
        output = await self._invoke_agent("improvement-orchestrator", prompt)
        data = self._extract_json(output)
        hypotheses = [ImprovementHypothesis(**h) for h in data.get("hypotheses", [])]
        # Sort by expected value descending
        hypotheses.sort(key=lambda h: h.expected_value_score, reverse=True)
        self._audit_step(
            state,
            target=f"adaptive-pool:round_{round_num}",
            structured_output={"pool_size": len(hypotheses), "hypotheses": [h.model_dump() for h in hypotheses]},
        )
        return hypotheses

    async def _rerank_remaining_pool(
        self,
        state: PipelineState,
        *,
        completed_results: list[PathResult],
        remaining: list[ImprovementHypothesis],
    ) -> list[ImprovementHypothesis]:
        """Re-rank remaining pool hypotheses based on completed batch results."""
        from backend.agents.prompts.improvement import ADAPTIVE_RERANK_PROMPT

        completed_desc = "\n".join(
            f"- **{r.path_id}** [{'SUCCESS' if r.success else 'FAILED'}]: {r.hypothesis}\n"
            f"  Metrics: {json.dumps(r.metrics)}\n"
            f"  {'Failure: ' + r.failure_notes if r.failure_notes else ''}"
            for r in completed_results
        )
        remaining_desc = json.dumps(
            [h.model_dump() for h in remaining], indent=2,
        )
        prompt = ADAPTIVE_RERANK_PROMPT.format(
            n_completed=len(completed_results),
            completed_results=completed_desc,
            remaining_candidates=remaining_desc,
        )
        output = await self._invoke_agent("improvement-orchestrator", prompt)
        data = self._extract_json(output)
        reranked = [ImprovementHypothesis(**h) for h in data.get("hypotheses", [])]
        reranked.sort(key=lambda h: h.expected_value_score, reverse=True)
        if not reranked:
            return remaining  # fallback if LLM returns empty
        return reranked

    async def _run_adaptive_batches(
        self,
        state: PipelineState,
        *,
        pool: list[ImprovementHypothesis],
        n_to_run: int,
        round_num: int,
        baseline_metrics: dict[str, Any],
    ) -> list[PathResult]:
        """Run hypotheses in adaptive batches.

        Splits the pool into batches sized by concurrency limit. After each
        batch, re-ranks the remaining pool using the LLM, then selects the
        next batch from the top of the re-ranked pool.
        """
        concurrency = self.execution_profile.max_concurrent_agents
        batch_size = min(concurrency, n_to_run)
        completed = self._load_completed_path_results()
        all_results: list[PathResult] = []
        remaining_pool = list(pool)
        paths_run = 0

        batch_num = 0
        while paths_run < n_to_run and remaining_pool:
            batch_num += 1
            # Select next batch from top of remaining pool
            this_batch_size = min(batch_size, n_to_run - paths_run, len(remaining_pool))
            batch_hypotheses = remaining_pool[:this_batch_size]
            remaining_pool = remaining_pool[this_batch_size:]

            # Skip already-completed paths
            to_run = [h for h in batch_hypotheses if h.path_id not in completed]
            already_done = [
                completed[h.path_id] for h in batch_hypotheses if h.path_id in completed
            ]
            all_results.extend(already_done)
            paths_run += len(already_done)

            if to_run:
                print(
                    f"  [adaptive batch {batch_num}] Running {len(to_run)} paths "
                    f"(pool remaining: {len(remaining_pool)})",
                    file=sys.stderr,
                    flush=True,
                )
                semaphore = asyncio.Semaphore(concurrency)
                tasks = [
                    self._run_single_improvement_path(h, state, semaphore)
                    for h in to_run
                ]
                batch_results = list(await asyncio.gather(*tasks))
                all_results.extend(batch_results)
                paths_run += len(batch_results)

                # Re-rank remaining pool if there are more batches to run
                if remaining_pool and paths_run < n_to_run:
                    print(
                        f"  [adaptive] Re-ranking {len(remaining_pool)} remaining "
                        f"candidates based on {len(all_results)} results...",
                        file=sys.stderr,
                        flush=True,
                    )
                    remaining_pool = await self._rerank_remaining_pool(
                        state,
                        completed_results=all_results,
                        remaining=remaining_pool,
                    )
                    # Refresh completed cache after re-ranking
                    completed = self._load_completed_path_results()

        return all_results

    def _build_prior_rounds_summary(self, rounds: list[ImprovementRound]) -> str:
        """Build a concise text summary of all prior rounds for the next-round prompt."""
        lines: list[str] = []
        for rnd in rounds:
            lines.append(f"## Round {rnd.round_number}")
            lines.append(f"Baseline: {rnd.baseline_path_id or 'original'} | Metrics: {json.dumps(rnd.baseline_metrics)}")
            for pr in rnd.path_results:
                status = "SUCCESS" if pr.success else "FAILED"
                lines.append(
                    f"  - {pr.path_id} [{status}]: {pr.hypothesis}"
                )
                if pr.metrics:
                    lines.append(f"    Metrics: {json.dumps(pr.metrics)}")
                if pr.failure_notes:
                    lines.append(f"    Failure: {pr.failure_notes}")
                if pr.recommendation:
                    lines.append(f"    Recommendation: {pr.recommendation}")
            if rnd.best_path_id:
                lines.append(
                    f"  Winner: {rnd.best_path_id} (+{rnd.improvement_pct:.2f}%)"
                    if rnd.improvement_pct is not None
                    else f"  Winner: {rnd.best_path_id}"
                )
            else:
                lines.append("  No winner — no path improved over baseline")
            lines.append("")
        return "\n".join(lines)

    def _evaluate_round(
        self,
        results: list[PathResult],
        baseline_metrics: dict[str, Any],
    ) -> tuple[PathResult | None, dict[str, Any], float | None]:
        """Find the best-performing path from a round's results.

        Returns (best_path, best_metrics, improvement_pct).
        improvement_pct is relative to the round's baseline.
        Uses a simple heuristic: average % improvement across all shared
        numeric metrics (higher is better assumed unless metric name
        contains 'loss', 'error', or 'time').
        """
        best_path: PathResult | None = None
        best_improvement: float | None = None
        best_metrics: dict[str, Any] = {}

        for result in results:
            if not result.success or not result.metrics:
                continue
            improvement = self._compute_aggregate_improvement(
                baseline_metrics, result.metrics,
            )
            if improvement is not None and improvement > 0:
                if best_improvement is None or improvement > best_improvement:
                    best_improvement = improvement
                    best_path = result
                    best_metrics = dict(result.metrics)

        return best_path, best_metrics, best_improvement

    @staticmethod
    def _compute_aggregate_improvement(
        baseline: dict[str, Any],
        candidate: dict[str, Any],
    ) -> float | None:
        """Compute average % improvement across shared numeric metrics.

        Metrics whose name contains 'loss', 'error', 'time', or 'cost'
        are treated as lower-is-better (improvement = baseline - candidate).
        All others are higher-is-better.
        """
        _LOWER_IS_BETTER = {"loss", "error", "time", "cost", "perplexity", "mse", "mae"}
        deltas: list[float] = []
        for key, b_val in baseline.items():
            if key not in candidate:
                continue
            try:
                b = float(b_val)
                c = float(candidate[key])
            except (TypeError, ValueError):
                continue
            if b == 0:
                continue
            lower = any(tok in key.lower() for tok in _LOWER_IS_BETTER)
            if lower:
                pct = ((b - c) / abs(b)) * 100
            else:
                pct = ((c - b) / abs(b)) * 100
            deltas.append(pct)
        if not deltas:
            return None
        return sum(deltas) / len(deltas)

    # ------------------------------------------------------------------
    # Composition phase: combine winning paths
    # ------------------------------------------------------------------

    async def run_composition(self, state: PipelineState) -> PipelineState:
        """Compose independently-successful improvement paths.

        1. If < 2 successful paths, skip.
        2. Try full composition of all winners.
        3. If full composition beats best individual, accept it.
        4. Otherwise, greedy forward selection: start with best individual,
           try adding each remaining winner, keep what helps.
        """
        baseline_metrics: dict[str, Any] = (
            state.experiment_artifacts.metrics if state.experiment_artifacts else {}
        )

        # Collect all successful paths
        winners = [r for r in state.path_results if r.success and r.metrics]
        if len(winners) < 2:
            state.composition_phase = CompositionPhase(
                winning_path_ids=[w.path_id for w in winners],
                strategy_used="skipped",
            )
            state.decision_log.append(
                f"Composition skipped: only {len(winners)} successful path(s)"
            )
            state.stage = PipelineStage.COMPOSITION_TESTED
            return state

        logger.info(
            "[composition] Composing %d winning paths", len(winners),
        )
        print(
            f"\n{'='*50}\n"
            f"  > Composition Phase: {len(winners)} winners\n"
            f"{'='*50}",
            file=sys.stderr,
            flush=True,
        )

        # Find the best individual path for comparison
        best_individual = max(
            winners,
            key=lambda w: self._compute_aggregate_improvement(
                baseline_metrics, w.metrics,
            ) or 0.0,
        )
        best_individual_improvement = (
            self._compute_aggregate_improvement(baseline_metrics, best_individual.metrics) or 0.0
        )
        winning_ids = [w.path_id for w in winners]

        phase = CompositionPhase(
            winning_path_ids=winning_ids,
            strategy_used="full_only",
        )

        # --- Step 1: Full composition ---
        full_result = await self._run_composition_attempt(
            state=state,
            attempt_id="compose_all",
            paths_to_compose=winners,
            baseline_metrics=baseline_metrics,
            best_individual_improvement=best_individual_improvement,
        )
        phase.full_composition = full_result

        if full_result.success and (
            full_result.improvement_pct_vs_best_individual is not None
            and full_result.improvement_pct_vs_best_individual > 0
        ):
            # Full composition is better than best individual — done
            phase.best_composition = full_result
            state.decision_log.append(
                f"Composition: full combo of {winning_ids} beats best individual "
                f"by {full_result.improvement_pct_vs_best_individual:+.2f}%"
            )
            print(
                f"  [composition] Full combo WINS "
                f"(+{full_result.improvement_pct_vs_best_individual:.2f}% vs best individual)",
                file=sys.stderr,
                flush=True,
            )
        else:
            # Full composition didn't help — greedy forward selection
            print(
                f"  [composition] Full combo did not improve over best individual — "
                f"running greedy ablation",
                file=sys.stderr,
                flush=True,
            )
            phase.strategy_used = "greedy_ablation"
            best_composition = await self._greedy_composition_search(
                state=state,
                winners=winners,
                best_individual=best_individual,
                baseline_metrics=baseline_metrics,
                best_individual_improvement=best_individual_improvement,
                phase=phase,
            )
            phase.best_composition = best_composition

        # Add the best composition as a PathResult so it's included in gate 3 and final report
        if phase.best_composition and phase.best_composition.success:
            composed_path_result = PathResult(
                path_id=phase.best_composition.attempt_id,
                hypothesis=f"Composition of {phase.best_composition.composed_path_ids}",
                diff_summary=phase.best_composition.diff_summary,
                metrics=phase.best_composition.metrics,
                success=True,
                recommendation=(
                    f"Composed result: {phase.best_composition.improvement_pct_vs_baseline:+.2f}% "
                    f"vs baseline"
                    if phase.best_composition.improvement_pct_vs_baseline is not None
                    else "Composed result"
                ),
            )
            state.path_results.append(composed_path_result)

        state.composition_phase = phase
        state.stage = PipelineStage.COMPOSITION_TESTED
        state.save_checkpoint(self.runs_root)
        self._enrich_workspace(
            "composition_phase",
            phase.model_dump(),
            "composition-agent",
        )
        return state

    async def _run_composition_attempt(
        self,
        *,
        state: PipelineState,
        attempt_id: str,
        paths_to_compose: list[PathResult],
        baseline_metrics: dict[str, Any],
        best_individual_improvement: float,
    ) -> CompositionAttempt:
        """Run a single composition attempt combining the given paths."""
        from backend.agents.prompts.improvement import COMPOSITION_AGENT_PROMPT

        path_ids = [p.path_id for p in paths_to_compose]
        compose_dir = self._project_dir / "compositions" / attempt_id
        compose_dir.mkdir(parents=True, exist_ok=True)

        # Build per-path description for the prompt
        path_descriptions: list[str] = []
        for p in paths_to_compose:
            path_descriptions.append(
                f"- **{p.path_id}**: {p.hypothesis}\n"
                f"  Diff: {p.diff_summary}\n"
                f"  Metrics: {json.dumps(p.metrics)}\n"
                f"  Code dir: {self._project_dir / 'improvements' / p.path_id}"
            )

        prompt = COMPOSITION_AGENT_PROMPT.format(
            paths_to_compose="\n".join(path_descriptions),
            compose_dir=compose_dir,
            compose_id=attempt_id,
            path_id_list=", ".join(path_ids),
        )
        prompt = (
            f"{prompt}\n\n"
            f"Baseline code: {self._project_dir / 'code'}\n"
            f"Environment:\n```json\n"
            f"{state.environment_spec.model_dump_json(indent=2) if state.environment_spec else '{}'}"
            f"\n```"
        )

        print(
            f"  [composition] Attempting {attempt_id}: {path_ids}",
            file=sys.stderr,
            flush=True,
        )

        try:
            output = await self._invoke_agent(
                "improvement-path", prompt, cwd=compose_dir,
            )
            data = self._extract_json(output)
            result_metrics = data.get("metrics", {})
            success = data.get("success", False)

            improvement_vs_baseline = self._compute_aggregate_improvement(
                baseline_metrics, result_metrics,
            )
            improvement_vs_best = (
                (improvement_vs_baseline - best_individual_improvement)
                if improvement_vs_baseline is not None
                else None
            )

            attempt = CompositionAttempt(
                attempt_id=attempt_id,
                composed_path_ids=path_ids,
                metrics=result_metrics,
                improvement_pct_vs_baseline=improvement_vs_baseline,
                improvement_pct_vs_best_individual=improvement_vs_best,
                success=success,
                diff_summary=data.get("diff_summary", ""),
                failure_notes=data.get("failure_notes", ""),
            )
            self._audit_step(
                state,
                target=f"composition:{attempt_id}",
                structured_output=attempt.model_dump(),
            )
            print(
                f"  [composition] {attempt_id}: "
                f"{'OK' if success else 'FAILED'}"
                f" | vs baseline: {improvement_vs_baseline:+.2f}%"
                if improvement_vs_baseline is not None
                else f"  [composition] {attempt_id}: no metrics",
                file=sys.stderr,
                flush=True,
            )
            return attempt

        except Exception as exc:
            logger.warning("Composition attempt %s failed: %s", attempt_id, exc)
            return CompositionAttempt(
                attempt_id=attempt_id,
                composed_path_ids=path_ids,
                success=False,
                failure_notes=str(exc),
            )

    async def _greedy_composition_search(
        self,
        *,
        state: PipelineState,
        winners: list[PathResult],
        best_individual: PathResult,
        baseline_metrics: dict[str, Any],
        best_individual_improvement: float,
        phase: CompositionPhase,
    ) -> CompositionAttempt | None:
        """Greedy forward selection: start with best individual, add winners one-by-one.

        At each step, try adding each remaining winner. Keep the addition
        that gives the biggest improvement. Stop when no addition helps.
        """
        current_set = [best_individual]
        current_improvement = best_individual_improvement
        remaining = [w for w in winners if w.path_id != best_individual.path_id]
        best_composition: CompositionAttempt | None = None
        step = 0

        while remaining:
            step += 1
            best_addition: PathResult | None = None
            best_attempt: CompositionAttempt | None = None
            best_new_improvement: float = current_improvement

            for candidate in remaining:
                trial_set = current_set + [candidate]
                trial_ids = [p.path_id for p in trial_set]
                attempt_id = f"greedy_s{step}_{'_'.join(trial_ids)}"

                attempt = await self._run_composition_attempt(
                    state=state,
                    attempt_id=attempt_id,
                    paths_to_compose=trial_set,
                    baseline_metrics=baseline_metrics,
                    best_individual_improvement=best_individual_improvement,
                )
                phase.ablation_attempts.append(attempt)

                if (
                    attempt.success
                    and attempt.improvement_pct_vs_baseline is not None
                    and attempt.improvement_pct_vs_baseline > best_new_improvement
                ):
                    best_addition = candidate
                    best_attempt = attempt
                    best_new_improvement = attempt.improvement_pct_vs_baseline

            if best_addition is None or best_attempt is None:
                # No addition helped — stop
                state.decision_log.append(
                    f"Greedy composition stopped at step {step}: "
                    f"no addition improved over current set {[p.path_id for p in current_set]}"
                )
                break

            # Accept the best addition
            current_set.append(best_addition)
            current_improvement = best_new_improvement
            best_composition = best_attempt
            remaining = [w for w in remaining if w.path_id != best_addition.path_id]
            state.decision_log.append(
                f"Greedy composition step {step}: added {best_addition.path_id} "
                f"(+{best_new_improvement:.2f}% vs baseline)"
            )
            print(
                f"  [composition] Greedy step {step}: +{best_addition.path_id} "
                f"→ {best_new_improvement:+.2f}% vs baseline",
                file=sys.stderr,
                flush=True,
            )

        return best_composition

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

    def _generate_final_report(self, state: PipelineState) -> PipelineState:
        """Generate the final delta report summarizing all parallel improvement paths."""
        logger.info("[Final] Generating delta report across all improvement paths")
        print(f"\n{'='*50}", file=sys.stderr, flush=True)
        print(f"  > Generating Final Report", file=sys.stderr, flush=True)
        print(f"{'='*50}", file=sys.stderr, flush=True)

        report = generate_final_report(
            project_id=self.project_id,
            paper_claim_map=state.paper_claim_map,
            experiment_artifacts=state.experiment_artifacts,
            improvement_hypotheses=state.improvement_hypotheses,
            path_results=state.path_results,
            research_map=state.research_map,
        )
        state.final_report = report

        # Write to disk
        json_path, md_path = write_final_report(report, self._project_dir)

        # Enrich workspace
        self._enrich_workspace(
            "final_report",
            report.model_dump(),
            "report-generator",
        )

        print(f"  [report] Reproduction score: {report.reproduction_score:.2f}", file=sys.stderr, flush=True)
        print(f"  [report] Paths: {report.paths_succeeded}/{report.total_paths_run} succeeded, "
              f"{report.paths_improved_over_baseline} improved over baseline", file=sys.stderr, flush=True)
        if report.best_path_id:
            print(f"  [report] Best: {report.best_path_id} (+{report.best_overall_improvement_pct:.2f}%)",
                  file=sys.stderr, flush=True)
        print(f"  [report] Verdict: {report.overall_verdict}", file=sys.stderr, flush=True)
        print(f"  [report] Written: {json_path}", file=sys.stderr, flush=True)

        return state

    def _format_gate_feedback(self, gate: GateDecision, report_entry: str) -> str:
        """Build a feedback prompt from a failed gate's findings."""
        lines = [
            f"GATE FAILED: {gate.gate} — status: {gate.status.value}",
            f"Decision log: {report_entry}",
        ]
        if gate.blocking_issues:
            lines.append("Blocking issues:")
            for issue in gate.blocking_issues:
                lines.append(f"  - {issue}")
        return "\n".join(lines)

    async def _retry_gate_1(self, state: PipelineState) -> PipelineState:
        """Retry the planning phase after gate 1 failure.

        Re-runs reproduction planner and environment detective with
        feedback from the gate's verifier findings, then re-verifies.
        """
        max_retries = self.execution_profile.max_gate_retries
        for attempt in range(1, max_retries + 1):
            if state.gate_1 and state.gate_1.passed:
                return state

            feedback = self._format_gate_feedback(
                state.gate_1,  # type: ignore[arg-type]
                state.decision_log[-1] if state.decision_log else "",
            )
            print(
                f"\n  [gate-retry] Gate 1 retry {attempt}/{max_retries}",
                file=sys.stderr,
                flush=True,
            )
            state.decision_log.append(
                f"Gate 1 retry {attempt}/{max_retries}: "
                f"re-running planner + environment with feedback"
            )

            # Re-run environment detective with feedback
            env_context = {
                "paper_claim_map": state.paper_claim_map.model_dump() if state.paper_claim_map else {},
                "artifact_index": state.artifact_index or {},
            }
            env_prompt = (
                f"Fix the Docker environment for project {self.project_id}.\n"
                f"The verification gate found problems:\n\n{feedback}\n\n"
                f"Context:\n```json\n{json.dumps(env_context, indent=2)}\n```\n"
                f"Read the current Dockerfile and environment_spec.json, fix the "
                f"issues, and write corrected versions to {self._project_dir}/.\n"
                f"Return the updated environment_spec JSON."
            )
            env_output = await self._invoke_agent("environment-detective", env_prompt)
            try:
                data = self._extract_json(
                    env_output,
                    fallback_file=str(self._project_dir / "environment_spec.json"),
                )
                state.environment_spec = EnvironmentSpec(**data)
            except Exception:
                logger.warning("Could not parse env spec from gate 1 retry")

            # Re-run reproduction planner with feedback
            plan_context = {
                "paper_claim_map": state.paper_claim_map.model_dump() if state.paper_claim_map else {},
                "environment_spec": state.environment_spec.model_dump() if state.environment_spec else {},
                "assumption_ledger": state.assumption_ledger,
            }
            plan_prompt = (
                f"Fix the reproduction plan for project {self.project_id}.\n"
                f"The verification gate found problems:\n\n{feedback}\n\n"
                f"Context:\n```json\n{json.dumps(plan_context, indent=2)}\n```\n"
                f"Write the corrected reproduction_contract.json to {self._project_dir}/.\n"
                f"Return the updated reproduction contract JSON."
            )
            plan_output = await self._invoke_agent("reproduction-planner", plan_prompt)
            try:
                data = self._normalize_reproduction_contract(
                    self._extract_json(
                        plan_output,
                        fallback_file=str(self._project_dir / "reproduction_contract.json"),
                    )
                )
                state.reproduction_contract = ReproductionContract(**data)
            except Exception:
                logger.warning("Could not parse contract from gate 1 retry")

            # Re-verify
            state.gate_1 = None
            state = await self.run_gate_1(state)

        return state

    async def _retry_gate_2(self, state: PipelineState) -> PipelineState:
        """Retry the baseline phase after gate 2 failure.

        Re-runs baseline implementation with feedback from the gate's
        verifier findings, re-runs the experiment, then re-verifies.
        """
        max_retries = self.execution_profile.max_gate_retries
        for attempt in range(1, max_retries + 1):
            if state.gate_2 and state.gate_2.passed:
                return state

            feedback = self._format_gate_feedback(
                state.gate_2,  # type: ignore[arg-type]
                state.decision_log[-1] if state.decision_log else "",
            )
            print(
                f"\n  [gate-retry] Gate 2 retry {attempt}/{max_retries}",
                file=sys.stderr,
                flush=True,
            )
            state.decision_log.append(
                f"Gate 2 retry {attempt}/{max_retries}: "
                f"re-running baseline implementation + experiment with feedback"
            )

            # Re-run baseline implementation with feedback
            code_dir = self._project_dir / "code"
            code_dir.mkdir(parents=True, exist_ok=True)
            context = {
                "paper_claim_map": state.paper_claim_map.model_dump() if state.paper_claim_map else {},
                "reproduction_contract": state.reproduction_contract.model_dump() if state.reproduction_contract else {},
                "environment_spec": state.environment_spec.model_dump() if state.environment_spec else {},
                "artifact_index": state.artifact_index or {},
                "assumption_ledger": state.assumption_ledger,
            }
            baseline_prompt = (
                f"Fix the baseline implementation for project {self.project_id}.\n"
                f"The verification gate found problems:\n\n{feedback}\n\n"
                f"Context:\n```json\n{json.dumps(context, indent=2)}\n```\n"
                f"Read the current code in {code_dir}, fix the issues listed above, "
                f"and return the updated baseline_result JSON."
            )
            output = await self._invoke_agent(
                "baseline-implementation", baseline_prompt, cwd=code_dir,
            )
            try:
                data = self._extract_json(
                    output,
                    fallback_file=str(self._project_dir / "baseline_result.json"),
                )
                state.baseline_result = BaselineResult(**data)
            except Exception:
                logger.warning("Could not parse baseline result from gate 2 retry")

            # Re-run experiment
            state.experiment_artifacts = None
            state = await self.run_experiment(state)

            # Re-verify
            state.gate_2 = None
            state = await self.run_gate_2(state)

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

            # Check gate results — retry with feedback before giving up
            if state.gate_1 and not state.gate_1.passed:
                print(f"  ! Gate 1 FAILED: {state.gate_1.status.value}", file=sys.stderr, flush=True)
                if self.execution_profile.max_gate_retries > 0:
                    state = await self._retry_gate_1(state)
                    current_idx = stages_order.index(state.stage)
                if state.gate_1 and not state.gate_1.passed:
                    print(f"  X Gate 1 FAILED after retries: {state.gate_1.status.value}", file=sys.stderr, flush=True)
                    return state
            if state.gate_2 and not state.gate_2.passed:
                print(f"  ! Gate 2 FAILED: {state.gate_2.status.value}", file=sys.stderr, flush=True)
                if self.execution_profile.max_gate_retries > 0:
                    state = await self._retry_gate_2(state)
                    current_idx = stages_order.index(state.stage)
                if state.gate_2 and not state.gate_2.passed:
                    print(f"  X Gate 2 FAILED after retries: {state.gate_2.status.value}", file=sys.stderr, flush=True)
                    return state

        # Improvement phase
        if current_idx < stages_order.index(PipelineStage.IMPROVEMENTS_RUN):
            state = await self.run_improvements(
                state, user_hints=user_hints, n_paths=n_improvement_paths,
            )
            current_idx = stages_order.index(state.stage)

        # Composition phase: combine winning paths
        if current_idx < stages_order.index(PipelineStage.COMPOSITION_TESTED):
            state = await self.run_composition(state)
            current_idx = stages_order.index(state.stage)

        if current_idx < stages_order.index(PipelineStage.GATE_3_PASSED):
            state = await self.run_gate_3(state)
            current_idx = stages_order.index(state.stage)

        if current_idx < stages_order.index(PipelineStage.RESEARCH_MAP_GENERATED):
            state = await self.generate_research_map(state)

        # Final report: compute deltas across all parallel improvement paths
        state = self._generate_final_report(state)

        self._close_workspace("pipeline_complete")
        logger.info("Pipeline complete for project %s", self.project_id)
        return state
