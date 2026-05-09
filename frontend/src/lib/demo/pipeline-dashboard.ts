import type {
  AgentNode,
  DashboardEvent,
  DashboardSnapshot,
  ProgressStatus
} from "@/lib/events/contract";
import type { DemoProvider } from "./demo-run-types";

type GateStatus =
  | "verified"
  | "verified_with_caveats"
  | "partial_reproduction"
  | "failed_reproduction"
  | "blocked_requires_human"
  | "invalid_claim";

interface GateDecisionLike {
  passed?: boolean;
  status?: GateStatus;
}

interface PaperClaimMapLike {
  core_contribution?: string;
  datasets?: Array<{ name?: string }>;
  metrics?: Array<{ name?: string; definition?: string }>;
  ambiguities?: Array<{
    assumption_id?: string;
    detail?: string;
    evidence?: string[];
    risk?: string;
  }>;
}

interface EnvironmentSpecLike {
  python_version?: string;
  framework?: string;
  framework_version?: string | Record<string, string>;
  assumptions?: Array<{
    assumption_id?: string;
    detail?: string;
    chosen_value?: string;
    risk?: string;
  }>;
}

interface BaselineResultLike {
  mode?: string;
  assumptions_applied?: string[];
}

interface ExperimentArtifactsLike {
  success?: boolean;
  metrics?: Record<string, unknown>;
  plots?: string[];
  log_path?: string;
  commands_log_path?: string;
  provenance_path?: string;
  error_message?: string;
}

interface PathResultLike {
  path_id?: string;
  hypothesis?: string;
  success?: boolean;
  metrics?: Record<string, unknown>;
  failure_notes?: string;
}

interface ResearchMapLike {
  promising_directions?: string[];
  dead_ends?: string[];
  inconclusive?: string[];
  next_experiments?: string[];
  overall_reproducibility_assessment?: string;
}

export interface PipelineStateDocument {
  project_id: string;
  stage: string;
  paper_claim_map?: PaperClaimMapLike;
  environment_spec?: EnvironmentSpecLike;
  baseline_result?: BaselineResultLike;
  experiment_artifacts?: ExperimentArtifactsLike;
  gate_1?: GateDecisionLike;
  gate_2?: GateDecisionLike;
  gate_3?: GateDecisionLike;
  path_results?: PathResultLike[];
  research_map?: ResearchMapLike;
  assumption_ledger?: Array<{ assumption_id?: string }>;
  decision_log?: string[];
}

export interface LiveDemoMeta {
  projectId: string;
  outputDir: string;
  sourceKind: "repo_pdf" | "workspace_fixture";
  runMode: "offline" | "sdk";
  llmProvider?: DemoProvider;
  sourceLabel: string;
  sourceNote: string;
}

export interface LiveDemoPayload extends LiveDemoMeta {
  generatedAt: string;
  log: string;
  initialSnapshot: DashboardSnapshot;
  events: DashboardEvent[];
  summary: {
    stage: string;
    meanReward: number | null;
    improvementCount: number;
    runModeLabel: string;
    llmProvider?: DemoProvider;
    sourceLabel: string;
  };
}

function runModeLabel(runMode: LiveDemoMeta["runMode"], provider?: DemoProvider): string {
  if (runMode !== "sdk") {
    return "Offline";
  }
  if (!provider) {
    return "SDK";
  }
  return provider === "openai" ? "SDK: OpenAI" : "SDK: Anthropic";
}

function toStatusTone(status?: GateStatus): ProgressStatus {
  switch (status) {
    case "verified":
      return "passed";
    case "verified_with_caveats":
    case "partial_reproduction":
      return "caveat";
    case "failed_reproduction":
    case "invalid_claim":
    case "blocked_requires_human":
      return "failed";
    default:
      return "pending";
  }
}

function buildRootAgent(meta: LiveDemoMeta, state: PipelineStateDocument): AgentNode {
  const complete = state.stage === "complete";

  return {
    id: "root-orchestrator",
    label: "Root Orchestrator",
    type: "orchestrator",
    status: complete ? "completed" : "running",
    currentTask: complete
      ? "Published the final research map and closed the demo run"
      : `Driving pipeline stage: ${state.stage}`,
    lastUpdated: "2026-05-09T15:00:00.000Z",
    outputTargetIds: [
      "paper-understanding",
      "environment-detective",
      "baseline-implementation",
      "experiment-runner",
      "supervisor-verifier"
    ],
    contextVariables: [meta.sourceKind === "repo_pdf" ? "raw_paper_pdf" : "ppo_workspace_fixture"]
  };
}

function buildInitialSnapshot(
  meta: LiveDemoMeta,
  state: PipelineStateDocument
): DashboardSnapshot {
  const ambiguityCount = state.paper_claim_map?.ambiguities?.length ?? 0;
  const assumptionCount = state.assumption_ledger?.length ?? 0;
  const improvementCount = state.path_results?.length ?? 0;
  const meanReward = state.experiment_artifacts?.metrics?.mean_reward;

  const planStatus = state.gate_1
    ? toStatusTone(state.gate_1.status)
    : state.stage === "plan_created" ||
        state.stage === "gate_1_passed" ||
        state.stage === "baseline_implemented" ||
        state.stage === "baseline_run" ||
        state.stage === "gate_2_passed" ||
        state.stage === "improvements_selected" ||
        state.stage === "improvements_run" ||
        state.stage === "gate_3_passed" ||
        state.stage === "research_map_generated" ||
        state.stage === "complete"
      ? "passed"
      : "pending";

  const baselineStatus = state.gate_2
    ? toStatusTone(state.gate_2.status)
    : state.stage === "baseline_run" ||
        state.stage === "gate_2_passed" ||
        state.stage === "improvements_selected" ||
        state.stage === "improvements_run" ||
        state.stage === "gate_3_passed" ||
        state.stage === "research_map_generated" ||
        state.stage === "complete"
      ? "passed"
      : "pending";

  const improvementStatus = state.gate_3
    ? toStatusTone(state.gate_3.status)
    : state.stage === "improvements_run" ||
        state.stage === "gate_3_passed" ||
        state.stage === "research_map_generated" ||
        state.stage === "complete"
      ? "passed"
      : "pending";

  return {
    agents: [buildRootAgent(meta, state)],
    reasoning: [],
    messages: [],
    citations: [],
    approvals: [],
    progress: [
      {
        stage: "plan",
        status: planStatus,
        detail: state.gate_1?.passed
          ? "Gate 1 passed. The plan is ready for baseline work."
          : "Waiting for Gate 1 verification."
      },
      {
        stage: "baseline",
        status: baselineStatus,
        detail: state.gate_2?.passed
          ? "Gate 2 passed. The baseline unlocked improvement work."
          : "Waiting for Gate 2 verification."
      },
      {
        stage: "improvement",
        status: improvementStatus,
        detail: state.gate_3?.passed
          ? "Gate 3 passed. The research map is ready."
          : "Waiting for Gate 3 verification."
      }
    ],
    dataPanels: [
      {
        id: "claims",
        title: "Claim Map",
        summary: `${ambiguityCount} ${ambiguityCount === 1 ? "ambiguity" : "ambiguities"} extracted from the paper context.`,
        items: [
          state.paper_claim_map?.core_contribution ?? "Core contribution pending.",
          `Dataset: ${state.paper_claim_map?.datasets?.[0]?.name ?? "Unknown"}`,
          `Metric: ${state.paper_claim_map?.metrics?.[0]?.name ?? "Unknown"}`
        ]
      },
      {
        id: "assumptions",
        title: "Assumption Ledger",
        summary: `${assumptionCount} assumption${assumptionCount === 1 ? "" : "s"} tracked through the run.`,
        items: [
          `Implementation mode: ${state.baseline_result?.mode ?? "pending"}`,
          `Applied assumptions: ${(state.baseline_result?.assumptions_applied ?? []).join(", ") || "Pending"}`,
          `Run mode: ${runModeLabel(meta.runMode, meta.llmProvider)}`,
          meta.sourceNote
        ]
      },
      {
        id: "artifacts",
        title: "Artifact Watch",
        summary: meanReward
          ? `Baseline reward captured at ${meanReward}.`
          : "Baseline artifacts have not been generated yet.",
        items: [
          `Output directory: ${meta.outputDir}`,
          `Improvement paths: ${improvementCount}`,
          `Research map: ${state.research_map ? "available" : "pending"}`
        ]
      }
    ]
  };
}

function isoAt(step: number): string {
  return new Date(Date.UTC(2026, 4, 9, 15, 0, step)).toISOString();
}

function eventAgent(
  id: string,
  label: string,
  type: AgentNode["type"],
  status: AgentNode["status"],
  currentTask: string,
  parentId = "root-orchestrator",
  contextVariables: string[] = [],
  outputTargetIds: string[] = ["root-orchestrator"]
): AgentNode {
  return {
    id,
    label,
    type,
    status,
    parentId,
    currentTask,
    lastUpdated: "2026-05-09T15:00:00.000Z",
    outputTargetIds,
    contextVariables
  };
}

export function buildLiveDemoDashboard(
  state: PipelineStateDocument,
  meta: LiveDemoMeta,
  log = ""
): LiveDemoPayload {
  const events: DashboardEvent[] = [];
  const ambiguities = state.paper_claim_map?.ambiguities ?? [];
  const envAssumptions = state.environment_spec?.assumptions ?? [];
  const pathResults = state.path_results ?? [];
  const meanRewardValue = state.experiment_artifacts?.metrics?.mean_reward;
  const meanReward =
    typeof meanRewardValue === "number" ? meanRewardValue : Number(meanRewardValue ?? NaN);

  events.push({
    event: "agent_started",
    timestamp: isoAt(0),
    agent: buildRootAgent(meta, { ...state, stage: "ingested" })
  });

  events.push({
    event: "agent_started",
    timestamp: isoAt(1),
    agent: eventAgent(
      "paper-understanding",
      "Paper Understanding",
      "builder",
      "running",
      "Extracting claims, datasets, metrics, and ambiguities from the paper context",
      "root-orchestrator",
      ["paper_excerpt_bundle"]
    )
  });
  events.push({
    event: "agent_reasoning_step",
    timestamp: isoAt(2),
    agentId: "paper-understanding",
    agentLabel: "Paper Understanding",
    stepType: "analysis",
    title: "Recovered the paper contract",
    detail: `${state.paper_claim_map?.core_contribution ?? "Core contribution unavailable"}${ambiguities.length ? ` with ${ambiguities.length} ambiguity check${ambiguities.length === 1 ? "" : "s"}` : ""}.`,
    citations: ambiguities.slice(0, 2).map((ambiguity, index) => ({
      id: ambiguity.assumption_id ?? `ambiguity-${index + 1}`,
      label: ambiguity.assumption_id ?? "Paper ambiguity",
      sourceType: "paper_section",
      excerpt: ambiguity.detail ?? "Ambiguity detail unavailable",
      trustLevel: "primary"
    }))
  });
  events.push({
    event: "context_enrichment",
    timestamp: isoAt(3),
    agentId: "paper-understanding",
    variableName: "paper_claim_map",
    summary: "Claim map published to shared context for downstream agents."
  });
  events.push({
    event: "agent_completed",
    timestamp: isoAt(4),
    agent: eventAgent(
      "paper-understanding",
      "Paper Understanding",
      "builder",
      "completed",
      "Claim map, dataset, and metric summaries were published",
      "root-orchestrator",
      ["paper_claim_map", "ambiguity_log"]
    )
  });

  events.push({
    event: "agent_started",
    timestamp: isoAt(5),
    agent: eventAgent(
      "environment-detective",
      "Environment Detective",
      "builder",
      "running",
      "Recovering the runtime and dependency envelope",
      "root-orchestrator",
      ["paper_claim_map"]
    )
  });
  events.push({
    event: "agent_reasoning_step",
    timestamp: isoAt(6),
    agentId: "environment-detective",
    agentLabel: "Environment Detective",
    stepType: "analysis",
    title: "Environment recommendation delivered",
    detail: `Pinned Python ${state.environment_spec?.python_version ?? "unknown"} with ${state.environment_spec?.framework ?? "framework"} ${typeof state.environment_spec?.framework_version === "string" ? state.environment_spec.framework_version : "resolved from assumptions"}.`,
    citations: envAssumptions.slice(0, 2).map((assumption, index) => ({
      id: assumption.assumption_id ?? `env-${index + 1}`,
      label: assumption.assumption_id ?? "Environment assumption",
      sourceType: "repo_file",
      excerpt: assumption.detail ?? "Environment detail unavailable",
      trustLevel: "strong_secondary"
    }))
  });
  events.push({
    event: "shared_state_updated",
    timestamp: isoAt(7),
    agentId: "environment-detective",
    changeType: "message",
    title: "Environment spec returned to orchestration",
    detail: "Docker and package constraints were captured for the baseline run.",
    fromAgentId: "environment-detective",
    toAgentId: "root-orchestrator"
  });
  events.push({
    event: "context_enrichment",
    timestamp: isoAt(8),
    agentId: "environment-detective",
    variableName: "environment_spec",
    summary: "Dockerfile and package versions added to shared context."
  });
  events.push({
    event: "agent_completed",
    timestamp: isoAt(9),
    agent: eventAgent(
      "environment-detective",
      "Environment Detective",
      "builder",
      "completed",
      "Environment specification published for implementation",
      "root-orchestrator",
      ["environment_spec"]
    )
  });

  events.push({
    event: "verification_gate_result",
    timestamp: isoAt(10),
    stage: "plan",
    status: toStatusTone(state.gate_1?.status),
    detail: state.gate_1?.passed
      ? "Gate 1 passed. The plan is ready for baseline work."
      : "Gate 1 did not pass cleanly."
  });

  events.push({
    event: "agent_started",
    timestamp: isoAt(11),
    agent: eventAgent(
      "baseline-implementation",
      "Baseline Implementation",
      "builder",
      "running",
      "Generating the runnable baseline implementation",
      "root-orchestrator",
      ["environment_spec", "paper_claim_map"]
    )
  });
  events.push({
    event: "agent_reasoning_step",
    timestamp: isoAt(12),
    agentId: "baseline-implementation",
    agentLabel: "Baseline Implementation",
    stepType: "analysis",
    title: "Baseline implementation prepared",
    detail: `Mode: ${state.baseline_result?.mode ?? "unknown"}. Applied assumptions: ${(state.baseline_result?.assumptions_applied ?? []).join(", ") || "none"}.`,
    citations: (state.baseline_result?.assumptions_applied ?? []).slice(0, 2).map((id) => ({
      id,
      label: id,
      sourceType: "paper_section",
      excerpt: "Implementation assumption carried from the paper understanding step.",
      trustLevel: "primary"
    }))
  });
  events.push({
    event: "agent_completed",
    timestamp: isoAt(13),
    agent: eventAgent(
      "baseline-implementation",
      "Baseline Implementation",
      "builder",
      "completed",
      "Baseline code and run commands are ready",
      "root-orchestrator",
      ["baseline_result"]
    )
  });

  events.push({
    event: "agent_started",
    timestamp: isoAt(14),
    agent: eventAgent(
      "experiment-runner",
      "Experiment Runner",
      "builder",
      "running",
      "Executing the baseline training job and collecting artifacts",
      "root-orchestrator",
      ["baseline_result"]
    )
  });
  events.push({
    event: "agent_reasoning_step",
    timestamp: isoAt(15),
    agentId: "experiment-runner",
    agentLabel: "Experiment Runner",
    stepType: "analysis",
    title: "Baseline artifacts captured",
    detail: state.experiment_artifacts?.success
      ? `Baseline run succeeded with mean reward ${Number.isFinite(meanReward) ? meanReward : "unknown"}.`
      : `Baseline run failed: ${state.experiment_artifacts?.error_message ?? "unknown error"}`,
    citations: [
      {
        id: "baseline-artifacts",
        label: "Baseline artifacts",
        sourceType: "run_log",
        excerpt: state.experiment_artifacts?.log_path ?? "Run log path unavailable",
        trustLevel: "strong_secondary"
      }
    ]
  });
  events.push({
    event: "context_enrichment",
    timestamp: isoAt(16),
    agentId: "experiment-runner",
    variableName: "baseline_artifacts",
    summary: "Run logs, metrics, provenance, and plots were attached to the shared context."
  });
  events.push({
    event: "agent_completed",
    timestamp: isoAt(17),
    agent: eventAgent(
      "experiment-runner",
      "Experiment Runner",
      "builder",
      "completed",
      "Baseline artifacts persisted for verification",
      "root-orchestrator",
      ["experiment_artifacts"]
    )
  });

  events.push({
    event: "verification_gate_result",
    timestamp: isoAt(18),
    stage: "baseline",
    status: toStatusTone(state.gate_2?.status),
    detail: state.gate_2?.passed
      ? "Gate 2 passed. The baseline is good enough to unlock improvement paths."
      : "Gate 2 did not pass cleanly."
  });

  events.push({
    event: "agent_started",
    timestamp: isoAt(19),
    agent: eventAgent(
      "improvement-orchestrator",
      "Improvement Orchestrator",
      "improvement",
      "running",
      `Selecting ${pathResults.length} path${pathResults.length === 1 ? "" : "s"} to explore`,
      "root-orchestrator",
      ["experiment_artifacts", "paper_claim_map"],
      pathResults.map((path) => `path-${path.path_id ?? "unknown"}`)
    )
  });

  pathResults.forEach((path, index) => {
    const agentId = `path-${path.path_id ?? `unknown-${index + 1}`}`;

    events.push({
      event: "agent_started",
      timestamp: isoAt(20 + index * 3),
      agent: eventAgent(
        agentId,
        `Improvement ${path.path_id ?? index + 1}`,
        "improvement",
        "running",
        path.hypothesis ?? "Exploring improvement hypothesis",
        "improvement-orchestrator",
        ["baseline_artifacts"]
      )
    });
    events.push({
      event: "agent_reasoning_step",
      timestamp: isoAt(21 + index * 3),
      agentId,
      agentLabel: `Improvement ${path.path_id ?? index + 1}`,
      stepType: "analysis",
      title: "Improvement path evaluated",
      detail: path.success
        ? `${path.hypothesis ?? "Path"} reached mean reward ${path.metrics?.mean_reward ?? "unknown"}.`
        : `${path.hypothesis ?? "Path"} failed: ${path.failure_notes ?? "No notes provided."}`,
      citations: [
        {
          id: `${agentId}-result`,
          label: path.path_id ?? `path-${index + 1}`,
          sourceType: "run_log",
          excerpt: path.success
            ? `Improvement delta: ${path.metrics?.improvement ?? "unknown"}`
            : path.failure_notes ?? "Failure notes unavailable",
          trustLevel: path.success ? "strong_secondary" : "secondary"
        }
      ]
    });
    events.push({
      event: path.success ? "agent_completed" : "agent_failed",
      timestamp: isoAt(22 + index * 3),
      agent: eventAgent(
        agentId,
        `Improvement ${path.path_id ?? index + 1}`,
        "improvement",
        path.success ? "completed" : "failed",
        path.success
          ? "Published an improvement result back to the orchestrator"
          : `Failed path recorded: ${path.failure_notes ?? "No notes provided."}`,
        "improvement-orchestrator",
        ["path_result"]
      )
    });
  });

  events.push({
    event: "agent_completed",
    timestamp: isoAt(20 + pathResults.length * 3),
    agent: eventAgent(
      "improvement-orchestrator",
      "Improvement Orchestrator",
      "improvement",
      "completed",
      "All configured improvement paths have finished running",
      "root-orchestrator",
      ["path_results"]
    )
  });

  events.push({
    event: "verification_gate_result",
    timestamp: isoAt(21 + pathResults.length * 3),
    stage: "improvement",
    status: toStatusTone(state.gate_3?.status),
    detail: state.gate_3?.passed
      ? "Gate 3 passed. The research map is ready to review."
      : "Gate 3 did not pass cleanly."
  });

  events.push({
    event: "context_enrichment",
    timestamp: isoAt(22 + pathResults.length * 3),
    agentId: "root-orchestrator",
    variableName: "research_map",
    summary: "Promising directions, dead ends, and next experiments were synthesized."
  });

  events.push({
    event: "agent_completed",
    timestamp: isoAt(23 + pathResults.length * 3),
    agent: buildRootAgent(meta, state)
  });

  return {
    ...meta,
    generatedAt: new Date().toISOString(),
    log,
    initialSnapshot: buildInitialSnapshot(meta, state),
    events,
    summary: {
      stage: state.stage,
      meanReward: Number.isFinite(meanReward) ? meanReward : null,
      improvementCount: pathResults.length,
      runModeLabel: runModeLabel(meta.runMode, meta.llmProvider),
      llmProvider: meta.llmProvider,
      sourceLabel: meta.sourceLabel
    }
  };
}
