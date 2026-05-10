import type {
  AgentNode,
  DashboardEvent,
  DashboardSnapshot,
  HermesPanel,
  ConceptCard,
  ProgressStatus
} from "@/lib/events/contract";
import type {
  DemoExecutionMode,
  DemoGpuMode,
  DemoProvider,
  DemoSandboxMode
} from "./demo-run-types";

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

interface ReproductionContractLike {
  reproduction_definition?: string;
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

type HermesAuditScopeLike = "step" | "checkpoint";
type HermesAuditStatusLike =
  | "grounded"
  | "caveat"
  | "unsupported"
  | "unavailable"
  | "system_error";
type HermesInterventionLike =
  | "annotate"
  | "retry_step"
  | "request_evidence"
  | "downgrade_claim"
  | "suppress_publication"
  | "escalate_human";

interface HermesEvidenceRefLike {
  kind?: string;
  path?: string;
  snippet?: string;
  description?: string;
}

interface HermesAuditReportLike {
  target?: string;
  scope?: HermesAuditScopeLike;
  status?: HermesAuditStatusLike;
  summary?: string;
  findings?: string[];
  unsupported_claims?: string[];
  evidence_refs?: HermesEvidenceRefLike[];
  recommended_intervention?: HermesInterventionLike;
  corrective_note?: string;
  confidence?: string;
  provider?: string;
  error_message?: string;
}

export interface PipelineStateDocument {
  project_id: string;
  stage: string;
  paper_claim_map?: PaperClaimMapLike;
  environment_spec?: EnvironmentSpecLike;
  reproduction_contract?: ReproductionContractLike;
  baseline_result?: BaselineResultLike;
  experiment_artifacts?: ExperimentArtifactsLike;
  gate_1?: GateDecisionLike;
  gate_2?: GateDecisionLike;
  gate_3?: GateDecisionLike;
  path_results?: PathResultLike[];
  research_map?: ResearchMapLike;
  assumption_ledger?: Array<{ assumption_id?: string }>;
  decision_log?: string[];
  hermes_step_reports?: Record<string, HermesAuditReportLike[]>;
  hermes_checkpoint_reports?: Record<string, HermesAuditReportLike[]>;
  hermes_interventions?: Array<{
    target?: string;
    scope?: HermesAuditScopeLike;
    action?: HermesInterventionLike;
    reason?: string;
    status?: HermesAuditStatusLike;
  }>;
}

export interface LiveDemoMeta {
  projectId: string;
  outputDir: string;
  sourceKind: "workspace_fixture" | "uploaded_pdf";
  runMode: "offline" | "sdk";
  llmProvider?: DemoProvider;
  verificationProvider?: DemoProvider;
  executionMode?: DemoExecutionMode;
  sandboxMode?: DemoSandboxMode;
  gpuMode?: DemoGpuMode;
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
    verificationProvider?: DemoProvider;
    executionMode?: DemoExecutionMode;
    sandboxMode?: DemoSandboxMode;
    gpuMode?: DemoGpuMode;
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

function sandboxLabel(mode?: DemoSandboxMode): string {
  if (mode === "local") {
    return "Local";
  }
  if (mode === "docker") {
    return "Docker sandbox";
  }
  if (mode === "runpod") {
    return "Runpod GPU";
  }
  return "Auto Docker sandbox";
}

function gpuLabel(mode?: DemoGpuMode): string {
  switch (mode) {
    case "off":
      return "CPU only";
    case "prefer":
      return "Prefer GPU";
    case "max":
      return "Max GPU";
    default:
      return "Auto GPU";
  }
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

function toHermesStatus(
  status?: HermesAuditStatusLike
): HermesPanel["overallStatus"] {
  switch (status) {
    case "grounded":
      return "verified";
    case "caveat":
      return "caveat";
    case "unsupported":
      return "unsupported";
    case "unavailable":
    case "system_error":
      return "caveat";
    default:
      return "pending";
  }
}

function latestHermesReport(
  reports?: HermesAuditReportLike[]
): HermesAuditReportLike | undefined {
  return reports?.at(-1);
}

function evidenceLabel(report?: HermesAuditReportLike): string | undefined {
  const ref = report?.evidence_refs?.[0];
  if (!ref) {
    return undefined;
  }
  return ref.path || ref.description || ref.snippet || ref.kind;
}

function reportDetail(
  report: HermesAuditReportLike | undefined,
  fallback: string
): string {
  if (!report) {
    return fallback;
  }

  const unsupported = report.unsupported_claims?.[0];
  const finding = report.findings?.[0];
  const intervention = report.recommended_intervention;
  const parts = [report.summary, unsupported, finding, intervention ? `Action: ${intervention}` : ""]
    .filter(Boolean);

  return parts.join(" ") || fallback;
}

function buildHermesPanelFromReports(
  state: PipelineStateDocument
): HermesPanel | null {
  const paperReport = latestHermesReport(state.hermes_step_reports?.["paper-understanding"]);
  const gate1Report = latestHermesReport(state.hermes_checkpoint_reports?.["gate_1"]);
  const implementationReport = latestHermesReport(
    state.hermes_step_reports?.["baseline-implementation"]
  );
  const artifactReport =
    latestHermesReport(state.hermes_checkpoint_reports?.["gate_2"]) ??
    latestHermesReport(state.hermes_step_reports?.["experiment-runner"]);
  const improvementReport =
    latestHermesReport(state.hermes_checkpoint_reports?.["gate_3"]) ??
    latestHermesReport(state.hermes_checkpoint_reports?.["research_map_generated"]);

  const reports = [
    paperReport,
    gate1Report,
    implementationReport,
    artifactReport,
    improvementReport
  ].filter((report): report is HermesAuditReportLike => Boolean(report));

  if (!reports.length) {
    return null;
  }

  const statuses = reports.map((report) => toHermesStatus(report.status));
  const overallStatus = statuses.includes("unsupported")
    ? "unsupported"
    : statuses.includes("caveat")
      ? "caveat"
      : statuses.includes("verified")
        ? "verified"
        : "checking";

  const interventions = (state.hermes_interventions ?? []).length;

  return {
    title: "Hermes Verification",
    summary:
      interventions > 0
        ? `Nous Hermes audited this run and recorded ${interventions} intervention${interventions === 1 ? "" : "s"} across steps and checkpoints.`
        : "Nous Hermes audited this run and did not need to intervene beyond annotations.",
    overallStatus,
    checks: [
      {
        id: "hermes-concept",
        label: "Paper concept extracted",
        detail: reportDetail(
          paperReport,
          "Waiting for the paper-understanding audit report."
        ),
        status: toHermesStatus(paperReport?.status),
        evidenceLabel: evidenceLabel(paperReport)
      },
      {
        id: "hermes-grounding",
        label: "Claim grounded in source text",
        detail: reportDetail(
          gate1Report ?? paperReport,
          "Source grounding will appear after the first checkpoint audit."
        ),
        status: toHermesStatus((gate1Report ?? paperReport)?.status),
        evidenceLabel: evidenceLabel(gate1Report ?? paperReport)
      },
      {
        id: "hermes-implementation",
        label: "Implementation matches concept",
        detail: reportDetail(
          implementationReport,
          "Implementation evidence is not available yet."
        ),
        status: toHermesStatus(implementationReport?.status),
        evidenceLabel: evidenceLabel(implementationReport)
      },
      {
        id: "hermes-artifacts",
        label: "Artifacts support reported result",
        detail: reportDetail(
          artifactReport,
          "Artifact verification will appear after the baseline run."
        ),
        status: toHermesStatus(artifactReport?.status),
        evidenceLabel: evidenceLabel(artifactReport)
      },
      {
        id: "hermes-improvement",
        label: "Improvement claim verified",
        detail: reportDetail(
          improvementReport,
          "Improvement verification unlocks after the final checkpoints."
        ),
        status: toHermesStatus(improvementReport?.status),
        evidenceLabel: evidenceLabel(improvementReport)
      }
    ]
  };
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
    contextVariables: [
      meta.sourceKind === "uploaded_pdf" ? "raw_paper_pdf" : "ppo_workspace_fixture"
    ]
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

  const hermesPanel = buildHermesPanel(state);
  const conceptCard = buildConceptCard(state);

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
          `Execution: ${meta.executionMode ?? "efficient"} / ${sandboxLabel(meta.sandboxMode)} / ${gpuLabel(meta.gpuMode)}`,
          `Verifier: ${meta.verificationProvider ? runModeLabel("sdk", meta.verificationProvider) : "Same provider as builder"}`,
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
    ],
    hermesPanel,
    conceptCard
  };
}

function buildHermesPanel(state: PipelineStateDocument): HermesPanel {
  const reportBackedPanel = buildHermesPanelFromReports(state);
  if (reportBackedPanel) {
    return reportBackedPanel;
  }

  const hasConcept = Boolean(state.paper_claim_map?.core_contribution);
  const hasImplementation = Boolean(state.baseline_result?.mode);
  const hasArtifacts = Boolean(state.experiment_artifacts?.success);
  const hasImprovements = Boolean(state.path_results?.length);
  const gate3Passed = Boolean(state.gate_3?.passed);

  return {
    title: "Hermes Verification",
    summary:
      "Hermes checks whether the current story is grounded in source text, implementation evidence, and run artifacts before the UI treats it as trustworthy.",
    overallStatus: gate3Passed
      ? "verified"
      : hasArtifacts
        ? "caveat"
        : hasConcept
          ? "checking"
          : "pending",
    checks: [
      {
        id: "hermes-concept",
        label: "Paper concept extracted",
        detail: hasConcept
          ? "The paper claim map has been extracted and attached to the run."
          : "Waiting for paper understanding output.",
        status: hasConcept ? "verified" : "pending",
        evidenceLabel: state.paper_claim_map?.datasets?.[0]?.name
      },
      {
        id: "hermes-grounding",
        label: "Claim grounded in source text",
        detail: hasConcept
          ? "The active concept is linked back to the extracted paper contract."
          : "Source grounding will start after claim extraction.",
        status: hasConcept ? "verified" : "pending",
        evidenceLabel: state.paper_claim_map?.metrics?.[0]?.name
      },
      {
        id: "hermes-implementation",
        label: "Implementation matches concept",
        detail: hasImplementation
          ? "Baseline implementation output has been produced from the paper plan."
          : "Implementation evidence is not available yet.",
        status: hasImplementation ? "checking" : "pending",
        evidenceLabel: state.baseline_result?.mode
      },
      {
        id: "hermes-artifacts",
        label: "Artifacts support reported result",
        detail: hasArtifacts
          ? "Hermes can now compare the UI claim against the baseline artifacts."
          : "No baseline artifact bundle has been verified yet.",
        status: hasArtifacts ? "caveat" : "pending",
        evidenceLabel: state.experiment_artifacts?.log_path
      },
      {
        id: "hermes-improvement",
        label: "Improvement claim verified",
        detail: hasImprovements
          ? "Improvement outcomes exist, but Hermes will not mark them verified until the final gate passes."
          : "Improvement verification unlocks after the baseline passes.",
        status: gate3Passed ? "verified" : hasImprovements ? "checking" : "pending",
        evidenceLabel: hasImprovements ? `${state.path_results?.length ?? 0} paths` : undefined
      }
    ]
  };
}

function buildConceptCard(state: PipelineStateDocument): ConceptCard {
  const baselineReward = state.experiment_artifacts?.metrics?.mean_reward;
  const bestImprovement = (state.path_results ?? [])
    .filter((path) => path.success)
    .sort(
      (left, right) =>
        Number(right.metrics?.improvement ?? -Infinity) -
        Number(left.metrics?.improvement ?? -Infinity)
    )[0];

  const hasImplementation = Boolean(state.baseline_result?.mode);
  const hasArtifacts = Boolean(state.experiment_artifacts?.success);
  const hasImprovement = Boolean(bestImprovement);

  return {
    id: "paper-concept-primary",
    title: state.paper_claim_map?.core_contribution ?? "Paper concept pending",
    interpretation:
      "This panel tracks how one paper idea moves from extraction and interpretation into code, evidence, and possible improvement.",
    status: hasImprovement
      ? "improved"
      : hasArtifacts
        ? "validated"
        : hasImplementation
          ? "active"
          : "planned",
    implementedSurface: hasImplementation
      ? `Baseline mode: ${state.baseline_result?.mode}`
      : "Implementation surface pending",
    artifactHint: state.experiment_artifacts?.log_path ?? "Waiting for validation artifacts",
    metricHint:
      baselineReward !== undefined ? `Baseline reward: ${baselineReward}` : "Metric pending",
    improvementHint: bestImprovement
      ? `${bestImprovement.path_id}: ${bestImprovement.metrics?.improvement ?? "n/a"} delta`
      : undefined,
    milestones: [
      {
        id: "depict-extracted",
        label: "Extracted",
        detail: state.paper_claim_map?.core_contribution
          ? "Concept recovered from the paper understanding stage."
          : "Paper concept not extracted yet.",
        status: state.paper_claim_map?.core_contribution ? "done" : "pending"
      },
      {
        id: "depict-interpreted",
        label: "Interpreted",
        detail: state.reproduction_contract?.reproduction_definition
          ? "The concept has been translated into a reproduction contract."
          : "Interpretation is waiting on the planner.",
        status: state.reproduction_contract?.reproduction_definition ? "done" : "pending"
      },
      {
        id: "depict-implemented",
        label: "Implemented",
        detail: hasImplementation
          ? "A baseline implementation has been produced from the concept."
          : "Implementation has not been emitted yet.",
        status: hasImplementation ? "done" : "active"
      },
      {
        id: "depict-validated",
        label: "Validated",
        detail: hasArtifacts
          ? "Run artifacts now support the implemented concept."
          : "Validation is waiting on experiment artifacts.",
        status: hasArtifacts ? "done" : hasImplementation ? "active" : "pending"
      },
      {
        id: "depict-improved",
        label: "Improved",
        detail: hasImprovement
          ? "At least one improvement path has produced a measurable outcome."
          : "Improvement evidence is not available yet.",
        status: hasImprovement ? "done" : "pending"
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
    event: "hermes_check_updated",
    timestamp: isoAt(4),
    panel: buildHermesPanel({
      ...state,
      baseline_result: undefined,
      experiment_artifacts: undefined,
      path_results: []
    })
  });
  events.push({
    event: "agent_completed",
    timestamp: isoAt(5),
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
    timestamp: isoAt(6),
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
    timestamp: isoAt(7),
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
    timestamp: isoAt(8),
    agentId: "environment-detective",
    changeType: "message",
    title: "Environment spec returned to orchestration",
    detail: "Docker and package constraints were captured for the baseline run.",
    fromAgentId: "environment-detective",
    toAgentId: "root-orchestrator"
  });
  events.push({
    event: "context_enrichment",
    timestamp: isoAt(9),
    agentId: "environment-detective",
    variableName: "environment_spec",
    summary: "Dockerfile and package versions added to shared context."
  });
  events.push({
    event: "agent_completed",
    timestamp: isoAt(10),
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
    timestamp: isoAt(11),
    stage: "plan",
    status: toStatusTone(state.gate_1?.status),
    detail: state.gate_1?.passed
      ? "Gate 1 passed. The plan is ready for baseline work."
      : "Gate 1 did not pass cleanly."
  });

  events.push({
    event: "agent_started",
    timestamp: isoAt(12),
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
    timestamp: isoAt(13),
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
    event: "concept_card_updated",
    timestamp: isoAt(14),
    card: buildConceptCard({
      ...state,
      experiment_artifacts: undefined,
      path_results: []
    })
  });
  events.push({
    event: "agent_completed",
    timestamp: isoAt(15),
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
    timestamp: isoAt(16),
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
    timestamp: isoAt(17),
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
    timestamp: isoAt(18),
    agentId: "experiment-runner",
    variableName: "baseline_artifacts",
    summary: "Run logs, metrics, provenance, and plots were attached to the shared context."
  });
  events.push({
    event: "hermes_check_updated",
    timestamp: isoAt(19),
    panel: buildHermesPanel({
      ...state,
      path_results: []
    })
  });
  events.push({
    event: "concept_card_updated",
    timestamp: isoAt(20),
    card: buildConceptCard({
      ...state,
      path_results: []
    })
  });
  events.push({
    event: "agent_completed",
    timestamp: isoAt(21),
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
    timestamp: isoAt(22),
    stage: "baseline",
    status: toStatusTone(state.gate_2?.status),
    detail: state.gate_2?.passed
      ? "Gate 2 passed. The baseline is good enough to unlock improvement paths."
      : "Gate 2 did not pass cleanly."
  });

  events.push({
    event: "agent_started",
    timestamp: isoAt(23),
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
    timestamp: isoAt(24 + pathResults.length * 3),
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
    timestamp: isoAt(25 + pathResults.length * 3),
    stage: "improvement",
    status: toStatusTone(state.gate_3?.status),
    detail: state.gate_3?.passed
      ? "Gate 3 passed. The research map is ready to review."
      : "Gate 3 did not pass cleanly."
  });

  events.push({
    event: "hermes_check_updated",
    timestamp: isoAt(26 + pathResults.length * 3),
    panel: buildHermesPanel(state)
  });
  events.push({
    event: "concept_card_updated",
    timestamp: isoAt(27 + pathResults.length * 3),
    card: buildConceptCard(state)
  });
  events.push({
    event: "context_enrichment",
    timestamp: isoAt(28 + pathResults.length * 3),
    agentId: "root-orchestrator",
    variableName: "research_map",
    summary: "Promising directions, dead ends, and next experiments were synthesized."
  });

  events.push({
    event: "agent_completed",
    timestamp: isoAt(29 + pathResults.length * 3),
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
      verificationProvider: meta.verificationProvider,
      executionMode: meta.executionMode,
      sandboxMode: meta.sandboxMode,
      gpuMode: meta.gpuMode,
      sourceLabel: meta.sourceLabel
    }
  };
}
