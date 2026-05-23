export type DemoRunMode = "rlm" | "rdr" | "rlm-pure";

export type DemoProvider = "anthropic" | "openai";

export type DemoExecutionMode = "efficient" | "max";

export type DemoSandboxMode = "auto" | "docker" | "local" | "runpod";

export type DemoGpuMode = "off" | "auto" | "prefer" | "max";

export type DemoModelChoice = "sonnet" | "opus";

export type DemoRunStatus =
  | "queued"
  | "running"
  | "stopped"
  | "completed"
  | "failed";

export interface SourcePdfArtifact {
  fileName: string;
  title: string;
  sizeBytes: number;
  sha256: string;
  pageCount?: number | null;
  runPath: string;
  codePath: string;
}

export interface DemoRubricArea {
  area: string;
  weight: number;
  score: number;
  justification: string;
  weak_points: string[];
}

export interface DemoPaperbenchBaseline {
  score: number;
  source: string;
  model: string;
}

export interface DemoBenchmarkSummary {
  benchmarkName: string;
  paperbenchTaskId: string;
  overallScore: number;
  targetMetric: string;
  targetValue: number;
  reproducedValue: number;
  deltaValue: number;
  verdict: string;
  reportPath: string;
  comparisonPath: string;
  logPath: string;
  // Track 3 — rubric-verifier comparison. Optional: offline runs and runs with
  // the verifier disabled won't carry these.
  paperbenchBaseline?: DemoPaperbenchBaseline | null;
  ourRubricScore?: number | null;
  verificationDelta?: number | null;
  improvementIterations?: number;
  meetsTarget?: boolean | null;
  comparisonSummary?: string;
  rubricAreas?: DemoRubricArea[];
  baselineRubricAreas?: DemoRubricArea[];
}

/** Shape of the enriched payload attached to a run by the backend's _build_payload. */
export interface LiveDemoRunPayload {
  events?: unknown[];
  [key: string]: unknown;
}

export interface LiveDemoRunState {
  projectId: string;
  outputDir: string;
  runMode: DemoRunMode;
  llmProvider?: DemoProvider;
  verificationProvider?: DemoProvider;
  executionMode?: DemoExecutionMode;
  sandboxMode?: DemoSandboxMode;
  gpuMode?: DemoGpuMode;
  model?: DemoModelChoice;
  status: DemoRunStatus;
  sourceKind?: "workspace_fixture" | "uploaded_pdf";
  sourceLabel?: string;
  sourceNote?: string;
  sourcePdf?: SourcePdfArtifact | null;
  benchmark?: DemoBenchmarkSummary | null;
  startedAt?: string;
  updatedAt?: string;
  completedAt?: string;
  error?: string;
  pid?: number;
  payload: LiveDemoRunPayload | null;
  log: string;
  telemetry?: TelemetryRecordPublic[];
}

/** Compact, public-safe view of one agent invocation telemetry record. */
export interface TelemetryRecordPublic {
  agent_id?: string;
  model?: string;
  started_at?: string;
  finished_at?: string;
  duration_seconds?: number;
  message_count?: number;
  output_chars?: number;
  success?: boolean;
  error_message?: string | null;
}

// ── RDR / Hybrid artifact types ─────────────────────────────────────────────
// Shapes match the backend _read_rdr_clusters / _read_rdr_repair_iterations /
// _read_rdr_leaf_scores functions in backend/app.py exactly.

export interface DemoRepairHistoryEntry {
  pass: number;
  failed: boolean;
  file_count: number;
}

export interface DemoClusterStatus {
  index: number;
  cluster_id: string;
  title: string;
  leaf_ids: string[];
  /** null when only repair checkpoints exist for this cluster (partial run) */
  failed: boolean | null;
  file_count: number;
  repair_history: DemoRepairHistoryEntry[];
}

export interface DemoClustersResponse {
  project_id: string;
  clusters: DemoClusterStatus[];
}

export interface DemoRepairPass {
  pass: number;
  cluster_count: number;
  failed_count: number;
}

export interface DemoRepairIterationsResponse {
  project_id: string;
  passes: DemoRepairPass[];
}

export interface DemoLeafScore {
  id: string;
  score: number;
  justification: string;
}

export interface DemoLeafScoresResponse {
  project_id: string;
  overall_score: number;
  leaf_scores: DemoLeafScore[];
}

// ── Auth status types (provider picker, D1) ─────────────────────────────────

export type RootProvider =
  | "anthropic_api"
  | "anthropic_oauth"
  | "openai_api"
  | "azure_openai"
  | "featherless";

export type SubagentAuth = "anthropic_api" | "anthropic_oauth";

export interface ProviderStatus {
  available: boolean;
  detail: string;
}

export interface AuthStatus {
  providers: Record<RootProvider, ProviderStatus>;
  subagent_auth: Record<SubagentAuth, boolean>;
  defaults: {
    root_provider: RootProvider;
    root_model: string;
    subagent_auth: SubagentAuth;
  };
}

export const RUN_MODE_OPTIONS: ReadonlyArray<{
  value: DemoRunMode;
  label: string;
  description: string;
}> = [
  {
    value: "rlm",
    label: "RLM (Hybrid)",
    description: "RDR rubric-decompose + RLM adaptive repair. Recommended."
  },
  {
    value: "rdr",
    label: "RDR (Rubric-driven)",
    description: "Deterministic rubric controller. No LLM repair. Predictable cost."
  },
  {
    value: "rlm-pure",
    label: "RLM pure",
    description: "Pre-hybrid RLM path for compatibility and regression checks."
  }
] as const;
