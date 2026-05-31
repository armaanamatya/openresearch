export type DemoRunMode = "rlm" | "rdr" | "rlm-pure";

export type DemoProvider = "anthropic" | "openai";

export type DemoExecutionMode = "efficient" | "max";

export type DemoSandboxMode = "auto" | "docker" | "local" | "runpod";

export type DemoGpuMode = "off" | "auto" | "prefer" | "max";

export type DemoGpuParallelism = "auto" | "single" | "multi";

export type DemoAccelerator = "off" | "auto" | "local" | "runpod" | "azure" | "endpoint";

export type DemoModelChoice = string;

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
  workerReports?: DemoWorkerReport[];
  [key: string]: unknown;
}

export interface DemoWorkerCommand {
  command: string;
  exit_code?: number | null;
  source?: string;
  cwd?: string;
  started_at?: string;
  finished_at?: string;
  stdout_tail?: string;
  stderr_tail?: string;
  status?: string;
}

export interface DemoWorkerBlocker {
  title: string;
  description: string;
  severity: "critical" | "high" | "medium" | "low";
  source: string;
  suggested_fix?: string;
}

export interface DemoWorkerError {
  message: string;
  stack_or_trace?: string | null;
  source_file?: string | null;
  recoverable?: boolean;
}

export interface DemoWorkerArtifact {
  path: string;
  type?: string;
  description?: string;
}

export interface DemoWorkerTest {
  command: string;
  status: string;
  passed_count?: number;
  failed_count?: number;
  skipped_count?: number;
  exit_code?: number | null;
  notes?: string;
}

export interface DemoWorkerAssignment {
  summary: string;
  detailed_prompt_or_task?: string;
  expected_outputs?: string[];
  constraints?: string[];
}

export interface DemoExecutionSummary {
  concise_summary?: string;
  implemented?: string[];
  partially_implemented?: string[];
  not_implemented?: string[];
  changed_files?: string[];
  created_files?: string[];
  deleted_files?: string[];
}

export type DemoWorkerType =
  | "rdr_cluster"
  | "rlm_primitive"
  | "sdk_agent"
  | "hybrid_iteration";

export interface DemoWorkerReport {
  report_id?: string;
  agent_id?: string;
  model?: string | null;
  provider?: string | null;
  status?: string;
  started_at?: string;
  finished_at?: string;
  implemented?: string[];
  left_undone?: string[];
  commands?: DemoWorkerCommand[];
  issues?: string[];
  procedures_followed?: boolean | null;
  procedure_notes?: string;
  error?: string | null;
  // Extended fields (2026-05-24)
  run_id?: string;
  worker_id?: string;
  worker_type?: DemoWorkerType;
  cluster_id?: string;
  task_id?: string;
  parent_task_id?: string;
  duration_ms?: number;
  assignment?: DemoWorkerAssignment;
  execution_summary?: DemoExecutionSummary;
  blockers?: DemoWorkerBlocker[];
  errors?: DemoWorkerError[];
  artifacts?: DemoWorkerArtifact[];
  tests?: DemoWorkerTest[];
  next_actions?: DemoNextAction[];
}

export interface DemoNextAction {
  priority?: string;
  action: string;
  owner_or_component?: string;
  rationale?: string;
}

export interface DemoReportsSummary {
  total_workers?: number;
  by_status?: Record<string, number>;
  critical_blockers?: DemoWorkerBlocker[];
  files_changed?: string[];
  commands_run?: number;
  failed_commands?: number;
  tests_summary?: { passed: number; failed: number };
  final_run_status?: string;
  top_next_actions?: DemoNextAction[];
  generated_at?: string;
}

export interface DemoReportsResponse {
  workers: DemoWorkerReport[];
  summary: DemoReportsSummary;
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

// ── Runpod status chip (U1) ────────────────────────────────────────────────

export type DemoRunpodStatusKind =
  | "not_runpod"
  | "not_yet"
  | "provisioning"
  | "ready"
  | "executing"
  | "stopping"
  | "destroyed"
  | "error";

export interface DemoRunpodStatusResponse {
  project_id: string;
  sandbox_mode?: DemoSandboxMode | null;
  status: DemoRunpodStatusKind;
  label: string;
  detail: string;
  source: "events" | "runpod_api";
  pod?: {
    id?: string | null;
    name?: string | null;
    desiredStatus?: string | null;
    currentStatus?: string | null;
  } | null;
  updated_at?: string | null;
  api_error?: string;
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
