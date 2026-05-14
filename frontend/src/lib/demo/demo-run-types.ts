import type { LiveDemoPayload } from "./pipeline-dashboard";

export type DemoRunMode = "offline" | "sdk";

export type DemoProvider = "anthropic" | "openai";

export type DemoExecutionMode = "efficient" | "max";

export type DemoSandboxMode = "auto" | "docker" | "local" | "runpod";

export type DemoGpuMode = "off" | "auto" | "prefer" | "max";

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
  payload: LiveDemoPayload | null;
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
