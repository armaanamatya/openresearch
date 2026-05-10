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
  startedAt?: string;
  updatedAt?: string;
  completedAt?: string;
  error?: string;
  pid?: number;
  payload: LiveDemoPayload | null;
  log: string;
}
