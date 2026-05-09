import type { LiveDemoPayload } from "./pipeline-dashboard";

export type DemoRunMode = "offline" | "sdk";

export type DemoProvider = "anthropic" | "openai";

export type DemoRunStatus = "queued" | "running" | "completed" | "failed";

export interface LiveDemoRunState {
  projectId: string;
  outputDir: string;
  runMode: DemoRunMode;
  llmProvider?: DemoProvider;
  status: DemoRunStatus;
  startedAt?: string;
  updatedAt?: string;
  completedAt?: string;
  error?: string;
  payload: LiveDemoPayload | null;
  log: string;
}
