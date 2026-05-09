import type { LiveDemoPayload } from "./pipeline-dashboard";

export type DemoRunMode = "offline" | "sdk";

export type DemoRunStatus = "queued" | "running" | "completed" | "failed";

export interface LiveDemoRunState {
  projectId: string;
  outputDir: string;
  runMode: DemoRunMode;
  status: DemoRunStatus;
  startedAt?: string;
  updatedAt?: string;
  completedAt?: string;
  error?: string;
  payload: LiveDemoPayload | null;
  log: string;
}
