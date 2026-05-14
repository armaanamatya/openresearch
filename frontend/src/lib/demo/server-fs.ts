import "server-only";

import { promises as fs } from "fs";
import path from "path";

import type {
  DemoExecutionMode,
  DemoGpuMode,
  DemoProvider,
  DemoRunMode,
  DemoRunStatus,
  DemoSandboxMode,
  LiveDemoRunState
} from "./demo-run-types";
import type { LiveDemoMeta, PipelineStateDocument } from "./pipeline-dashboard";

export interface DemoRunStatusFile {
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
  sourcePdf?: LiveDemoRunState["sourcePdf"];
  benchmark?: LiveDemoRunState["benchmark"];
  startedAt: string;
  updatedAt: string;
  completedAt?: string;
  error?: string;
  pid?: number;
}

/** Single agent invocation as recorded by AgentTelemetryRecorder. */
export interface TelemetryRecord {
  agent_id?: string;
  model?: string;
  started_at?: string;
  finished_at?: string;
  duration_seconds?: number;
  message_count?: number;
  output_chars?: number;
  success?: boolean;
  error_message?: string | null;
  usage?: Record<string, unknown>;
  [key: string]: unknown;
}

export function repoRoot(): string {
  const override = process.env.REPROLAB_REPO_ROOT?.trim();
  if (override) {
    return override;
  }
  return path.join(process.cwd(), "..");
}

export function runsRoot(): string {
  return path.join(repoRoot(), "runs");
}

export function runDir(projectId: string): string {
  return path.join(runsRoot(), projectId);
}

export function statusPath(projectId: string): string {
  return path.join(runDir(projectId), "demo_status.json");
}

export function logPath(projectId: string): string {
  return path.join(runDir(projectId), "runner.stderr.log");
}

export function pipelineStatePath(projectId: string): string {
  return path.join(runDir(projectId), "pipeline_state.json");
}

export function telemetryPath(projectId: string): string {
  return path.join(runDir(projectId), "agent_telemetry.jsonl");
}

export function buildFixtureMeta(
  projectId: string,
  outputDir: string,
  runMode: DemoRunMode,
  llmProvider?: DemoProvider,
  verificationProvider?: DemoProvider,
  executionMode: DemoExecutionMode = "efficient",
  sandboxMode: DemoSandboxMode = "runpod",
  gpuMode: DemoGpuMode = "auto"
): LiveDemoMeta {
  return {
    projectId,
    outputDir,
    sourceKind: "workspace_fixture",
    runMode,
    llmProvider,
    verificationProvider,
    executionMode,
    sandboxMode,
    gpuMode,
    sourceLabel: "ReproLab PPO demo paper",
    sourceNote:
      "This demo uses a checked-in PPO-style paper PDF, a deterministic generated codebase, and a PaperBench-style final benchmark comparison."
  };
}

export function buildUploadedPaperMeta(
  projectId: string,
  outputDir: string,
  runMode: DemoRunMode,
  llmProvider: DemoProvider | undefined,
  verificationProvider: DemoProvider | undefined,
  executionMode: DemoExecutionMode,
  sandboxMode: DemoSandboxMode,
  gpuMode: DemoGpuMode,
  fileName: string
): LiveDemoMeta {
  return {
    projectId,
    outputDir,
    sourceKind: "uploaded_pdf",
    runMode,
    llmProvider,
    verificationProvider,
    executionMode,
    sandboxMode,
    gpuMode,
    sourceLabel: fileName
  };
}

export async function readJsonFile<T>(filePath: string): Promise<T | null> {
  try {
    const raw = await fs.readFile(filePath, "utf8");
    return JSON.parse(raw) as T;
  } catch {
    return null;
  }
}

export async function readPipelineState(
  projectId: string
): Promise<PipelineStateDocument | null> {
  let raw: string;
  try {
    raw = await fs.readFile(pipelineStatePath(projectId), "utf8");
  } catch (error) {
    if ((error as NodeJS.ErrnoException)?.code === "ENOENT") return null;
    console.warn(`[server-fs] readPipelineState(${projectId}) failed:`, error);
    return null;
  }
  try {
    return JSON.parse(raw) as PipelineStateDocument;
  } catch {
    // Half-written JSON (orchestrator writes are non-atomic). Caller's
    // last-good cache will be used.
    return null;
  }
}

export async function readStatus(projectId: string): Promise<DemoRunStatusFile | null> {
  return readJsonFile<DemoRunStatusFile>(statusPath(projectId));
}

export async function readLogTail(projectId: string, maxChars = 12000): Promise<string> {
  try {
    const raw = await fs.readFile(logPath(projectId), "utf8");
    return raw.length > maxChars ? raw.slice(-maxChars) : raw;
  } catch {
    return "";
  }
}

/** Streamed-append JSONL files: tail without loading entire file into RAM. */
export async function readTelemetryTail(
  projectId: string,
  maxRecords = 50
): Promise<TelemetryRecord[]> {
  let raw: string;
  try {
    raw = await fs.readFile(telemetryPath(projectId), "utf8");
  } catch {
    return [];
  }
  const lines = raw.split(/\r?\n/).filter((line) => line.trim().length > 0);
  const tail = lines.slice(-maxRecords);
  const records: TelemetryRecord[] = [];
  for (const line of tail) {
    try {
      records.push(JSON.parse(line) as TelemetryRecord);
    } catch {
      // Skip partial / corrupt JSONL lines silently.
    }
  }
  return records;
}

export function metaFromStatus(
  projectId: string,
  outputDir: string,
  runMode: DemoRunMode,
  status?: Pick<
    DemoRunStatusFile,
    | "llmProvider"
    | "verificationProvider"
    | "executionMode"
    | "sandboxMode"
    | "gpuMode"
    | "sourceKind"
    | "sourceLabel"
    | "sourceNote"
  >
): LiveDemoMeta {
  const executionMode = status?.executionMode ?? "efficient";
  const sandboxMode = status?.sandboxMode ?? "runpod";
  const verificationProvider = status?.verificationProvider;
  const gpuMode = status?.gpuMode ?? "auto";

  if (
    status?.sourceKind === "uploaded_pdf" &&
    status.sourceLabel &&
    status.sourceNote
  ) {
    return {
      projectId,
      outputDir,
      runMode,
      llmProvider: status.llmProvider,
      verificationProvider,
      executionMode,
      sandboxMode,
      gpuMode,
      sourceKind: "uploaded_pdf",
      sourceLabel: status.sourceLabel,
      sourceNote: status.sourceNote
    };
  }

  return buildFixtureMeta(
    projectId,
    outputDir,
    runMode,
    status?.llmProvider,
    verificationProvider,
    executionMode,
    sandboxMode,
    gpuMode
  );
}
