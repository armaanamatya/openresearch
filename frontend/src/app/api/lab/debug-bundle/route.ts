import { NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";

import { readTelemetryTail } from "@/lib/demo/server-fs";

export const runtime = "nodejs";

const MAX_LOG_TAIL_BYTES = 24_000;
const MAX_TELEMETRY_RECORDS = 30;
const MAX_PIPELINE_STATE_BYTES = 32_000;

interface DebugBundle {
  generatedAt: string;
  projectId: string;
  status: Record<string, unknown> | null;
  pipelineStateStage: string | null;
  pipelineStateBytes: number;
  pipelineStatePreview: Record<string, unknown> | null;
  telemetry: ReturnType<typeof Number> extends never ? never : unknown[];
  lastError: string | null;
  logTail: string;
  paths: Record<string, string>;
}

function repoRoot(): string {
  const override = process.env.REPROLAB_REPO_ROOT?.trim();
  if (override) return override;
  return path.join(process.cwd(), "..");
}

function runDir(projectId: string): string {
  return path.join(repoRoot(), "runs", projectId);
}

async function readJson<T = unknown>(filePath: string): Promise<T | null> {
  try {
    const raw = await fs.readFile(filePath, "utf8");
    return JSON.parse(raw) as T;
  } catch {
    return null;
  }
}

async function readTextTail(filePath: string, maxBytes: number): Promise<string> {
  try {
    const raw = await fs.readFile(filePath, "utf8");
    return raw.length > maxBytes ? raw.slice(-maxBytes) : raw;
  } catch {
    return "";
  }
}

export async function GET(request: Request) {
  const url = new URL(request.url);
  const projectId = url.searchParams.get("projectId");
  if (!projectId) {
    return NextResponse.json(
      { error: "projectId query parameter is required" },
      { status: 400 }
    );
  }
  // Path-traversal guard.
  if (!/^[A-Za-z0-9._-]+$/.test(projectId)) {
    return NextResponse.json(
      { error: "Invalid projectId" },
      { status: 400 }
    );
  }

  const dir = runDir(projectId);
  const statusPath = path.join(dir, "demo_status.json");
  const pipelineStatePath = path.join(dir, "pipeline_state.json");
  const stderrPath = path.join(dir, "runner.stderr.log");
  const telemetryPath = path.join(dir, "agent_telemetry.jsonl");

  const [status, pipelineStateRaw, logTail, telemetry] = await Promise.all([
    readJson<Record<string, unknown>>(statusPath),
    readTextTail(pipelineStatePath, MAX_PIPELINE_STATE_BYTES),
    readTextTail(stderrPath, MAX_LOG_TAIL_BYTES),
    readTelemetryTail(projectId, MAX_TELEMETRY_RECORDS),
  ]);

  let pipelineStatePreview: Record<string, unknown> | null = null;
  let pipelineStateStage: string | null = null;
  let pipelineStateBytes = 0;
  if (pipelineStateRaw) {
    pipelineStateBytes = pipelineStateRaw.length;
    try {
      const parsed = JSON.parse(pipelineStateRaw) as Record<string, unknown>;
      pipelineStateStage = typeof parsed.stage === "string" ? parsed.stage : null;
      // Keep only the small, high-signal fields so the bundle stays compact.
      pipelineStatePreview = {
        stage: parsed.stage,
        gate_1: parsed.gate_1,
        gate_2: parsed.gate_2,
        gate_3: parsed.gate_3,
        decision_log: Array.isArray(parsed.decision_log)
          ? parsed.decision_log.slice(-10)
          : parsed.decision_log,
        assumption_ledger_count: Array.isArray(parsed.assumption_ledger)
          ? parsed.assumption_ledger.length
          : null,
      };
    } catch {
      pipelineStatePreview = null;
    }
  }

  const lastError =
    (status?.error as string | undefined) ??
    telemetry.find((t) => t.success === false)?.error_message ??
    null;

  const bundle: DebugBundle = {
    generatedAt: new Date().toISOString(),
    projectId,
    status,
    pipelineStateStage,
    pipelineStateBytes,
    pipelineStatePreview,
    telemetry,
    lastError: lastError ?? null,
    logTail,
    paths: {
      runDir: dir,
      status: statusPath,
      pipelineState: pipelineStatePath,
      runnerStderrLog: stderrPath,
      agentTelemetry: telemetryPath,
    },
  };

  return NextResponse.json(bundle);
}
