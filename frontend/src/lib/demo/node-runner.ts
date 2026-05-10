import "server-only";

import { promises as fs } from "fs";
import { existsSync } from "fs";
import path from "path";
import { createHash, randomUUID } from "crypto";
import { execFile, spawn } from "child_process";
import { promisify } from "util";

import type {
  DemoExecutionMode,
  DemoGpuMode,
  DemoProvider,
  DemoRunMode,
  DemoRunStatus,
  DemoSandboxMode,
  LiveDemoRunState
} from "./demo-run-types";
import { isStaleDemoRun, summarizeRunFailure } from "./run-staleness";
import {
  buildLiveDemoDashboard,
  type LiveDemoMeta,
  type PipelineStateDocument
} from "./pipeline-dashboard";

const DEMO_WORKSPACE = {
  project_id: "prj_e2e_test",
  entries: [
    {
      source_id: "src_1",
      title: "Abstract",
      excerpt:
        "We propose a new family of policy gradient methods for reinforcement learning, which alternate between sampling data and optimizing a surrogate objective."
    },
    {
      source_id: "src_2",
      title: "Experiments",
      excerpt:
        "We test on CartPole-v1 environment. We use Adam optimizer with learning rate 3e-4 and batch size 64. We report mean reward over 100 episodes after 500000 timesteps."
    },
    {
      source_id: "src_3",
      title: "Conclusion",
      excerpt:
        "We have introduced proximal policy optimization, a family of methods that use multiple epochs of stochastic gradient ascent."
    }
  ]
};

interface DemoRunStatusFile {
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
  startedAt: string;
  updatedAt: string;
  completedAt?: string;
  error?: string;
  pid?: number;
}

interface UploadedPaperInput {
  fileName: string;
  bytes: Uint8Array;
}

interface UploadedPaperLaunchConfig {
  sourcePath: string;
  fileName: string;
}

interface DemoRunStartOptions {
  uploadedPaper?: UploadedPaperInput;
  verificationProvider?: DemoProvider;
  gpuMode?: DemoGpuMode;
}

const execFileAsync = promisify(execFile);

export class DemoPreflightError extends Error {
  statusCode = 503;
  code = "sandbox_preflight_failed";

  constructor(message: string) {
    super(message);
    this.name = "DemoPreflightError";
  }
}

function repoRoot(): string {
  const override = process.env.REPROLAB_REPO_ROOT?.trim();
  if (override) {
    return override;
  }
  return path.join(process.cwd(), "..");
}

function pythonStringLiteral(value: string): string {
  return JSON.stringify(value);
}

/**
 * Resolve the Python interpreter to use for backend subprocesses.
 *
 * Resolution order:
 *   1. `REPROLAB_PYTHON_BIN` env var (explicit override).
 *   2. Local virtualenv at `<repo>/.venv/bin/python` (Linux/macOS) or
 *      `<repo>/.venv/Scripts/python.exe` (Windows).
 *   3. System `python3` (POSIX) or `py` (Windows) as a last resort.
 *
 * The venv path is preferred so the subprocess inherits the project's
 * pinned dependencies (pydantic, claude-agent-sdk, deepeval, etc).
 * Without this resolution, the runner falls back to system Python which
 * lacks the project's deps and fails with `ModuleNotFoundError`.
 */
function pythonBinary(): string {
  const override = process.env.REPROLAB_PYTHON_BIN?.trim();
  if (override) {
    return override;
  }
  const root = repoRoot();
  const venvCandidates =
    process.platform === "win32"
      ? [path.join(root, ".venv", "Scripts", "python.exe")]
      : [path.join(root, ".venv", "bin", "python")];
  for (const candidate of venvCandidates) {
    if (existsSync(candidate)) {
      return candidate;
    }
  }
  return process.platform === "win32" ? "py" : "python3";
}

function runsRoot(): string {
  return path.join(repoRoot(), "runs");
}

function runDir(projectId: string): string {
  return path.join(runsRoot(), projectId);
}

function statusPath(projectId: string): string {
  return path.join(runDir(projectId), "demo_status.json");
}

function logPath(projectId: string): string {
  return path.join(runDir(projectId), "runner.stderr.log");
}

function pipelineStatePath(projectId: string): string {
  return path.join(runDir(projectId), "pipeline_state.json");
}

function buildFixtureMeta(
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
    sourceLabel: "In-repo PPO workspace fixture",
    sourceNote:
      "The repo currently does not contain a checked-in paper PDF, so this UI demo uses the deterministic PPO workspace fixture that already drives the end-to-end pipeline tests."
  };
}

function buildUploadedPaperMeta(
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
    sourceLabel: fileName,
    sourceNote:
      "This run started from a PDF uploaded directly in the lab. The backend routed it through the repo's paper ingestion pipeline before running reproduction."
  };
}

async function readJsonFile<T>(filePath: string): Promise<T | null> {
  try {
    const raw = await fs.readFile(filePath, "utf8");
    return JSON.parse(raw) as T;
  } catch {
    return null;
  }
}

async function readPipelineState(projectId: string): Promise<PipelineStateDocument | null> {
  return readJsonFile<PipelineStateDocument>(pipelineStatePath(projectId));
}

async function readStatus(projectId: string): Promise<DemoRunStatusFile | null> {
  return readJsonFile<DemoRunStatusFile>(statusPath(projectId));
}

async function writeStatus(projectId: string, status: DemoRunStatusFile): Promise<void> {
  await fs.mkdir(runDir(projectId), { recursive: true });
  await fs.writeFile(statusPath(projectId), JSON.stringify(status, null, 2), "utf8");
}

async function readLogTail(projectId: string, maxChars = 12000): Promise<string> {
  try {
    const raw = await fs.readFile(logPath(projectId), "utf8");
    return raw.length > maxChars ? raw.slice(-maxChars) : raw;
  } catch {
    return "";
  }
}

function telemetryPath(projectId: string): string {
  return path.join(runDir(projectId), "agent_telemetry.jsonl");
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

function metaFromStatus(
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

async function payloadForProject(
  projectId: string,
  runMode: DemoRunMode,
  log = "",
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
) {
  const outputDir = runDir(projectId);
  const state = await readPipelineState(projectId);
  if (!state) {
    return null;
  }

  return buildLiveDemoDashboard(state, metaFromStatus(projectId, outputDir, runMode, status), log);
}

function buildPythonScript(
  projectId: string,
  runMode: DemoRunMode,
  llmProvider: DemoProvider,
  verificationProvider: DemoProvider | undefined,
  executionMode: DemoExecutionMode,
  sandboxMode: DemoSandboxMode,
  gpuMode: DemoGpuMode,
  uploadedPaper?: UploadedPaperLaunchConfig
): string {
  const projectIdLiteral = pythonStringLiteral(projectId);
  const runModeLiteral = pythonStringLiteral(runMode);
  const llmProviderLiteral = pythonStringLiteral(llmProvider);
  const verificationProviderLiteral =
    verificationProvider === undefined ? "None" : pythonStringLiteral(verificationProvider);
  const executionModeLiteral = pythonStringLiteral(executionMode);
  const sandboxModeLiteral = pythonStringLiteral(sandboxMode);
  const gpuModeLiteral = pythonStringLiteral(gpuMode);
  const runsRootLiteral = pythonStringLiteral(runsRoot());

  if (uploadedPaper) {
    const uploadedPaperPathLiteral = pythonStringLiteral(uploadedPaper.sourcePath);
    const uploadedFileNameLiteral = pythonStringLiteral(uploadedPaper.fileName);
    return `
import json
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

from backend.cli import cmd_reproduce
from backend.config import get_settings

project_id = ${projectIdLiteral}
llm_provider = ${llmProviderLiteral}
verification_provider = ${verificationProviderLiteral}
execution_mode = ${executionModeLiteral}
sandbox_mode = ${sandboxModeLiteral}
gpu_mode = ${gpuModeLiteral}
runs_root = Path(${runsRootLiteral})
output_dir = (runs_root / project_id).resolve()
output_dir.mkdir(parents=True, exist_ok=True)
status_path = output_dir / "demo_status.json"
uploaded_paper = Path(${uploadedPaperPathLiteral}).resolve()
uploaded_file_name = ${uploadedFileNameLiteral}

def now():
    return datetime.now(timezone.utc).isoformat()

def write_status(status, error=None, completed_at=None):
    existing = {}
    if status_path.exists():
        try:
            existing = json.loads(status_path.read_text())
        except Exception:
            existing = {}
    payload = {
        "projectId": project_id,
        "outputDir": str(output_dir),
        "runMode": ${runModeLiteral},
        "executionMode": execution_mode,
        "sandboxMode": sandbox_mode,
        "gpuMode": gpu_mode,
        "sourceKind": "uploaded_pdf",
        "sourceLabel": uploaded_file_name,
        "sourceNote": "This run started from a PDF uploaded directly in the lab. The backend routed it through the repo's paper ingestion pipeline before running reproduction.",
        "status": status,
        "startedAt": started_at,
        "updatedAt": now(),
    }
    if ${runModeLiteral} == "sdk":
        payload["llmProvider"] = llm_provider
        if verification_provider is not None:
            payload["verificationProvider"] = verification_provider
    if existing.get("pid") is not None:
        payload["pid"] = existing["pid"]
    if completed_at:
        payload["completedAt"] = completed_at
    if error:
        payload["error"] = error
    status_path.write_text(json.dumps(payload, indent=2))

started_at = now()
write_status("running")

try:
    exit_code = cmd_reproduce(Namespace(
        source=str(uploaded_paper),
        source_kind="pdf_path",
        agent="default",
        mode=${runModeLiteral},
        model=None,
        provider=llm_provider if ${runModeLiteral} == "sdk" else None,
        verification_provider=verification_provider if ${runModeLiteral} == "sdk" else None,
        execution_mode=execution_mode,
        sandbox=sandbox_mode,
        gpu_mode=gpu_mode,
        command_timeout=None,
        allow_sandbox_network=False,
        sandbox_platform=None,
        sandbox_memory=None,
        sandbox_cpus=None,
        hints="Keep this as a lightweight smoke test",
        n_paths=1,
        runs_root=str(runs_root),
        database_url=get_settings().database_url,
    ))
    if exit_code == 0:
        write_status("completed", completed_at=now())
    else:
        write_status("failed", error=f"Pipeline exited with status {exit_code}", completed_at=now())
except Exception as exc:
    write_status("failed", error=f"{type(exc).__name__}: {exc}", completed_at=now())
    raise
`;
  }

  const workspaceLiteral = pythonStringLiteral(JSON.stringify(DEMO_WORKSPACE));
  return `
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from backend.agents.execution import ExecutionProfile, SandboxMode
from backend.agents.pipeline import run_pipeline_offline, run_pipeline_sdk

workspace = json.loads(${workspaceLiteral})
project_id = ${projectIdLiteral}
llm_provider = ${llmProviderLiteral}
verification_provider = ${verificationProviderLiteral}
execution_mode = ${executionModeLiteral}
sandbox_mode = ${sandboxModeLiteral}
gpu_mode = ${gpuModeLiteral}
runs_root = Path(${runsRootLiteral})
output_dir = (runs_root / project_id).resolve()
output_dir.mkdir(parents=True, exist_ok=True)
status_path = output_dir / "demo_status.json"

def now():
    return datetime.now(timezone.utc).isoformat()

def write_status(status, error=None, completed_at=None):
    existing = {}
    if status_path.exists():
        try:
            existing = json.loads(status_path.read_text())
        except Exception:
            existing = {}
    payload = {
        "projectId": project_id,
        "outputDir": str(output_dir),
        "runMode": ${runModeLiteral},
        "executionMode": execution_mode,
        "sandboxMode": sandbox_mode,
        "gpuMode": gpu_mode,
        "status": status,
        "startedAt": started_at,
        "updatedAt": now(),
    }
    if ${runModeLiteral} == "sdk":
        payload["llmProvider"] = llm_provider
        if verification_provider is not None:
            payload["verificationProvider"] = verification_provider
    if existing.get("pid") is not None:
        payload["pid"] = existing["pid"]
    if completed_at:
        payload["completedAt"] = completed_at
    if error:
        payload["error"] = error
    status_path.write_text(json.dumps(payload, indent=2))

started_at = now()
write_status("running")
execution_profile = ExecutionProfile.from_mode(execution_mode, gpu_mode=gpu_mode)

try:
    if ${runModeLiteral} == "sdk":
        asyncio.run(run_pipeline_sdk(
            project_id,
            runs_root,
            workspace,
            provider=llm_provider,
            verification_provider=verification_provider,
            user_hints=["Keep this as a lightweight smoke test"],
            n_improvement_paths=1,
            execution_profile=execution_profile,
            sandbox_mode=SandboxMode(sandbox_mode),
        ))
    else:
        run_pipeline_offline(
            project_id,
            runs_root,
            workspace,
            execution_profile=execution_profile,
            sandbox_mode=SandboxMode(sandbox_mode),
        )
    write_status("completed", completed_at=now())
except Exception as exc:
    write_status("failed", error=f"{type(exc).__name__}: {exc}", completed_at=now())
    raise
`;
}

async function latestProjectId(
  runMode?: DemoRunMode,
  llmProvider?: DemoProvider,
  executionMode?: DemoExecutionMode,
  sandboxMode?: DemoSandboxMode,
  verificationProvider?: DemoProvider,
  gpuMode?: DemoGpuMode
): Promise<string | null> {
  try {
    const entries = await fs.readdir(runsRoot(), { withFileTypes: true });
    const candidates = await Promise.all(
      entries
        .filter((entry) => entry.isDirectory())
        .map(async (entry) => {
          const status = await readStatus(entry.name);
          if (!status) {
            return null;
          }
          if (runMode && status.runMode !== runMode) {
            return null;
          }
          if (runMode === "sdk" && llmProvider) {
            const statusProvider = status.llmProvider ?? providerFromProjectId(entry.name);
            if (statusProvider !== llmProvider) {
              return null;
            }
          }
          if (executionMode && (status.executionMode ?? "efficient") !== executionMode) {
            return null;
          }
          if (sandboxMode && (status.sandboxMode ?? "runpod") !== sandboxMode) {
            return null;
          }
          if (verificationProvider && status.verificationProvider !== verificationProvider) {
            return null;
          }
          if (gpuMode && (status.gpuMode ?? "auto") !== gpuMode) {
            return null;
          }
          const timestamp = Date.parse(status.updatedAt || status.startedAt || "");
          return Number.isFinite(timestamp)
            ? { projectId: entry.name, mtimeMs: timestamp }
            : null;
      })
    );

    const latest = candidates
      .filter((candidate): candidate is NonNullable<typeof candidate> => candidate !== null)
      .sort((left, right) => right.mtimeMs - left.mtimeMs)[0];

    return latest?.projectId ?? null;
  } catch {
    return null;
  }
}

async function inferState(projectId: string): Promise<LiveDemoRunState | null> {
  const status = await readStatus(projectId);
  const runMode: DemoRunMode =
    status?.runMode ?? (projectId.startsWith("ui_sdk_") ? "sdk" : "offline");
  const llmProvider = status?.llmProvider ?? providerFromProjectId(projectId);
  const verificationProvider = status?.verificationProvider;
  const executionMode = status?.executionMode ?? "efficient";
  const sandboxMode = status?.sandboxMode ?? "runpod";
  const gpuMode = status?.gpuMode ?? "auto";
  const log = await readLogTail(projectId);
  const payload = await payloadForProject(projectId, runMode, log, status ?? undefined);
  const telemetry = await readTelemetryTail(projectId);

  if (status) {
    return {
      projectId: status.projectId,
      outputDir: status.outputDir,
      runMode: status.runMode,
      llmProvider,
      verificationProvider,
      executionMode,
      sandboxMode,
      gpuMode,
      status: status.status,
      sourceKind: status.sourceKind,
      sourceLabel: status.sourceLabel,
      sourceNote: status.sourceNote,
      startedAt: status.startedAt,
      updatedAt: status.updatedAt,
      completedAt: status.completedAt,
      error: status.error,
      pid: status.pid,
      payload,
      log,
      telemetry
    };
  }

  if (payload) {
    return {
      projectId,
      outputDir: payload.outputDir,
      runMode,
      llmProvider,
      verificationProvider,
      executionMode,
      sandboxMode,
      gpuMode,
      status: "completed",
      sourceKind: payload.sourceKind,
      sourceLabel: payload.sourceLabel,
      sourceNote: payload.sourceNote,
      payload,
      log: payload.log,
      telemetry
    };
  }

  return null;
}

function providerFromProjectId(projectId: string): DemoProvider | undefined {
  if (projectId.startsWith("ui_sdk_openai_")) {
    return "openai";
  }
  if (projectId.startsWith("ui_sdk_anthropic_") || projectId.startsWith("ui_sdk_demo_")) {
    return "anthropic";
  }
  return undefined;
}

async function currentRunningRun(
  runMode: DemoRunMode,
  llmProvider?: DemoProvider,
  executionMode?: DemoExecutionMode,
  sandboxMode?: DemoSandboxMode,
  verificationProvider?: DemoProvider,
  gpuMode?: DemoGpuMode
): Promise<LiveDemoRunState | null> {
  const projectId = await latestProjectId(
    runMode,
    llmProvider,
    executionMode,
    sandboxMode,
    verificationProvider,
    gpuMode
  );
  if (!projectId) {
    return null;
  }

  const status = await readStatus(projectId);
  const log = await readLogTail(projectId);

  if (status && isStaleDemoRun(status)) {
    await writeStatus(projectId, {
      ...status,
      status: "failed",
      updatedAt: new Date().toISOString(),
      completedAt: new Date().toISOString(),
      error: summarizeRunFailure(log)
    });
    return null;
  }

  const state = await inferState(projectId);
  return state?.status === "running" || state?.status === "queued" ? state : null;
}

async function terminateRunProcess(pid: number): Promise<void> {
  if (process.platform === "win32") {
    try {
      await execFileAsync("taskkill", ["/PID", String(pid), "/T", "/F"]);
      return;
    } catch {
      // Fall back to process.kill if taskkill is unavailable or the process already exited.
    }
  }

  try {
    process.kill(pid, "SIGTERM");
  } catch {
    // Ignore processes that have already exited.
  }
}

function sandboxNeedsLocalDocker(sandboxMode: DemoSandboxMode): boolean {
  return sandboxMode === "auto" || sandboxMode === "docker";
}

async function ensureDemoSandboxReady(sandboxMode: DemoSandboxMode): Promise<void> {
  if (!sandboxNeedsLocalDocker(sandboxMode)) {
    return;
  }

  const script = `
from backend.agents.execution import ensure_sandbox_mode_available, resolve_sandbox_mode

mode = resolve_sandbox_mode(${pythonStringLiteral(sandboxMode)}, pipeline_mode="sdk")
ensure_sandbox_mode_available(mode)
print("sandbox preflight OK")
`;

  try {
    await execFileAsync(pythonBinary(), ["-c", script], {
      cwd: repoRoot(),
      timeout: 15000,
      env: process.env
    });
  } catch (error) {
    const detail =
      error && typeof error === "object" && "stderr" in error
        ? String((error as { stderr?: unknown }).stderr ?? "").trim()
        : "";
    const message =
      detail ||
      (error instanceof Error ? error.message : "Docker sandbox preflight failed");
    throw new DemoPreflightError(
      [
        "Docker sandbox is not ready for the Python backend.",
        "You do not need to launch the web app in Docker.",
        "Start Docker Desktop or Docker Engine, then install backend dependencies with `python -m pip install -r requirements.txt`.",
        message
      ].join(" ")
    );
  }
}

export async function startDemoRun(
  runMode: DemoRunMode,
  llmProvider: DemoProvider = "anthropic",
  executionMode: DemoExecutionMode = "efficient",
  sandboxMode: DemoSandboxMode = "runpod",
  options?: DemoRunStartOptions
): Promise<LiveDemoRunState> {
  const verificationProvider =
    runMode === "sdk" ? options?.verificationProvider : undefined;
  const gpuMode = options?.gpuMode ?? "auto";
  const existing = await currentRunningRun(
    runMode,
    runMode === "sdk" ? llmProvider : undefined,
    executionMode,
    sandboxMode,
    verificationProvider,
    gpuMode
  );
  if (existing) {
    return existing;
  }

  await ensureDemoSandboxReady(sandboxMode);

  const uploadedPaper = options?.uploadedPaper
    ? await stageUploadedPaper(options.uploadedPaper)
    : null;
  const projectId = uploadedPaper
    ? projectIdForUploadedPdfPath(uploadedPaper.sourcePath)
    : runMode === "sdk"
      ? `ui_sdk_${llmProvider}_review_${verificationProvider ?? "same"}_demo_${Date.now()}`
      : `ui_demo_${Date.now()}`;
  const outputDir = runDir(projectId);
  const meta = uploadedPaper
    ? buildUploadedPaperMeta(
        projectId,
        outputDir,
        runMode,
        runMode === "sdk" ? llmProvider : undefined,
        verificationProvider,
        executionMode,
        sandboxMode,
        gpuMode,
        uploadedPaper.fileName
      )
    : buildFixtureMeta(
        projectId,
        outputDir,
        runMode,
        runMode === "sdk" ? llmProvider : undefined,
        verificationProvider,
        executionMode,
        sandboxMode,
        gpuMode
      );
  await fs.mkdir(outputDir, { recursive: true });
  const now = new Date().toISOString();
  await writeStatus(projectId, {
    projectId,
    outputDir,
    runMode,
    llmProvider: runMode === "sdk" ? llmProvider : undefined,
    verificationProvider,
    executionMode,
    sandboxMode,
    gpuMode,
    sourceKind: meta.sourceKind,
    sourceLabel: meta.sourceLabel,
    sourceNote: meta.sourceNote,
    status: "queued",
    startedAt: now,
    updatedAt: now
  });

  const stderrFile = await fs.open(logPath(projectId), "a");
  const stdoutFile = await fs.open(path.join(outputDir, "runner.stdout.log"), "a");

  const command = pythonBinary();
  // When using a resolved interpreter (venv path or override), the `-3` shim
  // flag is meaningless. We only need it for the bare Windows `py` launcher.
  const usingPyLauncher =
    process.platform === "win32" && command === "py";
  const args = usingPyLauncher
    ? [
        "-3",
        "-u",
        "-c",
        buildPythonScript(
          projectId,
          runMode,
          llmProvider,
          verificationProvider,
          executionMode,
          sandboxMode,
          gpuMode,
          uploadedPaper ?? undefined
        )
      ]
    : [
        "-u",
        "-c",
        buildPythonScript(
          projectId,
          runMode,
          llmProvider,
          verificationProvider,
          executionMode,
          sandboxMode,
          gpuMode,
          uploadedPaper ?? undefined
        )
      ];

  const child = spawn(command, args, {
    cwd: repoRoot(),
    detached: true,
    stdio: ["ignore", stdoutFile.fd, stderrFile.fd],
    env: {
      ...process.env,
      REPROLAB_GPU_MODE: gpuMode,
      ...(runMode === "sdk" ? { REPROLAB_LLM_PROVIDER: llmProvider } : {}),
      ...(verificationProvider ? { REPROLAB_VERIFICATION_PROVIDER: verificationProvider } : {})
    }
  });

  await writeStatus(projectId, {
    projectId,
    outputDir,
    runMode,
    llmProvider: runMode === "sdk" ? llmProvider : undefined,
    verificationProvider,
    executionMode,
    sandboxMode,
    gpuMode,
    sourceKind: meta.sourceKind,
    sourceLabel: meta.sourceLabel,
    sourceNote: meta.sourceNote,
    status: "queued",
    startedAt: now,
    updatedAt: now,
    pid: child.pid
  });

  child.unref();
  await stderrFile.close();
  await stdoutFile.close();

  return {
    projectId,
    outputDir,
    runMode,
    llmProvider: runMode === "sdk" ? llmProvider : undefined,
    verificationProvider,
    executionMode,
    sandboxMode,
    gpuMode,
    sourceKind: meta.sourceKind,
    sourceLabel: meta.sourceLabel,
    sourceNote: meta.sourceNote,
    status: "queued",
    pid: child.pid,
    payload: null,
    log: ""
  };
}

export async function stopDemoRun(
  runMode: DemoRunMode,
  projectId?: string,
  llmProvider?: DemoProvider,
  executionMode?: DemoExecutionMode,
  sandboxMode?: DemoSandboxMode,
  verificationProvider?: DemoProvider,
  gpuMode?: DemoGpuMode
): Promise<LiveDemoRunState | null> {
  const resolvedProjectId =
    projectId ??
    (await latestProjectId(
      runMode,
      llmProvider,
      executionMode,
      sandboxMode,
      verificationProvider,
      gpuMode
    ));
  if (!resolvedProjectId) {
    return null;
  }

  const status = await readStatus(resolvedProjectId);
  if (!status) {
    return null;
  }

  if (status.status !== "queued" && status.status !== "running") {
    return inferState(resolvedProjectId);
  }

  if (typeof status.pid === "number" && status.pid > 0) {
    await terminateRunProcess(status.pid);
  }

  const stoppedAt = new Date().toISOString();
  await writeStatus(resolvedProjectId, {
    ...status,
    status: "stopped",
    updatedAt: stoppedAt,
    completedAt: stoppedAt,
    error: "Stopped by user"
  });

  return inferState(resolvedProjectId);
}

export async function loadDemoRun(
  projectId?: string,
  runMode?: DemoRunMode,
  llmProvider?: DemoProvider,
  executionMode?: DemoExecutionMode,
  sandboxMode?: DemoSandboxMode,
  verificationProvider?: DemoProvider,
  gpuMode?: DemoGpuMode
): Promise<LiveDemoRunState | null> {
  const resolvedProjectId =
    projectId ??
    (await latestProjectId(
      runMode,
      llmProvider,
      executionMode,
      sandboxMode,
      verificationProvider,
      gpuMode
    ));
  if (!resolvedProjectId) {
    return null;
  }

  return inferState(resolvedProjectId);
}

function projectIdForUploadedPdfPath(filePath: string): string {
  const digest = createHash("sha256")
    .update(`pdf_path:${path.resolve(filePath)}`)
    .digest("hex");
  return `prj_${digest.slice(0, 16)}`;
}

async function stageUploadedPaper(
  upload: UploadedPaperInput
): Promise<UploadedPaperLaunchConfig> {
  const uploadsRoot = path.join(runsRoot(), ".lab_uploads");
  await fs.mkdir(uploadsRoot, { recursive: true });
  const ext = path.extname(upload.fileName) || ".pdf";
  const base = path.basename(upload.fileName, ext).replace(/[^a-zA-Z0-9._-]+/g, "-");
  const stagedName = `${Date.now()}-${randomUUID()}-${base}${ext}`;
  const sourcePath = path.join(uploadsRoot, stagedName);
  await fs.writeFile(sourcePath, upload.bytes);
  return {
    sourcePath,
    fileName: upload.fileName
  };
}

export const __test__ = {
  buildPythonScript,
  ensureDemoSandboxReady,
  projectIdForUploadedPdfPath
};
