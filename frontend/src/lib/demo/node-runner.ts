import "server-only";

import { promises as fs } from "fs";
import { existsSync } from "fs";
import path from "path";
import { spawn } from "child_process";

import type {
  DemoProvider,
  DemoRunMode,
  DemoRunStatus,
  LiveDemoRunState
} from "./demo-run-types";
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
  status: DemoRunStatus;
  startedAt: string;
  updatedAt: string;
  completedAt?: string;
  error?: string;
}

function repoRoot(): string {
  return path.resolve(/* turbopackIgnore: true */ process.cwd(), "..");
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

function defaultMeta(
  projectId: string,
  outputDir: string,
  runMode: DemoRunMode,
  llmProvider?: DemoProvider
): LiveDemoMeta {
  return {
    projectId,
    outputDir,
    sourceKind: "workspace_fixture",
    runMode,
    llmProvider,
    sourceLabel: "In-repo PPO workspace fixture",
    sourceNote:
      "The repo currently does not contain a checked-in paper PDF, so this UI demo uses the deterministic PPO workspace fixture that already drives the end-to-end pipeline tests."
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

async function payloadForProject(
  projectId: string,
  runMode: DemoRunMode,
  log = "",
  llmProvider?: DemoProvider
) {
  const outputDir = runDir(projectId);
  const state = await readPipelineState(projectId);
  if (!state) {
    return null;
  }

  return buildLiveDemoDashboard(
    state,
    defaultMeta(projectId, outputDir, runMode, llmProvider),
    log
  );
}

function buildPythonScript(
  projectId: string,
  runMode: DemoRunMode,
  llmProvider: DemoProvider
): string {
  return `
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from backend.agents.pipeline import run_pipeline_offline, run_pipeline_sdk

workspace = json.loads(r'''${JSON.stringify(DEMO_WORKSPACE)}''')
project_id = r'''${projectId}'''
llm_provider = r'''${llmProvider}'''
runs_root = Path(r'''${runsRoot()}''')
output_dir = (runs_root / project_id).resolve()
output_dir.mkdir(parents=True, exist_ok=True)
status_path = output_dir / "demo_status.json"

def now():
    return datetime.now(timezone.utc).isoformat()

def write_status(status, error=None, completed_at=None):
    payload = {
        "projectId": project_id,
        "outputDir": str(output_dir),
        "runMode": "${runMode}",
        "status": status,
        "startedAt": started_at,
        "updatedAt": now(),
    }
    if "${runMode}" == "sdk":
        payload["llmProvider"] = llm_provider
    if completed_at:
        payload["completedAt"] = completed_at
    if error:
        payload["error"] = error
    status_path.write_text(json.dumps(payload, indent=2))

started_at = now()
write_status("running")

try:
    if "${runMode}" == "sdk":
        asyncio.run(run_pipeline_sdk(
            project_id,
            runs_root,
            workspace,
            provider=llm_provider,
            user_hints=["Keep this as a lightweight smoke test"],
            n_improvement_paths=1,
        ))
    else:
        run_pipeline_offline(project_id, runs_root, workspace)
    write_status("completed", completed_at=now())
except Exception as exc:
    write_status("failed", error=f"{type(exc).__name__}: {exc}", completed_at=now())
    raise
`;
}

async function latestProjectId(
  runMode?: DemoRunMode,
  llmProvider?: DemoProvider
): Promise<string | null> {
  try {
    const entries = await fs.readdir(runsRoot(), { withFileTypes: true });
    const filtered = entries.filter((entry) => {
      if (!entry.isDirectory()) {
        return false;
      }
      if (runMode === "sdk") {
        if (llmProvider === "anthropic") {
          return (
            entry.name.startsWith("ui_sdk_anthropic_demo_") ||
            entry.name.startsWith("ui_sdk_demo_")
          );
        }
        return llmProvider
          ? entry.name.startsWith(`ui_sdk_${llmProvider}_demo_`)
          : entry.name.startsWith("ui_sdk_");
      }
      if (runMode === "offline") {
        return entry.name.startsWith("ui_demo_");
      }
      return entry.name.startsWith("ui_demo_") || entry.name.startsWith("ui_sdk_");
    });

    const candidates = await Promise.all(
      filtered.map(async (entry) => {
        const statusStat = await fs.stat(statusPath(entry.name)).catch(() => null);
        const pipelineStat = await fs.stat(pipelineStatePath(entry.name)).catch(() => null);
        const mtimeMs = Math.max(statusStat?.mtimeMs ?? 0, pipelineStat?.mtimeMs ?? 0);
        return mtimeMs > 0 ? { projectId: entry.name, mtimeMs } : null;
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
  const runMode: DemoRunMode = projectId.startsWith("ui_sdk_") ? "sdk" : "offline";
  const llmProvider = status?.llmProvider ?? providerFromProjectId(projectId);
  const log = await readLogTail(projectId);
  const payload = await payloadForProject(projectId, runMode, log, llmProvider);

  if (status) {
    return {
      projectId: status.projectId,
      outputDir: status.outputDir,
      runMode: status.runMode,
      llmProvider,
      status: status.status,
      startedAt: status.startedAt,
      updatedAt: status.updatedAt,
      completedAt: status.completedAt,
      error: status.error,
      payload,
      log
    };
  }

  if (payload) {
    return {
      projectId,
      outputDir: payload.outputDir,
      runMode,
      llmProvider,
      status: "completed",
      payload,
      log: payload.log
    };
  }

  return null;
}

function providerFromProjectId(projectId: string): DemoProvider | undefined {
  if (projectId.startsWith("ui_sdk_openai_demo_")) {
    return "openai";
  }
  if (projectId.startsWith("ui_sdk_anthropic_demo_") || projectId.startsWith("ui_sdk_demo_")) {
    return "anthropic";
  }
  return undefined;
}

async function currentRunningRun(
  runMode: DemoRunMode,
  llmProvider?: DemoProvider
): Promise<LiveDemoRunState | null> {
  const projectId = await latestProjectId(runMode, llmProvider);
  if (!projectId) {
    return null;
  }

  const state = await inferState(projectId);
  return state?.status === "running" || state?.status === "queued" ? state : null;
}

export async function startDemoRun(
  runMode: DemoRunMode,
  llmProvider: DemoProvider = "anthropic"
): Promise<LiveDemoRunState> {
  const existing = await currentRunningRun(runMode, runMode === "sdk" ? llmProvider : undefined);
  if (existing) {
    return existing;
  }

  const projectId =
    runMode === "sdk"
      ? `ui_sdk_${llmProvider}_demo_${Date.now()}`
      : `ui_demo_${Date.now()}`;
  const outputDir = runDir(projectId);
  await fs.mkdir(outputDir, { recursive: true });
  const now = new Date().toISOString();
  await writeStatus(projectId, {
    projectId,
    outputDir,
    runMode,
    llmProvider: runMode === "sdk" ? llmProvider : undefined,
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
    ? ["-3", "-u", "-c", buildPythonScript(projectId, runMode, llmProvider)]
    : ["-u", "-c", buildPythonScript(projectId, runMode, llmProvider)];

  const child = spawn(command, args, {
    cwd: repoRoot(),
    detached: true,
    stdio: ["ignore", stdoutFile.fd, stderrFile.fd],
    env: {
      ...process.env,
      ...(runMode === "sdk" ? { REPROLAB_LLM_PROVIDER: llmProvider } : {})
    }
  });

  child.unref();
  await stderrFile.close();
  await stdoutFile.close();

  return {
    projectId,
    outputDir,
    runMode,
    llmProvider: runMode === "sdk" ? llmProvider : undefined,
    status: "queued",
    payload: null,
    log: ""
  };
}

export async function loadDemoRun(
  projectId?: string,
  runMode?: DemoRunMode,
  llmProvider?: DemoProvider
): Promise<LiveDemoRunState | null> {
  const resolvedProjectId = projectId ?? (await latestProjectId(runMode, llmProvider));
  if (!resolvedProjectId) {
    return null;
  }

  return inferState(resolvedProjectId);
}
