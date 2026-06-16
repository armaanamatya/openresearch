import "server-only";

import { promises as fs } from "fs";
import { existsSync } from "fs";
import path from "path";
import { execFile, spawn } from "child_process";
import { promisify } from "util";

const execFileAsync = promisify(execFile);

export interface PaperBenchBundleListing {
  bundles_root: string;
  bundles: Array<{
    paper_id?: string;
    metadata?: Record<string, unknown>;
    has_addendum?: boolean;
    rubric_path?: string;
    error?: string;
  }>;
}

export interface PaperBenchPublishedBaseline {
  mean: number;
  se: number;
}

export interface PaperBenchRubricSummary {
  node_count: number;
  leaf_count: number;
  max_depth: number;
  task_category_weights: Record<string, { weight: number; percent: number; leaf_count: number }>;
  finegrained_category_weights: Record<string, { weight: number; percent: number; leaf_count: number }>;
}

export interface PaperBenchAttempt {
  attempt_id: string;
  seed: number | null;
  status: string;
  elapsed_seconds?: number;
  project_id?: string;
  submission_dir?: string;
  submission_validation?: {
    ok: boolean;
    errors: string[];
    warnings: string[];
    total_bytes: number;
    file_count: number;
    committed_bytes: number | null;
  };
  score?: number | null;
}

export interface PaperBenchRunStatus {
  run_group_id: string;
  paper_id: string;
  bundle_root: string;
  runs_root: string;
  mode: "dry" | "with-pipeline";
  seeds: number[];
  max_parallel: number;
  provider: string | null;
  model: string | null;
  status: "pending" | "running" | "succeeded" | "failed";
  started_at: string;
  updated_at: string;
  completed_at: string | null;
  attempts: PaperBenchAttempt[];
  rubric_summary: PaperBenchRubricSummary;
  code_development_ceiling: number;
  published_baselines: Record<string, PaperBenchPublishedBaseline>;
  blacklist_entries: string[];
  mean_score: number | null;
  standard_error: number | null;
  n_attempts: number;
  error: string | null;
}

export interface StartRunOptions {
  paperId: string;
  seeds: number[];
  withPipeline?: boolean;
  provider?: "anthropic" | "openai";
  model?: string;
  maxParallel?: number;
}

function repoRoot(): string {
  const override = process.env.OPENRESEARCH_REPO_ROOT?.trim();
  if (override) return override;
  return path.join(process.cwd(), "..");
}

function pythonBinary(): string {
  const override = process.env.OPENRESEARCH_PYTHON_BIN?.trim();
  if (override) return override;
  const root = repoRoot();
  const venvPython =
    process.platform === "win32"
      ? path.join(root, ".venv", "Scripts", "python.exe")
      : path.join(root, ".venv", "bin", "python");
  if (existsSync(venvPython)) return venvPython;
  return process.platform === "win32" ? "py" : "python3";
}

function paperbenchRunsDir(): string {
  return path.join(repoRoot(), "runs", "paperbench");
}

async function readJsonFile<T>(filePath: string): Promise<T | null> {
  try {
    const raw = await fs.readFile(filePath, "utf8");
    return JSON.parse(raw) as T;
  } catch {
    return null;
  }
}

export async function listBundles(): Promise<PaperBenchBundleListing> {
  const { stdout } = await execFileAsync(
    pythonBinary(),
    ["-m", "backend.cli", "paperbench", "list"],
    { cwd: repoRoot(), maxBuffer: 8 * 1024 * 1024 }
  );
  return JSON.parse(stdout) as PaperBenchBundleListing;
}

export async function getBundleSummary(paperId: string): Promise<Record<string, unknown>> {
  const { stdout } = await execFileAsync(
    pythonBinary(),
    ["-m", "backend.cli", "paperbench", "summary", "--paper-id", paperId],
    { cwd: repoRoot(), maxBuffer: 8 * 1024 * 1024 }
  );
  return JSON.parse(stdout);
}

export async function listRuns(): Promise<PaperBenchRunStatus[]> {
  const dir = paperbenchRunsDir();
  let entries: string[] = [];
  try {
    entries = await fs.readdir(dir);
  } catch {
    return [];
  }
  const runs: PaperBenchRunStatus[] = [];
  for (const entry of entries) {
    const status = await readJsonFile<PaperBenchRunStatus>(
      path.join(dir, entry, "status.json")
    );
    if (status) runs.push(status);
  }
  runs.sort((a, b) => (a.started_at < b.started_at ? 1 : -1));
  return runs;
}

export async function loadRun(runGroupId: string): Promise<PaperBenchRunStatus | null> {
  return readJsonFile<PaperBenchRunStatus>(
    path.join(paperbenchRunsDir(), runGroupId, "status.json")
  );
}

export async function startRun(options: StartRunOptions): Promise<PaperBenchRunStatus> {
  const args = [
    "-m",
    "backend.cli",
    "paperbench",
    "run",
    "--paper-id",
    options.paperId,
    "--seeds",
    ...options.seeds.map((seed) => String(seed)),
  ];
  if (options.maxParallel) {
    args.push("--max-parallel", String(options.maxParallel));
  }
  if (options.withPipeline) {
    args.push("--with-pipeline");
  }
  if (options.provider) {
    args.push("--provider", options.provider);
  }
  if (options.model) {
    args.push("--model", options.model);
  }

  // For dry mode, run synchronously and return the resulting status.
  if (!options.withPipeline) {
    const { stdout } = await execFileAsync(pythonBinary(), args, {
      cwd: repoRoot(),
      maxBuffer: 16 * 1024 * 1024,
    });
    const handle = JSON.parse(stdout) as { run_group_id: string };
    const status = await loadRun(handle.run_group_id);
    if (!status) throw new Error(`status.json not found for ${handle.run_group_id}`);
    return status;
  }

  // For pipeline mode, spawn detached so the HTTP request can return quickly.
  const child = spawn(pythonBinary(), args, {
    cwd: repoRoot(),
    detached: true,
    stdio: ["ignore", "pipe", "pipe"],
    env: process.env,
  });

  // We need the run_group_id, which the child prints on stdout AFTER it has
  // already created the status file. Wait briefly for the directory to appear,
  // then resolve by reading the newest status.json.
  let captured = "";
  child.stdout.on("data", (chunk: Buffer) => {
    captured += chunk.toString("utf8");
  });

  const runGroupId = await new Promise<string>((resolve, reject) => {
    const start = Date.now();
    const tick = setInterval(() => {
      if (captured.includes("run_group_id")) {
        try {
          const parsed = JSON.parse(captured) as { run_group_id: string };
          clearInterval(tick);
          resolve(parsed.run_group_id);
          return;
        } catch {
          // still streaming
        }
      }
      if (Date.now() - start > 30_000) {
        clearInterval(tick);
        reject(new Error("Timed out waiting for paperbench run start"));
      }
    }, 200);
    child.on("error", (err) => {
      clearInterval(tick);
      reject(err);
    });
  });

  child.unref();
  const status = await loadRun(runGroupId);
  if (!status) throw new Error(`status.json not found for ${runGroupId}`);
  return status;
}
