import { LabShell } from "@/components/lab/lab-shell";
import { fetchRunById, backendBaseUrl } from "@/lib/demo/server-run";
import { fetchRecentRunsResult } from "@/lib/runs/server-list";
import { fetchModels } from "@/lib/models/server-fetch";
import { fetchLeaderboardRows } from "@/lib/leaderboard/server-fetch";
import { filterRecentRows } from "@/lib/leaderboard/filter-recent";
import type { AuthStatus, DemoSandboxMode } from "@/lib/demo/demo-run-types";

// The current run is identified by the `?projectId=` query param — that
// makes the URL the single source of truth, so a refresh or a shared
// link restores the exact run instead of dropping back to the upload
// view. No projectId → fresh upload view (the client may still
// auto-resume from localStorage).
export const dynamic = "force-dynamic";

async function fetchAuthStatus(): Promise<AuthStatus | null> {
  try {
    const response = await fetch(`${backendBaseUrl()}/auth-status`, { cache: "no-store" });
    if (!response.ok) return null;
    return (await response.json()) as AuthStatus;
  } catch {
    return null;
  }
}

export default async function LabPage({
  searchParams
}: {
  searchParams: Promise<{ projectId?: string | string[] }>;
}) {
  const params = await searchParams;
  const raw = params.projectId;
  const projectId = Array.isArray(raw) ? raw[0] : raw;
  const [initialRun, recentResult, models, authStatus, leaderboardResult] = await Promise.all([
    projectId ? fetchRunById(projectId) : Promise.resolve(null),
    fetchRecentRunsResult(8),
    fetchModels(),
    fetchAuthStatus(),
    fetchLeaderboardRows({ orderBy: "finished_at", limit: 30 }),
  ]);
  const rawSandbox = process.env.OPENRESEARCH_DEFAULT_SANDBOX;
  const serverDefaultSandbox: DemoSandboxMode | undefined =
    rawSandbox === "runpod" || rawSandbox === "docker" || rawSandbox === "local" || rawSandbox === "auto"
      ? rawSandbox
      : undefined;

  return (
    <LabShell
      initialRun={initialRun}
      initialRecents={recentResult.runs}
      initialRecentsError={recentResult.error}
      initialModels={models}
      initialAuthStatus={authStatus}
      serverDefaultSandbox={serverDefaultSandbox}
      presentationMode="internal"
      initialLeaderboardRows={filterRecentRows(leaderboardResult.rows)}
      initialLeaderboardError={leaderboardResult.error}
    />
  );
}
