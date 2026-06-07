import "server-only";
import { backendBaseUrl } from "@/lib/demo/server-run";

export interface RecentRunSummary {
  projectId: string;
  status: string;
  sourceLabel?: string;
  startedAt?: string;
  updatedAt?: string;
  paperTitle?: string;
  paper?: { id?: string; title?: string } | null;
  paperId?: string;
  outputDir?: string;
  runDir?: string;
}

/**
 * Superset of RecentRunSummary used by the /library page. The backend
 * returns full LiveDemoRunState dicts; we pluck only what the table
 * actually renders so the page stays cheap to serialize across the RSC
 * boundary. Anything optional that is absent on the backend payload
 * surfaces as undefined here — the table is responsible for falling
 * back to an em-dash.
 */
export interface RunSummary {
  projectId: string;
  status: string;
  sourceLabel?: string;
  startedAt?: string;
  updatedAt?: string;
  completedAt?: string;
  paperTitle?: string;
  paper?: { id?: string; title?: string } | null;
  paperId?: string;
  outputDir?: string;
  benchmark?: {
    overallScore?: number;
    deltaValue?: number;
  } | null;
  telemetry?: {
    totalMessages: number;
    totalOutputChars: number;
    totalDurationSeconds: number;
  };
}

interface RawTelemetryRecord {
  message_count?: number;
  output_chars?: number;
  duration_seconds?: number;
}

interface RawBackendRun {
  projectId?: string;
  status?: string;
  sourceLabel?: string;
  startedAt?: string;
  updatedAt?: string;
  completedAt?: string;
  paperTitle?: string;
  paper?: { id?: string; title?: string } | null;
  paperId?: string;
  outputDir?: string;
  benchmark?: {
    overallScore?: number;
    deltaValue?: number;
  } | null;
  telemetry?: RawTelemetryRecord[];
}

function rollupTelemetry(records: RawTelemetryRecord[] | undefined): RunSummary["telemetry"] {
  if (!records || records.length === 0) return undefined;
  let messages = 0;
  let chars = 0;
  let seconds = 0;
  for (const r of records) {
    if (typeof r.message_count === "number") messages += r.message_count;
    if (typeof r.output_chars === "number") chars += r.output_chars;
    if (typeof r.duration_seconds === "number") seconds += r.duration_seconds;
  }
  return {
    totalMessages: messages,
    totalOutputChars: chars,
    totalDurationSeconds: seconds
  };
}

function toRunSummary(raw: RawBackendRun): RunSummary | null {
  if (!raw || typeof raw.projectId !== "string" || typeof raw.status !== "string") {
    return null;
  }
  return {
    projectId: raw.projectId,
    status: raw.status,
    sourceLabel: raw.sourceLabel,
    startedAt: raw.startedAt,
    updatedAt: raw.updatedAt,
    completedAt: raw.completedAt,
    paperTitle: raw.paperTitle,
    paper: raw.paper ?? undefined,
    paperId: raw.paperId,
    outputDir: raw.outputDir,
    benchmark: raw.benchmark ?? undefined,
    telemetry: rollupTelemetry(raw.telemetry)
  };
}

export async function fetchRecentRuns(limit = 10): Promise<RecentRunSummary[]> {
  return (await fetchRecentRunsResult(limit)).runs;
}

export async function fetchRecentRunsResult(
  limit = 10
): Promise<{ runs: RecentRunSummary[]; error: string | null }> {
  try {
    const response = await fetch(
      `${backendBaseUrl()}/runs?limit=${encodeURIComponent(String(limit))}`,
      { cache: "no-store" }
    );
    if (!response.ok) {
      return { runs: [], error: `Recent runs unavailable (HTTP ${response.status}).` };
    }
    const body = (await response.json()) as RecentRunSummary[];
    return {
      runs: Array.isArray(body) ? body : [],
      error: Array.isArray(body) ? null : "Recent runs response was not a list.",
    };
  } catch {
    return { runs: [], error: "Recent runs unavailable." };
  }
}

export interface FetchRunListParams {
  limit?: number;
  status?: string;
  q?: string;
  order_by?: string;
}

export async function fetchRunList(
  params: FetchRunListParams = {}
): Promise<RunSummary[]> {
  return (await fetchRunListResult(params)).runs;
}

export async function fetchRunListResult(
  params: FetchRunListParams = {}
): Promise<{ runs: RunSummary[]; error: string | null }> {
  const qs = new URLSearchParams();
  if (typeof params.limit === "number") qs.set("limit", String(params.limit));
  if (params.status) qs.set("status", params.status);
  if (params.q) qs.set("q", params.q);
  if (params.order_by) qs.set("order_by", params.order_by);

  try {
    const response = await fetch(`${backendBaseUrl()}/runs?${qs.toString()}`, {
      cache: "no-store"
    });
    if (!response.ok) return { runs: [], error: `Run library unavailable (HTTP ${response.status}).` };
    const body = (await response.json()) as RawBackendRun[];
    if (!Array.isArray(body)) return { runs: [], error: "Run library response was not a list." };
    const out: RunSummary[] = [];
    for (const raw of body) {
      const row = toRunSummary(raw);
      if (row) out.push(row);
    }
    return { runs: out, error: null };
  } catch {
    return { runs: [], error: "Run library unavailable." };
  }
}
