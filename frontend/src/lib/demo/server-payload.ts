import "server-only";

import type { LiveDemoRunState } from "./demo-run-types";
import {
  metaFromStatus,
  readLogTail,
  readPipelineState,
  readTelemetryTail
} from "./server-fs";
import {
  buildLiveDemoDashboard,
  type PipelineStateDocument
} from "./pipeline-dashboard";

const TERMINAL_STATUSES: ReadonlySet<LiveDemoRunState["status"]> = new Set([
  "completed",
  "failed",
  "stopped"
]);

// Bound the cache so an abandoned queued/running run cannot leak forever.
// 64 entries × ~100 KB pipeline_state = ~6 MB worst case; TTL backs that up.
const MAX_CACHE_ENTRIES = 64;
const CACHE_TTL_MS = 30 * 60 * 1000;

interface CachedState {
  state: PipelineStateDocument;
  expiresAt: number;
}

// Map iteration order is insertion order — re-set on access to make this an LRU.
const pipelineStateCache = new Map<string, CachedState>();

function evictExpired(now: number): void {
  for (const [key, entry] of pipelineStateCache) {
    if (entry.expiresAt <= now) pipelineStateCache.delete(key);
  }
}

function cacheGet(projectId: string): PipelineStateDocument | null {
  const now = Date.now();
  evictExpired(now);
  const entry = pipelineStateCache.get(projectId);
  if (!entry) return null;
  pipelineStateCache.delete(projectId);
  pipelineStateCache.set(projectId, { state: entry.state, expiresAt: now + CACHE_TTL_MS });
  return entry.state;
}

function cacheSet(projectId: string, state: PipelineStateDocument): void {
  const now = Date.now();
  evictExpired(now);
  pipelineStateCache.delete(projectId);
  pipelineStateCache.set(projectId, { state, expiresAt: now + CACHE_TTL_MS });
  while (pipelineStateCache.size > MAX_CACHE_ENTRIES) {
    const oldest = pipelineStateCache.keys().next().value;
    if (oldest === undefined) break;
    pipelineStateCache.delete(oldest);
  }
}

async function pipelineStateForProject(
  projectId: string
): Promise<PipelineStateDocument | null> {
  const pipelineState = await readPipelineState(projectId);
  if (pipelineState) {
    cacheSet(projectId, pipelineState);
    return pipelineState;
  }
  return cacheGet(projectId);
}

export async function enrichRunStateWithPayload(
  state: LiveDemoRunState
): Promise<LiveDemoRunState> {
  const pipelineState = await pipelineStateForProject(state.projectId);
  if (TERMINAL_STATUSES.has(state.status)) {
    pipelineStateCache.delete(state.projectId);
  }
  if (!pipelineState) {
    return state;
  }

  const log = state.log || (await readLogTail(state.projectId));
  const meta = metaFromStatus(state.projectId, state.outputDir, state.runMode, state);
  const payload = buildLiveDemoDashboard(pipelineState, meta, log);
  const telemetry = state.telemetry?.length
    ? state.telemetry
    : await readTelemetryTail(state.projectId);

  return { ...state, payload, log, telemetry };
}
