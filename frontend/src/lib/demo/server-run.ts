import "server-only";

import type { LiveDemoRunState } from "./demo-run-types";
import { enrichRunStateWithPayload } from "./server-payload";

export function backendBaseUrl(): string {
  return (process.env.REPROLAB_BACKEND_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");
}

// Backend GET timeout for the lab page's status polling. Single-worker
// uvicorn (--reload) blocks on long SSE streams, so we cap how long the
// browser waits and let the client retry with backoff. Bumped from 4s →
// 10s on 2026-05-23 after a mid-run Playwright snapshot captured the
// SSR fetch 504-ing while the backend was mid-implement_baseline (lab
// page rendered with an empty <main>). 10s still feels snappy on the
// warm path while tolerating the occasional 5-8s tail when the FastAPI
// worker is processing a heavy SSE event payload.
export const BACKEND_GET_TIMEOUT_MS = 10000;

// Soft cap on payload enrichment so a slow filesystem read or
// buildLiveDemoDashboard pass cannot stall the GET response. On timeout
// we return the un-enriched state and let the next poll/SSE frame retry.
const ENRICH_TIMEOUT_MS = 750;

export async function enrichOrTimeout(state: LiveDemoRunState): Promise<LiveDemoRunState> {
  let timer: ReturnType<typeof setTimeout> | null = null;
  try {
    return await Promise.race<LiveDemoRunState>([
      enrichRunStateWithPayload(state),
      new Promise<LiveDemoRunState>((resolve) => {
        timer = setTimeout(() => resolve(state), ENRICH_TIMEOUT_MS);
      })
    ]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}

// Fetch a single run by id, server-side. NEVER throws — returns null on
// missing id / 404 / timeout / network error — so it is safe to call
// during an SSR render. A stale `?projectId=` link therefore falls
// through cleanly to the client's auto-resume / upload view.
export async function fetchRunById(projectId: string): Promise<LiveDemoRunState | null> {
  if (!projectId) {
    return null;
  }
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), BACKEND_GET_TIMEOUT_MS);
  try {
    const response = await fetch(`${backendBaseUrl()}/runs/${encodeURIComponent(projectId)}`, {
      cache: "no-store",
      signal: controller.signal
    });
    if (!response.ok) {
      return null;
    }
    const state = (await response.json()) as LiveDemoRunState | null;
    if (!state || !state.projectId) {
      return null;
    }
    return await enrichOrTimeout(state);
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}
