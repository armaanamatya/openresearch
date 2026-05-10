import { LiveDemoClient } from "@/components/lab/live-demo-client";
import type { LiveDemoRunState } from "@/lib/demo/demo-run-types";

export const dynamic = "force-dynamic";

const SSR_FETCH_TIMEOUT_MS = 1500;

function backendBaseUrl(): string {
  return (process.env.REPROLAB_BACKEND_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");
}

// SSR runs in the Next.js server process. If the FastAPI backend is busy
// (e.g. a single-worker uvicorn streaming SSE for an active pipeline),
// a no-timeout fetch will hang indefinitely and the page renders nothing
// — a white screen. Cap the SSR fetch with a tight AbortSignal and let
// the client hydrate from live state on its own when SSR can't get a
// snapshot in time.
async function loadInitialRun(): Promise<LiveDemoRunState | null> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), SSR_FETCH_TIMEOUT_MS);
  try {
    const response = await fetch(`${backendBaseUrl()}/runs/latest`, {
      cache: "no-store",
      signal: controller.signal
    });
    if (!response.ok) {
      return null;
    }
    return (await response.json()) as LiveDemoRunState;
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

export default async function LabPage() {
  const initialRun = await loadInitialRun();

  return <LiveDemoClient initialRun={initialRun} />;
}
