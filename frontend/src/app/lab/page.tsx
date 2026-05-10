import { LiveDemoClient } from "@/components/lab/live-demo-client";
import type { LiveDemoRunState } from "@/lib/demo/demo-run-types";

export const dynamic = "force-dynamic";

function backendBaseUrl(): string {
  return (process.env.REPROLAB_BACKEND_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");
}

async function loadInitialRun(): Promise<LiveDemoRunState | null> {
  try {
    const response = await fetch(`${backendBaseUrl()}/runs/latest`, {
      cache: "no-store"
    });
    if (!response.ok) {
      return null;
    }
    return (await response.json()) as LiveDemoRunState;
  } catch {
    return null;
  }
}

export default async function LabPage() {
  const initialRun = await loadInitialRun();

  return <LiveDemoClient initialRun={initialRun} />;
}
