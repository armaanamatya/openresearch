import { NextResponse } from "next/server";

import { backendBaseUrl } from "@/lib/demo/server-run";
import { gateSecret } from "@/lib/auth/demo-gate";

export const runtime = "nodejs";

function demoSecretHeaders(): Record<string, string> {
  const secret = gateSecret();
  return secret ? { "x-demo-secret": secret } : {};
}

/**
 * Thin proxy to backend ``POST /runs/{project_id}/resume`` — re-spawns the
 * orchestrator subprocess for the project. For RDR-mode runs this picks up
 * from the last on-disk checkpoint. The default RLM mode has no such
 * checkpoint: it restarts the reasoning loop from scratch under the same
 * project id, warm-started only via a preserved implementation cache,
 * cell-level resume, and prior-attempt lessons — never a resumed REPL
 * state. Optional JSON body overrides specific run config knobs (e.g.
 * ``{"executionMode": "max"}``) so a wall-clock timeout can be retried with
 * more headroom.
 */
export async function POST(request: Request): Promise<NextResponse> {
  const projectId = new URL(request.url).searchParams.get("projectId");
  if (!projectId) {
    return NextResponse.json(
      { error: "projectId is required" },
      { status: 400 }
    );
  }
  let body: unknown = null;
  try {
    const text = await request.text();
    body = text ? JSON.parse(text) : {};
  } catch {
    return NextResponse.json(
      { error: "Request body, when present, must be JSON." },
      { status: 400 }
    );
  }
  const response = await fetch(
    `${backendBaseUrl()}/runs/${encodeURIComponent(projectId)}/resume`,
    {
      method: "POST",
      headers: { "content-type": "application/json", ...demoSecretHeaders() },
      body: JSON.stringify(body ?? {})
    }
  );
  const text = await response.text();
  if (!response.ok) {
    return new NextResponse(text || "Upstream error", { status: response.status });
  }
  try {
    return NextResponse.json(JSON.parse(text));
  } catch {
    return new NextResponse(text, { status: response.status });
  }
}
