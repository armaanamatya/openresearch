import { NextResponse } from "next/server";

import { backendBaseUrl } from "@/lib/demo/server-run";
import { gateSecret } from "@/lib/auth/demo-gate";

export const runtime = "nodejs";

function demoSecretHeaders(): Record<string, string> {
  const secret = gateSecret();
  return secret ? { "x-demo-secret": secret } : {};
}

/**
 * Thin proxy to backend ``POST /runs/{project_id}/resume`` — picks the
 * project up from the last on-disk checkpoint and re-spawns the
 * orchestrator subprocess. Optional JSON body overrides specific run
 * config knobs (e.g. ``{"executionMode": "max"}``) so a wall-clock
 * timeout can be retried with more headroom without re-running every
 * earlier stage from scratch.
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
