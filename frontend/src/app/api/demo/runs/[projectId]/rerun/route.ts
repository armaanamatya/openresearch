import { NextResponse } from "next/server";

import { backendBaseUrl } from "@/lib/demo/server-run";
import { gateSecret } from "@/lib/auth/demo-gate";

export const runtime = "nodejs";

function demoSecretHeaders(): Record<string, string> {
  const secret = gateSecret();
  return secret ? { "x-demo-secret": secret } : {};
}

/**
 * Thin proxy to backend ``POST /runs/{project_id}/rerun``.
 *
 * Reads the original run's source PDF from disk and spawns a fresh run
 * with a new project_id.  Returns the new run's LiveRunState on 202,
 * passes through 404 (project not found) and 422 (source PDF gone).
 */
export async function POST(
  _request: Request,
  { params }: { params: Promise<{ projectId: string }> }
): Promise<NextResponse> {
  const { projectId } = await params;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 10_000);
  try {
    const response = await fetch(
      `${backendBaseUrl()}/runs/${encodeURIComponent(projectId)}/rerun`,
      {
        method: "POST",
        headers: { ...demoSecretHeaders() },
        cache: "no-store",
        signal: controller.signal,
      }
    );
    const text = await response.text();
    if (!response.ok) {
      try {
        return NextResponse.json(JSON.parse(text), { status: response.status });
      } catch {
        return new NextResponse(text || "Upstream error", { status: response.status });
      }
    }
    try {
      return NextResponse.json(JSON.parse(text), { status: 202 });
    } catch {
      return new NextResponse(text, { status: response.status });
    }
  } catch (err) {
    const aborted = err instanceof Error && (err.name === "AbortError" || err.name === "TimeoutError");
    return NextResponse.json(
      { error: aborted ? "Rerun request timed out" : "Backend unreachable" },
      { status: 504 }
    );
  } finally {
    clearTimeout(timer);
  }
}
