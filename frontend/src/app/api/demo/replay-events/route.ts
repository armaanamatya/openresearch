import { NextResponse } from "next/server";

export const runtime = "nodejs";

function backendBaseUrl(): string {
  return ((process.env.OPENRESEARCH_BACKEND_URL ?? process.env.REPROLAB_BACKEND_URL) ?? "http://127.0.0.1:8000").replace(/\/$/, "");
}

const EMPTY = { events: [], metadata: { count: 0, earliestTs: null, latestTs: null } };

/**
 * GET /api/demo/replay-events?projectId=<id>
 * Proxies the backend's persisted-event list for UI timeline replay. Returns the
 * same {events, metadata} shape; on any backend error returns an empty payload
 * (never throws into the client) so the replay surface degrades gracefully.
 */
export async function GET(request: Request) {
  const projectId = new URL(request.url).searchParams.get("projectId");
  if (!projectId) {
    return NextResponse.json({ error: "projectId is required" }, { status: 400 });
  }
  try {
    const response = await fetch(
      `${backendBaseUrl()}/runs/${encodeURIComponent(projectId)}/replay-events`,
      { cache: "no-store" }
    );
    if (!response.ok) {
      return NextResponse.json(EMPTY, { status: response.status });
    }
    const data = await response.json();
    return NextResponse.json(data, { headers: { "cache-control": "no-store" } });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unable to load replay events";
    return NextResponse.json({ ...EMPTY, error: message }, { status: 502 });
  }
}
