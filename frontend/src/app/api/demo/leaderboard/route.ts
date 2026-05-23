import { NextResponse } from "next/server";

export const runtime = "nodejs";

function backendBaseUrl(): string {
  return (process.env.REPROLAB_BACKEND_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");
}

/**
 * GET /api/demo/leaderboard — server-side proxy to the backend's
 * GET /leaderboard. Read-only; not gated by the demo secret.
 *
 * Spec: docs/superpowers/specs/2026-05-23-rubric-climb-leaderboard.md §4.4.
 */
export async function GET(request: Request) {
  const incoming = new URL(request.url);
  const search = incoming.search; // includes leading "?" when non-empty
  const target = `${backendBaseUrl()}/leaderboard${search}`;

  try {
    const response = await fetch(target, {
      method: "GET",
      headers: { accept: "application/json" },
      cache: "no-store",
    });
    const body = await response.text();
    return new NextResponse(body, {
      status: response.status,
      headers: {
        "content-type": response.headers.get("content-type") ?? "application/json",
        "cache-control": "no-store",
      },
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Backend unreachable";
    return NextResponse.json({ detail: message }, { status: 502 });
  }
}
