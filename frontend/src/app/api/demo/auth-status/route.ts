import { NextResponse } from "next/server";

export const runtime = "nodejs";

function backendBaseUrl(): string {
  return ((process.env.OPENRESEARCH_BACKEND_URL ?? process.env.REPROLAB_BACKEND_URL) ?? "http://127.0.0.1:8000").replace(/\/$/, "");
}

/**
 * GET /api/demo/auth-status — server-side proxy to the backend's GET /auth-status.
 * Returns which LLM providers have working credentials on the server so the
 * upload-view provider picker can disable unavailable options.
 * Not gated by the demo secret — it is a pure capability probe.
 */
export async function GET() {
  try {
    const response = await fetch(`${backendBaseUrl()}/auth-status`, {
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
