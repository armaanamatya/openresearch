import { NextResponse } from "next/server";

export const runtime = "nodejs";

export async function GET() {
  try {
    const backendUrl = (process.env.OPENRESEARCH_BACKEND_URL ?? process.env.REPROLAB_BACKEND_URL) ?? "http://localhost:8000";
    const res = await fetch(`${backendUrl}/health`, { cache: "no-store" });
    const body = await res.json();
    return NextResponse.json(body, { status: res.status });
  } catch {
    return NextResponse.json({ ok: false, error: "backend unreachable" }, { status: 503 });
  }
}
