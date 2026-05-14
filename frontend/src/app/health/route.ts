import { NextResponse } from "next/server";

export const runtime = "nodejs";

function backendBaseUrl(): string {
  return (process.env.REPROLAB_BACKEND_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");
}

export async function GET() {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 3000);
  try {
    const response = await fetch(`${backendBaseUrl()}/health`, {
      cache: "no-store",
      signal: controller.signal,
    });
    if (!response.ok) {
      return NextResponse.json({ status: "backend_unhealthy" }, { status: 503 });
    }
    return NextResponse.json({ status: "ok" });
  } catch {
    return NextResponse.json({ status: "backend_unreachable" }, { status: 503 });
  } finally {
    clearTimeout(timer);
  }
}
