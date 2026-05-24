import { NextResponse } from "next/server";

import { backendBaseUrl } from "@/lib/demo/server-run";

export const runtime = "nodejs";

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ projectId: string }> }
): Promise<NextResponse> {
  const { projectId } = await params;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 8000);
  try {
    const response = await fetch(
      `${backendBaseUrl()}/runs/${encodeURIComponent(projectId)}/reports`,
      {
        method: "GET",
        headers: { "Content-Type": "application/json" },
        cache: "no-store",
        signal: controller.signal,
      }
    );
    const text = await response.text();
    if (!response.ok) {
      return new NextResponse(text || "Upstream error", { status: 502 });
    }
    try {
      return NextResponse.json(JSON.parse(text));
    } catch {
      return new NextResponse(text, { status: response.status });
    }
  } catch {
    return new NextResponse(
      JSON.stringify({ workers: [], summary: {} }),
      { status: 200, headers: { "Content-Type": "application/json" } }
    );
  } finally {
    clearTimeout(timer);
  }
}
