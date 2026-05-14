import { NextResponse } from "next/server";

export const runtime = "nodejs";

function backendBaseUrl(): string {
  return (process.env.REPROLAB_BACKEND_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");
}

export async function GET(request: Request) {
  const projectId = new URL(request.url).searchParams.get("projectId");
  if (!projectId) {
    return NextResponse.json({ error: "projectId is required" }, { status: 400 });
  }

  try {
    const response = await fetch(
      `${backendBaseUrl()}/runs/${encodeURIComponent(projectId)}/final-report`,
      { cache: "no-store" }
    );
    if (!response.ok || !response.body) {
      return new NextResponse(await response.text(), { status: response.status });
    }
    return new Response(response.body, {
      status: response.status,
      headers: {
        "cache-control": "no-store",
        "content-disposition":
          response.headers.get("content-disposition") ??
          'inline; filename="final_benchmark_report.md"',
        "content-type": response.headers.get("content-type") ?? "text/markdown; charset=utf-8"
      }
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unable to load final report";
    return NextResponse.json({ error: message }, { status: 502 });
  }
}
