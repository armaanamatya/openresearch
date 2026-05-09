import { NextResponse } from "next/server";

import { loadDemoRun, startDemoRun } from "@/lib/demo/node-runner";

export const runtime = "nodejs";

function toRunMode(request: Request): "offline" | "sdk" | undefined {
  const url = new URL(request.url);
  const value = url.searchParams.get("mode");
  if (value === "offline" || value === "sdk") {
    return value;
  }
  return undefined;
}

function toProjectId(request: Request): string | undefined {
  const url = new URL(request.url);
  const value = url.searchParams.get("projectId");
  return value || undefined;
}

export async function GET(request: Request) {
  const latest = await loadDemoRun(toProjectId(request), toRunMode(request));
  return NextResponse.json(latest);
}

export async function POST(request: Request) {
  try {
    const run = await startDemoRun(toRunMode(request) ?? "offline");
    return NextResponse.json(run, { status: 202 });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Demo pipeline failed";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
