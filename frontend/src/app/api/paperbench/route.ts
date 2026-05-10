import { NextResponse } from "next/server";

import {
  getBundleSummary,
  listBundles,
  listRuns,
  loadRun,
  startRun,
} from "@/lib/paperbench/runner";

export const runtime = "nodejs";

export async function GET(request: Request) {
  const url = new URL(request.url);
  const view = url.searchParams.get("view");
  const runGroupId = url.searchParams.get("runGroupId");
  const paperId = url.searchParams.get("paperId");

  try {
    if (view === "bundles") {
      return NextResponse.json(await listBundles());
    }
    if (view === "summary" && paperId) {
      return NextResponse.json(await getBundleSummary(paperId));
    }
    if (runGroupId) {
      const run = await loadRun(runGroupId);
      if (!run) {
        return NextResponse.json({ error: "Run not found" }, { status: 404 });
      }
      return NextResponse.json(run);
    }
    return NextResponse.json({ runs: await listRuns() });
  } catch (error) {
    const message = error instanceof Error ? error.message : "PaperBench query failed";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

interface StartBody {
  paperId?: string;
  seeds?: number[];
  withPipeline?: boolean;
  provider?: "anthropic" | "openai";
  model?: string;
  maxParallel?: number;
}

export async function POST(request: Request) {
  try {
    const body = (await request.json().catch(() => ({}))) as StartBody;
    if (!body.paperId) {
      return NextResponse.json(
        { error: "paperId is required" },
        { status: 400 }
      );
    }
    const seeds =
      Array.isArray(body.seeds) && body.seeds.length > 0
        ? body.seeds.map((seed) => Number(seed))
        : [0];
    const status = await startRun({
      paperId: body.paperId,
      seeds,
      // Default to running the real pipeline; pass withPipeline=false to opt
      // into dry validation without LLM calls.
      withPipeline: body.withPipeline === false ? false : true,
      provider: body.provider,
      model: body.model,
      maxParallel: body.maxParallel,
    });
    return NextResponse.json(status, { status: 202 });
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "PaperBench run failed to start";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
