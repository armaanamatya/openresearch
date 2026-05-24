import { NextResponse } from "next/server";

import { backendBaseUrl } from "@/lib/demo/server-run";

export const runtime = "nodejs";

type RouteContext = {
  params: Promise<{ projectId: string }> | { projectId: string };
};

export async function GET(_request: Request, context: RouteContext) {
  const params = await context.params;
  const projectId = params.projectId;
  const response = await fetch(
    `${backendBaseUrl()}/runs/${encodeURIComponent(projectId)}/runpod-status`,
    { cache: "no-store" },
  );
  const text = await response.text();
  return new NextResponse(text || "null", {
    status: response.status,
    headers: {
      "content-type": response.headers.get("content-type") ?? "application/json",
    },
  });
}
