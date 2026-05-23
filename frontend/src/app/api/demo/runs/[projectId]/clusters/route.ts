import { NextResponse } from "next/server";

import { backendBaseUrl } from "@/lib/demo/server-run";
import { gateSecret } from "@/lib/auth/demo-gate";

export const runtime = "nodejs";

function demoSecretHeaders(): Record<string, string> {
  const secret = gateSecret();
  return secret ? { "x-demo-secret": secret } : {};
}

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ projectId: string }> }
): Promise<NextResponse> {
  const { projectId } = await params;
  try {
    const response = await fetch(
      `${backendBaseUrl()}/runs/${encodeURIComponent(projectId)}/clusters`,
      { method: "GET", headers: { ...demoSecretHeaders() }, cache: "no-store" }
    );
    const text = await response.text();
    if (!response.ok) {
      return new NextResponse(text || "Upstream error", { status: response.status });
    }
    try {
      return NextResponse.json(JSON.parse(text));
    } catch {
      return new NextResponse(text, { status: response.status });
    }
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "fetch failed" },
      { status: 502 }
    );
  }
}
