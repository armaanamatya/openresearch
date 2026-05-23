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
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 4000);
  try {
    const response = await fetch(
      `${backendBaseUrl()}/runs/${encodeURIComponent(projectId)}/repair-iterations`,
      { method: "GET", headers: { ...demoSecretHeaders() }, cache: "no-store", signal: controller.signal }
    );
    if (response.status >= 500) {
      return new NextResponse(null, { status: 404 });
    }
    const text = await response.text();
    if (!response.ok) {
      return new NextResponse(text || "Upstream error", { status: response.status });
    }
    try {
      return NextResponse.json(JSON.parse(text));
    } catch {
      return new NextResponse(text, { status: response.status });
    }
  } catch {
    return new NextResponse(null, { status: 404 });
  } finally {
    clearTimeout(timer);
  }
}
