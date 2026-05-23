import { NextResponse } from "next/server";

import { backendBaseUrl } from "@/lib/demo/server-run";
import { gateSecret } from "@/lib/auth/demo-gate";

export const runtime = "nodejs";

function demoSecretHeaders(): Record<string, string> {
  const secret = gateSecret();
  return secret ? { "x-demo-secret": secret } : {};
}

export async function POST(
  request: Request,
  { params }: { params: Promise<{ projectId: string }> }
): Promise<NextResponse> {
  const { projectId } = await params;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 4000);
  try {
    const body = await request.text();
    const response = await fetch(
      `${backendBaseUrl()}/runs/${encodeURIComponent(projectId)}/messages`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...demoSecretHeaders(),
        },
        body,
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
    return new NextResponse("Message delivery failed", { status: 502 });
  } finally {
    clearTimeout(timer);
  }
}
