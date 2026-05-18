import { NextResponse } from "next/server";

import { backendBaseUrl } from "@/lib/demo/server-run";
import { gateSecret } from "@/lib/auth/demo-gate";

export const runtime = "nodejs";

function demoSecretHeaders(): Record<string, string> {
  const secret = gateSecret();
  return secret ? { "x-demo-secret": secret } : {};
}

/**
 * Thin proxy to backend ``POST /runs/arxiv`` — the backend fetches the
 * paper server-side so we sidestep browser CORS (arxiv.org doesn't send
 * Access-Control-Allow-Origin headers). The body shape matches the
 * backend's StartArxivRunRequest model: { url, mode?, provider?, … }.
 */
export async function POST(request: Request): Promise<NextResponse> {
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json(
      { error: "Request body must be JSON with at least a 'url' field." },
      { status: 400 }
    );
  }
  if (!body || typeof body !== "object" || typeof (body as { url?: unknown }).url !== "string") {
    return NextResponse.json(
      { error: "Request body must include a 'url' string." },
      { status: 400 }
    );
  }
  const response = await fetch(`${backendBaseUrl()}/runs/arxiv`, {
    method: "POST",
    headers: { "content-type": "application/json", ...demoSecretHeaders() },
    body: JSON.stringify(body)
  });
  const text = await response.text();
  if (!response.ok) {
    return new NextResponse(text || "Upstream error", { status: response.status });
  }
  try {
    return NextResponse.json(JSON.parse(text));
  } catch {
    return new NextResponse(text, { status: response.status });
  }
}
