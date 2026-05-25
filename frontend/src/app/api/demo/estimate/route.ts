import { NextResponse } from "next/server";

import { backendBaseUrl } from "@/lib/demo/server-run";
import { gateSecret } from "@/lib/auth/demo-gate";

export const runtime = "nodejs";

function demoSecretHeaders(): Record<string, string> {
  const secret = gateSecret();
  return secret ? { "x-demo-secret": secret } : {};
}

const MAX_UPLOAD_BYTES = 50 * 1024 * 1024;

/**
 * Thin proxy to backend `POST /paper/estimate`.
 *
 * Two body shapes:
 *  - JSON: { source_kind: "arxiv_url" | "arxiv_id", source: string, recipe_mode? }
 *  - multipart/form-data with a `paper` PDF file (same shape as `/runs/upload`)
 *
 * Either way the result is forwarded verbatim. Estimator failure surfaces as
 * the same HTTP status the backend returned; the frontend BudgetPanel handles
 * it as a "skip and start anyway" path.
 */
export async function POST(request: Request): Promise<NextResponse> {
  const contentType = request.headers.get("content-type") ?? "";

  if (contentType.includes("multipart/form-data")) {
    // Stream multipart through to the backend without re-parsing — same
    // pattern as /api/demo/runs/upload.
    const declaredBytes = Number(request.headers.get("content-length") ?? 0);
    if (declaredBytes > MAX_UPLOAD_BYTES) {
      return NextResponse.json(
        { error: "PDF too large — the lab accepts uploads up to 50 MB." },
        { status: 413 }
      );
    }
    const upstream = await fetch(`${backendBaseUrl()}/paper/estimate`, {
      method: "POST",
      headers: { "content-type": contentType, ...demoSecretHeaders() },
      body: request.body,
      duplex: "half"
    } as RequestInit & { duplex: "half" });
    const text = await upstream.text();
    return new NextResponse(text || "null", {
      status: upstream.status,
      headers: { "content-type": upstream.headers.get("content-type") ?? "application/json" }
    });
  }

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json(
      { error: "Request body must be JSON with 'source_kind' and 'source' fields." },
      { status: 400 }
    );
  }
  const upstream = await fetch(`${backendBaseUrl()}/paper/estimate`, {
    method: "POST",
    headers: { "content-type": "application/json", ...demoSecretHeaders() },
    body: JSON.stringify(body)
  });
  const text = await upstream.text();
  if (!upstream.ok) {
    return new NextResponse(text || "Upstream error", { status: upstream.status });
  }
  try {
    return NextResponse.json(JSON.parse(text));
  } catch {
    return new NextResponse(text, { status: upstream.status });
  }
}
