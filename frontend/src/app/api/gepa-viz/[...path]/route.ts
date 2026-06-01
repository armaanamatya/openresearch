/**
 * Proxy: /api/gepa-viz/* → http://127.0.0.1:{GEPA_VIZ_PORT}/*
 *
 * gepa-viz runs as a standalone server (`gepa-viz live`) on GEPA_VIZ_PORT
 * (default 5151). This proxy lets the lab UI embed gepa-viz without CORS issues.
 * SSE /events and POST /ingest are proxied transparently.
 * Returns 503 when gepa-viz is not running.
 */
import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";

const GEPA_VIZ_PORT = process.env.GEPA_VIZ_PORT ?? "5151";
const GEPA_VIZ_BASE = `http://127.0.0.1:${GEPA_VIZ_PORT}`;

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const { path } = await params;
  const targetPath = "/" + path.join("/");
  const targetUrl = GEPA_VIZ_BASE + targetPath + (req.nextUrl.search || "");
  try {
    const upstream = await fetch(targetUrl, {
      headers: { ...Object.fromEntries(req.headers), host: `127.0.0.1:${GEPA_VIZ_PORT}` },
    });
    return new NextResponse(upstream.body, {
      status: upstream.status,
      headers: upstream.headers,
    });
  } catch {
    return NextResponse.json({ error: "gepa-viz unavailable" }, { status: 503 });
  }
}

export async function POST(
  req: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const { path } = await params;
  const targetPath = "/" + path.join("/");
  const targetUrl = GEPA_VIZ_BASE + targetPath;
  try {
    const body = await req.arrayBuffer();
    const upstream = await fetch(targetUrl, {
      method: "POST",
      headers: Object.fromEntries(req.headers),
      body,
    });
    return new NextResponse(upstream.body, { status: upstream.status });
  } catch {
    return NextResponse.json({ error: "gepa-viz unavailable" }, { status: 503 });
  }
}
