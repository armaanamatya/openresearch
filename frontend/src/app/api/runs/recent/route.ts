import { NextResponse } from "next/server";
import { backendBaseUrl } from "@/lib/demo/server-run";

export const runtime = "nodejs";

export async function GET(request: Request): Promise<Response> {
  const url = new URL(request.url);
  const limit = url.searchParams.get("limit") ?? "10";
  try {
    const response = await fetch(
      `${backendBaseUrl()}/runs?limit=${encodeURIComponent(limit)}`,
      { cache: "no-store" }
    );
    if (!response.ok) {
      return NextResponse.json([], { status: 200 });
    }
    const body = (await response.json()) as unknown;
    return NextResponse.json(body);
  } catch {
    return NextResponse.json([], { status: 200 });
  }
}
