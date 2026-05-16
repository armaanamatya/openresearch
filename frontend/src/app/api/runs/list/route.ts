import { NextResponse } from "next/server";
import { backendBaseUrl } from "@/lib/demo/server-run";

export const runtime = "nodejs";

export async function GET(request: Request): Promise<Response> {
  const url = new URL(request.url);
  const qs = new URLSearchParams();
  for (const key of ["limit", "status", "q", "order_by"]) {
    const value = url.searchParams.get(key);
    if (value !== null) qs.set(key, value);
  }
  try {
    const response = await fetch(`${backendBaseUrl()}/runs?${qs.toString()}`, {
      cache: "no-store"
    });
    if (!response.ok) return NextResponse.json([], { status: 200 });
    return NextResponse.json(await response.json());
  } catch {
    return NextResponse.json([], { status: 200 });
  }
}
