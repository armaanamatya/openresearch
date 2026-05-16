import { NextResponse } from "next/server";
import { backendBaseUrl } from "@/lib/demo/server-run";

export const runtime = "nodejs";

export async function GET(): Promise<Response> {
  try {
    const response = await fetch(`${backendBaseUrl()}/pipeline/topology`, {
      cache: "no-store"
    });
    if (!response.ok) {
      return NextResponse.json({ error: "Topology unavailable" }, { status: 503 });
    }
    return NextResponse.json(await response.json());
  } catch {
    return NextResponse.json({ error: "Topology unavailable" }, { status: 503 });
  }
}
