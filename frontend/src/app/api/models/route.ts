import { NextResponse } from "next/server";
import { backendBaseUrl } from "@/lib/demo/server-run";

export const runtime = "nodejs";

// Proxies the backend `/models` list so the upload-view dropdown can be
// populated without a client-side cross-origin call. Returns an empty
// list (NOT an error) when the backend is unreachable — the dropdown
// remains usable thanks to the resolved-model fallback (see
// `lib/user-prefs.ts` → "sonnet" default).
export async function GET(): Promise<Response> {
  try {
    const response = await fetch(`${backendBaseUrl()}/models`, { cache: "no-store" });
    if (!response.ok) {
      return NextResponse.json([], { status: 200 });
    }
    return NextResponse.json(await response.json());
  } catch {
    return NextResponse.json([], { status: 200 });
  }
}
