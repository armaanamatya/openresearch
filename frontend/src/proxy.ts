// NOTE: In Next.js 16, middleware.ts is deprecated. This file is proxy.ts,
// the correct filename. The exported function must be named `proxy`.
// Runs in the nodejs runtime (Edge is not supported in Next.js 16 proxy).

import { NextRequest, NextResponse } from "next/server";

import { COOKIE_NAME, gateSecret, safeEqual, sessionToken } from "@/lib/auth/demo-gate";

const PUBLIC_PATHS = ["/unlock", "/api/unlock", "/health"];

function isPublic(pathname: string): boolean {
  return PUBLIC_PATHS.some((p) => pathname === p || pathname.startsWith(p + "/"));
}

export async function proxy(request: NextRequest) {
  const secret = gateSecret();
  if (!secret) return NextResponse.next(); // gate disabled (local dev)

  const { pathname } = request.nextUrl;
  if (isPublic(pathname)) return NextResponse.next();

  const cookie = request.cookies.get(COOKIE_NAME)?.value;
  const expected = await sessionToken(secret);
  if (cookie && safeEqual(cookie, expected)) return NextResponse.next();

  if (pathname.startsWith("/api/")) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }
  const url = request.nextUrl.clone();
  url.pathname = "/unlock";
  url.search = "";
  return NextResponse.redirect(url);
}

// The unlock gate is scoped to the live lab surface. `/lab` and `/library`
// are open for internal use; `/api/demo/*` (the backend proxy for lab runs)
// requires the unlock cookie when REPROLAB_DEMO_SECRET is set.
export const config = {
  matcher: ["/api/demo/:path*"],
};
