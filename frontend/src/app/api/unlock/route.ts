import { NextResponse } from "next/server";

import { COOKIE_MAX_AGE, COOKIE_NAME, gateSecret, safeEqual, sessionToken } from "@/lib/auth/demo-gate";

export const runtime = "nodejs";

export async function POST(request: Request) {
  const secret = gateSecret();
  if (!secret) return NextResponse.json({ ok: true }); // gate disabled — nothing to unlock

  let password = "";
  try {
    const body = (await request.json()) as { password?: unknown };
    if (typeof body.password === "string") password = body.password;
  } catch {
    password = "";
  }

  const expected = await sessionToken(secret);
  const submitted = await sessionToken(password);
  if (!password || !safeEqual(submitted, expected)) {
    await new Promise((resolve) => setTimeout(resolve, 500)); // brute-force friction
    return NextResponse.json({ error: "Incorrect access code." }, { status: 401 });
  }

  const response = NextResponse.json({ ok: true });
  response.cookies.set(COOKIE_NAME, expected, {
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
    path: "/",
    maxAge: COOKIE_MAX_AGE,
  });
  return response;
}
