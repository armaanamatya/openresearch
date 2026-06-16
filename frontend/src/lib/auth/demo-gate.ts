import "server-only";

// Reads the OPENRESEARCH_DEMO_SECRET server env var, which must never reach the
// client bundle. The `server-only` import above turns an accidental client
// import into a build error instead of a silent failure.

export const COOKIE_NAME = "reprolab_session";
export const COOKIE_MAX_AGE = 60 * 60 * 12; // 12 hours

export function gateSecret(): string {
  return process.env.OPENRESEARCH_DEMO_SECRET ?? "";
}

export function isGateEnabled(): boolean {
  return gateSecret().length > 0;
}

/** SHA-256 hex digest. Works in Edge and Node runtimes. */
export async function sessionToken(secret: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(secret));
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

/** Constant-time string comparison. Inputs here are always 64-char hex digests. */
export function safeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let mismatch = 0;
  for (let i = 0; i < a.length; i++) {
    mismatch |= a.charCodeAt(i) ^ b.charCodeAt(i);
  }
  return mismatch === 0;
}
