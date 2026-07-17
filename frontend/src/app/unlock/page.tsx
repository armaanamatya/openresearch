"use client";

// NOTE: Do NOT import demo-gate.ts here — it is server-only and reads
// OPENRESEARCH_DEMO_SECRET, which must never reach the client bundle.

import { useState } from "react";

export default function UnlockPage() {
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setSubmitting(true);
    try {
      const res = await fetch("/api/unlock", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ password }),
      });
      if (res.ok) {
        window.location.href = "/";
        return;
      }
      const data = (await res.json()) as { error?: string };
      setError(data.error ?? "Incorrect access code.");
    } catch {
      setError("Network error. Please try again.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <main className="flex min-h-screen items-center justify-center bg-[var(--bg)] px-6 py-10 text-ink">
      <div className="w-full max-w-sm rounded-[28px] border border-line bg-[radial-gradient(circle_at_top_left,_rgba(232,162,74,0.10),_transparent_45%),var(--panel)] p-8 shadow-[0_24px_80px_-20px_rgba(0,0,0,0.7)]">
        <p className="mb-2 text-xs uppercase tracking-[0.35em] text-accent">
          OpenResearch
        </p>
        <h1 className="mb-1 text-xl font-semibold text-ink">Demo access</h1>
        <p className="mb-6 text-sm leading-6 text-muted">
          Enter the access code to continue.
        </p>
        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          <input
            type="password"
            placeholder="Access code"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            disabled={submitting}
            autoFocus
            className="w-full appearance-none rounded-xl border border-line bg-[var(--bg)] px-4 py-3 text-sm font-semibold text-ink outline-none transition placeholder:text-muted-2 focus:border-[var(--accent)] disabled:cursor-not-allowed disabled:text-muted"
          />
          {error && (
            <p className="rounded-xl border border-err-soft bg-err-soft px-4 py-2 text-sm text-err">
              {error}
            </p>
          )}
          <button
            type="submit"
            disabled={submitting || !password}
            className="inline-flex items-center justify-center rounded-full bg-accent px-5 py-3 text-sm font-semibold text-accent-ink transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {submitting ? "Verifying…" : "Unlock"}
          </button>
        </form>
      </div>
    </main>
  );
}
