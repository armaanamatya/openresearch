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
    <main className="flex min-h-screen items-center justify-center bg-stone-950 px-6 py-10 text-stone-100">
      <div className="w-full max-w-sm rounded-[28px] border border-emerald-400/20 bg-[radial-gradient(circle_at_top_left,_rgba(16,185,129,0.12),_transparent_40%),linear-gradient(135deg,_rgba(12,10,9,0.97),_rgba(28,25,23,0.94))] p-8 shadow-[0_20px_80px_rgba(0,0,0,0.4)]">
        <p className="mb-2 text-xs uppercase tracking-[0.35em] text-emerald-300">
          OpenResearch
        </p>
        <h1 className="mb-1 text-xl font-semibold text-white">Demo access</h1>
        <p className="mb-6 text-sm leading-6 text-stone-400">
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
            className="w-full appearance-none rounded-xl border border-white/10 bg-stone-950 px-4 py-3 text-sm font-semibold text-white outline-none transition placeholder:text-stone-600 focus:border-emerald-300/70 disabled:cursor-not-allowed disabled:text-stone-500"
          />
          {error && (
            <p className="rounded-xl border border-rose-400/30 bg-rose-400/10 px-4 py-2 text-sm text-rose-100">
              {error}
            </p>
          )}
          <button
            type="submit"
            disabled={submitting || !password}
            className="inline-flex items-center justify-center rounded-full bg-emerald-400 px-5 py-3 text-sm font-semibold text-stone-950 transition hover:bg-emerald-300 disabled:cursor-not-allowed disabled:bg-stone-700 disabled:text-stone-300"
          >
            {submitting ? "Verifying…" : "Unlock"}
          </button>
        </form>
      </div>
    </main>
  );
}
