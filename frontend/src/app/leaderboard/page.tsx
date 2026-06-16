import Link from "next/link";

import { LeaderboardTable } from "./leaderboard-table";
import type { LeaderboardRow } from "@/lib/leaderboard/types";
import styles from "./page.module.css";

export const dynamic = "force-dynamic";

export const metadata = {
  title: "Leaderboard — OpenResearch",
  description: "Ranked completed runs across reproduction models and papers.",
};

async function fetchRows(): Promise<{ rows: LeaderboardRow[]; error: string | null }> {
  const backendUrl = ((process.env.OPENRESEARCH_BACKEND_URL ?? process.env.REPROLAB_BACKEND_URL) ?? "http://127.0.0.1:8000").replace(/\/$/, "");
  try {
    const resp = await fetch(`${backendUrl}/leaderboard`, {
      cache: "no-store",
      headers: { accept: "application/json" },
    });
    if (!resp.ok) return { rows: [], error: `Leaderboard unavailable (HTTP ${resp.status}).` };
    return { rows: (await resp.json()) as LeaderboardRow[], error: null };
  } catch {
    return { rows: [], error: "Leaderboard unavailable." };
  }
}

export default async function LeaderboardPage() {
  const { rows, error } = await fetchRows();
  return (
    <main className={styles.shell}>
      <nav className={styles.nav}>
        <Link className={styles.navLink} href="/" prefetch={false}>Home</Link>
        <Link className={styles.navLink} href="/lab" prefetch={false}>Lab</Link>
        <Link className={styles.navLink} href="/leaderboard" prefetch={false}>Leaderboard</Link>
      </nav>
      <header className={styles.header}>
        <h1 className={styles.title}>Leaderboard</h1>
        <p className={styles.subtitle}>
          Ranked completed runs across reproduction models and papers.
        </p>
      </header>
      <LeaderboardTable rows={rows} error={error} />
    </main>
  );
}
