import Link from "next/link";

import { LeaderboardTable, type LeaderboardRow } from "./leaderboard-table";
import styles from "./page.module.css";

export const dynamic = "force-dynamic";

export const metadata = {
  title: "Leaderboard — ReproLab",
  description: "Ranked completed runs across reproduction models and papers.",
};

async function fetchRows(): Promise<LeaderboardRow[]> {
  const backendUrl = (process.env.REPROLAB_BACKEND_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");
  try {
    const resp = await fetch(`${backendUrl}/leaderboard`, {
      cache: "no-store",
      headers: { accept: "application/json" },
    });
    if (!resp.ok) return [];
    return (await resp.json()) as LeaderboardRow[];
  } catch {
    return [];
  }
}

export default async function LeaderboardPage() {
  const rows = await fetchRows();
  return (
    <main className={styles.shell}>
      <nav className={styles.nav}>
        <Link className={styles.navLink} href="/">Home</Link>
        <Link className={styles.navLink} href="/lab">Lab</Link>
        <Link className={styles.navLink} href="/leaderboard">Leaderboard</Link>
      </nav>
      <header className={styles.header}>
        <h1 className={styles.title}>Leaderboard</h1>
        <p className={styles.subtitle}>
          Ranked completed runs across reproduction models and papers.
        </p>
      </header>
      <LeaderboardTable rows={rows} />
    </main>
  );
}
