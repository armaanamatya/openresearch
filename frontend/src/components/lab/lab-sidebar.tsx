"use client";

import { useState } from "react";
import Link from "next/link";

import type { RecentRunSummary } from "@/lib/runs/server-list";
import { paperDisplayTitle, runDirName, numberRunsByPaper } from "@/lib/runs/paper-title";
import { ICONS, type IconKey } from "./icons";

import "./lab-sidebar.css";

type NavItem = {
  accent?: boolean;
  href: string;
  icon: IconKey;
  id: string;
  label: string;
};

const NAV: NavItem[] = [
  { id: "upload", label: "Upload", icon: "upload", href: "/lab?new=1" },
  { id: "lab", label: "Lab", icon: "lab", href: "/lab" },
  { id: "library", label: "Library", icon: "papers", href: "/library" },
  { id: "leaderboard", label: "Leaderboard", icon: "leaderboard", href: "/leaderboard" },
];

export function LabSidebar({
  active,
  recents,
  recentsError = null,
}: {
  active: string;
  recents: RecentRunSummary[];
  recentsError?: string | null;
}) {
  const [collapsed, setCollapsed] = useState(false);
  // Numbers runs that share a paper (1..N by recency); null when a paper has a
  // single run so we don't render a meaningless "run 1".
  const runNumbers = numberRunsByPaper(recents);

  return (
    <aside className={`sidebar${collapsed ? " collapsed" : ""}`}>
      <button
        className="sb-toggle"
        onClick={() => setCollapsed((value) => !value)}
        type="button"
        aria-label="Toggle sidebar"
      >
        <svg
          width="12"
          height="12"
          viewBox="0 0 16 16"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.8"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden="true"
        >
          <path d="M10 4l-4 4 4 4" />
        </svg>
      </button>
      <Link href="/" className="brand-row" aria-label="OpenResearch — back to landing">
        <span className="nav-icon">{ICONS.logo}</span>
        <span className="brand-text">OpenResearch</span>
      </Link>
      <div className="dotted" />
      {NAV.map((item) => (
        <a
          key={item.id}
          href={item.href}
          data-label={item.label}
          className={`navitem${active === item.id ? " active" : ""}`}
        >
          <span className="nav-icon" style={{ color: item.accent ? "var(--hermes)" : "var(--ink-2)" }}>
            {ICONS[item.icon]}
          </span>
          <span className="nav-label">{item.label}</span>
        </a>
      ))}
      <div className="dotted" />
      <div className="nav-section-title">Recent</div>
      {recentsError ? (
        <div className="navitem navitem-small" style={{ opacity: 0.7 }} role="status">
          <span className="nav-label">Recent runs unavailable.</span>
        </div>
      ) : recents.length === 0 ? (
        <div className="navitem navitem-small" style={{ opacity: 0.5 }}>
          <span className="nav-label">No recent runs.</span>
        </div>
      ) : (
        recents.map((run) => {
          const dotColor =
            run.status === "running"
              ? "var(--accent)"
              : run.status === "failed"
                ? "var(--warn)"
                : "var(--muted-2)";
          const runNumber = runNumbers.get(run);
          const label =
            runNumber !== null && runNumber !== undefined
              ? `${paperDisplayTitle(run)} · run ${runNumber}`
              : paperDisplayTitle(run);
          return (
            <a
              key={runDirName(run)}
              href={`/lab?projectId=${encodeURIComponent(run.projectId)}`}
              className="navitem navitem-small"
            >
              <span className="nav-icon nav-status-dot" style={{ background: dotColor }} />
              <span className="nav-label">{label}</span>
            </a>
          );
        })
      )}
      <div className="sidebar-footer">
        <div className="dotted" />
        {[
          { label: "Feedback", icon: "feedback" as IconKey },
          { label: "Help", icon: "help" as IconKey },
          { label: "Settings", icon: "settings" as IconKey }
        ].map((item) => (
          <a key={item.label} href="/lab" className="navitem">
            <span className="nav-icon" style={{ color: "var(--muted)" }}>
              {ICONS[item.icon]}
            </span>
            <span className="nav-label">{item.label}</span>
          </a>
        ))}
      </div>
    </aside>
  );
}
