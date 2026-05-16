"use client";

import { useState } from "react";

import type { RecentRunSummary } from "@/lib/runs/server-list";
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
  { id: "lab", label: "Lab", icon: "lab", href: "/lab" },
  { id: "library", label: "Library", icon: "papers", href: "/library" }
];

export function LabSidebar({
  active,
  onBrandClick,
  recents
}: {
  active: string;
  onBrandClick: () => void;
  recents: RecentRunSummary[];
}) {
  const [collapsed, setCollapsed] = useState(false);

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
      <button className="brand-row" type="button" onClick={onBrandClick}>
        <span className="nav-icon">{ICONS.logo}</span>
        <span className="brand-text">ReproLab</span>
      </button>
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
      {recents.length === 0 ? (
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
          const label = run.sourceLabel ?? run.projectId.slice(0, 14);
          return (
            <a
              key={run.projectId}
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
