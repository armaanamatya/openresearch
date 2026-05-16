"use client";

export type Status =
  | "auditing"
  | "completed"
  | "attention"
  | "queued"
  | "running"
  | "shipped"
  | "stopped";

export function statusTone(status: Status) {
  switch (status) {
    case "running":
    case "shipped":
      return { bg: "var(--accent-soft)", fg: "var(--accent-ink)", dot: "var(--accent)", pulse: true };
    case "auditing":
      return { bg: "var(--hermes-soft)", fg: "#5a3fd1", dot: "var(--hermes)", pulse: true };
    case "completed":
      return { bg: "var(--chip)", fg: "var(--ink-2)", dot: "var(--ink)", pulse: false };
    case "attention":
      return { bg: "var(--warn-soft)", fg: "var(--warn-ink)", dot: "var(--warn)", pulse: false };
    default:
      return { bg: "var(--chip)", fg: "var(--muted)", dot: "var(--muted-2)", pulse: false };
  }
}

export function statusLabel(status: Status) {
  if (status === "attention") {
    return "Needs attention";
  }
  return status;
}

export function StatusPill({ status }: { status: Status }) {
  const tone = statusTone(status);
  return (
    <span className="status-pill" style={{ background: tone.bg, color: tone.fg }}>
      <span className={`status-dot${tone.pulse ? " pulse-dot" : ""}`} style={{ background: tone.dot }} />
      {statusLabel(status)}
    </span>
  );
}
