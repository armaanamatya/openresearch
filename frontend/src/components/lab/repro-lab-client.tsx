"use client";

import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { useRouter } from "next/navigation";

import type { DemoModelChoice, LiveDemoRunState } from "@/lib/demo/demo-run-types";
import { AgentTimelineRail, type DashboardLiveEvent } from "./agent-timeline-rail";

const MAX_DASHBOARD_EVENTS = 200;

type NavItem = {
  accent?: boolean;
  href: string;
  icon: keyof typeof ICONS;
  id: string;
  label: string;
};

type Tone = "accent" | "hermes" | "info" | "neutral";
type NodeState = "done" | "running" | "upcoming";
type Status =
  | "auditing"
  | "completed"
  | "attention"
  | "queued"
  | "running"
  | "shipped"
  | "stopped";

type WorkflowNode = {
  detail: string;
  icon: keyof typeof ICONS;
  id: string;
  role: string;
  step: string;
  tone: Tone;
  x: number;
  y: number;
  agent: string;
};

type EventSourceLike = {
  addEventListener: (type: string, listener: EventListenerOrEventListenerObject) => void;
  close: () => void;
  onerror: ((this: EventSource, ev: Event) => unknown) | null;
};

type ReproLabClientProps = {
  initialRun?: LiveDemoRunState | null;
};

const DEFAULT_RUN_QUERY =
  "/api/demo?mode=sdk&provider=anthropic&executionMode=efficient&sandbox=runpod&gpuMode=auto";
const POLL_INTERVAL_MS = 3000;
const NODE_W = 200;
const NODE_H = 80;

// Per-browser pointer to the most recent run. Lets a closed tab or a
// fresh page load auto-resume an in-flight run; localStorage is
// per-browser, so a genuinely new browser still starts clean.
const LAST_RUN_KEY = "reprolab:lastRun";
// Which workflow drawer is expanded, persisted so the layout choice
// survives a refresh.
const DRAWER_KEY = "reprolab:openDrawer";

function writeLastRun(projectId: string): void {
  try {
    window.localStorage.setItem(LAST_RUN_KEY, projectId);
  } catch {
    // localStorage can throw in private mode / disabled storage — non-fatal.
  }
}

function clearLastRun(): void {
  try {
    window.localStorage.removeItem(LAST_RUN_KEY);
  } catch {
    // non-fatal
  }
}

function readLastRun(): string | null {
  try {
    return window.localStorage.getItem(LAST_RUN_KEY);
  } catch {
    return null;
  }
}

const NAV: NavItem[] = [
  { id: "lab", label: "Lab", icon: "lab", href: "/lab" },
  { id: "papers", label: "Library", icon: "papers", href: "/papers" },
  { id: "hermes", label: "Hermes", icon: "hermes", href: "/hermes", accent: true }
];

const NODES: WorkflowNode[] = [
  {
    id: "src",
    x: 20,
    y: 310,
    agent: "Paper",
    step: "Source intake",
    icon: "doc",
    tone: "neutral",
    role: "Receives the source artifact",
    detail:
      "This is the paper or workspace input that starts the run. Uploaded PDFs are ingested directly; fixture runs use the in-repo PPO workspace."
  },
  {
    id: "read",
    x: 260,
    y: 310,
    agent: "Reader",
    step: "Paper understanding",
    icon: "brain",
    tone: "info",
    role: "Extracts claims, metrics, and assumptions",
    detail:
      "Parses the paper, identifies the core contribution, and turns benchmarks and assumptions into a runnable reproduction plan."
  },
  {
    id: "env",
    x: 500,
    y: 200,
    agent: "Forge",
    step: "Environment",
    icon: "beaker",
    tone: "info",
    role: "Rebuilds the runtime environment",
    detail:
      "Resolves dependencies, creates the isolated execution environment, and prepares the run surface for the baseline implementation."
  },
  {
    id: "plan",
    x: 500,
    y: 420,
    agent: "Architect",
    step: "Reproduction plan",
    icon: "doc",
    tone: "info",
    role: "Defines the verification contract",
    detail:
      "Maps the paper claims to experiments and checkpoints so the baseline and follow-on improvements can be judged against a real contract."
  },
  {
    id: "impl",
    x: 740,
    y: 310,
    agent: "Builder",
    step: "Baseline implementation",
    icon: "zap",
    tone: "accent",
    role: "Builds and runs the baseline",
    detail:
      "Produces the baseline implementation, launches the run, and records the first metrics used for downstream verification."
  },
  {
    id: "opt",
    x: 1020,
    y: 60,
    agent: "Vesta",
    step: "Optimizer path",
    icon: "spark",
    tone: "info",
    role: "Explores optimizer changes",
    detail:
      "Tests alternative optimizers and schedules once the baseline is stable enough to compare against."
  },
  {
    id: "bb",
    x: 1020,
    y: 200,
    agent: "Athena",
    step: "Backbone path",
    icon: "copy",
    tone: "info",
    role: "Tests representation swaps",
    detail:
      "Evaluates backbone changes and logs the resulting deltas so Hermes can verify whether they are real improvements."
  },
  {
    id: "aug",
    x: 1020,
    y: 340,
    agent: "Orion",
    step: "Augmentation path",
    icon: "graph",
    tone: "info",
    role: "Explores robustness changes",
    detail:
      "Sweeps augmentation strategies and checks whether they help or hurt the reproduced baseline."
  },
  {
    id: "hor",
    x: 1020,
    y: 480,
    agent: "Lyra",
    step: "Horizon path",
    icon: "flag",
    tone: "info",
    role: "Extends planning horizon",
    detail:
      "Tests longer-horizon variants and re-runs evaluation to measure any tradeoff between reward and runtime."
  },
  {
    id: "div",
    x: 1020,
    y: 620,
    agent: "Pyxis",
    step: "Diffusion path",
    icon: "compute",
    tone: "info",
    role: "Sweeps diffusion settings",
    detail:
      "Compares DDIM and related inference-time changes, then records cost and metric impact for audit."
  },
  {
    id: "audit",
    x: 1300,
    y: 310,
    agent: "Hermes",
    step: "Result audit",
    icon: "shield",
    tone: "hermes",
    role: "Verifies claims against the run",
    detail:
      "Hermes checks whether claimed results are grounded in the actual run outputs, flags regressions, and records interventions."
  },
  {
    id: "report",
    x: 1540,
    y: 310,
    agent: "Scribe",
    step: "Final report",
    icon: "flag",
    tone: "neutral",
    role: "Packages the reproducibility output",
    detail:
      "Compiles manifests, logs, checkpoints, and the audit trail into the final reproducibility packet."
  }
];

const EDGES: Array<[string, string]> = [
  ["src", "read"],
  ["read", "env"],
  ["read", "plan"],
  ["env", "impl"],
  ["plan", "impl"],
  ["impl", "opt"],
  ["impl", "bb"],
  ["impl", "aug"],
  ["impl", "hor"],
  ["impl", "div"],
  ["opt", "audit"],
  ["bb", "audit"],
  ["aug", "audit"],
  ["hor", "audit"],
  ["div", "audit"],
  ["audit", "report"]
];

function icon(children: React.ReactNode, size = 18) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 18 18"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {children}
    </svg>
  );
}

const ICONS = {
  logo: (
    <svg width="22" height="22" viewBox="0 0 22 22" fill="none" aria-hidden="true">
      <path
        d="M4 6.5L11 3l7 3.5M4 6.5v9L11 19l7-3.5v-9M4 6.5L11 10l7-3.5M11 10v9"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  ),
  lab: icon(
    <>
      <path d="M7 2.5v4L3.5 12a1.5 1.5 0 0 0 1.3 2.5h8.4A1.5 1.5 0 0 0 14.5 12L11 6.5v-4" />
      <path d="M6.5 2.5h5" />
    </>
  ),
  papers: icon(
    <>
      <path d="M5 2.5h6l3 3v10H5z" />
      <path d="M11 2.5v3h3" />
      <path d="M7 9h6M7 12h4" />
    </>
  ),
  hermes: icon(
    <>
      <path d="M9 2l5.5 2v5c0 3.5-2.5 6.5-5.5 7.5C6 15.5 3.5 12.5 3.5 9V4z" />
      <path d="M6.5 9l2 2 3-4" />
    </>
  ),
  feedback: icon(<path d="M3 4h12v8H8l-3 3v-3H3z" />),
  help: icon(
    <>
      <circle cx="9" cy="9" r="6.5" />
      <path d="M7.5 7c.4-1 1.4-1.5 2.4-1.2 1 .3 1.6 1.4 1.2 2.4-.3.7-1.1 1.3-1.6 1.3v.8" />
      <circle cx="9" cy="13" r=".6" fill="currentColor" />
    </>
  ),
  settings: icon(
    <>
      <circle cx="9" cy="9" r="2" />
      <path d="M14.5 9c0 .4 0 .8-.1 1.1l1.4 1-1.6 2.7-1.7-.5c-.5.5-1.1.9-1.7 1.1L10.5 16h-3l-.3-1.6c-.6-.2-1.2-.6-1.7-1.1l-1.7.5L2.2 11l1.4-1c-.1-.3-.1-.7-.1-1.1s0-.8.1-1.1l-1.4-1L3.8 4l1.7.5c.5-.5 1.1-.9 1.7-1.1L7.5 2h3l.3 1.6c.6.2 1.2.6 1.7 1.1L14.2 4l1.6 2.7-1.4 1c.1.4.1.8.1 1.2z" />
    </>
  ),
  upload: icon(
    <>
      <path d="M9 11V3.5M9 3.5l-2.5 2.5M9 3.5l2.5 2.5" />
      <path d="M3.5 12v1.5A1.5 1.5 0 0 0 5 15h8a1.5 1.5 0 0 0 1.5-1.5V12" />
    </>
  ),
  play: icon(<path d="M5 3.5v11l9-5.5z" fill="currentColor" stroke="none" />),
  pause: icon(
    <>
      <rect x="5.5" y="4" width="2.2" height="10" rx="1" fill="currentColor" stroke="none" />
      <rect x="10.3" y="4" width="2.2" height="10" rx="1" fill="currentColor" stroke="none" />
    </>
  ),
  spark: icon(
    <>
      <path d="M9 2v3M9 13v3M2 9h3M13 9h3M4 4l2 2M12 12l2 2M4 14l2-2M12 6l2-2" />
    </>
  ),
  doc: icon(
    <>
      <path d="M5 2.5h6l3 3v10H5z" />
      <path d="M11 2.5v3h3" />
    </>
  ),
  brain: icon(
    <>
      <path d="M9 3.5a2.5 2.5 0 0 0-2.5 2.5v0a2 2 0 0 0-1 3.5 2 2 0 0 0 1 3.5v0A2.5 2.5 0 0 0 9 15.5" />
      <path d="M9 3.5a2.5 2.5 0 0 1 2.5 2.5v0a2 2 0 0 1 1 3.5 2 2 0 0 1-1 3.5v0a2.5 2.5 0 0 1-2.5 2.5" />
    </>
  ),
  beaker: icon(
    <>
      <path d="M7 2.5v4L3.5 12a1.5 1.5 0 0 0 1.3 2.5h8.4A1.5 1.5 0 0 0 14.5 12L11 6.5v-4" />
      <path d="M6.5 2.5h5" />
      <circle cx="9" cy="11" r=".7" fill="currentColor" />
      <circle cx="7" cy="9" r=".5" fill="currentColor" />
    </>
  ),
  shield: icon(
    <>
      <path d="M9 2l5.5 2v5c0 3.5-2.5 6.5-5.5 7.5C6 15.5 3.5 12.5 3.5 9V4z" />
    </>
  ),
  zap: icon(<path d="M10 2L4.5 10h3l-1 6 5.5-8h-3l1-6z" fill="currentColor" stroke="none" />),
  copy: icon(
    <>
      <rect x="5.5" y="5.5" width="9" height="9" rx="1.5" />
      <path d="M3.5 11V4A1.5 1.5 0 0 1 5 2.5h7" />
    </>
  ),
  graph: icon(
    <>
      <path d="M2.5 14.5l4-5 3 2 6-7" />
      <path d="M10 4.5h5.5V10" />
    </>
  ),
  flag: icon(
    <>
      <path d="M4 2v14" />
      <path d="M4 3h9l-2 3 2 3H4" />
    </>
  ),
  compute: icon(
    <>
      <rect x="3" y="3" width="12" height="12" rx="2" />
      <rect x="6" y="6" width="6" height="6" />
      <path d="M3 6.5h-1M3 11.5h-1M16 6.5h-1M16 11.5h-1M6.5 3v-1M11.5 3v-1M6.5 16v-1M11.5 16v-1" />
    </>
  )
};

function statusTone(status: Status) {
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

function statusLabel(status: Status) {
  if (status === "attention") {
    return "Needs attention";
  }
  return status;
}

function issueText(value?: string | null) {
  if (!value) {
    return "";
  }
  return value
    .replace(/\bfailed\b/gi, "needs attention")
    .replace(/\bfailure\b/gi, "issue");
}

function sourceTitle(run: LiveDemoRunState) {
  return run.sourceLabel || run.projectId;
}

function sourcePdfUrl(run: LiveDemoRunState) {
  return `/api/demo/source-pdf?projectId=${encodeURIComponent(run.projectId)}`;
}

function finalReportUrl(run: LiveDemoRunState) {
  return `/api/demo/final-report?projectId=${encodeURIComponent(run.projectId)}`;
}

function formatBytes(bytes?: number | null) {
  if (!bytes || bytes <= 0) {
    return "n/a";
  }
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let index = 0;
  while (value >= 1024 && index < units.length - 1) {
    value /= 1024;
    index += 1;
  }
  return `${value.toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

function shortHash(value?: string | null) {
  return value ? value.slice(0, 12) : "pending";
}

function verdictLabel(value?: string) {
  if (!value) {
    return "Pending";
  }
  return value
    .split("_")
    .map((part) => part.slice(0, 1).toUpperCase() + part.slice(1))
    .join(" ");
}

function deriveStage(run: LiveDemoRunState | null): string | null {
  return run?.payload?.summary.stage ?? null;
}

export function stateMapForRun(run: LiveDemoRunState | null): Record<string, NodeState> {
  const map = Object.fromEntries(NODES.map((node) => [node.id, "upcoming"])) as Record<
    string,
    NodeState
  >;

  if (!run) {
    return map;
  }

  const stage = deriveStage(run);
  const status = run.status;

  function mark(ids: string[], state: NodeState) {
    for (const id of ids) {
      map[id] = state;
    }
  }

  if (status === "queued" && !stage) {
    mark(["src"], "running");
    return map;
  }

  if (status === "completed") {
    mark(NODES.map((node) => node.id), "done");
    return map;
  }

  mark(["src"], "done");

  if (!stage) {
    mark(["read"], status === "failed" ? "done" : "running");
    return map;
  }

  if (stage === "ingested") {
    mark(["read"], "running");
    return map;
  }

  if (["paper_understood", "artifacts_discovered"].includes(stage)) {
    mark(["read"], "done");
    mark(["env", "plan"], "running");
    return map;
  }

  if (stage === "environment_built") {
    mark(["read"], "done");
    mark(["env"], "done");
    mark(["plan"], "running");
    return map;
  }

  if (["plan_created", "gate_1_passed"].includes(stage)) {
    // `env` finished at `environment_built` (an earlier stage) — it must stay
    // `done` here so the workflow counter never regresses (monotonic progress).
    mark(["read", "env"], "done");
    mark(["plan"], stage === "gate_1_passed" ? "done" : "running");
    return map;
  }

  if (["baseline_implemented", "baseline_run", "gate_2_passed"].includes(stage)) {
    mark(["read", "env", "plan"], "done");
    mark(["impl"], stage === "gate_2_passed" ? "done" : "running");
    return map;
  }

  if (["improvements_selected", "improvements_run", "gate_3_passed"].includes(stage)) {
    mark(["read", "env", "plan", "impl"], "done");
    const perPath = run.payload?.pathStates;
    for (const id of ["opt", "bb", "aug", "hor", "div"] as const) {
      const node = perPath?.[id];
      map[id] = node === "done" ? "done"
        : node === "running" || node === "attention" ? "running"
        : stage === "gate_3_passed" ? "done"
        : "upcoming";
    }
    return map;
  }

  if (stage === "research_map_generated") {
    mark(["read", "env", "plan", "impl", "opt", "bb", "aug", "hor", "div"], "done");
    mark(["audit"], "running");
    return map;
  }

  if (stage === "complete") {
    mark(NODES.map((node) => node.id), "done");
    return map;
  }

  mark(["read"], "running");
  return map;
}

// Merge a freshly-received run_state frame onto the current one. The GET
// route (750 ms) and the SSE route (250 ms) both cap payload enrichment
// and fall back to the *un-enriched* backend state on timeout — that
// frame carries no `payload`/`telemetry`. Applied raw it would blank the
// graph's per-path nodes (stateMapForRun reads `payload.pathStates`).
// Carry the last good values forward until the next enriched frame
// instead of letting the UI regress.
function coalesceRunState(
  prev: LiveDemoRunState | null,
  next: LiveDemoRunState
): LiveDemoRunState {
  if (!prev || prev.projectId !== next.projectId) {
    return next;
  }
  if (!next.payload && prev.payload && process.env.NODE_ENV !== "production") {
    console.warn(
      "[reprolab] un-enriched run_state frame (no payload) — retaining the last enriched payload so the graph does not regress"
    );
  }
  return {
    ...next,
    payload: next.payload ?? prev.payload,
    telemetry: next.telemetry?.length ? next.telemetry : prev.telemetry,
    log: next.log || prev.log
  };
}

function parseLogEntries(run: LiveDemoRunState | null) {
  if (!run?.log) {
    return [];
  }

  return run.log
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .slice(-80)
    .reverse()
    .map((line, index) => ({
      id: `${run.projectId}-${index}`,
      time: run.updatedAt ? new Date(run.updatedAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "--:--",
      // Log lines render verbatim — euphemising "failed" to "needs
      // attention" here hid real failures from anyone reading the log.
      msg: line
    }));
}

function telemetryForSelectedNode(run: LiveDemoRunState | null, selectedId: string | null) {
  if (!run?.telemetry?.length || !selectedId) {
    return [];
  }

  const agentMatchers: Record<string, string[]> = {
    read: ["paper-understanding", "artifact-discovery"],
    env: ["environment-detective", "environment-verifier"],
    plan: ["reproduction-planner", "root-orchestrator"],
    impl: [
      "baseline-implementation",
      "experiment-runner",
      "method-fidelity-verifier",
      "data-metrics-verifier",
      "artifact-diff-verifier"
    ],
    opt: ["improvement-orchestrator", "improvement-path"],
    bb: ["improvement-orchestrator", "improvement-path"],
    aug: ["improvement-orchestrator", "improvement-path"],
    hor: ["improvement-orchestrator", "improvement-path"],
    div: ["improvement-orchestrator", "improvement-path"],
    audit: ["supervisor-verifier", "verifier", "hermes"],
    report: ["supervisor-verifier", "root-orchestrator"]
  };

  const matches = agentMatchers[selectedId] ?? [];
  return run.telemetry
    .filter((record) => matches.some((match) => record.agent_id?.includes(match)))
    .slice(-6)
    .reverse();
}

function failedNodeIdForRun(run: LiveDemoRunState, stateMap: Record<string, NodeState>) {
  if (run.status !== "failed") {
    return null;
  }

  const runningNode = NODES.find((node) => stateMap[node.id] === "running");
  if (runningNode) {
    return runningNode.id;
  }

  const firstUpcoming = NODES.find((node) => stateMap[node.id] === "upcoming");
  if (firstUpcoming) {
    return firstUpcoming.id;
  }

  return "report";
}

function buildEdgePath(from: WorkflowNode, to: WorkflowNode) {
  const x1 = from.x + NODE_W;
  const y1 = from.y + NODE_H / 2;
  const x2 = to.x;
  const y2 = to.y + NODE_H / 2;
  const cx1 = x1 + Math.max(40, (x2 - x1) * 0.45);
  const cx2 = x2 - Math.max(40, (x2 - x1) * 0.45);
  return `M ${x1} ${y1} C ${cx1} ${y1}, ${cx2} ${y2}, ${x2} ${y2}`;
}

function Sidebar({ active, onBrandClick }: { active: string; onBrandClick: () => void }) {
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
          {item.id === "hermes" ? <span className="nav-aside">2</span> : null}
        </a>
      ))}
      <div className="dotted" />
      <div className="nav-section-title">Recent</div>
      {[
        { t: "Diffusion Policy", s: "running" },
        { t: "ACT Transformer", s: "shipped" },
        { t: "PerAct", s: "attention" }
      ].map((item) => (
        <a key={item.t} href="/lab" className="navitem navitem-small">
          <span
            className="nav-icon nav-status-dot"
            style={{
              background:
                item.s === "running" ? "var(--accent)" : item.s === "attention" ? "var(--warn)" : "var(--muted-2)"
            }}
          />
          <span className="nav-label">{item.t}</span>
        </a>
      ))}
      <div className="sidebar-footer">
        <div className="dotted" />
        {[
          { label: "Feedback", icon: "feedback" },
          { label: "Help", icon: "help" },
          { label: "Settings", icon: "settings" }
        ].map((item) => (
          <a key={item.label} href="/lab" className="navitem">
            <span className="nav-icon" style={{ color: "var(--muted)" }}>
              {ICONS[item.icon as keyof typeof ICONS]}
            </span>
            <span className="nav-label">{item.label}</span>
          </a>
        ))}
      </div>
    </aside>
  );
}

function StatusPill({ status }: { status: Status }) {
  const tone = statusTone(status);
  return (
    <span className="status-pill" style={{ background: tone.bg, color: tone.fg }}>
      <span className={`status-dot${tone.pulse ? " pulse-dot" : ""}`} style={{ background: tone.dot }} />
      {statusLabel(status)}
    </span>
  );
}

function UploadView({
  arxiv,
  busy,
  error,
  model,
  onArxivChange,
  onArxivSubmit,
  onFileSelected,
  onModelChange,
  over,
  setOver
}: {
  arxiv: string;
  busy: boolean;
  error: string | null;
  model: DemoModelChoice;
  onArxivChange: (value: string) => void;
  onArxivSubmit: () => void;
  onFileSelected: (file: File) => void;
  onModelChange: (value: DemoModelChoice) => void;
  over: boolean;
  setOver: (value: boolean) => void;
}) {
  const fileInput = useRef<HTMLInputElement | null>(null);

  return (
    <div className="upload-shell">
      <div
        className={`upload-zone${over ? " over" : ""}`}
        onDragOver={(event) => {
          event.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(event) => {
          event.preventDefault();
          setOver(false);
          const file = event.dataTransfer.files[0];
          if (file) {
            onFileSelected(file);
          }
        }}
        onClick={() => fileInput.current?.click()}
      >
        <input
          ref={fileInput}
          type="file"
          accept=".pdf"
          className="hidden-input"
          aria-label="Upload paper PDF"
          disabled={busy}
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) {
              onFileSelected(file);
              event.currentTarget.value = "";
            }
          }}
        />
        <div className="upload-icon">{ICONS.upload}</div>
        <h1 className="upload-title">Upload PDF</h1>
        <p className="upload-copy">
          Drop a paper here or click to browse. ReproLab will reproduce, verify, and report -
          independently.
        </p>
        <div className="upload-meta">PDF - max 50 MB - arXiv preprints recommended</div>
      </div>
      <div className="upload-divider">
        <span />
        <span className="upload-divider-label">or paste an arXiv link</span>
        <span />
      </div>
      <form
        className="upload-form"
        onSubmit={(event) => {
          event.preventDefault();
          if (!busy && arxiv.length >= 8) {
            onArxivSubmit();
          }
        }}
      >
        <span className="mono upload-prefix">https://</span>
        <input
          value={arxiv}
          onChange={(event) => onArxivChange(event.target.value)}
          placeholder="arxiv.org/abs/2303.04137"
          className="upload-text-input mono"
          disabled={busy}
        />
        <button type="submit" disabled={busy || arxiv.length < 8} className="begin-button">
          {busy ? "Starting..." : "Begin ->"}
        </button>
      </form>
      <div className="upload-config-row">
        <label className="upload-config-label" htmlFor="model-select">Model</label>
        <select
          id="model-select"
          className="upload-config-select"
          value={model}
          disabled={busy}
          onChange={(event) => onModelChange(event.target.value as DemoModelChoice)}
        >
          <option value="sonnet">Sonnet</option>
          <option value="opus">Opus</option>
        </select>
      </div>
      {error ? <p className="upload-error">{error}</p> : null}
    </div>
  );
}

function NodeCard({
  node,
  onClick,
  selected,
  state
}: {
  node: WorkflowNode;
  onClick: () => void;
  selected: boolean;
  state: NodeState;
}) {
  const tones = {
    info: { icBg: "var(--info-soft)", icFg: "#3b48d1" },
    accent: { icBg: "var(--accent-soft)", icFg: "var(--accent-ink)" },
    hermes: { icBg: "var(--hermes-soft)", icFg: "var(--hermes)" },
    neutral: { icBg: "var(--chip)", icFg: "var(--muted)" }
  } as const;

  const tone = tones[node.tone];
  let borderColor = "var(--line)";
  let glow = "none";
  let opacity = 1;
  let background = "#fff";
  let showProgress = false;

  if (node.tone === "hermes") {
    background = "linear-gradient(180deg,#faf8ff,#fff)";
  }
  if (state === "running") {
    borderColor = node.tone === "hermes" ? "var(--hermes)" : "var(--accent)";
    glow =
      node.tone === "hermes"
        ? "0 0 0 4px rgba(124,92,255,.10), 0 12px 32px -16px rgba(124,92,255,.55)"
        : "0 0 0 4px rgba(22,178,92,.10), 0 12px 32px -16px rgba(22,178,92,.5)";
    showProgress = true;
  }
  if (state === "done") {
    borderColor = "var(--line-2)";
  }
  if (state === "upcoming") {
    opacity = 0.4;
  }
  if (selected) {
    borderColor = "var(--ink)";
    glow = "0 0 0 4px rgba(14,14,16,.06), 0 16px 36px -18px rgba(14,14,16,.5)";
  }

  return (
    <div
      className={state === "upcoming" ? "" : "wf-pop"}
      data-node="1"
      onClick={onClick}
      style={{
        position: "absolute",
        left: node.x,
        top: node.y,
        width: NODE_W,
        height: NODE_H,
        background,
        border: `1px solid ${borderColor}`,
        borderRadius: 14,
        padding: "12px 14px",
        display: "flex",
        flexDirection: "column",
        gap: 8,
        boxShadow: glow,
        cursor: state === "upcoming" ? "default" : "pointer",
        transition: "border-color .25s ease, box-shadow .25s ease, opacity .3s ease, transform .25s ease",
        opacity,
        transform: selected ? "translateY(-2px) scale(1.015)" : "scale(1)",
        zIndex: selected ? 5 : state === "running" ? 3 : 2
      }}
    >
      <div className="node-head">
        <div className="node-icon" style={{ background: tone.icBg, color: tone.icFg }}>
          {ICONS[node.icon]}
          {state === "running" ? <span className="wf-ring node-ring" /> : null}
        </div>
        <div className="node-copy">
          <div className="node-agent">{node.agent}</div>
          <div className="node-step">{node.step}</div>
        </div>
        {state === "done" ? (
          <div className="node-check">
            <svg width="10" height="10" viewBox="0 0 16 16" aria-hidden="true">
              <path
                d="M3 8.5l3 3 7-7"
                stroke="currentColor"
                strokeWidth="2.5"
                fill="none"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </div>
        ) : null}
      </div>
      {showProgress ? (
        <div className="node-progress">
          <div
            className="wf-bar"
            style={{ background: node.tone === "hermes" ? "var(--hermes)" : "var(--accent)" }}
          />
        </div>
      ) : null}
    </div>
  );
}

function Canvas({
  onSelect,
  run,
  selectedId,
  stateMap
}: {
  onSelect: (id: string | null) => void;
  run: LiveDemoRunState;
  selectedId: string | null;
  stateMap: Record<string, NodeState>;
}) {
  function edgeState(from: string, to: string) {
    const source = stateMap[from];
    const target = stateMap[to];
    if (source === "done" && target === "done") {
      return "done" as const;
    }
    if (source === "done" && target === "running") {
      return "active" as const;
    }
    return "upcoming" as const;
  }

  return (
    <div className="canvas-surface">
      <svg width={1740} height={720} className="canvas-edges" aria-hidden="true">
        {EDGES.map(([fromId, toId]) => {
          const from = NODES.find((node) => node.id === fromId)!;
          const to = NODES.find((node) => node.id === toId)!;
          const state = edgeState(fromId, toId);
          const path = buildEdgePath(from, to);
          let color = "var(--line-2)";
          let strokeWidth = 1.5;
          let opacity = 1;

          if (state === "upcoming") {
            opacity = 0.5;
          } else if (state === "done") {
            color = "var(--ink-2)";
            strokeWidth = 1.6;
          } else {
            color = "var(--accent)";
            strokeWidth = 2;
          }

          return (
            <g key={`${fromId}-${toId}`} style={{ opacity }}>
              <path d={path} fill="none" stroke={color} strokeWidth={strokeWidth} strokeLinecap="round" />
              {state === "active" ? (
                <path
                  d={path}
                  fill="none"
                  stroke="var(--accent)"
                  strokeWidth="3"
                  strokeLinecap="round"
                  strokeDasharray="4 8"
                  className="wf-flow"
                  style={{ opacity: 0.7 }}
                />
              ) : null}
            </g>
          );
        })}
      </svg>
      {NODES.map((node) => (
        <NodeCard
          key={node.id}
          node={node}
          state={stateMap[node.id]}
          selected={selectedId === node.id}
          onClick={() =>
            stateMap[node.id] === "upcoming" ? undefined : onSelect(node.id === selectedId ? null : node.id)
          }
        />
      ))}
      <GateChips run={run} />
    </div>
  );
}

function gateChipState(
  gate: { passed?: boolean; status?: string; chipStatus?: string } | undefined,
  stage: string | null,
  passedStages: string[],
  runningStages: string[]
): { state: "pending" | "running" | "passed" | "caveat" | "failed"; label: string } {
  // Prefer the pre-normalized chipStatus from buildLiveDemoDashboard (handles
  // backend GateStatus enum values like "verified_with_caveats").
  const normalized = gate?.chipStatus;
  if (normalized === "caveat") return { state: "caveat", label: "caveat" };
  if (normalized === "failed") return { state: "failed", label: issueText("failed") };
  if (normalized === "passed" || (stage && passedStages.includes(stage))) {
    return { state: "passed", label: "passed" };
  }
  if (normalized === "running" || (stage && runningStages.includes(stage))) {
    return { state: "running", label: "checking" };
  }
  return { state: "pending", label: "pending" };
}

// Edge midpoints in the 1740x720 SVG. Computed from NODES coordinates
// (NODE_W=200, NODE_H=80): right-edge of source node + left-edge of target node.
//   Gate 1: plan(500,420)→impl(740,310)  midpoint (720, 405)
//   Gate 2: impl(740,310)→bb(1020,200)   midpoint (980, 295)  [impl right edge = 940]
//   Gate 3: bb(1020,200)→audit(1300,310) midpoint (1260, 295) [paths converge here]
const GATE_COORDS: Record<"gate_1" | "gate_2" | "gate_3", { x: number; y: number; label: string }> = {
  gate_1: { x: 720, y: 405, label: "Gate 1" },
  gate_2: { x: 980, y: 295, label: "Gate 2" },
  gate_3: { x: 1260, y: 295, label: "Gate 3" }
};

function GateChips({ run }: { run: LiveDemoRunState }) {
  const gates = run.payload?.gates;
  const stage = run.payload?.summary.stage ?? null;

  const g1 = gateChipState(
    gates?.gate_1,
    stage,
    ["gate_1_passed", "baseline_implemented", "baseline_run", "gate_2_passed",
     "improvements_selected", "improvements_run", "gate_3_passed",
     "research_map_generated", "complete"],
    ["plan_created"]
  );
  const g2 = gateChipState(
    gates?.gate_2,
    stage,
    ["gate_2_passed", "improvements_selected", "improvements_run",
     "gate_3_passed", "research_map_generated", "complete"],
    ["baseline_run"]
  );
  const g3 = gateChipState(
    gates?.gate_3,
    stage,
    ["gate_3_passed", "research_map_generated", "complete"],
    ["improvements_run"]
  );

  return (
    <>
      {[
        { id: "gate_1" as const, ...GATE_COORDS.gate_1, view: g1, detail: gates?.gate_1?.detail },
        { id: "gate_2" as const, ...GATE_COORDS.gate_2, view: g2, detail: gates?.gate_2?.detail },
        { id: "gate_3" as const, ...GATE_COORDS.gate_3, view: g3, detail: gates?.gate_3?.detail }
      ].map(({ id, x, y, label, view, detail }) => (
        <div
          key={id}
          className={`gate-chip gate-chip-${view.state}`}
          style={{ left: x, top: y }}
          title={detail ?? undefined}
        >
          {label} · {view.label}
        </div>
      ))}
    </>
  );
}

function PanCanvas({
  onSelect,
  run,
  selectedId,
  stateMap
}: {
  onSelect: (id: string | null) => void;
  run: LiveDemoRunState;
  selectedId: string | null;
  stateMap: Record<string, NodeState>;
}) {
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const dragRef = useRef({ active: false, moved: false, slx: 0, sx: 0, sty: 0, sy: 0 });

  useEffect(() => {
    const wrap = wrapRef.current;
    if (!wrap) {
      return;
    }
    wrap.scrollLeft = Math.max(0, 740 - wrap.clientWidth / 2 + 100);
    wrap.scrollTop = Math.max(0, 310 - wrap.clientHeight / 2 + 40);
  }, []);

  useEffect(() => {
    function onMove(event: MouseEvent) {
      const drag = dragRef.current;
      if (!drag.active || !wrapRef.current) {
        return;
      }
      wrapRef.current.scrollLeft = drag.slx - (event.clientX - drag.sx);
      wrapRef.current.scrollTop = drag.sty - (event.clientY - drag.sy);
      if (Math.abs(event.clientX - drag.sx) + Math.abs(event.clientY - drag.sy) > 4) {
        drag.moved = true;
      }
    }

    function onUp() {
      dragRef.current.active = false;
      if (wrapRef.current) {
        wrapRef.current.style.cursor = "grab";
      }
    }

    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, []);

  return (
    <div
      ref={wrapRef}
      className="pan-wrap"
      onMouseDown={(event) => {
        if ((event.target as HTMLElement).closest("[data-node]")) {
          return;
        }
        const wrap = wrapRef.current;
        if (!wrap) {
          return;
        }
        dragRef.current = {
          active: true,
          moved: false,
          slx: wrap.scrollLeft,
          sx: event.clientX,
          sty: wrap.scrollTop,
          sy: event.clientY
        };
        wrap.style.cursor = "grabbing";
      }}
    >
      <Canvas
        run={run}
        stateMap={stateMap}
        selectedId={selectedId}
        onSelect={(id) => {
          if (!dragRef.current.moved) {
            onSelect(id);
          }
        }}
      />
    </div>
  );
}

function AgentInfo({
  failedNodeId,
  logEntries,
  node,
  run,
  state,
  telemetry
}: {
  failedNodeId: string | null;
  logEntries: Array<{ id: string; msg: string; time: string }>;
  node: WorkflowNode;
  run: LiveDemoRunState;
  state: NodeState;
  telemetry: LiveDemoRunState["telemetry"];
}) {
  const tones = {
    info: { icBg: "var(--info-soft)", icFg: "#3b48d1" },
    accent: { icBg: "var(--accent-soft)", icFg: "var(--accent-ink)" },
    hermes: { icBg: "var(--hermes-soft)", icFg: "var(--hermes)" },
    neutral: { icBg: "var(--chip)", icFg: "var(--muted)" }
  } as const;
  const tone = tones[node.tone];
  const status: Status =
    failedNodeId === node.id
      ? "attention"
      : state === "done"
        ? "completed"
        : state === "running"
          ? node.tone === "hermes"
            ? "auditing"
            : "running"
          : "queued";
  const showFinalPackage = node.id === "report";

  return (
    <div>
      <div className="agent-head">
        <div className="agent-icon" style={{ background: tone.icBg, color: tone.icFg }}>
          {ICONS[node.icon]}
        </div>
        <div>
          <div className="eyebrow">Agent</div>
          <div className="agent-name">{node.agent}</div>
        </div>
      </div>
      <StatusPill status={status} />
      <div className="agent-section">
        <div className="eyebrow">Task</div>
        <div className="agent-task">{node.step}</div>
        <div className="agent-role">{node.role}</div>
      </div>
      <div className="agent-detail">{node.detail}</div>
      {run.payload?.summary.stage ? (
        <div className="agent-section">
          <div className="eyebrow">Backend stage</div>
          <div className="agent-task">{run.payload.summary.stage}</div>
        </div>
      ) : null}
      {telemetry && telemetry.length > 0 ? (
        <div className="agent-section">
          <div className="eyebrow">Telemetry</div>
          <div className="telemetry-list">
            {telemetry.map((record, index) => (
              <div key={`${record.agent_id ?? "agent"}-${index}`} className="telemetry-row">
                <span className="telemetry-name">{record.agent_id ?? "agent"}</span>
                <span className="telemetry-meta">
                  {record.duration_seconds ? `${record.duration_seconds.toFixed(1)}s` : "active"}
                </span>
              </div>
            ))}
          </div>
        </div>
      ) : null}
      {state === "running" ? (
        <div className="agent-section">
          <div className="eyebrow">Progress</div>
          <div className="agent-progress">
            <div
              className="wf-bar"
              style={{ background: node.tone === "hermes" ? "var(--hermes)" : "var(--accent)" }}
            />
          </div>
        </div>
      ) : null}
      {logEntries.length > 0 ? (
        <div className="agent-section">
          <div className="eyebrow">Latest log</div>
          <ul className="agent-log-list">
            {logEntries.slice(0, 6).map((entry) => (
              <li key={entry.id} className="agent-log-item">
                <span className="mono agent-log-time">{entry.time}</span>
                <span className="agent-log-msg">{entry.msg}</span>
              </li>
            ))}
          </ul>
        </div>
      ) : null}
      {node.id === "audit" ? <HermesAuditPanel run={run} /> : null}
      {showFinalPackage ? <ScriptPanel run={run} /> : null}
    </div>
  );
}

function HermesAuditPanel({ run }: { run: LiveDemoRunState }) {
  const hermes = run.payload?.hermes;
  if (!hermes) {
    return (
      <div className="agent-section">
        <div className="eyebrow">Hermes audit</div>
        <div className="agent-detail">No audit findings yet.</div>
      </div>
    );
  }
  // Defensive defaults: backend Hermes payload can ship with missing keys
  // when an early-stage agent fails, an offline run skips audit, or an old
  // pipeline_state.json predates these fields. Don't crash the UI on it.
  const stepReports = hermes.stepReports && typeof hermes.stepReports === "object" ? hermes.stepReports : {};
  const checkpointReports = hermes.checkpointReports && typeof hermes.checkpointReports === "object" ? hermes.checkpointReports : {};
  const interventions = Array.isArray(hermes.interventions) ? hermes.interventions : [];
  const stepEntries = Object.entries(stepReports);
  const checkpointEntries = Object.entries(checkpointReports);

  return (
    <div className="agent-section hermes-panel">
      <div className="eyebrow">Hermes audit timeline</div>
      {stepEntries.length === 0 && checkpointEntries.length === 0 ? (
        <div className="agent-detail">No audit findings yet.</div>
      ) : null}
      {stepEntries.map(([stage, reports]) => {
        const latest = reports[reports.length - 1];
        if (!latest) return null;
        return (
          <div key={`step-${stage}`} className="hermes-row">
            <div className="hermes-row-head">
              <span className="hermes-row-stage">step / {stage}</span>
              <span className={`hermes-status hermes-status-${latest.status ?? "unknown"}`}>
                {latest.status ?? "unknown"}
              </span>
            </div>
            {latest.summary ? <div className="hermes-row-summary">{latest.summary}</div> : null}
            {latest.findings && latest.findings.length > 0 ? (
              <ul className="hermes-findings">
                {latest.findings.slice(0, 3).map((finding, idx) => (
                  <li key={idx}>{finding}</li>
                ))}
              </ul>
            ) : null}
          </div>
        );
      })}
      {checkpointEntries.map(([stage, reports]) => {
        const latest = reports[reports.length - 1];
        if (!latest) return null;
        return (
          <div key={`ckpt-${stage}`} className="hermes-row">
            <div className="hermes-row-head">
              <span className="hermes-row-stage">checkpoint / {stage}</span>
              <span className={`hermes-status hermes-status-${latest.status ?? "unknown"}`}>
                {latest.status ?? "unknown"}
              </span>
            </div>
            {latest.summary ? <div className="hermes-row-summary">{latest.summary}</div> : null}
          </div>
        );
      })}
      {interventions.length > 0 ? (
        <>
          <div className="eyebrow hermes-section-eyebrow">Interventions</div>
          <ul className="hermes-interventions">
            {interventions.slice(-5).reverse().map((intervention, idx) => (
              <li key={idx} className="hermes-intervention">
                <span className="hermes-intervention-action">{intervention.action ?? "action"}</span>
                <span className="hermes-intervention-target">on {intervention.target ?? "target"}</span>
                {intervention.reason ? (
                  <div className="hermes-intervention-reason">{intervention.reason}</div>
                ) : null}
              </li>
            ))}
          </ul>
        </>
      ) : null}
    </div>
  );
}

function ScriptPanel({ run }: { run: LiveDemoRunState }) {
  const pdf = run.sourcePdf;
  const benchmark = run.benchmark;
  const pdfUrl = sourcePdfUrl(run);
  const reportUrl = finalReportUrl(run);
  const score =
    benchmark && benchmark.overallScore > 0 ? `${benchmark.overallScore.toFixed(1)}%` : "Pending";
  const delta =
    benchmark && benchmark.reproducedValue > 0
      ? `${benchmark.deltaValue >= 0 ? "+" : ""}${benchmark.deltaValue.toFixed(1)}`
      : "pending";

  return (
    <div className="agent-section script-panel">
      <div className="script-panel-head">
        <div>
          <div className="eyebrow">Script panel</div>
          <div className="agent-task">Source PDF and final benchmark</div>
        </div>
        <span className="script-chip">code root</span>
      </div>

      <div className="pdf-card">
        <div className="pdf-preview" aria-hidden="true">
          {pdf ? (
            <object data={`${pdfUrl}#toolbar=0&navpanes=0`} type="application/pdf">
              <div className="pdf-fallback">{ICONS.doc}</div>
            </object>
          ) : (
            <div className="pdf-fallback">{ICONS.doc}</div>
          )}
        </div>
        <div className="pdf-copy">
          <div className="pdf-title">{pdf?.title ?? sourceTitle(run)}</div>
          <div className="pdf-meta">
            {pdf?.fileName ?? "paper.pdf"} · {formatBytes(pdf?.sizeBytes)}
            {pdf?.pageCount ? ` · ${pdf.pageCount} pages` : ""}
          </div>
          <div className="pdf-hash mono">sha256:{shortHash(pdf?.sha256)}</div>
          <div className="pdf-actions">
            <a className="btn btn-sm btn-dark" href={pdfUrl} target="_blank" rel="noreferrer">
              Preview PDF
            </a>
            <a className="btn btn-sm" href={pdfUrl} download="paper.pdf">
              Download
            </a>
          </div>
        </div>
      </div>

      <div className="code-root-row">
        <span className="code-root-label">Generated root</span>
        <span className="mono code-root-path">{pdf?.codePath ?? `${run.outputDir}/code/paper.pdf`}</span>
      </div>

      <a className="final-report-link" href={reportUrl} target="_blank" rel="noreferrer">
        <span className="final-report-kicker">Final report</span>
        <span className="final-report-title">
          {benchmark ? verdictLabel(benchmark.verdict) : "Benchmark report"}
        </span>
        <span className="final-report-copy">
          Open the formatted Markdown report generated alongside the codebase.
        </span>
        <span className="final-report-action">Open final report</span>
      </a>

      <div className="benchmark-card">
        <div className="benchmark-head">
          <div>
            <div className="benchmark-title">
              {benchmark?.benchmarkName ?? "PaperBench-style final benchmark"}
            </div>
            <div className="benchmark-subtitle">
              {benchmark?.paperbenchTaskId ?? "pending evaluator output"}
            </div>
          </div>
          <div className="benchmark-score">{score}</div>
        </div>
        <div className="metric-compare">
          <div>
            <span className="metric-label">Paper target</span>
            <strong>{benchmark ? benchmark.targetValue.toFixed(1) : "n/a"}</strong>
          </div>
          <div>
            <span className="metric-label">Reproduced</span>
            <strong>{benchmark && benchmark.reproducedValue > 0 ? benchmark.reproducedValue.toFixed(1) : "n/a"}</strong>
          </div>
          <div>
            <span className="metric-label">Delta</span>
            <strong>{delta}</strong>
          </div>
        </div>
        <div className="benchmark-verdict">
          <span className="status-dot" />
          {verdictLabel(benchmark?.verdict)}
        </div>
      </div>
    </div>
  );
}

function RunOverview({
  error,
  logEntries,
  run,
  stateMap
}: {
  error: string | null;
  logEntries: Array<{ id: string; msg: string; time: string }>;
  run: LiveDemoRunState;
  stateMap: Record<string, NodeState>;
}) {
  const totals = NODES.reduce(
    (acc, node) => {
      const state = stateMap[node.id];
      acc[state] += 1;
      return acc;
    },
    { done: 0, running: 0, upcoming: 0 }
  );

  const subagents = ["opt", "bb", "aug", "hor", "div"].map((id) => ({
    id,
    node: NODES.find((entry) => entry.id === id)!,
    state: stateMap[id]
  }));

  const title =
    run.status === "completed"
      ? "Run complete"
      : run.status === "failed"
        ? "Run needs attention"
        : "Reproducing live backend run";

  return (
    <div>
      <div className="eyebrow">Run</div>
      <div className="overview-title">{title}</div>
      <div className="overview-copy">
        {run.payload?.summary.stage
          ? `Current backend stage: ${run.payload.summary.stage}`
          : run.sourceNote ?? "Waiting for the first backend update."}
      </div>
      <div className="overview-grid">
        <Stat label="Done" value={totals.done} dot="var(--ink)" />
        <Stat label="Running" value={totals.running} dot="var(--accent)" pulse />
        <Stat label="Queued" value={totals.upcoming} dot="var(--line-2)" />
        <Stat label="Agents" value={NODES.length} dot="var(--muted-2)" />
      </div>
      {error || run.error ? (
        <div className="agent-section">
          <div className="eyebrow">Issue</div>
          <div className="agent-detail">{issueText(error ?? run.error)}</div>
        </div>
      ) : null}
      {logEntries.length > 0 ? (
        <div className="agent-section">
          <div className="eyebrow">Latest backend log</div>
          <div className="agent-detail">{logEntries[0].msg}</div>
        </div>
      ) : null}
      <ScriptPanel run={run} />
      <div className="agent-section">
        <div className="eyebrow">Improvement sub-agents</div>
        <div className="subagent-list">
          {subagents.map((item) => (
            <div
              key={item.id}
              className="subagent-row"
              style={{
                background:
                  item.state === "running"
                    ? "var(--accent-soft)"
                    : item.state === "done"
                      ? "var(--bg)"
                      : "transparent"
              }}
            >
              <span
                className={item.state === "running" ? "pulse-dot subagent-dot" : "subagent-dot"}
                style={{
                  background:
                    item.state === "running"
                      ? "var(--accent)"
                      : item.state === "done"
                        ? "var(--ink)"
                        : "var(--line-2)"
                }}
              />
              <span className="subagent-name">{item.node.agent}</span>
              <span className="subagent-step">{item.node.step}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function Stat({
  dot,
  label,
  pulse,
  value
}: {
  dot: string;
  label: string;
  pulse?: boolean;
  value: number;
}) {
  return (
    <div className="stat-card">
      <div className="stat-head">
        <span className={pulse ? "pulse-dot stat-dot" : "stat-dot"} style={{ background: dot }} />
        <span className="stat-label">{label}</span>
      </div>
      <div className="stat-value">{value}</div>
    </div>
  );
}

function RightPanel({
  error,
  run,
  selectedId,
  stateMap
}: {
  error: string | null;
  run: LiveDemoRunState;
  selectedId: string | null;
  stateMap: Record<string, NodeState>;
}) {
  const selected = selectedId ? NODES.find((node) => node.id === selectedId) ?? null : null;
  const logEntries = parseLogEntries(run);
  const telemetry = telemetryForSelectedNode(run, selectedId);
  const failedNodeId = failedNodeIdForRun(run, stateMap);

  return (
    <aside className="card side-panel">
      <div className="side-panel-top">
        <div key={selectedId ?? "overview"} className="rp-pane side-panel-scroll">
          {selected ? (
            <AgentInfo
              failedNodeId={failedNodeId}
              node={selected}
              state={stateMap[selected.id]}
              run={run}
              telemetry={telemetry}
              logEntries={logEntries}
            />
          ) : (
            <RunOverview run={run} stateMap={stateMap} logEntries={logEntries} error={error} />
          )}
        </div>
      </div>
      <div className="side-panel-bottom">
        <div className="side-panel-heading">
          <div className="side-panel-title">{selected ? `${selected.agent} activity` : "Live activity"}</div>
          <span className="live-pill">
            <span className="pulse-dot live-pill-dot" />
            live
          </span>
        </div>
        <div className="side-panel-scroll">
          {logEntries.length > 0 ? (
            logEntries.map((entry, index) => (
              <div key={entry.id} className="event fadeup" style={{ animationDelay: `${index * 30}ms` }}>
                <span className="mono event-time">{entry.time}</span>
                <span className="event-dot" />
                <div className="mono event-message">{entry.msg}</div>
              </div>
            ))
          ) : (
            <div className="empty-activity">Waiting for backend activity...</div>
          )}
        </div>
      </div>
    </aside>
  );
}

function ChevronGlyph({ dir }: { dir: "left" | "right" }) {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d={dir === "left" ? "M15 6l-6 6 6 6" : "M9 6l6 6-6 6"}
        stroke="currentColor"
        strokeWidth="2.2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

// A panel docked to the right viewport edge. Collapsed it is a labelled
// tab; expanded it slides out as an overlay drawer ON TOP of the graph —
// the canvas never reflows. Controlled by WorkflowView so the two
// drawers behave as an accordion (at most one open at a time).
function EdgeDrawer({
  label,
  tabIndex,
  open,
  tabShifted,
  onToggle,
  children
}: {
  label: string;
  tabIndex: 0 | 1;
  open: boolean;
  tabShifted: boolean;
  onToggle: () => void;
  children: ReactNode;
}) {
  return (
    <>
      {!open && (
        <button
          type="button"
          className={`edge-drawer-tab edge-drawer-tab-${tabIndex}${tabShifted ? " shifted" : ""}`}
          onClick={onToggle}
          aria-expanded={false}
        >
          <ChevronGlyph dir="left" />
          <span className="edge-drawer-tab-label">{label}</span>
        </button>
      )}
      <aside className={`edge-drawer${open ? " open" : ""}`} aria-hidden={!open} inert={!open}>
        <div className="edge-drawer-head">
          <span className="edge-drawer-head-label">{label}</span>
          <button
            type="button"
            className="edge-drawer-close"
            onClick={onToggle}
            aria-label={`Collapse ${label}`}
          >
            <ChevronGlyph dir="right" />
          </button>
        </div>
        <div className="edge-drawer-body">{children}</div>
      </aside>
    </>
  );
}

function WorkflowView({
  busy,
  dashboardEvents,
  error,
  onClear,
  run
}: {
  busy: boolean;
  dashboardEvents: DashboardLiveEvent[];
  error: string | null;
  onClear: () => Promise<void>;
  run: LiveDemoRunState;
}) {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [openDrawer, setOpenDrawer] = useState<"details" | "agents" | null>(null);

  useEffect(() => {
    // Post-mount read of a browser-only preference: it cannot run during
    // render (the server has no localStorage and would hydrate-mismatch),
    // so the one-shot setState here is intentional, not a cascade.
    try {
      const stored = window.localStorage.getItem(DRAWER_KEY);
      if (stored === "details" || stored === "agents") {
        // eslint-disable-next-line react-hooks/set-state-in-effect
        setOpenDrawer(stored);
      }
    } catch {
      // localStorage unavailable — start with both drawers collapsed.
    }
  }, []);

  const toggleDrawer = useCallback((key: "details" | "agents") => {
    setOpenDrawer((prev) => {
      const next = prev === key ? null : key;
      try {
        window.localStorage.setItem(DRAWER_KEY, next ?? "");
      } catch {
        // non-fatal
      }
      return next;
    });
  }, []);

  const stateMap = useMemo(() => stateMapForRun(run), [run]);
  const doneCount = NODES.filter((node) => stateMap[node.id] === "done").length;
  const liveStatus: Status =
    run.status === "failed"
      ? "attention"
      : run.status === "completed"
        ? "completed"
        : deriveStage(run) === "research_map_generated"
          ? "auditing"
          : run.status === "stopped"
            ? "stopped"
            : "running";
  const decisions = run.payload?.decisionLog ?? [];

  return (
    <>
      <div className="workflow-header">
        <div>
          <div className="eyebrow">workflow - {run.projectId}</div>
          <h1 className="h1 workflow-title">{sourceTitle(run)}</h1>
          <div className="workflow-meta">
            <StatusPill status={liveStatus} />
            <span className="workflow-meta-sep">.</span>
            <span className="mono">{doneCount}/{NODES.length} agents complete</span>
          </div>
        </div>
        <div className="workflow-actions">
          <button className="btn btn-primary" onClick={() => void onClear()} type="button" disabled={busy}>
            {busy ? "Stopping…" : "Start New Run"}
          </button>
        </div>
      </div>
      <div className="workflow-stage">
        <div className="canvas-wrap canvas-wrap-full">
          <PanCanvas run={run} stateMap={stateMap} selectedId={selectedId} onSelect={setSelectedId} />
        </div>
        <EdgeDrawer
          label="Node details"
          tabIndex={0}
          open={openDrawer === "details"}
          tabShifted={openDrawer !== null}
          onToggle={() => toggleDrawer("details")}
        >
          <RightPanel run={run} selectedId={selectedId} stateMap={stateMap} error={error} />
        </EdgeDrawer>
        <EdgeDrawer
          label="Live agents"
          tabIndex={1}
          open={openDrawer === "agents"}
          tabShifted={openDrawer !== null}
          onToggle={() => toggleDrawer("agents")}
        >
          <AgentTimelineRail events={dashboardEvents} decisions={decisions} />
        </EdgeDrawer>
      </div>
    </>
  );
}

function PrototypeStyles() {
  return (
    <style jsx global>{`
      .reproLab {
        --bg: #f4f4f5;
        --panel: #ffffff;
        --ink: #0e0e10;
        --ink-2: #1f2024;
        --muted: #6b6b73;
        --muted-2: #9b9ba3;
        --line: #ececef;
        --line-2: #dcdce0;
        --dotted: rgba(155, 155, 163, 0.5);
        --chip: #f1f1f3;
        --accent: #16b25c;
        --accent-soft: #e6f7ed;
        --accent-ink: #0e7a3d;
        --err: #dc3545;
        --err-soft: #fde7ea;
        --warn: #d89500;
        --warn-soft: #fff3c4;
        --warn-ink: #8a5b00;
        --info-soft: #ecedff;
        --hermes: #7c5cff;
        --hermes-soft: #ede8ff;
        min-height: 100vh;
        background: var(--bg);
        color: var(--ink);
        font-family: "Plus Jakarta Sans", "Segoe UI", sans-serif;
      }
      .reproLab * {
        box-sizing: border-box;
      }
      .reproLab a {
        color: inherit;
        text-decoration: none;
      }
      .reproLab button {
        font: inherit;
      }
      .reproLab .mono {
        font-family: "JetBrains Mono", "SFMono-Regular", Consolas, monospace;
        font-feature-settings: "tnum" 1, "zero" 1;
        letter-spacing: 0;
      }
      .reproLab .layout {
        display: flex;
        min-height: 100vh;
        background: var(--bg);
      }
      .reproLab .sidebar {
        width: 212px;
        flex-shrink: 0;
        padding: 22px 14px 18px;
        display: flex;
        flex-direction: column;
        gap: 2px;
        position: sticky;
        top: 0;
        align-self: flex-start;
        height: 100vh;
        transition: width 0.32s cubic-bezier(0.2, 0.7, 0.2, 1);
        overflow: visible;
      }
      .reproLab .sidebar.collapsed {
        width: 64px;
        padding-left: 10px;
        padding-right: 10px;
      }
      .reproLab .sidebar.collapsed .navitem {
        justify-content: center;
        padding: 9px 0;
        gap: 0;
        overflow: visible;
      }
      .reproLab .sidebar.collapsed .nav-label,
      .reproLab .sidebar.collapsed .nav-aside,
      .reproLab .sidebar.collapsed .brand-text,
      .reproLab .sidebar.collapsed .nav-section-title,
      .reproLab .sidebar.collapsed .dotted {
        display: none;
      }
      .reproLab .sidebar.collapsed .brand-row {
        justify-content: center;
        padding: 4px 0 12px;
      }
      .reproLab .sidebar.collapsed .navitem.active {
        background: var(--ink);
        color: #fff;
      }
      .reproLab .sidebar.collapsed .navitem.active .nav-icon {
        color: #fff !important;
      }
      .reproLab .sb-toggle {
        position: absolute;
        top: 24px;
        right: -12px;
        width: 24px;
        height: 24px;
        border-radius: 999px;
        background: #fff;
        border: 1px solid var(--line-2);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        color: var(--muted);
        z-index: 10;
        cursor: pointer;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04), 0 4px 10px rgba(0, 0, 0, 0.04);
      }
      .reproLab .sidebar.collapsed .sb-toggle {
        transform: rotate(180deg);
      }
      .reproLab .brand-row {
        display: flex;
        align-items: center;
        gap: 11px;
        padding: 4px 11px 12px;
        width: 100%;
        border: 0;
        background: transparent;
        color: inherit;
        cursor: pointer;
        text-align: left;
      }
      .reproLab .brand-row:hover .brand-text {
        color: var(--ink);
      }
      .reproLab .brand-text {
        font-weight: 700;
        font-size: 17px;
        letter-spacing: -0.025em;
      }
      .reproLab .dotted {
        height: 1px;
        background-image: linear-gradient(to right, var(--dotted) 50%, transparent 50%);
        background-size: 6px 1px;
        background-repeat: repeat-x;
        margin: 14px 0;
      }
      .reproLab .navitem {
        display: flex;
        align-items: center;
        gap: 11px;
        padding: 8px 11px;
        border-radius: 10px;
        font-size: 13.5px;
        color: var(--ink-2);
        font-weight: 500;
        transition: background 0.12s ease, color 0.12s ease;
      }
      .reproLab .navitem:hover {
        background: rgba(0, 0, 0, 0.04);
      }
      .reproLab .navitem.active {
        background: #fff;
        box-shadow: 0 1px 0 rgba(0, 0, 0, 0.04), 0 1px 2px rgba(0, 0, 0, 0.04);
      }
      .reproLab .nav-icon {
        display: inline-flex;
        flex-shrink: 0;
      }
      .reproLab .nav-aside {
        margin-left: auto;
        font-size: 10px;
        font-weight: 600;
        background: var(--hermes-soft);
        color: var(--hermes);
        padding: 1px 7px;
        border-radius: 999px;
      }
      .reproLab .nav-section-title {
        padding: 0 10px 6px;
        font-size: 10.5px;
        color: var(--muted-2);
        letter-spacing: 0.06em;
        text-transform: uppercase;
        font-weight: 600;
      }
      .reproLab .navitem-small {
        padding: 6px 11px;
        font-size: 12.5px;
        color: var(--muted);
      }
      .reproLab .nav-status-dot {
        width: 7px;
        height: 7px;
        border-radius: 999px;
      }
      .reproLab .sidebar-footer {
        margin-top: auto;
      }
      .reproLab .content {
        flex: 1;
        min-width: 0;
        padding: 22px 28px 40px;
      }
      .reproLab .upload-shell {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 28px;
        min-height: calc(100vh - 80px);
        margin: -22px -28px;
        padding: 60px 40px;
        text-align: center;
      }
      .reproLab .upload-zone {
        width: 100%;
        max-width: 760px;
        border: 2px dashed var(--line-2);
        border-radius: 24px;
        background: #fafafb;
        padding: 68px 40px;
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 24px;
        cursor: pointer;
        transition: background 0.25s ease, border-color 0.25s ease;
      }
      .reproLab .upload-zone:hover,
      .reproLab .upload-zone.over {
        background: var(--accent-soft);
        border-color: var(--accent);
      }
      .reproLab .hidden-input {
        display: none;
      }
      .reproLab .upload-icon {
        width: 120px;
        height: 120px;
        border-radius: 28px;
        background: #fff;
        border: 1px solid var(--line);
        display: flex;
        align-items: center;
        justify-content: center;
        color: var(--ink);
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04), 0 12px 32px -16px rgba(0, 0, 0, 0.18);
      }
      .reproLab .upload-icon svg {
        width: 58px;
        height: 58px;
      }
      .reproLab .upload-title {
        font-size: 52px;
        font-weight: 700;
        letter-spacing: -0.04em;
        line-height: 1;
        margin: 0;
      }
      .reproLab .upload-copy {
        font-size: 17px;
        color: var(--muted);
        margin: 0;
        letter-spacing: -0.01em;
        line-height: 1.5;
        max-width: 460px;
      }
      .reproLab .upload-meta {
        font-size: 12.5px;
        color: var(--muted-2);
      }
      .reproLab .upload-divider {
        display: flex;
        align-items: center;
        gap: 14px;
        width: 100%;
        max-width: 560px;
      }
      .reproLab .upload-divider span:first-child,
      .reproLab .upload-divider span:last-child {
        flex: 1;
        height: 1px;
        background: var(--line);
      }
      .reproLab .upload-divider-label {
        font-size: 11.5px;
        color: var(--muted-2);
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }
      .reproLab .upload-form {
        display: flex;
        align-items: center;
        gap: 8px;
        width: 100%;
        max-width: 560px;
        border: 1px solid var(--line-2);
        border-radius: 999px;
        padding: 6px 6px 6px 18px;
        background: #fff;
      }
      .reproLab .upload-prefix {
        font-size: 12.5px;
        color: var(--muted-2);
      }
      .reproLab .upload-text-input {
        flex: 1;
        border: none;
        outline: none;
        background: none;
        font-size: 14px;
        color: var(--ink);
        padding: 8px 0;
      }
      .reproLab .begin-button {
        padding: 10px 22px;
        border-radius: 999px;
        background: var(--line);
        color: var(--muted-2);
        font-size: 13.5px;
        font-weight: 600;
        letter-spacing: -0.005em;
        cursor: not-allowed;
      }
      .reproLab .begin-button:enabled {
        background: var(--ink);
        color: #fff;
        cursor: pointer;
      }
      .reproLab .upload-error {
        margin: 0;
        font-size: 13px;
        color: var(--err);
      }
      .reproLab .upload-config-row {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-top: 12px;
      }
      .reproLab .upload-config-label {
        font-size: 12px;
        font-weight: 600;
        color: var(--muted);
        letter-spacing: 0.02em;
        text-transform: uppercase;
        white-space: nowrap;
      }
      .reproLab .upload-config-select {
        font: inherit;
        font-size: 13px;
        padding: 5px 10px;
        border: 1px solid var(--line-2);
        border-radius: 8px;
        background: var(--panel);
        color: var(--ink);
        cursor: pointer;
        outline: none;
      }
      .reproLab .upload-config-select:focus {
        border-color: var(--accent);
      }
      .reproLab .workflow-header {
        display: flex;
        align-items: flex-end;
        gap: 10px;
        padding: 4px 0 16px;
      }
      .reproLab .eyebrow {
        font-size: 11px;
        color: var(--muted);
        letter-spacing: 0.04em;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 4px;
      }
      .reproLab .h1 {
        font-size: 28px;
        font-weight: 700;
        letter-spacing: -0.03em;
        margin: 0;
      }
      .reproLab .workflow-title {
        font-size: 24px;
      }
      .reproLab .workflow-meta {
        font-size: 12.5px;
        color: var(--muted);
        margin-top: 6px;
        display: flex;
        align-items: center;
        gap: 10px;
      }
      .reproLab .workflow-meta-sep {
        color: var(--muted-2);
      }
      .reproLab .workflow-actions {
        margin-left: auto;
        display: flex;
        align-items: center;
        gap: 8px;
      }
      .reproLab .btn {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        height: 36px;
        padding: 0 14px;
        border-radius: 999px;
        font-size: 13.5px;
        font-weight: 500;
        letter-spacing: -0.01em;
        background: #fff;
        border: 1px solid var(--line);
        color: var(--ink-2);
      }
      .reproLab .btn-sm {
        height: 30px;
        padding: 0 11px;
        font-size: 12.5px;
      }
      .reproLab .btn-primary {
        background: var(--accent);
        border-color: var(--accent);
        color: #fff;
        font-weight: 600;
      }
      .reproLab .btn-primary:hover {
        background: var(--accent-ink);
        border-color: var(--accent-ink);
      }
      .reproLab .btn:disabled {
        opacity: 0.55;
        cursor: not-allowed;
      }
      .reproLab .workflow-stage {
        position: relative;
      }
      .reproLab .canvas-wrap-full {
        width: 100%;
      }
      /* Edge-docked drawers — the graph stays full-bleed; a drawer slides
         in as an overlay on top of it and never reflows the canvas. The
         closed drawer's tab shifts to the open drawer's edge so it stays
         reachable (accordion-style switching). */
      .reproLab .edge-drawer-tab {
        position: fixed;
        right: 0;
        z-index: 45;
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 9px;
        width: 34px;
        min-height: 132px;
        padding: 16px 0;
        background: var(--panel);
        border: 1px solid var(--line);
        border-right: none;
        border-radius: 12px 0 0 12px;
        color: var(--ink-2);
        cursor: pointer;
        box-shadow: -8px 0 24px -16px rgba(0, 0, 0, 0.3);
        transition: background 0.18s ease, color 0.18s ease,
          right 0.24s cubic-bezier(0.4, 0, 0.2, 1);
      }
      .reproLab .edge-drawer-tab:hover {
        background: var(--chip);
        color: var(--ink);
      }
      .reproLab .edge-drawer-tab-0 {
        top: calc(50% - 140px);
      }
      .reproLab .edge-drawer-tab-1 {
        top: calc(50% + 8px);
      }
      .reproLab .edge-drawer-tab.shifted {
        right: 384px;
      }
      .reproLab .edge-drawer-tab-label {
        writing-mode: vertical-rl;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.01em;
      }
      .reproLab .edge-drawer {
        position: fixed;
        top: 0;
        right: 0;
        bottom: 0;
        width: 384px;
        max-width: 92vw;
        z-index: 55;
        background: var(--panel);
        border-left: 1px solid var(--line-2);
        display: flex;
        flex-direction: column;
        box-shadow: -24px 0 60px -24px rgba(0, 0, 0, 0.34);
        transform: translateX(100%);
        transition: transform 0.24s cubic-bezier(0.4, 0, 0.2, 1);
        will-change: transform;
      }
      .reproLab .edge-drawer.open {
        transform: translateX(0);
      }
      .reproLab .edge-drawer-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
        padding: 14px 16px;
        border-bottom: 1px solid var(--line);
        flex-shrink: 0;
      }
      .reproLab .edge-drawer-head-label {
        font-size: 13px;
        font-weight: 700;
        letter-spacing: -0.015em;
        color: var(--ink);
      }
      .reproLab .edge-drawer-close {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 27px;
        height: 27px;
        border-radius: 999px;
        background: var(--chip);
        border: 1px solid var(--line);
        color: var(--ink-2);
        cursor: pointer;
        transition: background 0.16s ease, color 0.16s ease;
      }
      .reproLab .edge-drawer-close:hover {
        background: var(--line-2);
        color: var(--ink);
      }
      .reproLab .edge-drawer-body {
        flex: 1;
        min-height: 0;
        display: flex;
        flex-direction: column;
        overflow: hidden;
      }
      /* The wrapped panels were sized for the old 3-column grid — strip
         their standalone geometry so they fill the drawer body instead. */
      .reproLab .edge-drawer-body > .side-panel,
      .reproLab .edge-drawer-body > .timeline-rail {
        flex: 1;
        width: 100%;
        height: 100%;
        max-height: none;
        position: static;
        border: none;
        border-radius: 0;
      }
      .reproLab .timeline-rail {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 14px;
        height: calc(100vh - 180px);
        overflow-y: auto;
        font-size: 12.5px;
        color: var(--ink);
        display: flex;
        flex-direction: column;
        gap: 8px;
      }
      .reproLab .timeline-rail-head {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 6px;
      }
      .reproLab .timeline-rail-count {
        font-size: 11px;
        color: var(--muted);
      }
      .reproLab .timeline-list {
        list-style: none;
        margin: 0 0 4px 0;
        padding: 0;
        display: flex;
        flex-direction: column;
        gap: 6px;
      }
      .reproLab .timeline-empty {
        color: var(--muted);
        font-size: 11.5px;
        padding: 4px 0;
      }
      .reproLab .timeline-agent {
        display: grid;
        grid-template-columns: 8px minmax(0, 1fr) auto;
        gap: 8px;
        align-items: start;
        padding: 6px 8px;
        border-radius: 8px;
        background: var(--chip);
        cursor: pointer;
      }
      .reproLab .timeline-agent[role="button"]:hover {
        background: var(--line);
      }
      .reproLab .timeline-agent-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-top: 4px;
      }
      .reproLab .timeline-agent-running .timeline-agent-dot {
        animation: pulseDot 1.4s ease-in-out infinite;
      }
      @keyframes pulseDot {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
      }
      .reproLab .timeline-agent-label {
        font-weight: 600;
        font-size: 12.5px;
      }
      .reproLab .timeline-agent-task {
        font-size: 11.5px;
        color: var(--muted);
        margin-top: 2px;
      }
      .reproLab .timeline-agent-time {
        font-size: 11px;
        color: var(--muted-2);
        font-variant-numeric: tabular-nums;
      }
      .reproLab .timeline-reason,
      .reproLab .timeline-context {
        padding: 6px 8px;
        border-radius: 8px;
        background: var(--bg);
        border: 1px solid var(--line);
      }
      .reproLab .timeline-reason-head,
      .reproLab .timeline-context-head {
        display: flex;
        gap: 6px;
        align-items: center;
        font-size: 11px;
        color: var(--muted);
        margin-bottom: 2px;
      }
      .reproLab .timeline-reason-agent,
      .reproLab .timeline-context-route {
        font-weight: 600;
        color: var(--ink-2);
      }
      .reproLab .timeline-reason-type,
      .reproLab .timeline-context-type {
        text-transform: uppercase;
        font-size: 10px;
        letter-spacing: 0.04em;
      }
      .reproLab .timeline-reason-time {
        margin-left: auto;
        font-variant-numeric: tabular-nums;
      }
      .reproLab .timeline-reason-title,
      .reproLab .timeline-context-title {
        font-weight: 500;
        font-size: 12.5px;
      }
      .reproLab .timeline-reason-detail,
      .reproLab .timeline-context-detail {
        font-size: 11.5px;
        color: var(--muted);
        margin-top: 2px;
        white-space: pre-wrap;
        word-break: break-word;
      }
      .reproLab .timeline-decision {
        font-size: 12px;
        color: var(--ink);
        padding: 4px 0;
        border-bottom: 1px dashed var(--line);
      }
      .reproLab .timeline-decision:last-child {
        border-bottom: none;
      }
      .reproLab .agent-log-list {
        list-style: none;
        margin: 0;
        padding: 0;
        display: flex;
        flex-direction: column;
        gap: 4px;
      }
      .reproLab .agent-log-item {
        display: grid;
        grid-template-columns: 44px minmax(0, 1fr);
        gap: 8px;
        font-size: 12px;
        line-height: 1.45;
      }
      .reproLab .agent-log-time {
        color: var(--muted-2);
        font-variant-numeric: tabular-nums;
      }
      .reproLab .agent-log-msg {
        color: var(--ink-2);
        word-break: break-word;
      }
      .reproLab .hermes-panel {
        display: flex;
        flex-direction: column;
        gap: 8px;
      }
      .reproLab .hermes-row {
        padding: 8px 10px;
        border-radius: 10px;
        background: var(--chip);
        border: 1px solid var(--line);
      }
      .reproLab .hermes-row-head {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 11px;
        color: var(--muted);
        margin-bottom: 4px;
      }
      .reproLab .hermes-row-stage {
        text-transform: uppercase;
        letter-spacing: 0.04em;
        font-weight: 600;
      }
      .reproLab .hermes-status {
        font-size: 10.5px;
        padding: 2px 6px;
        border-radius: 999px;
        background: var(--bg);
        text-transform: uppercase;
        letter-spacing: 0.04em;
      }
      .reproLab .hermes-status-grounded { background: var(--info-soft); color: #2e3a8c; }
      .reproLab .hermes-status-caveat { background: var(--warn-soft); color: var(--warn-ink); }
      .reproLab .hermes-status-unsupported,
      .reproLab .hermes-status-system_error { background: var(--err-soft); color: var(--err); }
      .reproLab .hermes-row-summary {
        font-size: 12.5px;
        color: var(--ink-2);
      }
      .reproLab .hermes-findings {
        margin: 6px 0 0 16px;
        padding: 0;
        font-size: 12px;
        color: var(--muted);
      }
      .reproLab .hermes-section-eyebrow {
        margin-top: 4px;
      }
      .reproLab .hermes-interventions {
        list-style: none;
        margin: 0;
        padding: 0;
        display: flex;
        flex-direction: column;
        gap: 6px;
      }
      .reproLab .hermes-intervention {
        font-size: 12.5px;
      }
      .reproLab .hermes-intervention-action {
        font-weight: 600;
        margin-right: 6px;
      }
      .reproLab .hermes-intervention-target {
        color: var(--muted);
      }
      .reproLab .hermes-intervention-reason {
        font-size: 11.5px;
        color: var(--muted);
        margin-top: 2px;
      }
      .reproLab .gate-chip {
        position: absolute;
        transform: translate(-50%, -50%);
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 999px;
        padding: 2px 8px;
        font-size: 10.5px;
        font-weight: 600;
        color: var(--muted);
        pointer-events: auto;
        white-space: nowrap;
        cursor: default;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
      }
      .reproLab .gate-chip-running {
        color: var(--accent-ink);
        border-color: var(--accent-soft);
      }
      .reproLab .gate-chip-passed {
        color: #2c7a4a;
        border-color: #b9e0c8;
        background: #f0faf4;
      }
      .reproLab .gate-chip-caveat {
        color: var(--warn-ink);
        border-color: var(--warn-soft);
        background: var(--warn-soft);
      }
      .reproLab .gate-chip-failed {
        color: var(--err);
        border-color: var(--err-soft);
        background: var(--err-soft);
      }
      .reproLab .canvas-wrap {
        flex: 1;
        min-width: 0;
        height: calc(100vh - 180px);
        background: #fafafb;
        border: 1px solid var(--line);
        border-radius: 16px;
        overflow: hidden;
        position: relative;
      }
      .reproLab .pan-wrap {
        width: 100%;
        height: 100%;
        overflow: auto;
        cursor: grab;
        user-select: none;
      }
      .reproLab .canvas-surface {
        position: relative;
        width: 1740px;
        height: 720px;
        background-image: radial-gradient(#dcdce0 1px, transparent 1px);
        background-size: 22px 22px;
        background-color: #fafafb;
      }
      .reproLab .canvas-edges {
        position: absolute;
        inset: 0;
        pointer-events: none;
      }
      .reproLab .node-head {
        display: flex;
        align-items: center;
        gap: 9px;
      }
      .reproLab .node-icon {
        width: 30px;
        height: 30px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        position: relative;
      }
      .reproLab .node-ring {
        position: absolute;
        inset: -3px;
        border-radius: 11px;
        border: 1.5px solid currentColor;
        opacity: 0.5;
      }
      .reproLab .node-copy {
        min-width: 0;
        flex: 1;
      }
      .reproLab .node-agent {
        font-size: 10.5px;
        color: var(--muted-2);
        letter-spacing: 0.04em;
        text-transform: uppercase;
        font-weight: 600;
      }
      .reproLab .node-step {
        font-size: 13px;
        font-weight: 600;
        letter-spacing: -0.01em;
        line-height: 1.2;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }
      .reproLab .node-check {
        width: 18px;
        height: 18px;
        border-radius: 999px;
        background: var(--ink);
        color: #fff;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
      }
      .reproLab .node-progress,
      .reproLab .agent-progress {
        height: 5px;
        background: var(--line);
        border-radius: 999px;
        overflow: hidden;
      }
      .reproLab .node-progress {
        margin-top: auto;
        height: 3px;
      }
      .reproLab .wf-bar {
        height: 100%;
        border-radius: 999px;
        transform-origin: left;
        animation: wfBar 3s linear forwards;
      }
      .reproLab .card {
        background: var(--panel);
        border-radius: 16px;
        border: 1px solid var(--line);
      }
      .reproLab .side-panel {
        width: 360px;
        flex-shrink: 0;
        padding: 0;
        overflow: hidden;
        display: flex;
        flex-direction: column;
        height: calc(100vh - 180px);
        position: sticky;
        top: 22px;
      }
      .reproLab .side-panel-top {
        flex: 1 1 50%;
        min-height: 0;
        border-bottom: 1px solid var(--line);
      }
      .reproLab .side-panel-bottom {
        flex: 1 1 50%;
        min-height: 0;
        display: flex;
        flex-direction: column;
      }
      .reproLab .side-panel-scroll {
        padding: 18px 20px;
        overflow-y: auto;
        max-height: 100%;
      }
      .reproLab .side-panel-heading {
        padding: 12px 18px;
        display: flex;
        align-items: center;
        border-bottom: 1px solid var(--line);
      }
      .reproLab .side-panel-title {
        font-size: 13px;
        font-weight: 700;
        letter-spacing: -0.015em;
      }
      .reproLab .live-pill {
        margin-left: 8px;
        font-size: 10.5px;
        font-weight: 600;
        color: var(--accent-ink);
        background: var(--accent-soft);
        padding: 2px 8px;
        border-radius: 999px;
        display: inline-flex;
        align-items: center;
        gap: 5px;
      }
      .reproLab .live-pill-dot {
        width: 5px;
        height: 5px;
        border-radius: 999px;
        background: var(--accent);
      }
      .reproLab .agent-head {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 14px;
      }
      .reproLab .agent-icon {
        width: 44px;
        height: 44px;
        border-radius: 11px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
      }
      .reproLab .agent-name,
      .reproLab .overview-title {
        font-size: 18px;
        font-weight: 700;
        letter-spacing: -0.02em;
        line-height: 1.2;
      }
      .reproLab .agent-section {
        margin-top: 16px;
      }
      .reproLab .agent-task {
        font-size: 14px;
        font-weight: 600;
        letter-spacing: -0.01em;
        line-height: 1.3;
      }
      .reproLab .agent-role,
      .reproLab .overview-copy {
        font-size: 12.5px;
        color: var(--muted);
        margin-top: 4px;
      }
      .reproLab .agent-detail {
        margin-top: 14px;
        padding: 10px 12px;
        background: var(--bg);
        border-radius: 10px;
        font-size: 11.5px;
        color: var(--ink-2);
        line-height: 1.55;
      }
      .reproLab .script-panel {
        padding: 14px;
        border: 1px solid var(--line);
        border-radius: 14px;
        background: linear-gradient(180deg, #fff, #fafafb);
      }
      .reproLab .script-panel-head {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 12px;
      }
      .reproLab .script-chip {
        padding: 3px 8px;
        border-radius: 999px;
        background: var(--ink);
        color: #fff;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
      }
      .reproLab .pdf-card {
        display: grid;
        grid-template-columns: 96px minmax(0, 1fr);
        gap: 12px;
        align-items: stretch;
      }
      .reproLab .pdf-preview {
        min-height: 128px;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid var(--line);
        background:
          linear-gradient(180deg, rgba(22, 178, 92, 0.08), transparent 45%),
          #fff;
      }
      .reproLab .pdf-preview object {
        width: 100%;
        height: 100%;
        min-height: 128px;
        display: block;
      }
      .reproLab .pdf-fallback {
        height: 100%;
        min-height: 128px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: var(--accent-ink);
      }
      .reproLab .pdf-copy {
        min-width: 0;
        display: flex;
        flex-direction: column;
        justify-content: center;
        gap: 6px;
      }
      .reproLab .pdf-title {
        font-size: 14px;
        font-weight: 700;
        letter-spacing: -0.015em;
        line-height: 1.25;
      }
      .reproLab .pdf-meta,
      .reproLab .pdf-hash {
        font-size: 11px;
        color: var(--muted);
        line-height: 1.4;
      }
      .reproLab .pdf-actions {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-top: 4px;
      }
      .reproLab .btn-dark {
        background: var(--ink);
        color: #fff;
        border-color: var(--ink);
      }
      .reproLab .code-root-row {
        margin-top: 12px;
        padding: 9px 10px;
        border-radius: 10px;
        background: var(--bg);
      }
      .reproLab .code-root-label {
        display: block;
        margin-bottom: 4px;
        color: var(--muted-2);
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
      }
      .reproLab .code-root-path {
        display: block;
        overflow: hidden;
        text-overflow: ellipsis;
        color: var(--ink-2);
        font-size: 10.5px;
        white-space: nowrap;
      }
      .reproLab .final-report-link {
        margin-top: 12px;
        padding: 12px;
        border-radius: 12px;
        border: 1px solid rgba(14, 14, 16, 0.1);
        background:
          linear-gradient(180deg, rgba(255, 255, 255, 0.92), rgba(244, 244, 245, 0.96)),
          var(--panel);
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto;
        gap: 4px 12px;
        align-items: center;
        transition: border-color 0.16s ease, transform 0.16s ease, box-shadow 0.16s ease;
      }
      .reproLab .final-report-link:hover {
        border-color: rgba(14, 14, 16, 0.24);
        box-shadow: 0 10px 28px -20px rgba(14, 14, 16, 0.45);
        transform: translateY(-1px);
      }
      .reproLab .final-report-kicker {
        grid-column: 1 / -1;
        color: var(--muted-2);
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 0.05em;
        text-transform: uppercase;
      }
      .reproLab .final-report-title {
        min-width: 0;
        font-size: 13px;
        font-weight: 800;
        letter-spacing: -0.015em;
        color: var(--ink);
      }
      .reproLab .final-report-copy {
        grid-column: 1 / -1;
        color: var(--muted);
        font-size: 11.5px;
        line-height: 1.45;
      }
      .reproLab .final-report-action {
        padding: 6px 9px;
        border-radius: 999px;
        background: var(--ink);
        color: #fff;
        font-size: 10.5px;
        font-weight: 700;
        white-space: nowrap;
      }
      .reproLab .benchmark-card {
        margin-top: 12px;
        padding: 12px;
        border-radius: 12px;
        background: #101113;
        color: #fff;
      }
      .reproLab .benchmark-head {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 12px;
      }
      .reproLab .benchmark-title {
        font-size: 12.5px;
        font-weight: 700;
        letter-spacing: -0.01em;
      }
      .reproLab .benchmark-subtitle {
        margin-top: 3px;
        color: rgba(255, 255, 255, 0.58);
        font-size: 10.5px;
      }
      .reproLab .benchmark-score {
        font-size: 24px;
        line-height: 1;
        font-weight: 800;
        letter-spacing: -0.04em;
        color: #7df2a8;
      }
      .reproLab .metric-compare {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 6px;
        margin-top: 12px;
      }
      .reproLab .metric-compare > div {
        padding: 8px;
        border-radius: 9px;
        background: rgba(255, 255, 255, 0.08);
        min-width: 0;
      }
      .reproLab .metric-label {
        display: block;
        margin-bottom: 4px;
        color: rgba(255, 255, 255, 0.52);
        font-size: 9.5px;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
      }
      .reproLab .metric-compare strong {
        font-size: 13px;
        letter-spacing: -0.02em;
      }
      .reproLab .benchmark-verdict {
        margin-top: 10px;
        display: inline-flex;
        align-items: center;
        gap: 7px;
        color: rgba(255, 255, 255, 0.74);
        font-size: 11px;
        font-weight: 600;
      }
      .reproLab .benchmark-verdict .status-dot {
        background: #7df2a8;
      }
      .reproLab .overview-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
        margin-top: 14px;
      }
      .reproLab .stat-card {
        padding: 10px 12px;
        background: var(--bg);
        border-radius: 10px;
      }
      .reproLab .stat-head {
        display: flex;
        align-items: center;
        gap: 6px;
        margin-bottom: 4px;
      }
      .reproLab .stat-dot,
      .reproLab .subagent-dot,
      .reproLab .status-dot,
      .reproLab .event-dot {
        width: 6px;
        height: 6px;
        border-radius: 999px;
        display: inline-block;
      }
      .reproLab .event-dot {
        margin-top: 5px;
        width: 7px;
        height: 7px;
        background: var(--ink);
      }
      .reproLab .stat-label {
        font-size: 10.5px;
        color: var(--muted-2);
        letter-spacing: 0.04em;
        text-transform: uppercase;
        font-weight: 600;
      }
      .reproLab .stat-value {
        font-size: 22px;
        font-weight: 700;
        letter-spacing: -0.025em;
      }
      .reproLab .subagent-list,
      .reproLab .telemetry-list {
        display: flex;
        flex-direction: column;
        gap: 5px;
      }
      .reproLab .subagent-row,
      .reproLab .telemetry-row {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 6px 10px;
        border-radius: 8px;
      }
      .reproLab .telemetry-row {
        background: var(--bg);
      }
      .reproLab .subagent-name,
      .reproLab .telemetry-name {
        font-size: 12px;
        font-weight: 600;
        letter-spacing: -0.005em;
      }
      .reproLab .subagent-step,
      .reproLab .telemetry-meta {
        font-size: 11.5px;
        color: var(--muted);
        margin-left: auto;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }
      .reproLab .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 600;
      }
      .reproLab .event {
        display: grid;
        grid-template-columns: 60px 8px 1fr;
        gap: 10px;
        padding: 10px 18px;
        align-items: flex-start;
        transition: background 0.12s ease;
      }
      .reproLab .event:hover {
        background: #fafafb;
      }
      .reproLab .event-time {
        font-size: 10.5px;
        color: var(--muted-2);
      }
      .reproLab .event-message {
        font-size: 11px;
        color: var(--ink-2);
        line-height: 1.5;
      }
      .reproLab .empty-activity {
        padding: 20px;
        font-size: 12px;
        color: var(--muted-2);
        text-align: center;
      }
      .reproLab .rp-pane {
        animation: rpFade 0.32s cubic-bezier(0.2, 0.7, 0.2, 1) both;
      }
      .reproLab .fadeup {
        animation: fadeup 0.5s cubic-bezier(0.2, 0.7, 0.2, 1) both;
      }
      .reproLab .wf-pop {
        animation: wfPop 0.55s cubic-bezier(0.2, 0.7, 0.2, 1) both;
      }
      .reproLab .pulse-dot {
        position: relative;
      }
      .reproLab .pulse-dot::after {
        content: "";
        position: absolute;
        inset: -3px;
        border-radius: 999px;
        background: inherit;
        animation: rl-pulse 1.6s ease-out infinite;
        opacity: 0.45;
      }
      .reproLab .wf-ring {
        animation: wfRing 1.6s ease-out infinite;
      }
      .reproLab .wf-flow {
        animation: wfFlow 0.8s linear infinite;
      }
      @keyframes rl-pulse {
        0% {
          transform: scale(1);
          opacity: 0.55;
        }
        80% {
          transform: scale(2.4);
          opacity: 0;
        }
        100% {
          opacity: 0;
        }
      }
      @keyframes fadeup {
        from {
          opacity: 0;
          transform: translateY(6px);
        }
        to {
          opacity: 1;
          transform: none;
        }
      }
      @keyframes wfPop {
        from {
          opacity: 0;
          transform: scale(0.6) translateY(8px);
        }
        to {
          opacity: 1;
          transform: scale(1) translateY(0);
        }
      }
      @keyframes wfBar {
        from {
          transform: scaleX(0);
        }
        to {
          transform: scaleX(1);
        }
      }
      @keyframes wfRing {
        0% {
          transform: scale(1);
          opacity: 0.6;
        }
        80%,
        100% {
          transform: scale(1.6);
          opacity: 0;
        }
      }
      @keyframes wfFlow {
        to {
          stroke-dashoffset: -24;
        }
      }
      @keyframes rpFade {
        from {
          opacity: 0;
          transform: translateY(6px);
        }
        to {
          opacity: 1;
          transform: none;
        }
      }
      @media (max-width: 1200px) {
        .reproLab .workflow-layout {
          flex-direction: column;
        }
        .reproLab .side-panel {
          width: 100%;
          position: static;
          height: auto;
        }
      }
      @media (max-width: 900px) {
        .reproLab .layout {
          flex-direction: column;
        }
        .reproLab .sidebar {
          width: auto;
          height: auto;
          position: static;
          padding-bottom: 0;
        }
        .reproLab .content {
          padding-top: 8px;
        }
        .reproLab .upload-shell {
          margin: 0;
          padding: 24px 0 40px;
          min-height: auto;
        }
        .reproLab .upload-title {
          font-size: 40px;
        }
      }
    `}</style>
  );
}

export function ReproLabClient({ initialRun = null }: ReproLabClientProps) {
  const [run, setRun] = useState<LiveDemoRunState | null>(initialRun);
  const [arxiv, setArxiv] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [over, setOver] = useState(false);
  const [model, setModel] = useState<DemoModelChoice>("sonnet");
  const [dashboardEvents, setDashboardEvents] = useState<DashboardLiveEvent[]>([]);
  const eventSourceRef = useRef<EventSourceLike | null>(null);
  const pollTimer = useRef<number | null>(null);
  const dashboardProjectIdRef = useRef<string | null>(null);
  const router = useRouter();
  const didAutoResume = useRef(false);

  // Keep the URL in sync with the active run so a refresh or a shared
  // link restores it. `replace` (not `push`) avoids a history pile-up;
  // `scroll: false` keeps the viewport steady.
  const setRunUrl = useCallback(
    (projectId: string | null) => {
      router.replace(projectId ? `/lab?projectId=${encodeURIComponent(projectId)}` : "/lab", {
        scroll: false
      });
    },
    [router]
  );

  // Restore an in-flight run on mount so closing the tab or refreshing
  // doesn't lose progress. Precedence: a server-provided initialRun
  // (from ?projectId=) wins; otherwise fall back to the per-browser
  // localStorage pointer. A genuinely new browser has neither and lands
  // on the fresh upload view.
  useEffect(() => {
    if (didAutoResume.current) {
      return;
    }
    didAutoResume.current = true;

    if (initialRun) {
      writeLastRun(initialRun.projectId);
      return;
    }

    const urlPid = new URLSearchParams(window.location.search).get("projectId");
    if (urlPid) {
      // ?projectId= was in the URL but the server could not load it
      // (deleted / expired) — clear the stale link and stay on upload.
      clearLastRun();
      setRunUrl(null);
      return;
    }

    const stored = readLastRun();
    if (!stored) {
      return;
    }

    void (async () => {
      try {
        const response = await fetch(`/api/demo?projectId=${encodeURIComponent(stored)}`, {
          cache: "no-store"
        });
        if (response.status === 504) {
          // Transient backend outage — keep the pointer so the next
          // visit retries instead of discarding a live run.
          return;
        }
        if (!response.ok) {
          clearLastRun();
          return;
        }
        const restored = (await response.json()) as LiveDemoRunState | null;
        if (!restored || !restored.projectId) {
          clearLastRun();
          return;
        }
        setRun(restored);
        setRunUrl(restored.projectId);
        writeLastRun(restored.projectId);
      } catch {
        // Network error — keep the pointer; the next visit retries.
      }
    })();
    // Mount-only: initialRun is server-rendered and stable for this
    // mount. The didAutoResume ref makes this idempotent under
    // StrictMode's double-invoke without stranding the in-flight fetch.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (run?.projectId !== dashboardProjectIdRef.current) {
      dashboardProjectIdRef.current = run?.projectId ?? null;
      setDashboardEvents([]);
    }

    eventSourceRef.current?.close();
    eventSourceRef.current = null;
    if (pollTimer.current) {
      window.clearTimeout(pollTimer.current);
      pollTimer.current = null;
    }

    if (!run || !["queued", "running"].includes(run.status)) {
      return;
    }

    if (typeof EventSource !== "undefined") {
      const source = new EventSource(
        `/api/demo/events?projectId=${encodeURIComponent(run.projectId)}`
      ) as unknown as EventSourceLike;
      eventSourceRef.current = source;
      source.addEventListener("run_state", (event) => {
        try {
          const next = JSON.parse((event as MessageEvent).data) as LiveDemoRunState;
          setRun((current) => coalesceRunState(current, next));
          if (next.status === "failed") {
            setError(next.error ? issueText(next.error) : "Run needs attention");
            setBusy(false);
          }
          if (next.status === "completed" || next.status === "stopped") {
            setBusy(false);
          }
        } catch {
          setError("Unable to parse live run update");
        }
      });
      source.addEventListener("agent_log", (event) => {
        try {
          const update = JSON.parse((event as MessageEvent).data) as {
            log?: string;
            text?: string;
          };
          setRun((current) =>
            current && current.projectId === run.projectId
              ? {
                  ...current,
                  log:
                    typeof update.log === "string"
                      ? update.log
                      : `${current.log}${update.text ?? ""}`
                }
              : current
          );
        } catch {
          setError("Unable to parse live log update");
        }
      });
      source.addEventListener("dashboard_event", (event) => {
        try {
          const evt = JSON.parse((event as MessageEvent).data) as DashboardLiveEvent;
          setDashboardEvents((prev) => {
            const next = [...prev, evt];
            return next.length > MAX_DASHBOARD_EVENTS
              ? next.slice(next.length - MAX_DASHBOARD_EVENTS)
              : next;
          });
        } catch {
          // Malformed dashboard events should never break the live UI.
        }
      });
      source.onerror = () => {
        source.close();
        if (eventSourceRef.current === source) {
          eventSourceRef.current = null;
        }
      };

      return () => {
        source.close();
        if (eventSourceRef.current === source) {
          eventSourceRef.current = null;
        }
      };
    }

    pollTimer.current = window.setTimeout(async () => {
      try {
        const response = await fetch(`/api/demo?projectId=${encodeURIComponent(run.projectId)}`, {
          cache: "no-store"
        });
        if (!response.ok) {
          throw new Error("Unable to refresh run");
        }
        const next = (await response.json()) as LiveDemoRunState | null;
        if (next) {
          setRun((current) => coalesceRunState(current, next));
        }
      } catch (pollError) {
        setError(pollError instanceof Error ? pollError.message : "Unable to refresh run");
      }
    }, POLL_INTERVAL_MS);

    return () => {
      if (pollTimer.current) {
        window.clearTimeout(pollTimer.current);
        pollTimer.current = null;
      }
    };
    // We intentionally depend ONLY on the stable identifiers (projectId,
    // status). Depending on the full `run` object would tear down and recreate
    // the EventSource on every run_state/agent_log event, causing a continuous
    // reconnect loop and dropping mid-flight events. Listener closures only
    // reference run.projectId for filtering, which is stable for the lifetime
    // of this effect.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [run?.projectId, run?.status]);

  async function startFixtureRun() {
    setBusy(true);
    setError(null);
    try {
      const response = await fetch(`${DEFAULT_RUN_QUERY}&model=${encodeURIComponent(model)}`, { method: "POST" });
      if (!response.ok) {
        const payload = (await response.json().catch(() => null)) as { error?: string } | null;
        throw new Error(payload?.error ?? "Unable to start run");
      }
      const next = (await response.json()) as LiveDemoRunState;
      setRun(next);
      setArxiv("");
      setRunUrl(next.projectId);
      writeLastRun(next.projectId);
    } catch (startError) {
      setError(startError instanceof Error ? startError.message : "Unable to start run");
      setBusy(false);
    }
  }

  async function startUploadedRun(file: File) {
    setBusy(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.set("mode", "sdk");
      formData.set("provider", "anthropic");
      formData.set("executionMode", "efficient");
      formData.set("sandbox", "runpod");
      formData.set("gpuMode", "auto");
      formData.set("model", model);
      formData.set("paper", file);
      const response = await fetch("/api/demo", {
        method: "POST",
        body: formData
      });
      if (!response.ok) {
        const payload = (await response.json().catch(() => null)) as { error?: string } | null;
        throw new Error(payload?.error ?? "Unable to start uploaded run");
      }
      const next = (await response.json()) as LiveDemoRunState;
      setRun(next);
      setRunUrl(next.projectId);
      writeLastRun(next.projectId);
    } catch (startError) {
      setError(startError instanceof Error ? startError.message : "Unable to start uploaded run");
      setBusy(false);
    }
  }

  async function clearRun() {
    setBusy(true);
    try {
      if (run) {
        await fetch(`/api/demo?projectId=${encodeURIComponent(run.projectId)}`, {
          method: "DELETE"
        }).catch(() => null);
      }
    } finally {
      resetToUpload();
    }
  }

  function resetToUpload() {
    setRun(null);
    setArxiv("");
    setBusy(false);
    setError(null);
    setOver(false);
    eventSourceRef.current?.close();
    eventSourceRef.current = null;
    if (pollTimer.current) {
      window.clearTimeout(pollTimer.current);
      pollTimer.current = null;
    }
    clearLastRun();
    setRunUrl(null);
  }

  return (
    <div className="reproLab">
      <PrototypeStyles />
      <div className="layout">
        <Sidebar active="lab" onBrandClick={resetToUpload} />
        <main className="content">
          {run ? (
            <WorkflowView
              run={run}
              onClear={clearRun}
              busy={busy}
              error={error}
              dashboardEvents={dashboardEvents}
            />
          ) : (
            <UploadView
              arxiv={arxiv}
              busy={busy}
              error={error}
              model={model}
              onArxivChange={setArxiv}
              onArxivSubmit={() => void startFixtureRun()}
              onFileSelected={(file) => void startUploadedRun(file)}
              onModelChange={setModel}
              over={over}
              setOver={setOver}
            />
          )}
        </main>
      </div>
    </div>
  );
}
