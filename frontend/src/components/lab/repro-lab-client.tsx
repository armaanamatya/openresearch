"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import type { LiveDemoRunState } from "@/lib/demo/demo-run-types";

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
  | "failed"
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
    case "failed":
      return { bg: "var(--err-soft)", fg: "var(--err)", dot: "var(--err)", pulse: false };
    default:
      return { bg: "var(--chip)", fg: "var(--muted)", dot: "var(--muted-2)", pulse: false };
  }
}

function sourceTitle(run: LiveDemoRunState) {
  return run.sourceLabel || run.projectId;
}

function deriveStage(run: LiveDemoRunState | null): string | null {
  return run?.payload?.summary.stage ?? null;
}

function stateMapForRun(run: LiveDemoRunState | null): Record<string, NodeState> {
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

  mark(["src"], "done");

  if (!stage) {
    mark(["read"], status === "failed" ? "done" : "running");
    return map;
  }

  if (stage === "ingested") {
    mark(["read"], "running");
    return map;
  }

  if (["plan_created", "gate_1_passed"].includes(stage)) {
    mark(["read"], "done");
    mark(["env", "plan"], stage === "gate_1_passed" ? "done" : "running");
    return map;
  }

  if (["baseline_implemented", "baseline_run", "gate_2_passed"].includes(stage)) {
    mark(["read", "env", "plan"], "done");
    mark(["impl"], stage === "gate_2_passed" ? "done" : "running");
    return map;
  }

  if (["improvements_selected", "improvements_run", "gate_3_passed"].includes(stage)) {
    mark(["read", "env", "plan", "impl"], "done");
    mark(
      ["opt", "bb", "aug", "hor", "div"],
      stage === "gate_3_passed" ? "done" : "running"
    );
    return map;
  }

  if (stage === "research_map_generated") {
    mark(["read", "env", "plan", "impl", "opt", "bb", "aug", "hor", "div"], "done");
    mark(["audit"], "running");
    return map;
  }

  if (stage === "complete" || status === "completed") {
    mark(NODES.map((node) => node.id), "done");
    return map;
  }

  mark(["read"], "running");
  return map;
}

/** Markers that identify meaningful pipeline log lines. */
const LOG_MARKERS = /^\[|^={2,}|^\s*>|^\s*>>|^\s*X |^\s*\|\||^Execution|^Starting|^Workspace|^Sandbox|^Pipeline|^Agent |^Step |^Gate |^Checkpoint|WARNING|ERROR|FAILED|STOPPED|completed|queued|running/i;

function parseLogEntries(run: LiveDemoRunState | null) {
  if (!run?.log) {
    return [];
  }

  return run.log
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line && LOG_MARKERS.test(line))
    .slice(-20)
    .reverse()
    .map((line, index) => ({
      id: `${run.projectId}-${index}`,
      time: run.updatedAt ? new Date(run.updatedAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "--:--",
      msg: line
    }));
}

function telemetryForSelectedNode(run: LiveDemoRunState | null, selectedId: string | null) {
  if (!run?.telemetry?.length || !selectedId) {
    return [];
  }

  const agentMatchers: Record<string, string[]> = {
    read: ["paper-understanding"],
    env: ["environment-detective"],
    plan: ["paper-understanding", "root-orchestrator"],
    impl: ["baseline-implementation", "experiment-runner"],
    audit: ["supervisor-verifier"]
  };

  const matches = agentMatchers[selectedId] ?? [];
  return run.telemetry
    .filter((record) => matches.some((match) => record.agent_id?.includes(match)))
    .slice(-6)
    .reverse();
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
      <span className="navitem navitem-small" style={{ color: "var(--muted)", cursor: "default" }}>
        <span className="nav-label">No recent runs</span>
      </span>
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
      {status}
    </span>
  );
}

function UploadView({
  arxiv,
  busy,
  error,
  onArxivChange,
  onArxivSubmit,
  onFileSelected,
  over,
  setOver
}: {
  arxiv: string;
  busy: boolean;
  error: string | null;
  onArxivChange: (value: string) => void;
  onArxivSubmit: () => void;
  onFileSelected: (file: File) => void;
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
  selectedId,
  stateMap
}: {
  onSelect: (id: string | null) => void;
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
    </div>
  );
}

function PanCanvas({
  onSelect,
  selectedId,
  stateMap
}: {
  onSelect: (id: string | null) => void;
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
  logEntries,
  node,
  run,
  state,
  telemetry
}: {
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
    run.status === "failed"
      ? "failed"
      : state === "done"
        ? "completed"
        : state === "running"
          ? node.tone === "hermes"
            ? "auditing"
            : "running"
          : "queued";

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
          <div className="agent-detail">{logEntries[0].msg}</div>
        </div>
      ) : null}
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
        ? "Run failed"
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
          <div className="agent-detail">{error ?? run.error}</div>
        </div>
      ) : null}
      {logEntries.length > 0 ? (
        <div className="agent-section">
          <div className="eyebrow">Latest backend log</div>
          <div className="agent-detail">{logEntries[0].msg}</div>
        </div>
      ) : null}
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

  return (
    <aside className="card side-panel">
      <div className="side-panel-top">
        <div key={selectedId ?? "overview"} className="rp-pane side-panel-scroll">
          {selected ? (
            <AgentInfo
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

function WorkflowView({
  busy,
  error,
  onClear,
  run
}: {
  busy: boolean;
  error: string | null;
  onClear: () => Promise<void>;
  run: LiveDemoRunState;
}) {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const stateMap = useMemo(() => stateMapForRun(run), [run]);
  const doneCount = NODES.filter((node) => stateMap[node.id] === "done").length;
  const liveStatus: Status =
    run.status === "failed"
      ? "failed"
      : run.status === "completed"
        ? "completed"
        : deriveStage(run) === "research_map_generated"
          ? "auditing"
          : run.status === "stopped"
            ? "stopped"
            : "running";

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
          <button className="btn btn-sm" onClick={() => void onClear()} type="button" disabled={busy}>
            {busy ? "Stopping..." : "New paper"}
          </button>
        </div>
      </div>
      <div className="workflow-layout">
        <div className="canvas-wrap">
          <PanCanvas stateMap={stateMap} selectedId={selectedId} onSelect={setSelectedId} />
        </div>
        <RightPanel run={run} selectedId={selectedId} stateMap={stateMap} error={error} />
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
      .reproLab .workflow-layout {
        display: flex;
        gap: 16px;
        align-items: flex-start;
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
  const [showUploadOnly, setShowUploadOnly] = useState(false);
  const eventSourceRef = useRef<EventSourceLike | null>(null);
  const pollTimer = useRef<number | null>(null);

  useEffect(() => {
    if (run || showUploadOnly) {
      return;
    }

    let cancelled = false;

    async function loadLatest() {
      try {
        const response = await fetch("/api/demo", { cache: "no-store" });
        if (!response.ok || cancelled) {
          return;
        }
        const next = (await response.json()) as LiveDemoRunState | null;
        // Only resume runs that are still active — don't auto-load
        // completed/failed/stopped runs from previous sessions.
        if (next && !cancelled && (next.status === "queued" || next.status === "running")) {
          setRun(next);
        }
      } catch {
        // Ignore initial load failures and let the user start a new run.
      }
    }

    void loadLatest();
    return () => {
      cancelled = true;
    };
  }, [run, showUploadOnly]);

  useEffect(() => {
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
          setRun(next);
          if (next.status === "failed") {
            setError(next.error ?? "Run failed");
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
          setRun(next);
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
  }, [run]);

  async function startFixtureRun() {
    setBusy(true);
    setError(null);
    setShowUploadOnly(false);
    try {
      const response = await fetch(DEFAULT_RUN_QUERY, { method: "POST" });
      if (!response.ok) {
        const payload = (await response.json().catch(() => null)) as { error?: string } | null;
        throw new Error(payload?.error ?? "Unable to start run");
      }
      const next = (await response.json()) as LiveDemoRunState;
      setRun(next);
      setArxiv("");
    } catch (startError) {
      setError(startError instanceof Error ? startError.message : "Unable to start run");
      setBusy(false);
    }
  }

  async function startUploadedRun(file: File) {
    setBusy(true);
    setError(null);
    setShowUploadOnly(false);
    try {
      const formData = new FormData();
      formData.set("mode", "sdk");
      formData.set("provider", "anthropic");
      formData.set("executionMode", "efficient");
      formData.set("sandbox", "runpod");
      formData.set("gpuMode", "auto");
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
    setShowUploadOnly(true);
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
  }

  return (
    <div className="reproLab">
      <PrototypeStyles />
      <div className="layout">
        <Sidebar active="lab" onBrandClick={resetToUpload} />
        <main className="content">
          {run ? (
            <WorkflowView run={run} onClear={clearRun} busy={busy} error={error} />
          ) : (
            <UploadView
              arxiv={arxiv}
              busy={busy}
              error={error}
              onArxivChange={setArxiv}
              onArxivSubmit={() => void startFixtureRun()}
              onFileSelected={(file) => void startUploadedRun(file)}
              over={over}
              setOver={setOver}
            />
          )}
        </main>
      </div>
    </div>
  );
}
