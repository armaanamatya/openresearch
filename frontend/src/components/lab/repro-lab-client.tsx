"use client";

import { useEffect, useMemo, useRef, useState } from "react";

type RunState = {
  t: number;
  title: string;
};

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
  accentClass?: string;
  detail?: string;
  icon: keyof typeof ICONS;
  id: string;
  role: string;
  step: string;
  tone: Tone;
  x: number;
  y: number;
  agent: string;
};

type Phase = {
  dur: number;
  ids: string[];
};

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
    step: "Source PDF",
    icon: "doc",
    tone: "neutral",
    role: "Diffusion Policy - arXiv:2303.04137"
  },
  {
    id: "read",
    x: 260,
    y: 310,
    agent: "Reader",
    step: "Paper understanding",
    icon: "brain",
    tone: "info",
    role: "Extracts claims, metrics, datasets",
    detail:
      "Parses the paper structure, extracts verifiable claims, and cross-references citations to the reported metrics."
  },
  {
    id: "env",
    x: 500,
    y: 200,
    agent: "Forge",
    step: "Environment",
    icon: "beaker",
    tone: "info",
    role: "Reproduces the build environment",
    detail:
      "Resolves dependencies, builds an isolated container, and applies compatibility patches needed to run the code."
  },
  {
    id: "plan",
    x: 500,
    y: 420,
    agent: "Architect",
    step: "Baseline plan",
    icon: "doc",
    tone: "info",
    role: "12-step reproduction contract",
    detail:
      "Drafts the experiment plan that maps each paper claim to a concrete verification step before implementation."
  },
  {
    id: "impl",
    x: 740,
    y: 310,
    agent: "Builder",
    step: "Baseline implementation",
    icon: "zap",
    tone: "accent",
    role: "Generates code and runs the baseline",
    detail:
      "Generates the reproduction code, launches the baseline training runs, and compares the first results against the paper."
  },
  {
    id: "opt",
    x: 1020,
    y: 60,
    agent: "Vesta",
    step: "Optimizer pathway",
    icon: "spark",
    tone: "info",
    role: "AdamW -> Lion and LR cosine",
    detail:
      "Explores alternative optimizers and learning-rate schedules, tracking gains against the reproduced baseline."
  },
  {
    id: "bb",
    x: 1020,
    y: 200,
    agent: "Athena",
    step: "Backbone swap",
    icon: "copy",
    tone: "info",
    role: "ResNet-18 -> DINOv2",
    detail:
      "Swaps the visual encoder and compares frozen versus fine-tuned backbones on the paper benchmark."
  },
  {
    id: "aug",
    x: 1020,
    y: 340,
    agent: "Orion",
    step: "Augmentation pathway",
    icon: "graph",
    tone: "info",
    role: "RandAug plus temporal jitter",
    detail:
      "Tests augmentation strategies and tracks whether robustness improvements hold up on the evaluation tasks."
  },
  {
    id: "hor",
    x: 1020,
    y: 480,
    agent: "Lyra",
    step: "Horizon extension",
    icon: "flag",
    tone: "info",
    role: "16 -> 24 step planning horizon",
    detail:
      "Extends the planning horizon and reruns the evaluation to measure the tradeoff between accuracy and runtime."
  },
  {
    id: "div",
    x: 1020,
    y: 620,
    agent: "Pyxis",
    step: "DDIM step ablation",
    icon: "compute",
    tone: "info",
    role: "12 -> 16 -> 20 inference steps",
    detail:
      "Sweeps DDIM inference steps and logs accuracy deltas against wall-clock cost for each setting."
  },
  {
    id: "audit",
    x: 1300,
    y: 310,
    agent: "Hermes",
    step: "Result audit",
    icon: "shield",
    tone: "hermes",
    role: "Verification and merge gate",
    detail:
      "Cross-checks every sub-agent claim against the reproduced runs, accepts faithful improvements, and rejects regressions."
  },
  {
    id: "report",
    x: 1540,
    y: 310,
    agent: "Scribe",
    step: "Final report",
    icon: "flag",
    tone: "neutral",
    role: "Manifest, report, and checkpoints",
    detail:
      "Compiles the reproducibility manifest, checkpoint hashes, report, and the full Hermes audit trail."
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

const PHASES: Phase[] = [
  { ids: ["src"], dur: 1400 },
  { ids: ["read"], dur: 3200 },
  { ids: ["env", "plan"], dur: 3400 },
  { ids: ["impl"], dur: 3800 },
  { ids: ["opt", "bb", "aug", "hor", "div"], dur: 5200 },
  { ids: ["audit"], dur: 3400 },
  { ids: ["report"], dur: 2200 }
];

const PER_AGENT_LOG: Record<string, Array<{ msg: string; t: string }>> = {
  src: [{ t: "09:32:00", msg: "PDF received - 4.2 MB - 14 pages" }],
  read: [
    { t: "09:32:18", msg: "Parsing structure - 8 sections" },
    { t: "09:32:42", msg: "Extracted 17 claims" },
    { t: "09:33:05", msg: "4 reported metrics resolved" }
  ],
  env: [
    { t: "09:33:55", msg: "py3.10 - cu12.1 - resolving 38 deps" },
    { t: "09:34:30", msg: "Patch applied: torch.compile fix" },
    { t: "09:35:10", msg: "Container ready - 2.1 GB" }
  ],
  plan: [
    { t: "09:34:00", msg: "Drafted 12-step contract" },
    { t: "09:34:48", msg: "Mapped claims to experiments" },
    { t: "09:35:20", msg: "Plan v3 frozen" }
  ],
  impl: [
    { t: "09:35:30", msg: "Generated 14 files - 1,420 LOC" },
    { t: "09:36:18", msg: "Type-check passed" },
    { t: "09:37:00", msg: "Seed 1/5 - loss 0.052" },
    { t: "09:38:14", msg: "Seed 5/5 - within +/-1.5% of paper" }
  ],
  opt: [
    { t: "09:38:30", msg: "Spawned - LR sweep started" },
    { t: "09:39:12", msg: "Lion beta1=0.95 - ckpt-128" },
    { t: "09:40:00", msg: "Delta +0.8% on Push-T" }
  ],
  bb: [
    { t: "09:38:30", msg: "Spawned - downloading DINOv2 weights" },
    { t: "09:39:48", msg: "Frozen backbone - ablation A" },
    { t: "09:40:33", msg: "Delta +1.4% (frozen)" }
  ],
  aug: [
    { t: "09:38:30", msg: "Spawned - RandAug N=2 M=9" },
    { t: "09:39:20", msg: "Temporal jitter tau=0.1" },
    { t: "09:40:11", msg: "Delta -0.3% - regression" }
  ],
  hor: [
    { t: "09:38:30", msg: "Spawned - horizon=24" },
    { t: "09:39:50", msg: "Re-eval Push-T - 5 seeds" },
    { t: "09:40:42", msg: "Delta +0.5%" }
  ],
  div: [
    { t: "09:38:30", msg: "Spawned - DDIM sweep" },
    { t: "09:39:35", msg: "steps=12 - accuracy down, cost down 2.4x" },
    { t: "09:40:25", msg: "steps=20 - Delta +0.6%" }
  ],
  audit: [
    { t: "09:41:48", msg: "Cross-checking 5 deltas" },
    { t: "09:41:58", msg: "Vesta accepted (+0.8%)" },
    { t: "09:42:11", msg: "Athena accepted (+1.4%)" },
    { t: "09:42:20", msg: "Orion stopped - regression" }
  ],
  report: [
    { t: "09:42:45", msg: "Compiling manifest" },
    { t: "09:42:58", msg: "PDF rendering - 12 pages" }
  ]
};

const GLOBAL_LOG = [
  { t: "09:42:11", who: "Hermes", msg: "Result hashes match across 5 seeds" },
  { t: "09:41:58", who: "Vesta", msg: "Lion optimizer - Delta +0.8% accepted" },
  { t: "09:41:12", who: "Builder", msg: "checkpoint saved - ckpt-540" },
  { t: "09:40:47", who: "Pyxis", msg: "DDIM 12 -> 20 steps - +0.6%" },
  { t: "09:39:20", who: "Hermes", msg: "Orion regressed - stopped" },
  { t: "09:38:02", who: "Forge", msg: "DINOv2 weights downloaded" },
  { t: "09:36:41", who: "Architect", msg: "plan v3 verified - 12 steps" },
  { t: "09:35:10", who: "Builder", msg: "epoch 100/300 - loss 0.0521" },
  { t: "09:33:55", who: "System", msg: "compute pool - 7 -> 11 A100s" },
  { t: "09:32:18", who: "Reader", msg: "17 claims extracted - 4 metrics" }
];

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
  shield: icon(<path d="M9 2l5.5 2v5c0 3.5-2.5 6.5-5.5 7.5C6 15.5 3.5 12.5 3.5 9V4z" />),
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

function Sidebar({ active }: { active: string }) {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <aside className={`sidebar${collapsed ? " collapsed" : ""}`}>
      <button className="sb-toggle" onClick={() => setCollapsed((value) => !value)} type="button" aria-label="Toggle sidebar">
        <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M10 4l-4 4 4 4" />
        </svg>
      </button>
      <div className="brand-row">
        <span className="nav-icon">{ICONS.logo}</span>
        <span className="brand-text">ReproLab</span>
      </div>
      <div className="dotted" />
      {NAV.map((item) => (
        <a key={item.id} href={item.href} data-label={item.label} className={`navitem${active === item.id ? " active" : ""}`}>
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
        { t: "PerAct", s: "failed" }
      ].map((item) => (
        <a key={item.t} href="/lab" className="navitem navitem-small">
          <span
            className="nav-icon nav-status-dot"
            style={{
              background:
                item.s === "running" ? "var(--accent)" : item.s === "failed" ? "var(--err)" : "var(--muted-2)"
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
      {status}
    </span>
  );
}

function UploadView({ onLaunch }: { onLaunch: (run: RunState) => void }) {
  const [arxiv, setArxiv] = useState("");
  const [over, setOver] = useState(false);
  const fileInput = useRef<HTMLInputElement | null>(null);

  function launch(label?: string) {
    const title =
      label ?? (arxiv ? arxiv.replace(/^https?:\/\//, "").slice(0, 60) : "diffusion_policy.pdf");
    onLaunch({ title, t: Date.now() });
  }

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
          launch(file?.name);
        }}
        onClick={() => fileInput.current?.click()}
      >
        <input
          ref={fileInput}
          type="file"
          accept=".pdf"
          className="hidden-input"
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) {
              launch(file.name);
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
          if (arxiv.length >= 8) {
            launch();
          }
        }}
        onClick={(event) => event.stopPropagation()}
      >
        <span className="mono upload-prefix">https://</span>
        <input
          value={arxiv}
          onChange={(event) => setArxiv(event.target.value)}
          placeholder="arxiv.org/abs/2303.04137"
          className="upload-text-input mono"
        />
        <button type="submit" disabled={arxiv.length < 8} className="begin-button">
          Begin -&gt;
        </button>
      </form>
    </div>
  );
}

function useWorkflowProgress() {
  const [phaseIdx, setPhaseIdx] = useState(0);
  const [playing, setPlaying] = useState(true);

  useEffect(() => {
    if (!playing || phaseIdx >= PHASES.length) {
      return;
    }

    const timer = window.setTimeout(() => setPhaseIdx((value) => value + 1), PHASES[phaseIdx].dur);
    return () => window.clearTimeout(timer);
  }, [phaseIdx, playing]);

  const stateMap = useMemo(() => {
    const stateById: Record<string, NodeState> = {};
    for (let phase = 0; phase < PHASES.length; phase += 1) {
      const state: NodeState = phase < phaseIdx ? "done" : phase === phaseIdx ? "running" : "upcoming";
      for (const id of PHASES[phase].ids) {
        stateById[id] = state;
      }
    }
    return stateById;
  }, [phaseIdx]);

  return {
    finished: phaseIdx >= PHASES.length,
    phaseIdx,
    playing,
    reset: () => setPhaseIdx(0),
    setPlaying,
    stateMap
  };
}

function NodeCard({
  node,
  onClick,
  popDelay,
  selected,
  state
}: {
  node: WorkflowNode;
  onClick: () => void;
  popDelay: number;
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
        animationDelay: `${popDelay}ms`,
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
              <path d="M3 8.5l3 3 7-7" stroke="currentColor" strokeWidth="2.5" fill="none" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </div>
        ) : null}
      </div>
      {showProgress ? (
        <div className="node-progress">
          <div className="wf-bar" style={{ background: node.tone === "hermes" ? "var(--hermes)" : "var(--accent)" }} />
        </div>
      ) : null}
    </div>
  );
}

function bezier(a: WorkflowNode, b: WorkflowNode) {
  const x1 = a.x + NODE_W;
  const y1 = a.y + NODE_H / 2;
  const x2 = b.x;
  const y2 = b.y + NODE_H / 2;
  const cx1 = x1 + Math.max(40, (x2 - x1) * 0.45);
  const cx2 = x2 - Math.max(40, (x2 - x1) * 0.45);
  return `M ${x1} ${y1} C ${cx1} ${y1}, ${cx2} ${y2}, ${x2} ${y2}`;
}

function Edge({
  from,
  state,
  to
}: {
  from: WorkflowNode;
  state: "active" | "done" | "upcoming";
  to: WorkflowNode;
}) {
  const d = bezier(from, to);
  let color = "var(--line-2)";
  let strokeWidth = 1.5;
  let opacity = 1;

  if (state === "upcoming") {
    opacity = 0.5;
  }
  if (state === "done") {
    color = "var(--ink-2)";
    strokeWidth = 1.6;
  }
  if (state === "active") {
    color = "var(--accent)";
    strokeWidth = 2;
  }

  return (
    <g style={{ opacity }}>
      <path d={d} fill="none" stroke={color} strokeWidth={strokeWidth} strokeLinecap="round" />
      {state === "active" ? (
        <path
          d={d}
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
}

function Canvas({
  onSelect,
  phaseIdx,
  selectedId,
  stateMap
}: {
  onSelect: (id: string | null) => void;
  phaseIdx: number;
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
          return <Edge key={`${fromId}-${toId}`} from={from} to={to} state={edgeState(fromId, toId)} />;
        })}
      </svg>
      {phaseIdx === 4 ? (
        <div className="spawn-banner wf-pop">
          <span className="pulse-dot spawn-dot" />
          5 sub-agents spawning
        </div>
      ) : null}
      {NODES.map((node) => (
        <NodeCard
          key={node.id}
          node={node}
          state={stateMap[node.id]}
          selected={selectedId === node.id}
          onClick={() => (stateMap[node.id] === "upcoming" ? undefined : onSelect(node.id === selectedId ? null : node.id))}
          popDelay={(PHASES.findIndex((phase) => phase.ids.includes(node.id)) % 2) * 60}
        />
      ))}
    </div>
  );
}

function AgentInfo({ node, state }: { node: WorkflowNode; state: NodeState }) {
  const tones = {
    info: { icBg: "var(--info-soft)", icFg: "#3b48d1" },
    accent: { icBg: "var(--accent-soft)", icFg: "var(--accent-ink)" },
    hermes: { icBg: "var(--hermes-soft)", icFg: "var(--hermes)" },
    neutral: { icBg: "var(--chip)", icFg: "var(--muted)" }
  } as const;
  const tone = tones[node.tone];
  const status: Status =
    state === "done" ? "completed" : state === "running" ? (node.tone === "hermes" ? "auditing" : "running") : "queued";

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
      {node.detail ? <div className="agent-detail">{node.detail}</div> : null}
      {state === "running" ? (
        <div className="agent-section">
          <div className="eyebrow">Progress</div>
          <div className="agent-progress">
            <div className="wf-bar" style={{ background: node.tone === "hermes" ? "var(--hermes)" : "var(--accent)" }} />
          </div>
        </div>
      ) : null}
    </div>
  );
}

function RunOverview({ finished, stateMap }: { finished: boolean; stateMap: Record<string, NodeState> }) {
  const totals = NODES.reduce(
    (acc, node) => {
      const state = stateMap[node.id];
      acc[state] += 1;
      return acc;
    },
    { done: 0, running: 0, upcoming: 0 }
  );

  const subagents = ["opt", "bb", "aug", "hor", "div"].map((id) => {
    const node = NODES.find((entry) => entry.id === id)!;
    return { id, node, state: stateMap[id] };
  });

  return (
    <div>
      <div className="eyebrow">Run</div>
      <div className="overview-title">{finished ? "Run complete" : "Reproducing baseline"}</div>
      <div className="overview-copy">Click any agent in the graph to inspect its work.</div>
      <div className="overview-grid">
        <Stat label="Done" value={totals.done} dot="var(--ink)" />
        <Stat label="Running" value={totals.running} dot="var(--accent)" pulse />
        <Stat label="Queued" value={totals.upcoming} dot="var(--line-2)" />
        <Stat label="Agents" value={NODES.length} dot="var(--muted-2)" />
      </div>
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
              <span className="subagent-step">{item.node.step.replace(" pathway", "")}</span>
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
  finished,
  selectedId,
  stateMap
}: {
  finished: boolean;
  selectedId: string | null;
  stateMap: Record<string, NodeState>;
}) {
  const selected = selectedId ? NODES.find((node) => node.id === selectedId) ?? null : null;
  const state = selected ? stateMap[selected.id] : null;
  const agentLog = selected ? [...(PER_AGENT_LOG[selected.id] ?? [])].reverse() : null;

  return (
    <aside className="card side-panel">
      <div className="side-panel-top">
        <div key={selectedId ?? "overview"} className="rp-pane side-panel-scroll">
          {selected && state ? <AgentInfo node={selected} state={state} /> : <RunOverview finished={finished} stateMap={stateMap} />}
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
        <div className="side-panel-scroll" key={`act-${selectedId ?? "all"}`}>
          {selected ? (
            agentLog && agentLog.length > 0 ? (
              agentLog.map((entry, index) => (
                <div key={`${entry.t}-${index}`} className="event fadeup" style={{ animationDelay: `${index * 40}ms` }}>
                  <span className="mono event-time">{entry.t}</span>
                  <span className="event-dot" />
                  <div className="mono event-message">{entry.msg}</div>
                </div>
              ))
            ) : (
              <div className="empty-activity">Waiting for activity...</div>
            )
          ) : (
            GLOBAL_LOG.map((entry, index) => (
              <div key={`${entry.t}-${entry.who}`} className="event fadeup" style={{ animationDelay: `${index * 30}ms` }}>
                <span className="mono event-time">{entry.t}</span>
                <span className="event-dot" />
                <div>
                  <div className="event-who">{entry.who}</div>
                  <div className="mono event-copy">{entry.msg}</div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </aside>
  );
}

function PanCanvas({
  onSelect,
  phaseIdx,
  selectedId,
  stateMap
}: {
  onSelect: (id: string | null) => void;
  phaseIdx: number;
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
        phaseIdx={phaseIdx}
        onSelect={(id) => {
          if (!dragRef.current.moved) {
            onSelect(id);
          }
        }}
      />
    </div>
  );
}

function WorkflowView({ onClear, run }: { onClear: () => void; run: RunState }) {
  const { finished, phaseIdx, playing, reset, setPlaying, stateMap } = useWorkflowProgress();
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const doneCount = NODES.filter((node) => stateMap[node.id] === "done").length;

  return (
    <>
      <div className="workflow-header">
        <div>
          <div className="eyebrow">workflow - run_8a4f</div>
          <h1 className="h1 workflow-title">{run.title}</h1>
          <div className="workflow-meta">
            <StatusPill status={finished ? "completed" : "running"} />
            <span className="workflow-meta-sep">.</span>
            <span className="mono">{doneCount}/{NODES.length} agents complete</span>
          </div>
        </div>
        <div className="workflow-actions">
          <button className="btn btn-sm" onClick={onClear} type="button">
            {ICONS.upload} New paper
          </button>
          {finished ? (
            <button className="btn btn-sm" onClick={reset} type="button">
              {ICONS.spark} Replay
            </button>
          ) : (
            <button className="btn btn-sm" onClick={() => setPlaying((value) => !value)} type="button">
              {playing ? (
                <>
                  {ICONS.pause} Pause
                </>
              ) : (
                <>
                  {ICONS.play} Play
                </>
              )}
            </button>
          )}
        </div>
      </div>
      <div className="workflow-layout">
        <div className="canvas-wrap">
          <PanCanvas stateMap={stateMap} selectedId={selectedId} onSelect={setSelectedId} phaseIdx={phaseIdx} />
        </div>
        <RightPanel selectedId={selectedId} stateMap={stateMap} finished={finished} />
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
        transition: transform 0.25s ease, color 0.15s ease, border-color 0.15s ease;
      }
      .reproLab .sidebar.collapsed .sb-toggle {
        transform: rotate(180deg);
      }
      .reproLab .brand-row {
        display: flex;
        align-items: center;
        gap: 11px;
        padding: 4px 11px 12px;
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
        transition: all 0.15s;
      }
      .reproLab .begin-button:enabled {
        background: var(--ink);
        color: #fff;
        cursor: pointer;
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
      .reproLab .spawn-banner {
        position: absolute;
        left: 1020px;
        top: 14px;
        width: 200px;
        text-align: center;
        font-size: 10.5px;
        color: var(--accent-ink);
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        background: var(--accent-soft);
        padding: 4px 10px;
        border-radius: 999px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 6px;
      }
      .reproLab .spawn-dot {
        width: 6px;
        height: 6px;
        border-radius: 999px;
        background: var(--accent);
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
      .reproLab .subagent-list {
        display: flex;
        flex-direction: column;
        gap: 5px;
      }
      .reproLab .subagent-row {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 6px 10px;
        border-radius: 8px;
      }
      .reproLab .subagent-name {
        font-size: 12px;
        font-weight: 600;
        letter-spacing: -0.005em;
      }
      .reproLab .subagent-step {
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
      .reproLab .event-who {
        font-size: 11.5px;
        font-weight: 600;
        letter-spacing: -0.005em;
      }
      .reproLab .event-copy {
        font-size: 11px;
        color: var(--muted);
        line-height: 1.4;
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

export function ReproLabClient() {
  const [run, setRun] = useState<RunState | null>(null);

  useEffect(() => {
    try {
      const saved = window.localStorage.getItem("rl-run");
      if (saved) {
        setRun(JSON.parse(saved) as RunState);
      }
    } catch {
      // Ignore client storage read failures.
    }
  }, []);

  function launch(nextRun: RunState) {
    setRun(nextRun);
    try {
      window.localStorage.setItem("rl-run", JSON.stringify(nextRun));
    } catch {
      // Ignore client storage write failures.
    }
  }

  function clear() {
    setRun(null);
    try {
      window.localStorage.removeItem("rl-run");
    } catch {
      // Ignore client storage write failures.
    }
  }

  return (
    <div className="reproLab">
      <PrototypeStyles />
      <div className="layout">
        <Sidebar active="lab" />
        <main className="content">{run ? <WorkflowView run={run} onClear={clear} /> : <UploadView onLaunch={launch} />}</main>
      </div>
    </div>
  );
}
