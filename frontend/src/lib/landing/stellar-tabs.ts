import type { LucideIcon } from "lucide-react";
import { BarChart3, BookOpen, Rocket, ShieldCheck } from "lucide-react";

export type StellarTabId = "analyse" | "train" | "testing" | "deploy";

export type StellarTab = {
  id: StellarTabId;
  label: string;
  icon: LucideIcon;
};

export type StellarMetric = {
  label: string;
  value: string;
};

export type StellarChecklistItem = {
  label: string;
  complete: boolean;
};

export type StellarOverlay = {
  eyebrow: string;
  title: string;
  description: string;
  accentClass: string;
  progress?: number;
  metrics?: StellarMetric[];
  steps?: string[];
  checklist?: StellarChecklistItem[];
  successLabel?: string;
  ctaLabel?: string;
};

export const stellarTabs: StellarTab[] = [
  {
    id: "analyse",
    label: "Ingest",
    icon: BarChart3
  },
  {
    id: "train",
    label: "Reproduce",
    icon: BookOpen
  },
  {
    id: "testing",
    label: "Audit",
    icon: ShieldCheck
  },
  {
    id: "deploy",
    label: "Report",
    icon: Rocket
  }
];

export const stellarVideoSource =
  "https://d8j0ntlcm91z4.cloudfront.net/user_38xzZboKViGWJOttwIXH07lWA1P/hf_20260319_165750_358b1e72-c921-48b7-aaac-f200994f32fb.mp4";

export const stellarOverlays: Record<StellarTabId, StellarOverlay> = {
  analyse: {
    eyebrow: "Ingest",
    title: "Read The Paper First",
    description:
      "ReproLab parses the paper, extracts claims, identifies benchmarks, and builds a structured reproduction plan before any code runs.",
    accentClass: "bg-violet-500",
    progress: 25,
    steps: ["Paper parsing", "Claim extraction", "Benchmark mapping", "Execution plan"]
  },
  train: {
    eyebrow: "Reproduce",
    title: "Build And Run The Stack",
    description:
      "The system recreates the environment, generates the baseline implementation, and launches controlled runs against the paper setup.",
    accentClass: "bg-orange-400",
    progress: 67,
    metrics: [
      { label: "Packages", value: "38" },
      { label: "Seeds", value: "5" },
      { label: "Runtime", value: "2.4h" },
      { label: "GPU Load", value: "71%" }
    ]
  },
  testing: {
    eyebrow: "Audit",
    title: "Verify Every Claimed Result",
    description:
      "Hermes checks reported deltas against actual runs, flags regressions, and separates faithful reproduction from optimistic storytelling.",
    accentClass: "bg-emerald-500",
    successLabel: "14 claims verified",
    metrics: [
      { label: "Accepted", value: "11" },
      { label: "Flagged", value: "2" },
      { label: "Regressed", value: "1" },
      { label: "Re-run", value: "5 seeds" }
    ]
  },
  deploy: {
    eyebrow: "Report",
    title: "Ship The Reproducibility Packet",
    description:
      "Export the final verdict with logs, manifests, checkpoints, and a clean audit trail your team can actually trust.",
    accentClass: "bg-sky-500",
    checklist: [
      { label: "Manifest compiled", complete: true },
      { label: "Run logs attached", complete: true },
      { label: "Checkpoint hashes saved", complete: true },
      { label: "Final PDF exported", complete: false }
    ],
    ctaLabel: "Export Report"
  }
};

export const stellarLogos = [
  "INTERSCOPE",
  "SPOTIFY",
  "Nexera",
  "M3",
  "LAURA COLE",
  "vertex"
] as const;
